"""Enhanced HART Transformer with VAR-style caching mechanism.

This file provides enhanced HART transformer that closely follows VAR's caching implementation.
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union, List
import argparse

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, PreTrainedModel

from hart.modules.diffusion.diffloss import DiffLoss
from hart.modules.models.autoencoder import (
    HARTAutoEncoder,
    HARTAutoEncoderWithDisc,
    HARTHybridQuantizer,
)
from hart.modules.models.transformer.configuration import HARTForT2IConfig
from hart.modules.networks.basic_hart_enhanced import (
    AdaLNBeforeHead,
    AdaLNSelfAttnEnhanced,
    CacheConfig,
    LlamaRMSNormFused,
    TimestepEmbedder,
)
from hart.modules.networks.utils import (
    gumbel_softmax_with_rng,
    sample_with_top_k_top_p_,
)
from hart.utils import get_device


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking


class CopyableGenerator(torch.Generator):
    def __deepcopy__(self, memo):
        new_generator = CopyableGenerator(device=self.device)
        new_generator.set_state(self.get_state())
        return new_generator


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class HARTForT2IEnhanced(PreTrainedModel):
    """Enhanced HART Transformer with VAR-style caching"""
    config_class = HARTForT2IConfig

    def __init__(self, config: HARTForT2IConfig):
        super().__init__(config)
        self.supports_gradient_checkpointing = True
        
        # 0. hyperparameters
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        vae_path = config.vae_path
        if vae_path is None:
            vae_path = os.path.join(
                os.path.dirname(config._name_or_path.rstrip("/")), "tokenizer"
            )
        depth = config.depth
        drop_rate = config.drop_rate
        cond_drop_rate = config.cond_drop_rate
        drop_path_rate = config.drop_path_rate
        attn_drop_rate = config.attn_drop_rate
        mlp_ratio = config.mlp_ratio
        norm_eps = config.norm_eps
        shared_aln = config.shared_aln
        attn_l2_norm = config.attn_l2_norm
        context_token = config.context_token
        context_dim = config.context_dim
        patch_nums = config.patch_nums
        flash_if_available, fused_if_available = (
            config.flash_if_available,
            config.fused_if_available,
        )
        self.mlp_type = mlp_type = config.mlp_type
        self.attn_type = attn_type = config.attn_type
        if self.attn_type == "gpt2":
            norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        elif self.attn_type == "llama":
            norm_layer = partial(LlamaRMSNormFused, eps=norm_eps)
        else:
            raise NotImplementedError
        self.disable_aln = config.disable_aln
        self.use_timestep_embed = use_timestep_embed = config.use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode = config.sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn = config.use_cross_attn

        self.diffusion_head_repeats = diffusion_head_repeats = (
            config.diffusion_head_repeats
        )
        # MAR variant masking ratio
        mask_ratio_min = 0.5
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        vae_local = HARTAutoEncoderWithDisc.from_pretrained(vae_path).vae
        vae_local = vae_local.cuda()
        vae_local.requires_grad_(False)

        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = tuple(patch_nums)
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = context_token
        self.begin_ends = []
        self.begin_ends.append((0, context_token))
        cur = context_token
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = CopyableGenerator(device=get_device())

        # Enhanced cache configuration
        self.cache_config = self._create_cache_config(config)
        
        # Initialize enhanced caching mechanism
        self.use_cache = getattr(config, 'use_cache', False) or self.cache_config.skip_stages
        self.calibration = getattr(config, 'calibration', False)
        self.sim_path = getattr(config, 'sim_path', None)
        self.threshold = getattr(config, 'threshold', 0.7)
        
        self.cache_mlp = [None] * depth
        self.cache_attn = [None] * depth
        self.cache_similarity_mlp = [[0.0] * 15 for _ in range(depth)]  # 15 iterations (0-14)
        self.cache_similarity_attn = [[0.0] * 15 for _ in range(depth)]
        
        # Load similarity data if path provided
        if self.sim_path and os.path.exists(self.sim_path):
            import pickle
            with open(self.sim_path, 'rb') as f:
                sim_data = pickle.load(f)
                self.cache_similarity_mlp = sim_data.get('mlp', self.cache_similarity_mlp)
                self.cache_similarity_attn = sim_data.get('attn', self.cache_similarity_attn)

        # 1. input (word) embedding
        quant: HARTHybridQuantizer = vae_local.quantize
        self.vae_proxy: Tuple[HARTAutoEncoder] = (vae_local,)
        self.vae_quant_proxy: Tuple[HARTHybridQuantizer] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.context_token = context_token
        self.context_token_proj = nn.Linear(context_dim, self.C)
        self.to_logits = nn.Linear(self.C, self.V + 1)
        if self.use_timestep_embed:
            self.t_embedder = TimestepEmbedder(self.C)
        self.sep_aln_context_position_embed = nn.Parameter(
            torch.randn(1, context_token, self.C) * init_std
        )

        # 3. absolute position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.L + context_token, self.C))
        nn.init.trunc_normal_(self.pos_embed.data, mean=0, std=init_std)

        # 4. enhanced transformer blocks
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        
        # Create enhanced blocks with cache configuration
        self.blocks = nn.ModuleList([
            AdaLNSelfAttnEnhanced(
                block_idx=block_idx,
                last_drop_p=drop_path_rates[block_idx],
                embed_dim=embed_dim,
                cond_dim=self.C,
                shared_aln=shared_aln,
                norm_layer=norm_layer,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[block_idx],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,
                fused_if_available=fused_if_available,
                # Enhanced cache parameters
                cache_config=self.cache_config,
                use_cache=self.use_cache,
                calibration=self.calibration,
                threshold=self.threshold,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask used in training
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        )
        dT = d.unsqueeze(0)
        d = d.unsqueeze(1)
        self.register_buffer("d", d, persistent=False)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf)
        self.register_buffer("attn_bias_for_masking", attn_bias_for_masking, persistent=False)

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.C, norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # 7. diffusion loss
        self.diffloss = DiffLoss(
            target_channels=config.target_channels,
            z_channels=self.C,
            width=config.width,
            height=config.height,
            num_sampling_steps=config.num_sampling_steps,
            sampler=config.sampler,
        )
        self.diffusion_batch_mul = config.diffusion_batch_mul
    
    def _create_cache_config(self, config) -> CacheConfig:
        """Create cache configuration from config or command line args"""
        # Get cache parameters from config with defaults
        skip_stages = getattr(config, 'cache_skip_stages', [])
        cache_stages = getattr(config, 'cache_cache_stages', [])
        enable_attn_cache = getattr(config, 'cache_enable_attn', True)
        enable_mlp_cache = getattr(config, 'cache_enable_mlp', True)
        threshold = getattr(config, 'cache_threshold', 0.7)
        adaptive_threshold = getattr(config, 'cache_adaptive_threshold', False)
        interpolation_mode = getattr(config, 'cache_interpolation_mode', 'bilinear')
        
        return CacheConfig(
            skip_stages=skip_stages,
            cache_stages=cache_stages,
            enable_attn_cache=enable_attn_cache,
            enable_mlp_cache=enable_mlp_cache,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold,
            interpolation_mode=interpolation_mode
        )

    @classmethod
    def add_cache_arguments(cls, parser: argparse.ArgumentParser):
        """Add cache-related command line arguments"""
        cache_group = parser.add_argument_group('Enhanced Caching Options')
        
        cache_group.add_argument('--cache-skip-stages', type=int, nargs='*', default=[],
                               help='Stages to skip computation (e.g., 169 256)')
        cache_group.add_argument('--cache-cache-stages', type=int, nargs='*', default=[],
                               help='Stages to cache results (e.g., 100 169)')
        cache_group.add_argument('--cache-enable-attn', action='store_true', default=True,
                               help='Enable attention layer caching')
        cache_group.add_argument('--cache-disable-attn', action='store_false', dest='cache_enable_attn',
                               help='Disable attention layer caching')
        cache_group.add_argument('--cache-enable-mlp', action='store_true', default=True,
                               help='Enable MLP layer caching')
        cache_group.add_argument('--cache-disable-mlp', action='store_false', dest='cache_enable_mlp',
                               help='Disable MLP layer caching')
        cache_group.add_argument('--cache-threshold', type=float, default=0.7,
                               help='Similarity threshold for cache usage')
        cache_group.add_argument('--cache-adaptive-threshold', action='store_true', default=False,
                               help='Use adaptive thresholding')
        cache_group.add_argument('--cache-interpolation-mode', choices=['bilinear', 'nearest', 'bicubic'],
                               default='bilinear', help='Interpolation mode for feature upsampling')
        
        # Preset configurations
        cache_group.add_argument('--cache-preset', choices=[
            'no-cache', 'conservative', 'original', 'aggressive', 'ultra-fast',
            'attn-only', 'mlp-only', 'high-quality', 'memory-efficient'
        ], help='Use predefined cache configuration preset')
        
        # Legacy compatibility
        cache_group.add_argument('--use-cache', action='store_true', default=False,
                               help='Enable caching (legacy compatibility)')
        cache_group.add_argument('--calibration', action='store_true', default=False,
                               help='Run in calibration mode')
        cache_group.add_argument('--sim-path', type=str, default=None,
                               help='Path to similarity data file')
        cache_group.add_argument('--threshold', type=float, default=0.7,
                               help='Cache threshold (legacy compatibility)')

    @classmethod
    def from_args_and_config(cls, args, config: HARTForT2IConfig):
        """Create model from command line arguments and config"""
        # Apply cache preset if specified
        if hasattr(args, 'cache_preset') and args.cache_preset:
            cache_presets = {
                'no-cache': {
                    'cache_skip_stages': [],
                    'cache_cache_stages': [],
                    'cache_enable_attn': False,
                    'cache_enable_mlp': False,
                },
                'conservative': {
                    'cache_skip_stages': [256],
                    'cache_cache_stages': [144],
                    'cache_threshold': 0.8,
                },
                'original': {
                    'cache_skip_stages': [144, 256],
                    'cache_cache_stages': [81, 144],
                    'cache_threshold': 0.7,
                },
                'aggressive': {
                    'cache_skip_stages': [49, 81, 144, 256],
                    'cache_cache_stages': [25, 49, 81, 144],
                    'cache_threshold': 0.6,
                },
                'ultra-fast': {
                    'cache_skip_stages': [25, 49, 81, 144, 256, 441],
                    'cache_cache_stages': [16, 25, 49, 81, 144, 256],
                    'cache_threshold': 0.5,
                    'cache_adaptive_threshold': True,
                },
                'attn-only': {
                    'cache_skip_stages': [144, 256],
                    'cache_cache_stages': [81, 144],
                    'cache_enable_attn': True,
                    'cache_enable_mlp': False,
                },
                'mlp-only': {
                    'cache_skip_stages': [144, 256],
                    'cache_cache_stages': [81, 144],
                    'cache_enable_attn': False,
                    'cache_enable_mlp': True,
                },
                'high-quality': {
                    'cache_skip_stages': [256],
                    'cache_cache_stages': [144],
                    'cache_threshold': 0.9,
                    'cache_interpolation_mode': 'bicubic',
                },
                'memory-efficient': {
                    'cache_skip_stages': [144, 256],
                    'cache_cache_stages': [144],
                    'cache_enable_attn': True,
                    'cache_enable_mlp': False,
                    'cache_threshold': 0.8,
                }
            }
            
            preset = cache_presets.get(args.cache_preset, {})
            for key, value in preset.items():
                setattr(config, key, value)
        
        # Override with command line arguments
        cache_arg_mapping = {
            'cache_skip_stages': 'cache_skip_stages',
            'cache_cache_stages': 'cache_cache_stages', 
            'cache_enable_attn': 'cache_enable_attn',
            'cache_enable_mlp': 'cache_enable_mlp',
            'cache_threshold': 'cache_threshold',
            'cache_adaptive_threshold': 'cache_adaptive_threshold',
            'cache_interpolation_mode': 'cache_interpolation_mode',
            # Legacy compatibility
            'use_cache': 'use_cache',
            'calibration': 'calibration',
            'sim_path': 'sim_path',
            'threshold': 'threshold',
        }
        
        for arg_name, config_name in cache_arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(config, config_name, value)
        
        return cls(config)

    def set_cache_config(self, cache_config: CacheConfig):
        """Update cache configuration and propagate to all blocks"""
        self.cache_config = cache_config
        self.use_cache = bool(cache_config.skip_stages or cache_config.cache_stages)
        
        # Update all transformer blocks
        for block in self.blocks:
            block.cache_config = cache_config
            block.use_cache = self.use_cache
            block.attn.cache_config = cache_config
            block.attn.use_cache = self.use_cache
            block.ffn.cache_config = cache_config
            block.ffn.use_cache = self.use_cache

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
    ) -> torch.Tensor:
        """Autoregressive inference with enhanced caching"""
        if g_seed is None:
            rng = self.rng
        else:
            rng = torch.Generator(device=self.rng.device)
            rng.manual_seed(g_seed)

        if label_B is None:
            label_B = torch.multinomial(
                torch.ones(B, self.num_classes, dtype=torch.float, device=self.rng.device),
                num_samples=1,
                generator=rng,
            ).view(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=label_B, dtype=torch.long, device=self.rng.device)

        # Enhanced caching: reset cache state for new generation
        self.cache_mlp = [None] * self.depth
        self.cache_attn = [None] * self.depth

        # Progressive generation with enhanced caching
        sos = self.context_token_proj(torch.randn(B, self.context_token, self.context_dim, device=self.rng.device))

        if self.use_timestep_embed:
            t = torch.randint(0, 1000, (B,), device=self.rng.device)
            cond_BD = self.t_embedder(t).unsqueeze(1)
        else:
            cond_BD = torch.zeros(B, 1, self.C, device=self.rng.device)

        lvl_pos = None
        h_BLC = sos
        for si in range(final_stage, self.num_stages_minus_1 + 1):
            if self.context_token == 0:
                kv_B, kv_L, kv_C = h_BLC.shape
            else:
                kv_B, kv_L, kv_C = h_BLC.shape

            # Enhanced caching: pass cache parameters to blocks
            for block in self.blocks:
                block.attn.kv_caching(True)

            cur_L = self.patch_nums[si] ** 2
            h_BLC = torch.cat(
                [
                    h_BLC,
                    torch.empty(kv_B, cur_L, kv_C, dtype=h_BLC.dtype, device=h_BLC.device),
                ],
                dim=1,
            )

            for i in range(cur_L):
                for b_idx, block in enumerate(self.blocks):
                    h_BLC = block(
                        h_BLC,
                        cond_BD,
                        attn_bias=None,
                        si=si,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        cache_mlp=self.cache_mlp,
                        cache_attn=self.cache_attn,
                        cache_similarity_mlp=self.cache_similarity_mlp,
                        cache_similarity_attn=self.cache_similarity_attn,
                    )
                
                logits_BlV = self.get_logits(h_BLC, cond_BD)
                if si == self.num_stages_minus_1:
                    last_layer_cond = h_BLC

                idx_in_sequence = self.context_token + sum(pn**2 for pn in self.patch_nums[:si]) + i
                logits_BlV = logits_BlV[:, idx_in_sequence]

                # Apply sampling
                if cfg > 1:
                    # Classifier-free guidance
                    pass

                idx_next_BL = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                ).view(B, 1)

                h_BLC[:, idx_in_sequence] = self.word_embed(
                    F.one_hot(idx_next_BL, num_classes=self.V).squeeze(1).float()
                )

            for block in self.blocks:
                block.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(
            self.vae_quant_proxy[0].embedding(h_BLC[:, self.context_token :])
        ).clamp_(0, 1)

    def save_similarity_data(self):
        """Save similarity data after calibration"""
        if self.sim_path:
            import pickle
            sim_data = {
                'mlp': self.cache_similarity_mlp,
                'attn': self.cache_similarity_attn
            }
            with open(self.sim_path, 'wb') as f:
                pickle.dump(sim_data, f)
            print(f"Similarity data saved to {self.sim_path}")

    def get_cache_statistics(self):
        """Get cache usage statistics"""
        stats = {
            'cache_config': {
                'skip_stages': self.cache_config.skip_stages,
                'cache_stages': self.cache_config.cache_stages,
                'enable_attn_cache': self.cache_config.enable_attn_cache,
                'enable_mlp_cache': self.cache_config.enable_mlp_cache,
                'threshold': self.cache_config.threshold,
            },
            'similarity_stats': {
                'mlp': self.cache_similarity_mlp,
                'attn': self.cache_similarity_attn,
            }
        }
        return stats