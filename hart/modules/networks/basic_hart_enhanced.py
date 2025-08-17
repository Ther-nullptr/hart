"""Enhanced HART basic blocks with VAR-style caching mechanism.

This file provides enhanced HART transformer blocks that closely follow VAR's caching implementation
for maximum compatibility and performance.
"""

import functools
import math
from typing import Tuple, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from hart.modules.networks.utils import DropPath, drop_path

# Import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    pass
try:
    import xformers
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func
except ImportError:
    pass
try:
    from torch.nn.functional import scaled_dot_product_attention as slow_attn
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)
        if attn_mask is not None:
            attn.add_(attn_mask)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value


# VAR-style cache configuration
class CacheConfig:
    """Cache configuration for enhanced HART blocks"""
    def __init__(
        self,
        skip_stages: List[int] = None,
        cache_stages: List[int] = None,
        enable_attn_cache: bool = True,
        enable_mlp_cache: bool = True,
        threshold: float = 0.7,
        adaptive_threshold: bool = False,
        interpolation_mode: str = 'bilinear'
    ):
        self.skip_stages = skip_stages or []
        self.cache_stages = cache_stages or []
        self.enable_attn_cache = enable_attn_cache
        self.enable_mlp_cache = enable_mlp_cache
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.interpolation_mode = interpolation_mode


# Cache helper functions - updated for HART
length2iteration = {
    1: 0, 4: 1, 9: 2, 16: 3, 25: 4, 49: 6, 81: 7, 144: 8, 256: 9, 441: 10, 729: 11, 1296: 12, 2304: 13, 4096: 14
}

next_patch_size = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 7: 9, 9: 12, 12: 16, 16: 21, 21: 27, 27: 36, 36: 48, 48: 64
}


def feature_interpolate(x, mode='bilinear', align_corners=True):
    """Enhanced feature interpolation following VAR's implementation"""
    if x is None:
        return None
    
    # input: [B, L1, C]
    # output: [B, L2, C]
    hw = int(math.sqrt(x.shape[1]))
    if hw not in next_patch_size:
        return x  # Return original if no interpolation needed
    
    result = x.view(x.shape[0], hw, hw, -1)
    result = torch.nn.functional.interpolate(
        result.permute(0, 3, 1, 2), 
        size=(next_patch_size[hw], next_patch_size[hw]), 
        mode=mode, 
        align_corners=align_corners
    ).permute(0, 2, 3, 1)
    result = result.view(result.shape[0], -1, result.shape[-1])
    return result


class FFNEnhanced(nn.Module):
    """Enhanced FFN with VAR-style caching"""
    
    def __init__(
        self, 
        block_idx,
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_func="gelu",
        drop=0., 
        fused_if_available=True,
        # Cache parameters
        cache_config: Optional[CacheConfig] = None,
        use_cache=False,
        calibration=False,
        threshold=0.7,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        if act_func == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif act_func == "silu":
            self.act = nn.SiLU()
        else:
            raise NotImplementedError
            
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        
        # Cache configuration
        self.cache_config = cache_config or CacheConfig()
        self.block_idx = block_idx
        self.use_cache = use_cache or bool(self.cache_config.skip_stages or self.cache_config.cache_stages)
        self.calibration = calibration
        self.threshold = threshold
        
        # Cache state
        self.temp_cache = None
        self.iter = 0
        
        # Adaptive threshold tracking
        self.similarity_history = []
        self.adaptive_threshold = self.cache_config.threshold

    def forward(self, x, cache_similarity_mlp=None, cache_mlp=None):
        L = x.shape[1]
        
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        
        # Enhanced caching logic following VAR's approach
        if self.use_cache and cache_similarity_mlp is not None and cache_mlp is not None:
            if self.calibration:
                # Calibration mode: compute features and track similarity
                result = self.drop(self.fc2(self.act(self.fc1(x))))
                
                if L == 1 and self.temp_cache is not None:
                    self.temp_cache = None
                    self.iter += 1
                
                if self.temp_cache is not None and L in length2iteration:
                    cache = feature_interpolate(
                        self.temp_cache, 
                        mode=self.cache_config.interpolation_mode
                    )
                    
                    # Compute similarity and update tracking
                    cos_sim = torch.nn.functional.cosine_similarity(
                        cache.view(-1, cache.shape[-1]), 
                        result.view(-1, cache.shape[-1]), 
                        dim=1
                    ).mean().item()
                    
                    if length2iteration[L] - 1 >= 0:
                        prev_idx = length2iteration[L] - 1
                        cache_similarity_mlp[self.block_idx][prev_idx] = (
                            cos_sim + cache_similarity_mlp[self.block_idx][prev_idx] * self.iter
                        ) / (self.iter + 1)
                
                self.temp_cache = result.clone()
                return result
            
            else:
                # Inference mode with caching
                effective_threshold = self.threshold
                if self.cache_config.adaptive_threshold and hasattr(self, 'adaptive_threshold'):
                    effective_threshold = self.adaptive_threshold
                
                # Check if we should skip computation and use cache
                if (L in length2iteration and 
                    L in self.cache_config.skip_stages and
                    cache_similarity_mlp[self.block_idx][length2iteration[L] - 1] > effective_threshold):
                    
                    result = feature_interpolate(
                        cache_mlp[self.block_idx],
                        mode=self.cache_config.interpolation_mode
                    )
                else:
                    result = self.drop(self.fc2(self.act(self.fc1(x))))
                
                # Clear previous cache
                cache_mlp[self.block_idx] = None
                
                # Check if we should cache the result
                if (L in self.cache_config.cache_stages and 
                    L in length2iteration and 
                    cache_similarity_mlp[self.block_idx][length2iteration[L]] > effective_threshold):
                    
                    cache_mlp[self.block_idx] = result.clone()
                else:
                    cache_mlp[self.block_idx] = None
                
                return result
        
        # No caching - standard computation
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttentionEnhanced(nn.Module):
    """Enhanced Self-Attention with VAR-style caching"""
    
    def __init__(
        self, 
        block_idx, 
        embed_dim=768, 
        num_heads=12,
        attn_drop=0., 
        proj_drop=0., 
        attn_l2_norm=False, 
        flash_if_available=True,
        # Cache parameters
        cache_config: Optional[CacheConfig] = None,
        use_cache=False,
        calibration=False,
        threshold=0.7,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # Cache configuration
        self.cache_config = cache_config or CacheConfig()
        self.use_cache = use_cache or bool(self.cache_config.skip_stages or self.cache_config.cache_stages)
        self.calibration = calibration
        self.threshold = threshold
        
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), 
                requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # KV caching for autoregressive generation
        self.caching, self.cached_k, self.cached_v = False, None, None
        
        # Feature caching state
        self.temp_cache = None
        self.iter = 0

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward_inner(self, x, attn_bias):
        """Core attention computation"""
        B, L, C = x.shape
        qkv = self.mat_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
                dropout_p=dropout_p, softmax_scale=self.scale
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
                attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), 
                p=dropout_p, scale=self.scale
            ).view(B, L, C)
        else:
            attn = q.mul(self.scale) @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn.add_(attn_bias)
            oup = self.softmax(attn) @ v
            oup = oup.transpose(1, 2).reshape(B, L, C)
        
        result = self.proj_drop(self.proj(oup))
        return result

    def forward(self, x, attn_bias, cache_similarity_attn=None, cache_attn=None):
        L = x.shape[1]
        
        # Enhanced caching logic following VAR's approach
        if self.use_cache and cache_similarity_attn is not None and cache_attn is not None:
            if self.calibration:
                # Calibration mode
                result = self.forward_inner(x, attn_bias)
                
                if L == 1 and self.temp_cache is not None:
                    self.temp_cache = None
                    self.iter += 1
                
                if self.temp_cache is not None and L in length2iteration:
                    cache = feature_interpolate(
                        self.temp_cache,
                        mode=self.cache_config.interpolation_mode
                    )
                    
                    # Compute similarity and update tracking
                    cos_sim = torch.nn.functional.cosine_similarity(
                        cache.view(-1, cache.shape[-1]),
                        result.view(-1, cache.shape[-1]),
                        dim=1
                    ).mean().item()
                    
                    if length2iteration[L] - 1 >= 0:
                        prev_idx = length2iteration[L] - 1
                        cache_similarity_attn[self.block_idx][prev_idx] = (
                            cos_sim + cache_similarity_attn[self.block_idx][prev_idx] * self.iter
                        ) / (self.iter + 1)
                
                self.temp_cache = result.clone()
                return result
            
            else:
                # Inference mode with caching
                effective_threshold = self.threshold
                
                # Check if we should skip computation and use cache
                if (L in length2iteration and 
                    L in self.cache_config.skip_stages and
                    cache_similarity_attn[self.block_idx][length2iteration[L] - 1] > effective_threshold):
                    
                    result = feature_interpolate(
                        cache_attn[self.block_idx],
                        mode=self.cache_config.interpolation_mode
                    )
                else:
                    result = self.forward_inner(x, attn_bias)
                
                # Clear previous cache
                cache_attn[self.block_idx] = None
                
                # Check if we should cache the result
                if (L in self.cache_config.cache_stages and 
                    L in length2iteration and 
                    cache_similarity_attn[self.block_idx][length2iteration[L]] > effective_threshold):
                    
                    cache_attn[self.block_idx] = result.clone()
                else:
                    cache_attn[self.block_idx] = None
                
                return result
        
        # No caching - standard computation
        return self.forward_inner(x, attn_bias)


class AdaLNSelfAttnEnhanced(nn.Module):
    """Enhanced AdaLN Self-Attention with VAR-style caching"""
    
    def __init__(
        self, 
        block_idx, 
        last_drop_p, 
        embed_dim, 
        cond_dim, 
        shared_aln: bool, 
        norm_layer,
        num_heads, 
        mlp_ratio=4., 
        drop=0., 
        attn_drop=0., 
        drop_path=0., 
        attn_l2_norm=False,
        flash_if_available=False, 
        fused_if_available=True,
        # Cache parameters
        cache_config: Optional[CacheConfig] = None,
        use_cache=False,
        calibration=False,
        threshold=0.7,
    ):
        super(AdaLNSelfAttnEnhanced, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Cache configuration
        self.cache_config = cache_config or CacheConfig()
        self.use_cache = use_cache or bool(self.cache_config.skip_stages or self.cache_config.cache_stages)
        self.calibration = calibration
        self.threshold = threshold
        
        self.attn = SelfAttentionEnhanced(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, 
            flash_if_available=flash_if_available, 
            cache_config=cache_config, use_cache=use_cache, 
            calibration=calibration, threshold=threshold
        )
        self.ffn = FFNEnhanced(
            block_idx=block_idx, in_features=embed_dim, 
            hidden_features=round(embed_dim * mlp_ratio), drop=drop, 
            fused_if_available=fused_if_available,
            cache_config=cache_config, use_cache=use_cache,
            calibration=calibration, threshold=threshold
        )
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None

    def forward(
        self, x, cond_BD, attn_bias, si=-1, context_position_ids=None, context_mask=None,
        m_maskgit=None, cache_mlp=None, cache_attn=None, 
        cache_similarity_mlp=None, cache_similarity_attn=None
    ):
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        # Attention with enhanced caching
        attn_input = self.ln_wo_grad(x) * (scale1.add(1)) + (shift1)
        attn_out = self.attn(
            attn_input, attn_bias=attn_bias, 
            cache_attn=cache_attn, cache_similarity_attn=cache_similarity_attn
        )
        x = x + self.drop_path(attn_out.mul_(gamma1))
        
        # FFN with enhanced caching
        ffn_input = self.ln_wo_grad(x) * (scale2.add(1)) + (shift2)
        ffn_out = self.ffn(
            ffn_input, cache_mlp=cache_mlp, 
            cache_similarity_mlp=cache_similarity_mlp
        )
        x = x + self.drop_path(ffn_out.mul(gamma2))
        
        return x


# Legacy exports for backward compatibility
FFN = FFNEnhanced
AdaLNSelfAttn = AdaLNSelfAttnEnhanced