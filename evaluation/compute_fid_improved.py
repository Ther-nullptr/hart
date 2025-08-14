"""
Improved FID computation for HART model evaluation
Refactored from original evaluation with better structure and modularity
"""
import argparse
import copy
import datetime
import json
import os
import pathlib
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed

# Add HART modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from hart.modules.models.transformer import HARTForT2I
    from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
    HART_AVAILABLE = True
except ImportError:
    print("Warning: HART modules not found. Generation features disabled.")
    HART_AVAILABLE = False

# Import FID calculation utilities
try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    print("Warning: cleanfid not available, using custom implementation")
    CLEANFID_AVAILABLE = False

from utils import tracker


class ImagePathDataset(torch.utils.data.Dataset):
    """Dataset for loading images from file paths."""
    
    def __init__(self, files: List[str], transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {path}: {e}")
        return img


class HARTFIDEvaluator:
    """Improved FID evaluator for HART model with better structure and error handling."""
    
    def __init__(
        self,
        model_path: str = None,
        text_model_path: str = None,
        device: str = "cuda",
        use_ema: bool = True,
        img_size: int = 1024,
        batch_size: int = 4,
        max_token_length: int = 300,
        use_llm_system_prompt: bool = True
    ):
        self.device = torch.device(device)
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.use_llm_system_prompt = use_llm_system_prompt
        
        # Initialize models if paths provided and HART is available
        self.model = None
        self.ema_model = None
        self.text_tokenizer = None
        self.text_model = None
        self.use_ema = use_ema
        
        if HART_AVAILABLE and model_path and text_model_path:
            self._load_models(model_path, text_model_path)
        
        # Initialize FID computation model
        self.fid_model = None
        
    def _load_models(self, model_path: str, text_model_path: str):
        """Load HART and text models."""
        try:
            # Load HART model
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            
            # Load EMA model if specified
            if self.use_ema:
                self.ema_model = copy.deepcopy(self.model)
                ema_path = os.path.join(model_path, "ema_model.bin")
                if os.path.exists(ema_path):
                    self.ema_model.load_state_dict(torch.load(ema_path))
                else:
                    print(f"Warning: EMA model not found at {ema_path}, using main model")
                    self.use_ema = False
            
            # Load text model
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            self.text_model = AutoModel.from_pretrained(text_model_path).to(self.device)
            self.text_model.eval()
            
            print("HART models loaded successfully")
            
        except Exception as e:
            print(f"Error loading HART models: {e}")
            self.model = None
        
    def initialize_fid_model(self, dims: int = 2048):
        """Initialize InceptionV3 model for FID computation."""
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.fid_model = InceptionV3([block_idx]).to(self.device)
        self.fid_model.eval()
    
    def generate_images(
        self,
        prompts: List[str],
        output_dir: str,
        cfg: float = 4.5,
        seed: Optional[int] = None,
        more_smooth: bool = True,
        save_individually: bool = True
    ) -> List[str]:
        """Generate images from prompts using HART model."""
        
        if not self.model:
            raise ValueError("HART model not loaded. Cannot generate images.")
            
        if seed is not None:
            set_seed(seed)
            
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        infer_func = (
            self.ema_model.autoregressive_infer_cfg
            if self.use_ema and self.ema_model
            else self.model.autoregressive_infer_cfg
        )
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            try:
                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                        # Encode prompt
                        (
                            context_tokens,
                            context_mask,
                            context_position_ids,
                            context_tensor,
                        ) = encode_prompts(
                            [prompt],
                            self.text_model,
                            self.text_tokenizer,
                            self.max_token_length,
                            llm_system_prompt,
                            self.use_llm_system_prompt,
                        )
                        
                        # Generate image
                        output_imgs = infer_func(
                            B=context_tensor.size(0),
                            label_B=context_tensor,
                            cfg=cfg,
                            g_seed=seed,
                            more_smooth=more_smooth,
                            context_position_ids=context_position_ids,
                            context_mask=context_mask,
                        )
                
                # Save image
                if save_individually:
                    for j, img in enumerate(output_imgs):
                        filename = f"{i:06d}_{j:02d}.png"
                        filepath = os.path.join(output_dir, filename)
                        self._save_image(img, filepath)
                        generated_files.append(filepath)
                else:
                    filename = f"{i:06d}.png"
                    filepath = os.path.join(output_dir, filename)
                    self._save_image(output_imgs[0], filepath)
                    generated_files.append(filepath)
                    
            except Exception as e:
                print(f"Error generating image for prompt '{prompt[:50]}...': {e}")
                continue
                
        return generated_files
    
    def _save_image(self, img_tensor: torch.Tensor, filepath: str):
        """Save a single image tensor to file."""
        img_np = img_tensor.mul_(255).cpu().numpy()
        img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(filepath)
    
    def compute_fid(
        self,
        real_path: str,
        gen_path: str,
        batch_size: int = 50,
        dims: int = 2048,
        num_workers: int = 4
    ) -> float:
        """Compute FID score between real and generated images."""
        
        # Use cleanfid for robust FID computation if available
        if CLEANFID_AVAILABLE:
            try:
                fid_score = fid.compute_fid(
                    real_path,
                    gen_path,
                    batch_size=batch_size,
                    device=self.device,
                    num_workers=num_workers
                )
                return fid_score
            except Exception as e:
                print(f"Error computing FID with cleanfid: {e}")
                print("Falling back to custom FID computation...")
        
        # Fallback to custom implementation
        return self._compute_fid_custom(real_path, gen_path, batch_size, dims, num_workers)
    
    def _compute_fid_custom(
        self,
        real_path: str,
        gen_path: str,
        batch_size: int,
        dims: int,
        num_workers: int
    ) -> float:
        """Custom FID computation as fallback."""
        
        if self.fid_model is None:
            self.initialize_fid_model(dims)
        
        def get_image_files(path: str) -> List[str]:
            """Get all image files from a directory."""
            extensions = {"png", "jpg", "jpeg", "bmp", "tiff"}
            path_obj = pathlib.Path(path)
            files = []
            for ext in extensions:
                files.extend(path_obj.glob(f"*.{ext}"))
            return [str(f) for f in sorted(files)]
        
        real_files = get_image_files(real_path)
        gen_files = get_image_files(gen_path)
        
        if len(real_files) == 0:
            raise ValueError(f"No images found in real path: {real_path}")
        if len(gen_files) == 0:
            raise ValueError(f"No images found in generated path: {gen_path}")
        
        print(f"Computing FID: {len(real_files)} real images vs {len(gen_files)} generated images")
        
        # Compute statistics for real and generated images
        real_stats = self._compute_activation_statistics(real_files, batch_size, dims, num_workers)
        gen_stats = self._compute_activation_statistics(gen_files, batch_size, dims, num_workers)
        
        # Calculate FID
        fid_score = self._calculate_frechet_distance(*real_stats, *gen_stats)
        return fid_score
    
    def _compute_activation_statistics(
        self,
        files: List[str],
        batch_size: int,
        dims: int,
        num_workers: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute activation statistics for a set of images."""
        
        transform = T.Compose([
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor()
        ])
        
        dataset = ImagePathDataset(files, transforms=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            drop_last=False, num_workers=num_workers
        )
        
        activations = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing activations"):
                batch = batch.to(self.device)
                pred = self.fid_model(batch)[0]
                
                # Global spatial average pooling if needed
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                activations.append(pred)
        
        activations = np.vstack(activations)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        
        return mu, sigma
    
    def _calculate_frechet_distance(
        self, 
        mu1: np.ndarray, 
        sigma1: np.ndarray, 
        mu2: np.ndarray, 
        sigma2: np.ndarray, 
        eps: float = 1e-6
    ) -> float:
        """Calculate Frechet distance between two multivariate Gaussians."""
        
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f"Adding {eps} to diagonal of covariance estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def evaluate_on_mjhq30k(
    evaluator: HARTFIDEvaluator,
    mjhq_metadata_path: str,
    mjhq_images_path: str,
    output_dir: str,
    category_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
    cfg: float = 4.5,
    seed: int = 1
) -> dict:
    """Evaluate HART on MJHQ-30K dataset."""
    
    # Load MJHQ metadata
    with open(mjhq_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Filter by category if specified
    if category_filter:
        metadata = {
            k: v for k, v in metadata.items() 
            if category_filter in v.get('category', [])
        }
        print(f"Filtered to {len(metadata)} samples with category '{category_filter}'")
    
    # Limit samples if specified
    if max_samples:
        items = list(metadata.items())[:max_samples]
        metadata = dict(items)
        print(f"Limited to {len(metadata)} samples")
    
    # Extract prompts
    prompts = [data['prompt'] for data in metadata.values()]
    
    # Generate images
    print(f"Generating {len(prompts)} images...")
    gen_output_dir = os.path.join(output_dir, "generated")
    generated_files = evaluator.generate_images(
        prompts, gen_output_dir, cfg=cfg, seed=seed
    )
    
    # Compute FID
    if category_filter:
        real_path = os.path.join(mjhq_images_path, category_filter)
    else:
        real_path = mjhq_images_path
        
    print(f"Computing FID between {real_path} and {gen_output_dir}...")
    fid_score = evaluator.compute_fid(real_path, gen_output_dir)
    
    results = {
        'fid_score': fid_score,
        'num_samples': len(prompts),
        'category_filter': category_filter,
        'generated_files': generated_files[:10],  # Just first 10 for logging
        'config': {
            'cfg': cfg,
            'seed': seed,
            'img_size': evaluator.img_size
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, "fid_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"FID Score: {fid_score:.4f}")
    print(f"Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Improved FID evaluation for HART model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to HART model"
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="Qwen/Qwen2-VL-1.5B-Instruct",
        help="Path to text encoder model"
    )
    
    # Dataset paths
    parser.add_argument(
        "--mjhq_metadata_path",
        type=str,
        help="Path to MJHQ-30K metadata.json file"
    )
    parser.add_argument(
        "--mjhq_images_path",
        type=str,
        help="Path to MJHQ-30K images directory"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fid_evaluation",
        help="Output directory for generated images and results"
    )
    parser.add_argument(
        "--category_filter",
        type=str,
        help="Filter to specific category (e.g., 'people')"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    
    # Generation settings
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model")
    parser.add_argument("--more_smooth", action="store_true", help="More smooth generation")
    
    # Technical settings
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--fid_batch_size", type=int, default=50, help="FID computation batch size")
    parser.add_argument("--img_size", type=int, default=1024, help="Image resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_token_length", type=int, default=300, help="Max token length for text encoding")
    
    # Alternative: direct FID computation
    parser.add_argument(
        "--compute_fid_only",
        nargs=2,
        metavar=("REAL_PATH", "GEN_PATH"),
        help="Only compute FID between two directories"
    )
    
    # Experiment tracking
    parser.add_argument("--exp_name", type=str, default="fid_experiment", help="Experiment name")
    parser.add_argument("--report_to", type=str, default="none", help="Where to report metrics")
    parser.add_argument("--tracker_project_name", type=str, default="hart-evaluation", help="Project name for tracking")
    
    args = parser.parse_args()
    
    # Direct FID computation mode
    if args.compute_fid_only:
        evaluator = HARTFIDEvaluator(
            device=args.device,
            img_size=args.img_size
        )
        
        fid_score = evaluator.compute_fid(
            args.compute_fid_only[0],
            args.compute_fid_only[1],
            batch_size=args.fid_batch_size,
            num_workers=args.num_workers
        )
        print(f"FID Score: {fid_score:.4f}")
        
        # Log to tracker if specified
        if args.report_to == "wandb":
            result_dict = {args.exp_name: fid_score}
            tracker(args, result_dict, label="", pattern="epoch_step", metric="FID")
        
        return
    
    # Full evaluation mode
    if not args.mjhq_metadata_path or not args.mjhq_images_path:
        if not args.model_path:
            raise ValueError("For FID-only mode, use --compute_fid_only. For full evaluation, --model_path, --mjhq_metadata_path and --mjhq_images_path are required")
    
    # Initialize evaluator
    evaluator = HARTFIDEvaluator(
        model_path=args.model_path,
        text_model_path=args.text_model_path,
        device=args.device,
        use_ema=args.use_ema,
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length
    )
    
    # Run evaluation
    if args.mjhq_metadata_path and args.mjhq_images_path:
        results = evaluate_on_mjhq30k(
            evaluator=evaluator,
            mjhq_metadata_path=args.mjhq_metadata_path,
            mjhq_images_path=args.mjhq_images_path,
            output_dir=args.output_dir,
            category_filter=args.category_filter,
            max_samples=args.max_samples,
            cfg=args.cfg,
            seed=args.seed
        )
        
        # Log to tracker if specified
        if args.report_to == "wandb":
            result_dict = {args.exp_name: results['fid_score']}
            tracker(args, result_dict, label="", pattern="epoch_step", metric="FID")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Final FID Score: {results['fid_score']:.4f}")
    else:
        print("No MJHQ dataset provided. Use --compute_fid_only for direct FID computation.")


if __name__ == "__main__":
    main()