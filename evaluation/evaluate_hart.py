"""
Comprehensive evaluation script for HART model
Integrates multiple evaluation metrics and benchmarks
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# Import evaluation modules
from compute_fid import calculate_fid_given_paths
from compute_clip_score import CLIPEvaluator

# Import HART modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check


class HARTEvaluator:
    """
    Comprehensive HART model evaluator
    Supports FID, CLIP Score, GenEval, and custom metrics
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.results = {}
        
        # Initialize CLIP evaluator if needed
        if args.eval_clip_score:
            self.clip_evaluator = CLIPEvaluator(device=self.device)
        else:
            self.clip_evaluator = None
        
        # Initialize HART models if generation is enabled
        if args.generate_samples:
            self._load_hart_models()
    
    def _load_hart_models(self):
        """Load HART model and related components"""
        print("Loading HART models...")
        
        # Load main HART model
        self.hart_model = AutoModel.from_pretrained(self.args.model_path)
        self.hart_model = self.hart_model.to(self.device)
        self.hart_model.eval()
        
        # Load EMA model if specified
        if self.args.use_ema:
            self.ema_model = copy.deepcopy(self.hart_model)
            ema_path = os.path.join(self.args.model_path, "ema_model.bin")
            if os.path.exists(ema_path):
                self.ema_model.load_state_dict(torch.load(ema_path))
            else:
                print(f"Warning: EMA model not found at {ema_path}, using main model")
                self.ema_model = self.hart_model
        else:
            self.ema_model = None
        
        # Load text model
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.args.text_model_path)
        self.text_model = AutoModel.from_pretrained(self.args.text_model_path).to(self.device)
        self.text_model.eval()
        
        # Load safety checker if specified
        if self.args.shield_model_path:
            self.safety_checker_tokenizer = AutoTokenizer.from_pretrained(self.args.shield_model_path)
            self.safety_checker_model = AutoModelForCausalLM.from_pretrained(
                self.args.shield_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        else:
            self.safety_checker_tokenizer = None
            self.safety_checker_model = None
    
    def _save_images_batch(self, sample_imgs, prompts, output_dir, start_idx=0):
        """Save a batch of generated images"""
        # Clone tensor to avoid inplace operation in inference mode
        sample_imgs_np = sample_imgs.clone().mul(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(output_dir, exist_ok=True)
        
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            output_path = os.path.join(output_dir, f"{start_idx + img_idx:06d}.png")
            cur_img_store.save(output_path)
    
    def _generate_batch(self, batch_prompts, batch_idx=0):
        """Generate images for a batch of prompts"""
        # Safety check
        if self.safety_checker_tokenizer and self.safety_checker_model:
            for idx, prompt in enumerate(batch_prompts):
                if safety_check.is_dangerous(
                    self.safety_checker_tokenizer, self.safety_checker_model, prompt
                ):
                    batch_prompts[idx] = random.sample(default_prompts, 1)[0]
                    print(f"Detected unsafe prompt in batch {batch_idx}, replacing with default prompt")
        
        # Encode prompts
        (
            context_tokens,
            context_mask,
            context_position_ids,
            context_tensor,
        ) = encode_prompts(
            batch_prompts,
            self.text_model,
            self.text_tokenizer,
            self.args.max_token_length,
            llm_system_prompt,
            self.args.use_llm_system_prompt,
        )
        
        # Generate images
        infer_func = (
            self.ema_model.autoregressive_infer_cfg
            if self.args.use_ema and self.ema_model is not None
            else self.hart_model.autoregressive_infer_cfg
        )
        
        output_imgs = infer_func(
            B=context_tensor.size(0),
            label_B=context_tensor,
            cfg=self.args.cfg,
            g_seed=self.args.seed,
            more_smooth=self.args.more_smooth,
            context_position_ids=context_position_ids,
            context_mask=context_mask,
        )
        
        return output_imgs
    
    def generate_samples(self):
        """
        Generate samples using HART model with batched inference
        """
        if not self.args.prompts_file:
            print("Sample generation skipped. No prompts file provided.")
            return
            
        print("Generating samples using HART model with batched inference...")
        
        # Read prompts
        with open(self.args.prompts_file, 'r') as f:
            if self.args.prompts_file.endswith('.json'):
                prompts_data = json.load(f)
                if isinstance(prompts_data, list):
                    prompts = [item.get('prompt', str(item)) for item in prompts_data]
                else:
                    prompts = list(prompts_data.values())
            else:
                prompts = [line.strip() for line in f.readlines()]
        
        # Limit samples if specified
        if hasattr(self.args, 'max_samples') and self.args.max_samples > 0:
            prompts = prompts[:self.args.max_samples]
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Process in batches
        batch_size = getattr(self.args, 'generation_batch_size', 4)  # Default batch size for generation
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    batch_idx = i // batch_size
                    
                    # Check if all images in this batch already exist
                    if not self.args.overwrite:
                        batch_exists = all(
                            os.path.exists(os.path.join(self.args.output_dir, f"{i+j:06d}.png"))
                            for j in range(len(batch_prompts))
                        )
                        if batch_exists:
                            print(f"Batch {batch_idx+1}/{total_batches} already exists, skipping...")
                            continue
                    
                    try:
                        # Generate batch
                        print(f"Generating batch {batch_idx+1}/{total_batches} ({len(batch_prompts)} prompts)")
                        batch_output_imgs = self._generate_batch(batch_prompts, batch_idx)
                        
                        # Save batch images
                        self._save_images_batch(batch_output_imgs, batch_prompts, self.args.output_dir, start_idx=i)
                        
                        print(f"Completed batch {batch_idx+1}/{total_batches}")
                        
                    except Exception as e:
                        print(f"Error generating batch {batch_idx+1}: {e}")
                        continue
        
        total_time = time.time() - start_time
        print(f"Generated {len(prompts)} images in {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(prompts):.2f}s")
        print(f"Average time per batch: {total_time/total_batches:.2f}s")
    
    def evaluate_fid(self):
        """
        Evaluate FID (Fréchet Inception Distance)
        """
        if not self.args.eval_fid:
            return
        
        print("Evaluating FID...")
        
        try:
            fid_score = calculate_fid_given_paths(
                [self.args.reference_path, self.args.output_dir],
                batch_size=self.args.batch_size,
                device=self.device,
                dims=self.args.fid_dims,
                img_size=self.args.img_size
            )
            
            self.results['FID'] = float(fid_score)
            print(f"FID Score: {fid_score:.4f}")
            
        except Exception as e:
            print(f"Error computing FID: {e}")
            self.results['FID'] = None
    
    def evaluate_clip_score(self):
        """
        Evaluate CLIP Score
        """
        if not self.args.eval_clip_score or not self.clip_evaluator:
            return
        
        print("Evaluating CLIP Score...")
        
        try:
            # Load prompts and generated images
            with open(self.args.prompts_file, 'r') as f:
                if self.args.prompts_file.endswith('.json'):
                    prompts_data = json.load(f)
                    if isinstance(prompts_data, list):
                        prompts = [item.get('prompt', str(item)) for item in prompts_data]
                    else:
                        prompts = list(prompts_data.values())
                else:
                    prompts = [line.strip() for line in f.readlines()]
            
            # Get generated images
            image_files = sorted([
                os.path.join(self.args.output_dir, f) 
                for f in os.listdir(self.args.output_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            # Limit to available samples
            min_samples = min(len(prompts), len(image_files), self.args.max_samples)
            prompts = prompts[:min_samples]
            image_files = image_files[:min_samples]
            
            # Compute CLIP scores
            scores = self.clip_evaluator.compute_clip_score(
                image_files, prompts, batch_size=self.args.batch_size
            )
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            self.results['CLIP_Score'] = {
                'mean': float(mean_score),
                'std': float(std_score),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
            
            print(f"CLIP Score - Mean: {mean_score:.4f}, Std: {std_score:.4f}")
            
        except Exception as e:
            print(f"Error computing CLIP Score: {e}")
            self.results['CLIP_Score'] = None
    
    def evaluate_geneval(self):
        """
        Evaluate using GenEval benchmark
        """
        if not self.args.eval_geneval:
            return
        
        print("Evaluating GenEval...")
        
        try:
            # Run GenEval evaluation script
            cmd = [
                'python', 'evaluate_geneval.py',
                '--img_path', self.args.output_dir,
                '--prompt_file', self.args.geneval_prompts,
                '--model_path', self.args.geneval_model_path,
                '--config_path', self.args.geneval_config_path,
                '--device', self.device,
                '--exp_name', self.args.exp_name
            ]
            
            if self.args.geneval_class_names:
                cmd.extend(['--class_names_file', self.args.geneval_class_names])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results from output or result file
            result_file = f"{self.args.exp_name}_geneval_results.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    geneval_results = json.load(f)
                self.results['GenEval'] = geneval_results
                print(f"GenEval Overall Accuracy: {geneval_results.get('overall_accuracy', 'N/A'):.4f}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running GenEval: {e}")
            self.results['GenEval'] = None
        except Exception as e:
            print(f"Error processing GenEval results: {e}")
            self.results['GenEval'] = None
    
    def evaluate_image_quality_metrics(self):
        """
        Evaluate additional image quality metrics
        """
        if not self.args.eval_quality_metrics:
            return
        
        print("Evaluating image quality metrics...")
        
        try:
            # Get generated images
            image_files = [
                os.path.join(self.args.output_dir, f) 
                for f in os.listdir(self.args.output_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if not image_files:
                print("No images found for quality evaluation")
                return
            
            # Sample subset for efficiency
            if len(image_files) > self.args.max_samples:
                image_files = np.random.choice(image_files, self.args.max_samples, replace=False)
            
            # Compute image quality metrics
            quality_scores = []
            resolution_stats = []
            
            for img_path in tqdm(image_files, desc="Computing quality metrics"):
                try:
                    img = Image.open(img_path)
                    
                    # Resolution
                    resolution_stats.append(img.size)
                    
                    # Simple quality metrics (can be extended)
                    img_array = np.array(img)
                    if len(img_array.shape) == 3:
                        # Color diversity (std of pixel values)
                        color_diversity = np.std(img_array)
                        quality_scores.append(color_diversity)
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            
            if quality_scores and resolution_stats:
                self.results['Quality_Metrics'] = {
                    'color_diversity_mean': float(np.mean(quality_scores)),
                    'color_diversity_std': float(np.std(quality_scores)),
                    'avg_resolution': [
                        float(np.mean([r[0] for r in resolution_stats])),
                        float(np.mean([r[1] for r in resolution_stats]))
                    ]
                }
                
                print(f"Average Color Diversity: {np.mean(quality_scores):.2f}")
                print(f"Average Resolution: {np.mean([r[0] for r in resolution_stats]):.0f}x{np.mean([r[1] for r in resolution_stats]):.0f}")
        
        except Exception as e:
            print(f"Error computing quality metrics: {e}")
            self.results['Quality_Metrics'] = None
    
    def benchmark_inference_speed(self):
        """
        Benchmark inference speed and memory usage
        """
        if not self.args.eval_speed:
            return
        
        print("Benchmarking inference speed...")
        
        try:
            # Run latency profiling script
            if self.args.latency_script:
                cmd = [
                    'python', self.args.latency_script,
                    '--model_path', self.args.model_path,
                    '--text_model_path', self.args.text_model_path
                ]
                
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                end_time = time.time()
                
                # Parse timing information from output
                self.results['Inference_Speed'] = {
                    'total_time': end_time - start_time,
                    'output': result.stdout
                }
                
                print(f"Inference benchmark completed in {end_time - start_time:.2f}s")
        
        except Exception as e:
            print(f"Error benchmarking inference speed: {e}")
            self.results['Inference_Speed'] = None
    
    def save_results(self):
        """
        Save all evaluation results
        """
        # Add metadata
        self.results['metadata'] = {
            'model_path': self.args.model_path,
            'text_model_path': self.args.text_model_path,
            'experiment_name': self.args.exp_name,
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'args': vars(self.args)
        }
        
        # Save to JSON file
        output_file = f"{self.args.exp_name}_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nAll evaluation results saved to {output_file}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        
        if 'FID' in self.results and self.results['FID'] is not None:
            print(f"FID Score: {self.results['FID']:.4f}")
        
        if 'CLIP_Score' in self.results and self.results['CLIP_Score'] is not None:
            print(f"CLIP Score: {self.results['CLIP_Score']['mean']:.4f} ± {self.results['CLIP_Score']['std']:.4f}")
        
        if 'GenEval' in self.results and self.results['GenEval'] is not None:
            print(f"GenEval Accuracy: {self.results['GenEval'].get('overall_accuracy', 'N/A'):.4f}")
        
        print("-" * 50)
    
    def run_evaluation(self):
        """
        Run comprehensive evaluation
        """
        print(f"Starting HART evaluation: {self.args.exp_name}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Device: {self.device}")
        
        # Generate samples if requested
        if self.args.generate_samples:
            self.generate_samples()
        
        # Run evaluations
        self.evaluate_fid()
        self.evaluate_clip_score()
        self.evaluate_geneval()
        self.evaluate_image_quality_metrics()
        self.benchmark_inference_speed()
        
        # Save results
        self.save_results()
        
        print("\nEvaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive HART Model Evaluation')
    
    # Model and paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to HART model')
    parser.add_argument('--text_model_path', type=str, required=True,
                        help='Path to text model (Qwen2)')
    parser.add_argument('--shield_model_path', type=str,
                        help='Path to shield model (ShieldGemma)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_output',
                        help='Directory to save generated images and results')
    parser.add_argument('--prompts_file', type=str, required=True,
                        help='File containing prompts for evaluation')
    
    # Evaluation options
    parser.add_argument('--eval_fid', action='store_true',
                        help='Evaluate FID score')
    parser.add_argument('--eval_clip_score', action='store_true',
                        help='Evaluate CLIP score')
    parser.add_argument('--eval_geneval', action='store_true',
                        help='Evaluate using GenEval benchmark')
    parser.add_argument('--eval_quality_metrics', action='store_true',
                        help='Evaluate additional image quality metrics')
    parser.add_argument('--eval_speed', action='store_true',
                        help='Benchmark inference speed')
    
    # Sample generation
    parser.add_argument('--generate_samples', action='store_true',
                        help='Generate samples using HART model')
    parser.add_argument('--sample_script', type=str, default='sample.py',
                        help='Path to HART sample generation script (deprecated - now using direct generation)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing generated samples')
    parser.add_argument('--generation_batch_size', type=int, default=4,
                        help='Batch size for image generation')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model for generation')
    parser.add_argument('--max_token_length', type=int, default=300,
                        help='Maximum token length for text encoding')
    parser.add_argument('--use_llm_system_prompt', action='store_true', default=True,
                        help='Use LLM system prompt')
    parser.add_argument('--cfg', type=float, default=4.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--more_smooth', action='store_true', default=True,
                        help='Turn on for more visually smooth samples')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for generation')
    
    # FID evaluation
    parser.add_argument('--reference_path', type=str,
                        help='Path to reference dataset for FID computation')
    parser.add_argument('--fid_dims', type=int, default=2048,
                        help='Dimensionality for FID computation')
    
    # GenEval evaluation
    parser.add_argument('--geneval_prompts', type=str,
                        help='Path to GenEval prompts file')
    parser.add_argument('--geneval_model_path', type=str,
                        help='Path to MMDetection model for GenEval')
    parser.add_argument('--geneval_config_path', type=str,
                        help='Path to MMDetection config for GenEval')
    parser.add_argument('--geneval_class_names', type=str,
                        help='Path to class names file for GenEval')
    
    # Speed benchmarking
    parser.add_argument('--latency_script', type=str, default='latency_profile.py',
                        help='Path to latency profiling script')
    
    # General options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Image size for evaluation')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--exp_name', type=str, default='hart_evaluation',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = HARTEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()