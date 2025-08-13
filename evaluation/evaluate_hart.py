"""
Comprehensive evaluation script for HART model
Integrates multiple evaluation metrics and benchmarks
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Import evaluation modules
from compute_fid import calculate_fid_given_paths
from compute_clip_score import CLIPEvaluator


class HARTEvaluator:
    """
    Comprehensive HART model evaluator
    Supports FID, CLIP Score, GenEval, and custom metrics
    """
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.results = {}
        
        # Initialize CLIP evaluator if needed
        if args.eval_clip_score:
            self.clip_evaluator = CLIPEvaluator(device=self.device)
        else:
            self.clip_evaluator = None
    
    def generate_samples(self):
        """
        Generate samples using HART model
        """
        if self.args.sample_script and self.args.prompts_file:
            print("Generating samples using HART model...")
            
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
            
            # Create output directory
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Generate samples for each prompt
            for i, prompt in enumerate(tqdm(prompts, desc="Generating samples")):
                output_path = os.path.join(self.args.output_dir, f"{i:06d}.png")
                
                if os.path.exists(output_path) and not self.args.overwrite:
                    continue
                
                # Run sample script
                cmd = [
                    'python', self.args.sample_script,
                    '--model_path', self.args.model_path,
                    '--text_model_path', self.args.text_model_path,
                    '--prompt', prompt,
                    '--sample_folder_dir', os.path.dirname(output_path),
                    '--store_seperately'
                ]
                
                if self.args.shield_model_path:
                    cmd.extend(['--shield_model_path', self.args.shield_model_path])
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"Generated sample {i+1}/{len(prompts)}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating sample {i}: {e}")
        else:
            print("Sample generation skipped. Using existing images.")
    
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
                        help='Path to HART sample generation script')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing generated samples')
    
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