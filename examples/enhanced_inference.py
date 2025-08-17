#!/usr/bin/env python3
"""
Example script demonstrating enhanced HART inference with improved caching.
Shows how to use different cache configurations and benchmark performance.
"""

import argparse
import json
import os
import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

# Add hart modules to path
import sys
sys.path.append('/home/ther-nullptr/research/hart')

from hart.modules.enhanced_hart import create_enhanced_hart_model, EnhancedHARTForT2I
from hart.modules.cache_config import HARTCacheConfig, get_cache_presets
from hart.modules.models.transformer.configuration import HARTForT2IConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced HART inference with caching")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to HART model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to tokenizer")
    
    # Generation arguments
    parser.add_argument("--prompts", type=str, nargs="+",
                       default=["A beautiful sunset over mountains"],
                       help="Text prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of samples to generate")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Cache arguments
    parser.add_argument("--cache_preset", type=str, default="original-cache",
                       choices=list(get_cache_presets().keys()),
                       help="Cache configuration preset")
    parser.add_argument("--custom_cache_config", type=str,
                       help="Path to custom cache configuration JSON")
    parser.add_argument("--skip_stages", type=int, nargs="*",
                       help="Custom skip stages (overrides preset)")
    parser.add_argument("--cache_stages", type=int, nargs="*",
                       help="Custom cache stages (overrides preset)")
    parser.add_argument("--threshold", type=float,
                       help="Custom similarity threshold (overrides preset)")
    parser.add_argument("--interpolation_mode", type=str,
                       choices=["bilinear", "nearest", "bicubic"],
                       help="Interpolation mode (overrides preset)")
    
    # Analysis arguments
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--calibrate", action="store_true",
                       help="Run threshold calibration")
    parser.add_argument("--compare_configs", action="store_true",
                       help="Compare multiple cache configurations")
    parser.add_argument("--output_dir", type=str, default="./enhanced_hart_output",
                       help="Output directory for results")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, cache_config: HARTCacheConfig):
    """Load enhanced HART model and tokenizer"""
    print(f"Loading model from {model_path}")
    print(f"Loading tokenizer from {tokenizer_path}")
    
    # Load configuration (you may need to adjust this based on your setup)
    config = HARTForT2IConfig.from_pretrained(model_path)
    
    # Create enhanced model
    model = create_enhanced_hart_model(
        config=config,
        custom_cache_config=cache_config
    )
    
    # Load model weights
    model = model.from_pretrained(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer


def create_cache_config(args) -> HARTCacheConfig:
    """Create cache configuration based on arguments"""
    if args.custom_cache_config:
        cache_config = HARTCacheConfig.from_json(args.custom_cache_config)
    else:
        cache_config = HARTCacheConfig.from_preset(args.cache_preset)
    
    # Override with custom arguments if provided
    if args.skip_stages is not None:
        cache_config.skip_stages = args.skip_stages
    if args.cache_stages is not None:
        cache_config.cache_stages = args.cache_stages
    if args.threshold is not None:
        cache_config.threshold = args.threshold
    if args.interpolation_mode is not None:
        cache_config.interpolation_mode = args.interpolation_mode
    
    return cache_config


def generate_samples(model: EnhancedHARTForT2I, prompts: list, args) -> Dict[str, Any]:
    """Generate samples with the model"""
    print(f"Generating {args.num_samples} samples for {len(prompts)} prompts")
    
    results = {
        'prompts': prompts,
        'samples': [],
        'generation_stats': []
    }
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"Generating for prompt {i+1}/{len(prompts)}: {prompt}")
            
            prompt_results = []
            prompt_times = []
            
            for sample_idx in range(args.num_samples):
                start_time = time.time()
                
                # Generate sample (you may need to adjust this based on your setup)
                # This is a placeholder - actual implementation depends on your HART setup
                try:
                    sample = model.autoregressive_infer_cfg(
                        B=args.batch_size,
                        label_B=None,  # or encode prompt to label
                        g_seed=args.seed + sample_idx,
                        cfg=args.cfg_scale
                    )
                    
                    generation_time = time.time() - start_time
                    prompt_times.append(generation_time)
                    prompt_results.append(sample)
                    
                    print(f"  Sample {sample_idx+1}/{args.num_samples} - Time: {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Error generating sample {sample_idx+1}: {e}")
                    continue
            
            results['samples'].append(prompt_results)
            results['generation_stats'].append({
                'prompt': prompt,
                'mean_time': sum(prompt_times) / len(prompt_times) if prompt_times else 0,
                'total_samples': len(prompt_results)
            })
    
    return results


def run_benchmark(model: EnhancedHARTForT2I, args) -> Dict[str, Any]:
    """Run performance benchmark"""
    print("Running performance benchmark...")
    
    benchmark_results = model.benchmark_performance(
        num_samples=10,
        batch_size=args.batch_size
    )
    
    print("\nBenchmark Results:")
    print(f"  Mean time per sample: {benchmark_results['benchmark']['mean_time']:.3f}s")
    print(f"  Throughput: {benchmark_results['benchmark']['throughput_img_per_sec']:.2f} images/sec")
    print(f"  Cache hit rate: {benchmark_results['benchmark']['cache_hit_rate']:.2f}")
    
    if torch.cuda.is_available():
        print(f"  Memory usage: {benchmark_results['benchmark']['mean_memory_mb']:.1f} MB")
    
    return benchmark_results


def run_calibration(model: EnhancedHARTForT2I, args) -> Dict[str, float]:
    """Run threshold calibration"""
    print("Running threshold calibration...")
    
    recommendations = model.calibrate_thresholds(num_samples=20)
    
    print("\nCalibration Results:")
    for key, value in recommendations.items():
        print(f"  Recommended {key}: {value:.3f}")
    
    return recommendations


def compare_configurations(model_path: str, tokenizer_path: str, args) -> Dict[str, Any]:
    """Compare multiple cache configurations"""
    print("Comparing cache configurations...")
    
    configs_to_compare = [
        "no-cache",
        "conservative-cache", 
        "original-cache",
        "aggressive-cache"
    ]
    
    results = {}
    
    for config_name in configs_to_compare:
        print(f"\nTesting configuration: {config_name}")
        
        # Create model with specific cache config
        cache_config = HARTCacheConfig.from_preset(config_name)
        model, _ = load_model_and_tokenizer(model_path, tokenizer_path, cache_config)
        
        # Run benchmark
        benchmark_result = model.benchmark_performance(num_samples=5, batch_size=1)
        
        results[config_name] = {
            'mean_time': benchmark_result['benchmark']['mean_time'],
            'throughput': benchmark_result['benchmark']['throughput_img_per_sec'],
            'cache_hit_rate': benchmark_result['benchmark']['cache_hit_rate'],
            'config': benchmark_result['cache_config']
        }
        
        print(f"  Time: {results[config_name]['mean_time']:.3f}s")
        print(f"  Throughput: {results[config_name]['throughput']:.2f} img/s")
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, filename: str):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


def main():
    args = parse_args()
    
    # Create cache configuration
    cache_config = create_cache_config(args)
    print(f"Using cache configuration:")
    print(f"  Skip stages: {cache_config.skip_stages}")
    print(f"  Cache stages: {cache_config.cache_stages}")
    print(f"  Threshold: {cache_config.threshold}")
    print(f"  Interpolation mode: {cache_config.interpolation_mode}")
    
    # Special case: compare configurations
    if args.compare_configs:
        comparison_results = compare_configurations(args.model_path, args.tokenizer_path, args)
        save_results(comparison_results, args.output_dir, "cache_comparison.json")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path, cache_config)
    
    # Run calibration if requested
    if args.calibrate:
        calibration_results = run_calibration(model, args)
        save_results(calibration_results, args.output_dir, "calibration_results.json")
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_results = run_benchmark(model, args)
        save_results(benchmark_results, args.output_dir, "benchmark_results.json")
    
    # Generate samples
    generation_results = generate_samples(model, args.prompts, args)
    save_results(generation_results, args.output_dir, "generation_results.json")
    
    # Save final cache statistics
    cache_stats = model.get_cache_statistics()
    save_results(cache_stats, args.output_dir, "cache_statistics.json")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()