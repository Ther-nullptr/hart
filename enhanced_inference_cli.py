#!/usr/bin/env python3
"""
Enhanced HART inference script with command line cache control.
Demonstrates VAR-style caching implementation in HART.
"""

import argparse
import json
import os
import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

from hart.modules.models.transformer.hart_transformer_t2i_enhanced import HARTForT2IEnhanced
from hart.modules.models.transformer.configuration import HARTForT2IConfig
from hart.modules.networks.basic_hart_enhanced import CacheConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced HART inference with VAR-style caching")
    
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
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.0,
                       help="Top-p sampling")
    
    # Add cache arguments using the model's method
    HARTForT2IEnhanced.add_cache_arguments(parser)
    
    # Analysis arguments
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--compare_configs", action="store_true",
                       help="Compare multiple cache configurations")
    parser.add_argument("--output_dir", type=str, default="./enhanced_hart_output",
                       help="Output directory for results")
    parser.add_argument("--save_images", action="store_true",
                       help="Save generated images")
    
    return parser.parse_args()


def load_model(model_path: str, args):
    """Load enhanced HART model with cache configuration"""
    print(f"Loading model from {model_path}")
    
    # Load configuration
    config = HARTForT2IConfig.from_pretrained(model_path)
    
    # Create enhanced model with cache configuration from args
    model = HARTForT2IEnhanced.from_args_and_config(args, config)
    
    # Load model weights
    model = model.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def generate_samples(model: HARTForT2IEnhanced, prompts: list, args) -> Dict[str, Any]:
    """Generate samples with the model"""
    print(f"Generating {args.num_samples} samples for {len(prompts)} prompts")
    print(f"Cache configuration:")
    print(f"  Skip stages: {model.cache_config.skip_stages}")
    print(f"  Cache stages: {model.cache_config.cache_stages}")
    print(f"  Attention cache: {model.cache_config.enable_attn_cache}")
    print(f"  MLP cache: {model.cache_config.enable_mlp_cache}")
    print(f"  Threshold: {model.cache_config.threshold}")
    print(f"  Interpolation: {model.cache_config.interpolation_mode}")
    
    results = {
        'prompts': prompts,
        'samples': [],
        'generation_stats': [],
        'cache_config': {
            'skip_stages': model.cache_config.skip_stages,
            'cache_stages': model.cache_config.cache_stages,
            'enable_attn_cache': model.cache_config.enable_attn_cache,
            'enable_mlp_cache': model.cache_config.enable_mlp_cache,
            'threshold': model.cache_config.threshold,
        }
    }
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"\\nGenerating for prompt {i+1}/{len(prompts)}: {prompt}")
            
            prompt_results = []
            prompt_times = []
            
            for sample_idx in range(args.num_samples):
                start_time = time.time()
                
                try:
                    # Generate sample
                    sample = model.autoregressive_infer_cfg(
                        B=args.batch_size,
                        label_B=None,  # Random class
                        g_seed=args.seed + sample_idx,
                        cfg=args.cfg_scale,
                        top_k=args.top_k,
                        top_p=args.top_p
                    )
                    
                    generation_time = time.time() - start_time
                    prompt_times.append(generation_time)
                    prompt_results.append(sample)
                    
                    print(f"  Sample {sample_idx+1}/{args.num_samples} - Time: {generation_time:.2f}s")
                    
                    # Save image if requested
                    if args.save_images:
                        os.makedirs(args.output_dir, exist_ok=True)
                        import torchvision
                        for b in range(sample.shape[0]):
                            image_path = os.path.join(
                                args.output_dir, 
                                f"prompt_{i}_sample_{sample_idx}_batch_{b}.png"
                            )
                            torchvision.utils.save_image(sample[b], image_path)
                    
                except Exception as e:
                    print(f"  Error generating sample {sample_idx+1}: {e}")
                    continue
            
            results['samples'].append(prompt_results)
            results['generation_stats'].append({
                'prompt': prompt,
                'mean_time': sum(prompt_times) / len(prompt_times) if prompt_times else 0,
                'total_samples': len(prompt_results),
                'successful_samples': len(prompt_results)
            })
    
    return results


def run_benchmark(model: HARTForT2IEnhanced, args) -> Dict[str, Any]:
    """Run performance benchmark"""
    print("\\nRunning performance benchmark...")
    
    # Warm up
    print("Warming up...")
    try:
        _ = model.autoregressive_infer_cfg(
            B=1, label_B=None, g_seed=42, cfg=args.cfg_scale
        )
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    times = []
    memory_usage = []
    
    num_benchmark_samples = 5
    print(f"Benchmarking with {num_benchmark_samples} samples...")
    
    for i in range(num_benchmark_samples):
        # Monitor memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        try:
            # Generate sample
            _ = model.autoregressive_infer_cfg(
                B=args.batch_size,
                label_B=None,
                g_seed=i,
                cfg=args.cfg_scale
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Monitor memory after
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                memory_usage.append({
                    'allocated_mb': (mem_after - mem_before) / 1024 / 1024,
                    'peak_mb': peak_mem / 1024 / 1024
                })
            
            print(f"  Benchmark {i+1}/{num_benchmark_samples} - Time: {times[-1]:.3f}s")
            
        except Exception as e:
            print(f"  Benchmark {i+1} failed: {e}")
            continue
    
    if not times:
        print("No successful benchmark runs!")
        return {}
    
    # Calculate statistics
    import numpy as np
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = args.batch_size / mean_time
    
    mean_memory = np.mean([m['allocated_mb'] for m in memory_usage]) if memory_usage else 0
    peak_memory = np.max([m['peak_mb'] for m in memory_usage]) if memory_usage else 0
    
    results = {
        'benchmark': {
            'mean_time': mean_time,
            'std_time': std_time,
            'throughput_img_per_sec': throughput,
            'mean_memory_mb': mean_memory,
            'peak_memory_mb': peak_memory,
            'num_samples': len(times)
        },
        'cache_config': {
            'skip_stages': model.cache_config.skip_stages,
            'cache_stages': model.cache_config.cache_stages,
            'threshold': model.cache_config.threshold,
            'interpolation_mode': model.cache_config.interpolation_mode
        }
    }
    
    print("\\nBenchmark Results:")
    print(f"  Mean time per sample: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"  Throughput: {throughput:.2f} images/sec")
    if torch.cuda.is_available():
        print(f"  Memory usage: {mean_memory:.1f} MB (peak: {peak_memory:.1f} MB)")
    
    return results


def compare_configurations(model_path: str, args) -> Dict[str, Any]:
    """Compare multiple cache configurations"""
    print("\\nComparing cache configurations...")
    
    configs_to_compare = [
        ('no-cache', {'cache_preset': 'no-cache'}),
        ('conservative', {'cache_preset': 'conservative'}),
        ('original', {'cache_preset': 'original'}),
        ('aggressive', {'cache_preset': 'aggressive'}),
    ]
    
    results = {}
    
    for config_name, config_args in configs_to_compare:
        print(f"\\nTesting configuration: {config_name}")
        
        # Create args with specific config
        test_args = argparse.Namespace(**vars(args))
        for key, value in config_args.items():
            setattr(test_args, key, value)
        
        try:
            # Create model with specific cache config
            model = load_model(model_path, test_args)
            
            # Run benchmark
            benchmark_result = run_benchmark(model, test_args)
            
            if benchmark_result:
                results[config_name] = {
                    'mean_time': benchmark_result['benchmark']['mean_time'],
                    'throughput': benchmark_result['benchmark']['throughput_img_per_sec'],
                    'memory_mb': benchmark_result['benchmark']['mean_memory_mb'],
                    'config': benchmark_result['cache_config']
                }
                
                print(f"  Time: {results[config_name]['mean_time']:.3f}s")
                print(f"  Throughput: {results[config_name]['throughput']:.2f} img/s")
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Configuration {config_name} failed: {e}")
            results[config_name] = {'error': str(e)}
    
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
    
    print("Enhanced HART Inference with VAR-style Caching")
    print("=" * 50)
    
    # Special case: compare configurations
    if args.compare_configs:
        comparison_results = compare_configurations(args.model_path, args)
        save_results(comparison_results, args.output_dir, "cache_comparison.json")
        return
    
    # Load model
    model = load_model(args.model_path, args)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_results = run_benchmark(model, args)
        if benchmark_results:
            save_results(benchmark_results, args.output_dir, "benchmark_results.json")
    
    # Generate samples
    generation_results = generate_samples(model, args.prompts, args)
    save_results(generation_results, args.output_dir, "generation_results.json")
    
    # Save cache statistics
    cache_stats = model.get_cache_statistics()
    save_results(cache_stats, args.output_dir, "cache_statistics.json")
    
    print(f"\\nAll results saved to {args.output_dir}")
    
    # Print summary
    print("\\n" + "=" * 50)
    print("Generation Summary:")
    for i, stats in enumerate(generation_results['generation_stats']):
        print(f"  Prompt {i+1}: {stats['successful_samples']}/{args.num_samples} samples, "
              f"avg {stats['mean_time']:.2f}s per sample")


if __name__ == "__main__":
    main()