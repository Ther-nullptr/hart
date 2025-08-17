#!/usr/bin/env python3
"""
Example script demonstrating HART cache evaluation on GenEval.
Shows basic usage without requiring full model setup.
"""

import os
import json
from pathlib import Path

def create_example_evaluation():
    """Create example evaluation results to demonstrate the output format"""
    
    # Example cache configurations
    cache_configs = [
        {
            "name": "no-cache",
            "skip_stages": [],
            "cache_stages": [],
            "threshold": 0.7,
            "enable_attn_cache": False,
            "enable_mlp_cache": False
        },
        {
            "name": "conservative",
            "skip_stages": [256],
            "cache_stages": [144],
            "threshold": 0.8,
            "enable_attn_cache": True,
            "enable_mlp_cache": True
        },
        {
            "name": "aggressive",
            "skip_stages": [49, 81, 144, 256],
            "cache_stages": [25, 49, 81, 144],
            "threshold": 0.6,
            "enable_attn_cache": True,
            "enable_mlp_cache": True
        }
    ]
    
    # Example performance results
    performance_results = [
        {
            "config": "no-cache",
            "mean_time": 3.45,
            "throughput": 0.29,
            "mean_memory_mb": 1024.0,
            "overall_quality_score": 0.92,
            "category_scores": {
                "object": 0.94,
                "count": 0.89,
                "spatial": 0.91,
                "attribute": 0.93,
                "color": 0.95,
                "shape": 0.88,
                "texture": 0.86
            }
        },
        {
            "config": "conservative",
            "mean_time": 2.87,
            "throughput": 0.35,
            "mean_memory_mb": 1076.5,
            "overall_quality_score": 0.89,
            "category_scores": {
                "object": 0.91,
                "count": 0.86,
                "spatial": 0.88,
                "attribute": 0.90,
                "color": 0.92,
                "shape": 0.85,
                "texture": 0.83
            }
        },
        {
            "config": "aggressive",
            "mean_time": 1.78,
            "throughput": 0.56,
            "mean_memory_mb": 1145.2,
            "overall_quality_score": 0.78,
            "category_scores": {
                "object": 0.82,
                "count": 0.74,
                "spatial": 0.76,
                "attribute": 0.79,
                "color": 0.83,
                "shape": 0.72,
                "texture": 0.69
            }
        }
    ]
    
    return cache_configs, performance_results

def generate_example_report():
    """Generate an example evaluation report"""
    
    cache_configs, performance_results = create_example_evaluation()
    
    # Create output directory
    output_dir = Path("./example_cache_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save individual configuration results
    for config, perf in zip(cache_configs, performance_results):
        config_dir = output_dir / f"config_{config['name']}"
        config_dir.mkdir(exist_ok=True)
        
        # Detailed results for this configuration
        detailed_results = {
            "cache_config": config,
            "performance": {
                "mean_time": perf["mean_time"],
                "throughput": perf["throughput"], 
                "mean_memory_mb": perf["mean_memory_mb"]
            },
            "overall_metrics": {
                "overall_quality_score": perf["overall_quality_score"]
            },
            "category_results": {
                category: {
                    "num_prompts": 10,
                    "averages": {
                        "avg_detection_confidence": score,
                        "avg_object_accuracy": score * 0.9
                    }
                }
                for category, score in perf["category_scores"].items()
            }
        }
        
        # Save detailed results
        with open(config_dir / "evaluation_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    # Generate comparison results
    comparison_results = {
        "configurations": [
            {
                "cache_config": config,
                "performance": {
                    "mean_time": perf["mean_time"],
                    "throughput": perf["throughput"],
                    "mean_memory_mb": perf["mean_memory_mb"]
                },
                "overall_metrics": {
                    "overall_quality_score": perf["overall_quality_score"]
                }
            }
            for config, perf in zip(cache_configs, performance_results)
        ],
        "comparison_metrics": {
            "mean_time": {
                "values": [perf["mean_time"] for perf in performance_results],
                "best_idx": 2,  # aggressive config is fastest
                "worst_idx": 0   # no-cache is slowest
            },
            "quality_score": {
                "values": [perf["overall_quality_score"] for perf in performance_results],
                "best_idx": 0,   # no-cache has best quality
                "worst_idx": 2   # aggressive has worst quality
            }
        }
    }
    
    # Save comparison results
    with open(output_dir / "cache_comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Generate markdown report
    report_content = """# HART Cache Configuration Comparison Report

## Configuration Summary

| Config | Skip Stages | Cache Stages | Threshold | Attn Cache | MLP Cache |
|--------|-------------|--------------|-----------|------------|-----------|
| 1 | [] | [] | 0.7 | False | False |
| 2 | [256] | [144] | 0.8 | True | True |
| 3 | [49, 81, 144, 256] | [25, 49, 81, 144] | 0.6 | True | True |

## Performance Comparison

### Generation Time
- Best (fastest): Config 3 with 1.78s
- Times: ['3.45s', '2.87s', '1.78s']

### Throughput
- Best: Config 3 with 0.56 img/s
- Throughputs: ['0.29', '0.35', '0.56']

### Quality Score
- Best: Config 1 with score 0.92
- Scores: ['0.92', '0.89', '0.78']

## Category-wise Performance

### Object Detection
| Config | Score | Change from Baseline |
|--------|-------|---------------------|
| no-cache | 0.94 | - |
| conservative | 0.91 | -3.2% |
| aggressive | 0.82 | -12.8% |

### Counting Accuracy
| Config | Score | Change from Baseline |
|--------|-------|---------------------|
| no-cache | 0.89 | - |
| conservative | 0.86 | -3.4% |
| aggressive | 0.74 | -16.9% |

### Spatial Relationships
| Config | Score | Change from Baseline |
|--------|-------|---------------------|
| no-cache | 0.91 | - |
| conservative | 0.88 | -3.3% |
| aggressive | 0.76 | -16.5% |

## Recommendations

- **Balanced Performance**: Config 2 offers the best speed/quality trade-off
- **Speed Priority**: Config 3 for maximum generation speed (1.9x faster)
- **Quality Priority**: Config 1 for best generation quality

## Analysis

### Speed vs Quality Trade-off
The evaluation shows the expected trade-off between generation speed and quality:
- **No Cache**: Highest quality (0.92) but slowest generation (3.45s)
- **Conservative**: Good balance with 17% speedup and only 3% quality loss
- **Aggressive**: Significant speedup (48% faster) but notable quality degradation (15% loss)

### Category Impact
Caching affects different compositional aspects differently:
- **Most Affected**: Texture understanding and counting accuracy
- **Least Affected**: Color representation and object detection
- **Spatial Relationships**: Moderately affected by aggressive caching

### Memory Usage
Cache configurations show modest memory increases:
- Conservative: +5% memory for 17% speedup
- Aggressive: +12% memory for 48% speedup

### Recommendations by Use Case

**Research & Development**: Use no-cache for highest quality
**Production Balanced**: Use conservative for good speed/quality balance  
**Real-time Applications**: Use aggressive for maximum speed
**Memory-Constrained**: Consider attention-only caching configurations
"""
    
    with open(output_dir / "comparison_report.md", 'w') as f:
        f.write(report_content)
    
    print(f"Example evaluation results generated in: {output_dir}")
    print("\nGenerated files:")
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            print(f"  {file_path}")
    
    return output_dir

def print_usage_examples():
    """Print usage examples for the evaluation script"""
    
    print("\n" + "="*60)
    print("HART Cache GenEval Evaluation - Usage Examples")
    print("="*60)
    
    print("\n1. Compare all cache presets:")
    print("./run_hart_cache_geneval.sh \\")
    print("    --model_path /path/to/hart/model \\")
    print("    --geneval_path /path/to/geneval.jsonl \\")
    print("    --compare_presets")
    
    print("\n2. Evaluate specific preset:")
    print("./run_hart_cache_geneval.sh \\")
    print("    --model_path /path/to/hart/model \\")
    print("    --geneval_path /path/to/geneval.jsonl \\")
    print("    --cache_preset aggressive")
    
    print("\n3. Custom cache configuration:")
    print("./run_hart_cache_geneval.sh \\")
    print("    --model_path /path/to/hart/model \\")
    print("    --geneval_path /path/to/geneval.jsonl \\")
    print("    --skip_stages 144 256 \\")
    print("    --cache_stages 81 144 \\")
    print("    --threshold 0.65")
    
    print("\n4. Memory-efficient evaluation:")
    print("./run_hart_cache_geneval.sh \\")
    print("    --model_path /path/to/hart/model \\")
    print("    --geneval_path /path/to/geneval.jsonl \\")
    print("    --skip_stages 144 256 \\")
    print("    --cache_stages 144 \\")
    print("    --disable_mlp_cache \\")
    print("    --batch_size 2")
    
    print("\n5. With object detection:")
    print("./run_hart_cache_geneval.sh \\")
    print("    --model_path /path/to/hart/model \\")
    print("    --geneval_path /path/to/geneval.jsonl \\")
    print("    --detector_config /path/to/detector_config.py \\")
    print("    --detector_model /path/to/detector_model.pth \\")
    print("    --compare_presets")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("HART Cache GenEval Evaluation - Example Generator")
    
    # Generate example results
    output_dir = generate_example_report()
    
    # Print usage examples
    print_usage_examples()
    
    print(f"\n✅ Example evaluation completed!")
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 View comparison report: {output_dir}/comparison_report.md")
    print(f"📈 View detailed results: {output_dir}/cache_comparison_results.json")
    
    print(f"\n🚀 To run actual evaluation:")
    print(f"./run_hart_cache_geneval.sh --model_path YOUR_MODEL --geneval_path YOUR_DATA --compare_presets")