# Enhanced HART Caching System

Enhanced HART implementation with improved caching mechanisms inspired by VAR's multi-stage caching system. Provides flexible control over attention and MLP layer caching with adaptive thresholding.

## 🚀 Key Features

- **Flexible Stage Control**: Support for skipping any combination of stages (1-256)
- **Layer-Specific Caching**: Independent control over attention and MLP layer caching
- **Adaptive Thresholding**: Dynamic threshold adjustment based on similarity patterns
- **Multiple Interpolation Modes**: Bilinear, nearest, and bicubic upsampling
- **Comprehensive Benchmarking**: Performance comparison across different configurations
- **Easy Configuration**: JSON-based configuration with preset options
- **Production Ready**: Optimized for both research and deployment scenarios

## 📁 Enhanced File Structure

```
hart/
├── hart/modules/
│   ├── cache_config.py           # Cache configuration classes
│   ├── enhanced_hart.py          # Enhanced HART model with caching
│   └── models/transformer/
│       └── hart_transformer_t2i.py  # Original HART transformer
├── examples/
│   └── enhanced_inference.py     # Example usage script
├── hart_cache_presets.json       # Configuration presets
└── README_ENHANCED_CACHING.md    # This file
```

## 🔧 Quick Start

### 1. Basic Usage

```python
from hart.modules.enhanced_hart import create_enhanced_hart_model
from hart.modules.cache_config import HARTCacheConfig
from hart.modules.models.transformer.configuration import HARTForT2IConfig

# Load configuration
config = HARTForT2IConfig.from_pretrained("path/to/model")

# Create enhanced model with preset
model = create_enhanced_hart_model(
    config=config,
    cache_preset="original-cache"
)

# Generate with caching
images = model.autoregressive_infer_cfg(
    B=4, label_B=None, g_seed=42, cfg=1.5
)
```

### 2. Custom Configuration

```python
# Create custom cache configuration
cache_config = HARTCacheConfig(
    skip_stages=[100, 169, 256],
    cache_stages=[64, 100, 169],
    enable_attn_cache=True,
    enable_mlp_cache=True,
    threshold=0.65,
    adaptive_threshold=True,
    interpolation_mode='bilinear'
)

# Create model with custom config
model = create_enhanced_hart_model(
    config=config,
    custom_cache_config=cache_config
)
```

### 3. Configuration from Presets

```python
# Available presets
presets = [
    "no-cache",           # Baseline performance
    "conservative-cache", # Minimal quality loss
    "original-cache",     # Original HART caching
    "aggressive-cache",   # Maximum speedup
    "ultra-fast",         # Ultra-fast with adaptive thresholding
    "high-quality",       # Quality-focused
    "memory-efficient",   # Memory-optimized
    "production-balanced" # Production-ready balanced
]

# Use preset
cache_config = HARTCacheConfig.from_preset("aggressive-cache")
```

## ⚙️ Configuration Options

### Cache Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `skip_stages` | List[int] | Stages to skip computation | `[]` |
| `cache_stages` | List[int] | Stages to cache results | `[]` |
| `enable_attn_cache` | bool | Enable attention caching | `True` |
| `enable_mlp_cache` | bool | Enable MLP caching | `True` |
| `threshold` | float | Similarity threshold | `0.7` |
| `adaptive_threshold` | bool | Use adaptive thresholding | `False` |
| `interpolation_mode` | str | Upsampling mode | `'bilinear'` |
| `calibration_samples` | int | Samples for calibration | `100` |
| `similarity_window` | int | Window for adaptive threshold | `20` |

### Valid Stage Numbers

| Stage | Resolution | Characteristics |
|-------|------------|-----------------|
| 1 | 1×1 | Initial stage - never cached |
| 4 | 2×2 | Early stage - rarely cached |
| 9 | 3×3 | Early stage - rarely cached |
| 16 | 4×4 | Early-mid stage |
| 25 | 5×5 | Mid stage - good for caching |
| 36 | 6×6 | Mid stage - excellent for caching |
| 64 | 8×8 | Mid-late stage - excellent for caching |
| 100 | 10×10 | Late stage - prime for caching |
| 169 | 13×13 | Late stage - most commonly cached |
| 256 | 16×16 | Final stage - often skipped |

## 🚀 Example Usage Script

```bash
# Basic inference with original caching
python examples/enhanced_inference.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache_preset original-cache \
    --prompts "A beautiful sunset" "A cat in a garden"

# Aggressive caching for speed
python examples/enhanced_inference.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache_preset aggressive-cache \
    --benchmark

# Custom configuration
python examples/enhanced_inference.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --skip_stages 100 169 256 \
    --cache_stages 64 100 169 \
    --threshold 0.65 \
    --interpolation_mode bicubic

# Compare configurations
python examples/enhanced_inference.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --compare_configs
```

## 📊 Performance Analysis

### Benchmarking

```python
# Run performance benchmark
benchmark_results = model.benchmark_performance(
    num_samples=10,
    batch_size=1
)

print(f"Mean time: {benchmark_results['benchmark']['mean_time']:.3f}s")
print(f"Throughput: {benchmark_results['benchmark']['throughput_img_per_sec']:.2f} img/s")
print(f"Cache hit rate: {benchmark_results['benchmark']['cache_hit_rate']:.2f}")
```

### Expected Performance Improvements

| Configuration | Speedup | Memory | Quality |
|---------------|---------|---------|---------|
| No Cache | 1.0× | Baseline | Best |
| Conservative | 1.2-1.5× | +5% | Excellent |
| Original | 1.5-2.0× | +10% | Very Good |
| Aggressive | 2.0-3.0× | +15% | Good |
| Ultra Fast | 3.0-4.0× | +20% | Acceptable |

## 🔬 Advanced Features

### 1. Adaptive Thresholding

```python
# Enable adaptive thresholding
cache_config = HARTCacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    adaptive_threshold=True,
    threshold=0.7,  # Initial threshold
    similarity_window=20  # History window
)
```

Adaptive thresholding automatically adjusts cache usage thresholds based on observed similarity patterns.

### 2. Threshold Calibration

```python
# Run calibration to optimize thresholds
recommendations = model.calibrate_thresholds(num_samples=50)
print("Recommended thresholds:", recommendations)

# Apply recommendations
cache_config.threshold = recommendations['attn_threshold']
model.set_cache_config(cache_config)
```

### 3. Layer-Specific Control

```python
# Attention-only caching (memory efficient)
cache_config = HARTCacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    enable_attn_cache=True,
    enable_mlp_cache=False
)

# MLP-only caching
cache_config = HARTCacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    enable_attn_cache=False,
    enable_mlp_cache=True
)
```

### 4. Multiple Interpolation Modes

```python
# Quality-focused: bicubic interpolation
cache_config = HARTCacheConfig(
    interpolation_mode='bicubic',
    threshold=0.8
)

# Speed-focused: nearest neighbor
cache_config = HARTCacheConfig(
    interpolation_mode='nearest',
    threshold=0.6
)
```

## 🎯 Tuning Guidelines

### For Maximum Speed
- Use aggressive skip stages: `[64, 100, 169, 256]`
- Lower threshold: `0.5-0.6`
- Enable both attention and MLP caching
- Consider adaptive thresholding
- Use bilinear or nearest interpolation

### For Best Quality
- Minimal skip stages: `[256]` only
- Higher threshold: `0.8-0.9`
- Use bicubic interpolation
- Conservative cache stages
- Disable adaptive thresholding

### For Memory Efficiency
- Attention-only caching (`enable_mlp_cache=False`)
- Fewer cache stages
- Higher threshold to reduce cache usage
- Use bilinear interpolation

### For Production Balance
- Original configuration: skip `[169, 256]`, cache `[100, 169]`
- Threshold: `0.7`
- Both attention and MLP caching
- Adaptive thresholding enabled
- Bilinear interpolation

## 📈 Monitoring and Statistics

### Cache Usage Statistics

```python
# Get comprehensive statistics
stats = model.get_cache_statistics()

print("Cache hit rate:", stats['cache_hit_rate'])
print("Similarity stats:", stats['similarity_stats'])

if model.cache_config.adaptive_threshold:
    print("Adaptive thresholds:", stats['adaptive_thresholds'])
```

### Save/Load Configurations

```python
# Save current configuration
model.save_cache_config("my_config.json")

# Load configuration
model.load_cache_config("my_config.json")

# Save from cache config object
cache_config.to_json("preset_config.json")

# Load into cache config object
cache_config = HARTCacheConfig.from_json("preset_config.json")
```

## 🔄 Integration with Existing HART

The enhanced caching system is designed to be backward compatible with existing HART implementations:

1. **Drop-in Replacement**: `EnhancedHARTForT2I` extends the original `HARTForT2I`
2. **Legacy Support**: Supports original `use_cache`, `calibration`, and `sim_path` parameters
3. **No Breaking Changes**: Existing code continues to work without modification
4. **Enhanced Features**: Additional capabilities available through new configuration system

### Migration Example

```python
# Original HART usage
model = HARTForT2I(config)
model.use_cache = True

# Enhanced HART usage (backward compatible)
enhanced_model = EnhancedHARTForT2I(config)
enhanced_model.use_cache = True  # Still works

# New enhanced features
cache_config = HARTCacheConfig.from_preset("aggressive-cache")
enhanced_model.set_cache_config(cache_config)
```

## ❗ Limitations and Considerations

### Quality vs Speed Trade-off
- More aggressive caching increases speed but may reduce output quality
- Higher resolution stages (169, 256) are most beneficial to skip but also most impactful on quality
- Adaptive thresholding helps balance this trade-off automatically

### Memory Usage
- Caching increases memory usage proportional to number of cached stages
- Attention-only caching significantly reduces memory overhead
- Consider memory constraints when selecting cache stages

### Interpolation Artifacts
- Feature interpolation may introduce slight artifacts
- Higher quality interpolation modes (bicubic) reduce artifacts but increase computation
- Bilinear interpolation provides good balance for most use cases

## 🤝 Contributing

To extend the enhanced caching system:

1. **Add New Interpolation Modes**: Extend `enhanced_feature_interpolate()` method
2. **Custom Similarity Metrics**: Modify similarity computation in cache decision logic
3. **Advanced Scheduling**: Implement dynamic stage-specific threshold schedules
4. **Memory Optimization**: Add cache eviction strategies for long sequences

## 📚 References

- Original HART Paper: [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/abs/2410.10812)
- VAR Caching Implementation: [FastVAR Post-training Speedup](https://github.com/csguoh/FastVAR)
- Enhanced VAR Caching: Local VAR implementation with multi-stage caching

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or number of cache stages
   - Use attention-only caching (`enable_mlp_cache=False`)
   - Consider gradient checkpointing

2. **Quality Degradation**
   - Increase similarity threshold
   - Use fewer skip stages
   - Try bicubic interpolation
   - Run threshold calibration

3. **No Speedup Observed**
   - Ensure skip stages are computationally expensive (e.g., 169, 256)
   - Check that cache stages provide good interpolation base
   - Run calibration to optimize thresholds
   - Verify cache hit rate in statistics

4. **Configuration Errors**
   - Validate configuration with `cache_config.validate()`
   - Check stage numbers are valid (1, 4, 9, 16, 25, 36, 64, 100, 169, 256)
   - Ensure threshold is between 0 and 1

This enhanced caching system provides fine-grained control over the speed-quality trade-off in HART models, enabling efficient deployment across different computational constraints and quality requirements.