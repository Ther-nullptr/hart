# VAR-Style Caching Implementation for HART

This implementation provides true VAR-style caching for HART transformers, closely following the original VAR caching mechanism for maximum compatibility and performance.

## 🚀 Key Features

- **True VAR Implementation**: Exact replication of VAR's caching logic and data structures
- **Command Line Control**: Full command line interface for all cache parameters
- **Backward Compatibility**: Works with existing HART models without modification
- **Enhanced Blocks**: Drop-in replacement transformer blocks with caching
- **Preset Configurations**: 9 predefined cache configurations for different use cases
- **Real-time Statistics**: Comprehensive cache usage and performance tracking

## 📁 File Structure

```
hart/
├── hart/modules/networks/
│   └── basic_hart_enhanced.py          # Enhanced transformer blocks with VAR-style caching
├── hart/modules/models/transformer/
│   └── hart_transformer_t2i_enhanced.py # Enhanced HART model with caching
├── enhanced_inference_cli.py            # Command line inference script
└── README_VAR_STYLE_CACHING.md         # This file
```

## 🔧 Core Implementation

### Enhanced Transformer Blocks

The implementation provides three enhanced transformer blocks that follow VAR's caching approach:

1. **`FFNEnhanced`** - MLP block with VAR-style feature caching
2. **`SelfAttentionEnhanced`** - Attention block with VAR-style feature caching  
3. **`AdaLNSelfAttnEnhanced`** - Complete transformer block combining both

### Cache Configuration

```python
class CacheConfig:
    def __init__(
        self,
        skip_stages: List[int] = None,        # Stages to skip (e.g., [169, 256])
        cache_stages: List[int] = None,       # Stages to cache (e.g., [100, 169])
        enable_attn_cache: bool = True,       # Enable attention caching
        enable_mlp_cache: bool = True,        # Enable MLP caching
        threshold: float = 0.7,               # Similarity threshold
        adaptive_threshold: bool = False,     # Use adaptive thresholding
        interpolation_mode: str = 'bilinear'  # Interpolation mode
    ):
```

## 🖥️ Command Line Usage

### Basic Usage

```bash
# Standard inference with original caching
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset original \
    --prompts "A beautiful sunset" "A cat in a garden"

# Aggressive caching for maximum speed
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset aggressive \
    --benchmark
```

### Cache Presets

Available presets:
- `no-cache` - Baseline performance, no caching
- `conservative` - Minimal caching, best quality
- `original` - Original HART caching implementation  
- `aggressive` - High speedup with good quality
- `ultra-fast` - Maximum speedup with adaptive thresholding
- `attn-only` - Memory-efficient attention-only caching
- `mlp-only` - MLP-only caching
- `high-quality` - Quality-focused with bicubic interpolation
- `memory-efficient` - Minimal memory usage

### Custom Configuration

```bash
# Custom skip and cache stages
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-skip-stages 100 169 256 \
    --cache-cache-stages 64 100 169 \
    --cache-threshold 0.65 \
    --cache-interpolation-mode bicubic

# Attention-only caching with adaptive threshold
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-skip-stages 169 256 \
    --cache-cache-stages 100 169 \
    --cache-disable-mlp \
    --cache-adaptive-threshold
```

### Performance Analysis

```bash
# Compare multiple configurations
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --compare_configs

# Benchmark specific configuration
python enhanced_inference_cli.py \
    --model_path /path/to/hart/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset aggressive \
    --benchmark
```

## ⚙️ Command Line Arguments

### Cache Control Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--cache-skip-stages` | int list | Stages to skip computation |
| `--cache-cache-stages` | int list | Stages to cache results |
| `--cache-enable-attn` | flag | Enable attention caching |
| `--cache-disable-attn` | flag | Disable attention caching |
| `--cache-enable-mlp` | flag | Enable MLP caching |
| `--cache-disable-mlp` | flag | Disable MLP caching |
| `--cache-threshold` | float | Similarity threshold (0-1) |
| `--cache-adaptive-threshold` | flag | Use adaptive thresholding |
| `--cache-interpolation-mode` | choice | bilinear/nearest/bicubic |
| `--cache-preset` | choice | Use predefined preset |

### Legacy Compatibility

| Argument | Type | Description |
|----------|------|-------------|
| `--use-cache` | flag | Enable caching (legacy) |
| `--calibration` | flag | Run calibration mode |
| `--sim-path` | string | Path to similarity data |
| `--threshold` | float | Cache threshold (legacy) |

## 🔬 Implementation Details

### VAR-Style Caching Logic

The implementation follows VAR's exact caching approach:

1. **Calibration Mode**: Computes both cached and non-cached results, tracks similarity
2. **Inference Mode**: Uses cached results when similarity exceeds threshold
3. **Feature Interpolation**: Bilinear interpolation to match target resolution
4. **Similarity Tracking**: Cosine similarity between cached and computed features

### Cache State Management

```python
# Cache storage (per layer)
self.cache_mlp = [None] * depth
self.cache_attn = [None] * depth

# Similarity tracking (per layer, per stage)
self.cache_similarity_mlp = [[0.0] * 15 for _ in range(depth)]
self.cache_similarity_attn = [[0.0] * 15 for _ in range(depth)]

# Stage mapping (updated for HART)
length2iteration = {
    1: 0, 4: 1, 9: 2, 16: 3, 25: 4, 49: 6, 81: 7, 144: 8, 256: 9, 441: 10, 729: 11, 1296: 12, 2304: 13, 4096: 14
}
```

### Enhanced Block Integration

```python
# In transformer forward pass
for b_idx, block in enumerate(self.blocks):
    h_BLC = block(
        h_BLC,
        cond_BD,
        attn_bias=None,
        si=si,
        context_position_ids=context_position_ids,
        context_mask=context_mask,
        # Enhanced caching parameters
        cache_mlp=self.cache_mlp,
        cache_attn=self.cache_attn,
        cache_similarity_mlp=self.cache_similarity_mlp,
        cache_similarity_attn=self.cache_similarity_attn,
    )
```

## 📊 Expected Performance

### Speedup Comparison

| Configuration | Speedup | Memory | Quality |
|---------------|---------|---------|---------|
| No Cache | 1.0× | Baseline | Best |
| Conservative | 1.2-1.5× | +5% | Excellent |
| Original | 1.5-2.0× | +10% | Very Good |
| Aggressive | 2.0-3.0× | +15% | Good |
| Ultra Fast | 3.0-4.0× | +20% | Acceptable |

### Stage Selection Guidelines

| Stage | Resolution | Caching Benefit | Quality Impact |
|-------|------------|-----------------|----------------|
| 1-16 | 1×1 to 4×4 | Low | Minimal |
| 25-49 | 5×5 to 7×7 | Medium | Low |
| 81-144 | 9×9 to 12×12 | High | Medium |
| 256-441 | 16×16 to 21×21 | Very High | High |
| 729-4096 | 27×27 to 64×64 | Maximum | Maximum |

## 🔧 Integration Guide

### Using Enhanced Model

```python
from hart.modules.models.transformer.hart_transformer_t2i_enhanced import HARTForT2IEnhanced
from hart.modules.networks.basic_hart_enhanced import CacheConfig

# Load configuration
config = HARTForT2IConfig.from_pretrained("path/to/model")

# Set cache parameters in config
config.cache_skip_stages = [169, 256]
config.cache_cache_stages = [100, 169]
config.cache_threshold = 0.7

# Create enhanced model
model = HARTForT2IEnhanced(config)

# Or create from command line args
model = HARTForT2IEnhanced.from_args_and_config(args, config)
```

### Dynamic Cache Configuration

```python
# Update cache configuration at runtime
new_cache_config = CacheConfig(
    skip_stages=[64, 100, 169, 256],
    cache_stages=[36, 64, 100, 169],
    threshold=0.6,
    adaptive_threshold=True
)

model.set_cache_config(new_cache_config)
```

### Monitoring Cache Performance

```python
# Get cache statistics
stats = model.get_cache_statistics()
print(f"Cache configuration: {stats['cache_config']}")
print(f"Similarity stats: {stats['similarity_stats']}")

# Save similarity data after calibration
model.save_similarity_data()
```

## 🎯 Usage Examples

### Research and Development

```bash
# High quality for research
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset high-quality \
    --save_images \
    --prompts "Research quality image"

# Compare multiple configurations
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --compare_configs \
    --benchmark
```

### Production Deployment

```bash
# Balanced performance for production
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset original \
    --batch_size 4 \
    --num_samples 1

# Memory-constrained deployment
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset memory-efficient \
    --batch_size 1
```

### Interactive Demos

```bash
# Fast interactive generation
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --cache-preset ultra-fast \
    --num_samples 1 \
    --save_images
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure enhanced modules are in Python path
   export PYTHONPATH=/path/to/hart:$PYTHONPATH
   ```

2. **CUDA Out of Memory**
   ```bash
   # Use memory-efficient preset
   --cache-preset memory-efficient
   # Or disable MLP caching
   --cache-disable-mlp
   ```

3. **Quality Degradation**
   ```bash
   # Increase threshold
   --cache-threshold 0.8
   # Use higher quality interpolation
   --cache-interpolation-mode bicubic
   ```

4. **No Speedup**
   ```bash
   # Ensure aggressive caching
   --cache-preset aggressive
   # Or use custom aggressive settings
   --cache-skip-stages 64 100 169 256 --cache-threshold 0.6
   ```

### Performance Debugging

```bash
# Run benchmark to identify bottlenecks
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --benchmark \
    --cache-preset original

# Compare with no-cache baseline
python enhanced_inference_cli.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --compare_configs
```

## 📚 References

- Original VAR Paper: [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- HART Paper: [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/abs/2410.10812)
- FastVAR Implementation: Caching mechanisms and optimizations

## 🤝 Contributing

This implementation closely follows VAR's caching approach for maximum compatibility. When extending:

1. Maintain compatibility with VAR's similarity tracking format
2. Preserve the exact stage mapping (`length2iteration`)
3. Keep feature interpolation logic consistent
4. Follow VAR's calibration and inference modes

This ensures that cache configurations and similarity data can be shared between VAR and enhanced HART implementations.