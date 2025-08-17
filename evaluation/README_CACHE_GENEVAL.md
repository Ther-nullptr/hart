# HART Cache GenEval Evaluation

This evaluation framework assesses the impact of VAR-style caching mechanisms on HART model performance using the GenEval dataset for compositional text-to-image generation.

## 🎯 Overview

The evaluation framework provides:
- **Performance Analysis**: Measures generation speed, memory usage, and throughput
- **Quality Assessment**: Evaluates compositional generation quality using object detection
- **Cache Impact**: Analyzes the trade-offs between caching aggressiveness and output quality
- **Comprehensive Comparison**: Compares multiple cache configurations systematically

## 📁 Files

- `evaluate_hart_cache_geneval.py` - Main evaluation script
- `run_hart_cache_geneval.sh` - Convenient shell wrapper
- `sample_geneval_prompts.jsonl` - Sample GenEval prompts for testing
- `README_CACHE_GENEVAL.md` - This documentation

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch torchvision pillow tqdm numpy pandas opencv-python

# Optional: Install MMDetection for object detection evaluation
pip install mmdet mmcv
```

### Basic Usage

```bash
# Compare all cache presets
./run_hart_cache_geneval.sh \
    --model_path /path/to/hart/model \
    --geneval_path /path/to/geneval.jsonl \
    --compare_presets

# Evaluate specific preset
./run_hart_cache_geneval.sh \
    --model_path /path/to/hart/model \
    --geneval_path /path/to/geneval.jsonl \
    --cache_preset aggressive

# Custom cache configuration
./run_hart_cache_geneval.sh \
    --model_path /path/to/hart/model \
    --geneval_path /path/to/geneval.jsonl \
    --skip_stages 144 256 \
    --cache_stages 81 144 \
    --threshold 0.65
```

## 📊 Evaluation Categories

The framework evaluates compositional generation across multiple categories:

### Object Detection
- **Prompt**: "A red apple on a wooden table"
- **Evaluation**: Presence and accuracy of expected objects

### Counting
- **Prompt**: "Three cats sitting on a fence"
- **Evaluation**: Correct counting of specified objects

### Spatial Relationships
- **Prompt**: "A blue car next to a yellow house"
- **Evaluation**: Correct spatial positioning between objects

### Attributes
- **Prompt**: "A large green tree with small red flowers"
- **Evaluation**: Correct size, color, and other attributes

### Color Accuracy
- **Prompt**: "Two white dogs playing in a park"
- **Evaluation**: Correct color representation

### Shape Recognition
- **Prompt**: "A round clock on a square wall"
- **Evaluation**: Correct geometric shapes

### Texture Understanding
- **Prompt**: "A fluffy cat on a smooth marble floor"
- **Evaluation**: Appropriate texture representation

## ⚙️ Cache Configurations

### Predefined Presets

| Preset | Skip Stages | Cache Stages | Threshold | Speed | Quality |
|--------|-------------|--------------|-----------|--------|---------|
| `no-cache` | [] | [] | - | 1.0× | Best |
| `conservative` | [256] | [144] | 0.8 | 1.2× | Excellent |
| `original` | [144,256] | [81,144] | 0.7 | 1.5× | Very Good |
| `aggressive` | [49,81,144,256] | [25,49,81,144] | 0.6 | 2.0× | Good |
| `ultra-fast` | [25,49,81,144,256,441] | [16,25,49,81,144,256] | 0.5 | 3.0× | Acceptable |

### Custom Configuration

```bash
python evaluate_hart_cache_geneval.py \
    --model_path /path/to/hart \
    --geneval_path /path/to/geneval.jsonl \
    --skip_stages 81 144 256 \
    --cache_stages 49 81 144 \
    --threshold 0.65 \
    --adaptive_threshold \
    --interpolation_mode bicubic
```

## 📈 Output Analysis

### Results Structure

```
output_dir/
├── cache_comparison_results.json          # Overall comparison data
├── comparison_report.md                   # Human-readable report
└── config_specific_dirs/
    ├── skip_81-144_cache_49-81_thresh_0.65_type_attn_mlp/
    │   ├── CONFIG_DESCRIPTION.md          # Human-readable config description
    │   ├── config_info.json               # Configuration summary and metadata
    │   ├── evaluation_results.json        # Detailed evaluation results
    │   └── generated_images/               # Generated images in standard format
    │       ├── 00000/                      # First prompt
    │       │   ├── metadata.jsonl          # Prompt metadata and quality metrics
    │       │   ├── grid.png                # Grid view of all samples
    │       │   └── samples/
    │       │       ├── 0000.png            # First sample
    │       │       ├── 0001.png            # Second sample (if multiple samples)
    │       │       └── ...
    │       ├── 00001/                      # Second prompt
    │       │   ├── metadata.jsonl
    │       │   ├── grid.png
    │       │   └── samples/
    │       │       └── 0000.png
    │       └── ...
    ├── skip_none_cache_none_thresh_0.7_type_none/  # No-cache baseline
    └── skip_25-49-81-144-256-441_cache_16-25-49-81-144-256_thresh_0.5_type_attn_mlp_adaptive/  # Ultra-fast
```

### Key Metrics

**Performance Metrics:**
- `mean_time`: Average generation time per image
- `throughput`: Images generated per second
- `mean_memory_mb`: Average memory usage
- `peak_memory_mb`: Peak memory usage

**Quality Metrics:**
- `overall_quality_score`: Composite quality score
- `detection_confidence`: Object detection confidence
- `object_accuracy`: Object presence accuracy
- `count_accuracy`: Counting accuracy

**Cache Statistics:**
- `cache_similarity_mlp`: MLP layer similarity scores
- `cache_similarity_attn`: Attention layer similarity scores

### Example Results

```json
{
  "cache_config": {
    "skip_stages": [144, 256],
    "cache_stages": [81, 144],
    "threshold": 0.7
  },
  "performance": {
    "mean_time": 2.34,
    "throughput": 0.43,
    "mean_memory_mb": 1024.5
  },
  "overall_metrics": {
    "overall_quality_score": 0.78,
    "num_images_generated": 70
  },
  "category_results": {
    "object": {
      "num_prompts": 10,
      "averages": {
        "avg_detection_confidence": 0.82,
        "avg_object_accuracy": 0.75
      }
    }
  }
}
```

## 🔧 Advanced Usage

### With Object Detection

```bash
# Download YOLOv3 or other detector
wget https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth

./run_hart_cache_geneval.sh \
    --model_path /path/to/hart \
    --geneval_path /path/to/geneval.jsonl \
    --detector_config /path/to/yolov3_config.py \
    --detector_model /path/to/yolov3_model.pth \
    --compare_presets
```

### Batch Processing

```bash
# Process with larger batches for speed
python evaluate_hart_cache_geneval.py \
    --model_path /path/to/hart \
    --geneval_path /path/to/geneval.jsonl \
    --cache_preset aggressive \
    --batch_size 8 \
    --num_samples 3
```

### Memory-Constrained Environment

```bash
# Use memory-efficient settings
./run_hart_cache_geneval.sh \
    --model_path /path/to/hart \
    --geneval_path /path/to/geneval.jsonl \
    --skip_stages 144 256 \
    --cache_stages 144 \
    --disable_mlp_cache \
    --batch_size 2
```

## 📊 Interpreting Results

### Performance Analysis

1. **Speed vs Quality Trade-off**:
   - Higher skip stages = faster generation but potentially lower quality
   - Lower thresholds = more aggressive caching = faster but less accurate

2. **Memory Usage**:
   - More cache stages = higher memory usage
   - MLP caching typically uses more memory than attention caching

3. **Quality Metrics**:
   - `detection_confidence > 0.7`: Good object detection
   - `object_accuracy > 0.8`: Strong compositional understanding
   - `count_accuracy > 0.6`: Reasonable counting ability

### Optimization Guidelines

**For Speed Priority**:
```bash
--cache_preset ultra-fast
# or
--skip_stages 25 49 81 144 256 441 --threshold 0.5
```

**For Quality Priority**:
```bash
--cache_preset conservative
# or
--skip_stages 256 --cache_stages 144 --threshold 0.9
```

**For Balanced Performance**:
```bash
--cache_preset original
# or
--skip_stages 144 256 --cache_stages 81 144 --threshold 0.7
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   --batch_size 1 --disable_mlp_cache
   ```

2. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=/path/to/hart:$PYTHONPATH
   ```

3. **No Object Detection**:
   - Install MMDetection: `pip install mmdet mmcv`
   - Or run without detector (quality metrics will be simplified)

4. **Low Quality Scores**:
   - Increase threshold: `--threshold 0.8`
   - Use fewer skip stages: `--skip_stages 256`
   - Use bicubic interpolation: `--interpolation_mode bicubic`

### Performance Debugging

```bash
# Enable verbose logging
python evaluate_hart_cache_geneval.py \
    --model_path /path/to/hart \
    --geneval_path /path/to/geneval.jsonl \
    --cache_preset original \
    --batch_size 1 \
    --num_samples 1 \
    2>&1 | tee evaluation.log
```

## 📚 Integration Examples

### Research Workflow

```bash
# 1. Baseline evaluation
./run_hart_cache_geneval.sh --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --cache_preset no-cache

# 2. Quick comparison
./run_hart_cache_geneval.sh --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --compare_presets

# 3. Fine-tuning
./run_hart_cache_geneval.sh --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --skip_stages 81 144 --threshold 0.75

# 4. Production evaluation
./run_hart_cache_geneval.sh --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --cache_preset original --batch_size 4
```

### Automated Testing

```bash
#!/bin/bash
# Automated cache configuration testing

MODEL_PATH="/path/to/hart"
GENEVAL_PATH="/path/to/geneval.jsonl"

for threshold in 0.5 0.6 0.7 0.8; do
    echo "Testing threshold: $threshold"
    ./run_hart_cache_geneval.sh \
        --model_path "$MODEL_PATH" \
        --geneval_path "$GENEVAL_PATH" \
        --skip_stages 144 256 \
        --cache_stages 81 144 \
        --threshold $threshold \
        --output_dir "./results_thresh_$threshold"
done
```

## 🤝 Contributing

To extend the evaluation framework:

1. **Add New Metrics**: Extend `evaluate_generation_quality()` method
2. **Custom Categories**: Add new GenEval categories in `geneval_categories`
3. **Advanced Detectors**: Integrate additional object detection models
4. **Performance Metrics**: Add new timing and memory measurements

## 📖 References

- GenEval Dataset: Compositional text-to-image generation evaluation
- HART Paper: [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/abs/2410.10812)
- VAR Caching: Visual Autoregressive Modeling with caching mechanisms
- MMDetection: OpenMMLab Detection Toolbox

This evaluation framework provides comprehensive analysis of cache mechanisms' impact on HART's compositional generation capabilities, enabling informed decisions about speed-quality trade-offs in different deployment scenarios.