# HART Model Evaluation Framework

This evaluation framework provides comprehensive benchmarking for HART (Hybrid Autoregressive Transformer) models, integrating evaluation methodologies from SANA, FastVAR, and VAR repositories.

## Features

- **FID (Fréchet Inception Distance)** - Image generation quality assessment
- **CLIP Score** - Text-image alignment evaluation  
- **GenEval** - Compositional generation evaluation
- **Image Quality Metrics** - Additional quality assessments
- **Inference Speed Benchmarking** - Performance profiling
- **Multi-dataset Support** - MJHQ-30K, GenEval, custom datasets

## Installation

1. Install required dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers diffusers
pip install clip-by-openai
pip install pytorch-fid
pip install pillow numpy scipy tqdm
pip install mmdetection  # For GenEval object detection
pip install wandb  # For experiment tracking
```

2. For GenEval evaluation, install MMDetection:
```bash
pip install mmdet
# Download required models using the GenEval setup scripts
```

## Quick Start

### Basic CLIP Score Evaluation
```bash
cd /path/to/hart/evaluation
./run_evaluation.sh --model_path /path/to/hart/model \
                   --text_model_path /path/to/qwen2 \
                   --prompts_file prompts.txt \
                   --eval_clip
```

### Full Evaluation Suite
```bash
./run_evaluation.sh --model_path /path/to/hart/model \
                   --text_model_path /path/to/qwen2 \
                   --prompts_file prompts.txt \
                   --reference_path /path/to/mjhq30k \
                   --generate_samples \
                   --eval_fid \
                   --eval_clip \
                   --eval_quality \
                   --eval_speed
```

### GenEval Benchmark
```bash
./run_evaluation.sh --model_path /path/to/hart/model \
                   --text_model_path /path/to/qwen2 \
                   --prompts_file prompts.txt \
                   --eval_geneval \
                   --geneval_prompts geneval_prompts.json \
                   --geneval_model_path /path/to/mmdet/model \
                   --geneval_config_path /path/to/mmdet/config
```

## Evaluation Scripts

### Core Scripts

- `evaluate_hart.py` - Main comprehensive evaluation script
- `run_evaluation.sh` - Bash wrapper with convenient CLI
- `utils.py` - Shared utilities and logging functions

### Individual Metric Scripts

- `compute_fid.py` - FID computation using PyTorch-FID
- `compute_clip_score.py` - CLIP Score evaluation
- `evaluate_geneval.py` - GenEval compositional evaluation

## Supported Benchmarks

### 1. MJHQ-30K (Aesthetic Quality)
- 30K high-quality images from Midjourney
- FID evaluation against aesthetically pleasing references
- Categories: animals, art, fashion, food, indoor, landscape, logo, people, plants, vehicles

### 2. GenEval (Compositional Understanding)
- Object detection-based evaluation
- Tests: object presence, counting, spatial relationships, attributes
- Uses MMDetection models for object detection

### 3. Custom Prompt Sets
- Support for any text prompt file (.txt or .json)
- Flexible prompt format handling
- Batch evaluation capabilities

## Configuration Options

### Model Paths
```bash
--model_path PATH              # HART model path (required)
--text_model_path PATH         # Qwen2 text model path (required)
--shield_model_path PATH       # ShieldGemma safety model (optional)
```

### Evaluation Options
```bash
--eval_fid                     # Enable FID evaluation
--eval_clip                    # Enable CLIP Score evaluation
--eval_geneval                 # Enable GenEval benchmark
--eval_quality                 # Enable image quality metrics
--eval_speed                   # Enable inference speed benchmarking
--generate_samples             # Generate new samples
```

### Performance Settings
```bash
--device cuda                  # Computation device
--batch_size 32                # Evaluation batch size
--img_size 1024                # Image resolution
--max_samples 10000            # Maximum samples to evaluate
```

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "FID": 5.38,
  "CLIP_Score": {
    "mean": 0.2845,
    "std": 0.0123,
    "min": 0.1234,
    "max": 0.4567
  },
  "GenEval": {
    "overall_accuracy": 0.7234,
    "object_presence": [0.8, 0.9, 0.7],
    "object_counting": [0.6, 0.8, 0.5],
    "spatial_relations": [0.7, 0.6, 0.8]
  },
  "Quality_Metrics": {
    "color_diversity_mean": 45.67,
    "avg_resolution": [1024, 1024]
  },
  "Inference_Speed": {
    "total_time": 125.34
  },
  "metadata": {
    "model_path": "/path/to/model",
    "experiment_name": "hart_evaluation",
    "timestamp": "2024-01-01 12:00:00"
  }
}
```

## Integration with Tracking Systems

The evaluation framework supports Weights & Biases (wandb) for experiment tracking:

```python
# Enable wandb logging in evaluation scripts
args.report_to = "wandb"
args.tracker_project_name = "hart-evaluation"
```

## Benchmarking Results

### Expected Performance Ranges

Based on HART paper results:
- **FID on MJHQ-30K**: ~5.38 (HART-0.7B)
- **Reconstruction FID**: ~0.30 (hybrid tokenizer improvement)
- **CLIP Score**: Typically 0.25-0.35 for good text-image alignment
- **GenEval Accuracy**: 0.6-0.8 for compositional understanding

### Comparison with Other Models

The framework enables fair comparison with:
- VAR (Visual Autoregressive models)
- Diffusion models (DDPM, DiT)
- Autoregressive models (MAR, VAR-based)

## Extending the Framework

### Adding New Metrics

1. Create new evaluation script following the pattern:
```python
def evaluate_custom_metric(args):
    # Your evaluation logic
    return metric_score

# Add to evaluate_hart.py
if args.eval_custom:
    self.evaluate_custom_metric()
```

2. Update CLI options in `run_evaluation.sh`

### Adding New Benchmarks

1. Implement benchmark-specific data loading
2. Add evaluation logic following GenEval pattern
3. Integrate with main evaluation pipeline

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--max_samples`
2. **MMDetection not found**: Install mmdet for GenEval evaluation
3. **Model loading errors**: Check model paths and formats
4. **Image loading errors**: Verify image file formats and paths

### Performance Tips

- Use smaller batch sizes for high-resolution images
- Enable mixed precision for faster evaluation
- Use SSD storage for faster image I/O
- Pre-download and cache reference datasets

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{tang2024hart,
  title={HART: Efficient Visual Generation with Hybrid Autoregressive Transformer},
  author={Tang, Haotian and Wu, Yecheng and Yang, Shang and Xie, Enze and Chen, Junsong and Chen, Junyu and Zhang, Zhuoyang and Cai, Han and Lu, Yao and Han, Song},
  journal={arXiv preprint},
  year={2024}
}
```