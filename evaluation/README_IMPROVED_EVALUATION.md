# HART Improved Evaluation Framework

This directory contains comprehensive and improved evaluation scripts for the HART (Hybrid Autoregressive Rectification Transformer) model, implementing both enhanced FID evaluation and complete GenEval framework for compositional text-to-image alignment assessment.

## 🎯 Overview

This evaluation framework provides:

1. **Enhanced FID Evaluation** (`compute_fid_improved.py`) - Improved version with better modularity, error handling, and MJHQ-30K integration
2. **Complete GenEval Implementation** (`evaluate_geneval_improved.py`) - Full GenEval framework with automated prompt generation and comprehensive evaluation
3. **Automated Evaluation Pipeline** (`run_hart_evaluation.sh`) - One-click comprehensive evaluation script

## 🚀 Quick Start

### 1. Update Configuration

Edit the paths in `run_hart_evaluation.sh`:

```bash
# Essential - Update these paths
MODEL_PATH="/path/to/your/hart/model"           # REQUIRED

# Optional - For FID evaluation on MJHQ-30K
MJHQ_METADATA="/path/to/MJHQ30K/meta_data.json"
MJHQ_IMAGES="/path/to/MJHQ30K/mjhq30k_imgs"

# Optional - Detection checkpoint (auto-downloaded if not present)
DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
```

### 2. Run Comprehensive Evaluation

```bash
./run_hart_evaluation.sh
```

This will automatically:
- ✅ Validate paths and dependencies
- 📊 Run FID evaluation (if MJHQ dataset is configured)
- 🎯 Download detection models (if needed)
- 🧠 Generate GenEval prompts and evaluate compositional reasoning
- 📈 Provide comprehensive results summary

## 📦 Installation

### Core Dependencies

```bash
# HART and basic dependencies
pip install torch torchvision transformers
pip install numpy pillow scipy tqdm

# FID computation
pip install pytorch-fid cleanfid

# GenEval dependencies
pip install mmdet mmcv-full
pip install open-clip-torch clip-benchmark

# Alternative MMCV installation (if needed)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### Detection Models (Auto-downloaded)

The evaluation script automatically downloads required detection models. Manual download:

```bash
mkdir -p checkpoints && cd checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth
```

## 🔧 Individual Usage

### Enhanced FID Evaluation

#### Full MJHQ-30K Evaluation
```bash
python compute_fid_improved.py \
    --model_path /path/to/hart/model \
    --text_model_path Qwen/Qwen2-VL-1.5B-Instruct \
    --mjhq_metadata_path /path/to/MJHQ30K/meta_data.json \
    --mjhq_images_path /path/to/MJHQ30K/mjhq30k_imgs \
    --output_dir ./fid_results \
    --category_filter people \
    --max_samples 1000 \
    --use_ema
```

#### Direct FID Computation
```bash
python compute_fid_improved.py \
    --compute_fid_only /path/to/real/images /path/to/generated/images
```

### Complete GenEval Evaluation

#### Full Pipeline (Generate + Evaluate)
```bash
python evaluate_geneval_improved.py \
    --model_path /path/to/hart/model \
    --text_model_path Qwen/Qwen2-VL-1.5B-Instruct \
    --detector_checkpoint ./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
    --output_dir ./geneval_results \
    --generate_prompts \
    --generate_images \
    --use_ema
```

#### Evaluation Only
```bash
python evaluate_geneval_improved.py \
    --prompts_dir /path/to/existing/prompts \
    --detector_checkpoint ./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
    --evaluate_only
```

## 🎛️ Configuration Options

### Generation Settings
- `--cfg`: Classifier-free guidance scale (default: 4.5)
- `--seed`: Random seed for reproducibility (default: 1)
- `--use_ema`: Use EMA model weights
- `--more_smooth`: Enable smoother generation
- `--img_size`: Image resolution (default: 1024)

### FID Settings
- `--category_filter`: Filter MJHQ category (e.g., "people")
- `--max_samples`: Limit evaluation samples
- `--fid_batch_size`: Batch size for FID computation (default: 50)

### GenEval Settings
- `--images_per_prompt`: Images per prompt (default: 4)
- `--threshold`: Object detection threshold (default: 0.3)
- `--counting_threshold`: Higher threshold for counting (default: 0.9)
- `--max_objects`: Max objects per class (default: 16)
- `--position_threshold`: Spatial relationship tolerance (default: 0.1)

### Technical Settings
- `--device`: Computation device (default: "cuda")
- `--batch_size`: Generation batch size (default: 4)
- `--num_workers`: Data loading workers (default: 4)
- `--max_token_length`: Text encoding length (default: 300)

## 📊 Output Structure

```
hart_evaluation_results/
├── fid/
│   ├── generated/           # Generated images
│   ├── fid_results.json     # FID score and metadata
│   └── ...
├── geneval/
│   ├── prompts/            # GenEval prompt structure
│   │   ├── 0/
│   │   │   ├── metadata.jsonl
│   │   │   └── samples/    # Generated images
│   │   ├── 1/
│   │   └── ...
│   ├── geneval_results.jsonl   # Detailed per-image results
│   └── geneval_summary.json    # Aggregate statistics
└── ...
```

## 🧠 GenEval Tasks

The framework evaluates four compositional reasoning capabilities:

### 1. **Counting** 🔢
Tests ability to generate correct number of objects
- Examples: "A photo of 3 cats", "A photo of a single apple"

### 2. **Color** 🎨
Tests color attribute understanding
- Examples: "A photo of a red car", "A photo of a blue bicycle"

### 3. **Spatial Relations** 📐
Tests spatial relationship comprehension
- Examples: "A photo of a cat left of a dog", "A photo of an apple above a banana"

### 4. **Co-occurrence** 👥
Tests multi-object generation capability
- Examples: "A photo of a cat and a dog", "A photo of an apple and a banana"

## 📈 Evaluation Metrics

### FID Metrics
- **FID Score**: Lower is better (measures distributional similarity)
- **Computed using**: InceptionV3 features
- **Supports**: Category filtering, sample limiting

### GenEval Metrics
- **Overall Accuracy**: Fraction of correctly generated images
- **Task-specific Accuracies**: Performance per task type
- **Detailed Analysis**: Per-prompt and per-category breakdowns

## 🔄 Key Improvements

### Over Original FID Implementation
1. **🏗️ Modular Architecture**: Clean `HARTFIDEvaluator` class
2. **🛡️ Robust Error Handling**: Graceful failure recovery
3. **⚙️ Flexible Configuration**: Easy dataset adaptation
4. **📚 Better Documentation**: Comprehensive docstrings
5. **🔄 Fallback Mechanisms**: Multiple FID computation methods

### GenEval Implementation Features
1. **🤖 Automated Prompt Generation**: No external files needed
2. **🔗 Integrated Pipeline**: End-to-end evaluation
3. **🎯 Complete Framework**: All GenEval tasks implemented
4. **🔧 Extensible Design**: Easy to add new tasks
5. **📊 Comprehensive Metrics**: Detailed performance analysis

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch sizes
   --batch_size 2 --fid_batch_size 25
   ```

2. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   ```

3. **Detection Model Not Found**
   ```bash
   # Manual download
   mkdir -p checkpoints && cd checkpoints
   wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/...
   ```

4. **HART Model Not Loading**
   - Verify model path is correct
   - Check if all HART dependencies are installed
   - Ensure model files are not corrupted

### Performance Optimization

- **🖥️ Memory**: Reduce `--img_size` for development
- **⚡ Speed**: Use `--max_samples` for quick testing  
- **🔄 Parallel**: Increase `--num_workers` for faster data loading
- **💾 Storage**: Use `--category_filter` to limit evaluation scope

## 📖 Usage Examples

### Development Testing
```bash
# Quick test with limited samples
./run_hart_evaluation.sh
# Then edit script to set MAX_SAMPLES=10
```

### Full Evaluation
```bash
# Complete evaluation (may take hours)
./run_hart_evaluation.sh
# With all samples and categories
```

### Custom Evaluation
```bash
# Only specific category
python compute_fid_improved.py ... --category_filter people

# Only specific tasks in GenEval
# (modify GenEvalDataGenerator.generate_all_prompts to customize)
```

## 📄 Results Interpretation

### FID Scores
- **< 10**: Excellent quality, very similar to real images
- **10-20**: Good quality, noticeable but minor differences
- **20-50**: Moderate quality, some visible artifacts
- **> 50**: Poor quality, significant differences from real images

### GenEval Scores
- **> 0.8**: Excellent compositional understanding
- **0.6-0.8**: Good performance with room for improvement
- **0.4-0.6**: Moderate performance, significant issues
- **< 0.4**: Poor compositional understanding

## 📚 Citations

If you use this evaluation framework, please cite:

```bibtex
@article{geneval2023,
  title={GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment},
  author={Ghosh, Dhruba and Hajishirzi, Hanna and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2310.11513},
  year={2023}
}
```

And the original HART paper when available.

---

**🔗 Related Files:**
- `compute_fid_improved.py` - Enhanced FID evaluation
- `evaluate_geneval_improved.py` - Complete GenEval implementation  
- `run_hart_evaluation.sh` - Comprehensive evaluation script
- Original files: `compute_fid.py`, `evaluate_geneval.py` - Basic implementations