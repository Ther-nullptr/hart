# HART Evaluation Scripts - Separated Evaluations

This directory now contains **separated evaluation scripts** that allow you to run FID and GenEval evaluations independently, providing more flexibility and focused evaluation workflows.

## 📁 Available Scripts

### 🎯 Individual Evaluation Scripts

1. **`run_fid_evaluation.sh`** - FID evaluation only
2. **`run_geneval_evaluation.sh`** - GenEval evaluation only

### 🔄 Legacy Scripts (Still Available)

3. **`run_hart_evaluation_corrected.sh`** - Combined evaluation (both FID + GenEval)

## 🚀 Quick Start Guide

### FID Evaluation Only

```bash
# Full MJHQ-30K FID evaluation
./run_fid_evaluation.sh --model-path /path/to/hart/model \
                       --mjhq-metadata /path/to/MJHQ30K/meta_data.json \
                       --mjhq-images /path/to/MJHQ30K/mjhq30k_imgs

# Direct FID computation between two directories
./run_fid_evaluation.sh --direct-fid /path/to/real /path/to/generated

# Quick test with limited samples
./run_fid_evaluation.sh --max-samples 100 --category-filter people
```

### GenEval Evaluation Only

```bash
# Full GenEval evaluation (downloads official dataset)
./run_geneval_evaluation.sh --model-path /path/to/hart/model

# Quick test with limited prompts
./run_geneval_evaluation.sh --max-prompts 100

# Evaluate existing images only
./run_geneval_evaluation.sh --evaluate-only /path/to/prompts \
                           --detector-checkpoint /path/to/checkpoint.pth
```

## 📊 Detailed Usage

### FID Evaluation Script (`run_fid_evaluation.sh`)

#### Purpose
Evaluates image quality using Fréchet Inception Distance on MJHQ-30K dataset.

#### Two Modes

**1. Full MJHQ-30K Evaluation**
```bash
./run_fid_evaluation.sh \
    --model-path /path/to/hart/model \
    --mjhq-metadata /path/to/MJHQ30K/meta_data.json \
    --mjhq-images /path/to/MJHQ30K/mjhq30k_imgs \
    --category-filter people \
    --max-samples 1000
```

**2. Direct FID Mode**
```bash
./run_fid_evaluation.sh --direct-fid /path/to/real /path/to/generated
```

#### Key Options
- `--model-path`: Path to HART model (required for generation mode)
- `--mjhq-metadata`: Path to MJHQ metadata JSON
- `--mjhq-images`: Path to MJHQ images directory
- `--direct-fid`: Compute FID between two existing directories
- `--category-filter`: Filter to specific category (people, animals, etc.)
- `--max-samples`: Limit number of samples
- `--cfg`: CFG scale for generation
- `--seed`: Random seed

#### Output
```
📊 FID Evaluation Results:
🎯 FID Score: 15.2340
📋 Evaluation Details:
  • Samples Evaluated: 1000
  • Category Filter: people
  • Generation Config: CFG=4.5, Size=1024px
```

### GenEval Evaluation Script (`run_geneval_evaluation.sh`)

#### Purpose
Evaluates compositional reasoning using the official GenEval dataset.

#### Three Modes

**1. Full Evaluation (Generate + Evaluate)**
```bash
./run_geneval_evaluation.sh \
    --model-path /path/to/hart/model \
    --max-prompts 500
```

**2. Evaluate Existing Images Only**
```bash
./run_geneval_evaluation.sh \
    --evaluate-only /path/to/prompts \
    --detector-checkpoint /path/to/checkpoint.pth
```

**3. Generate Images Only**
```bash
./run_geneval_evaluation.sh \
    --skip-generation \
    --model-path /path/to/hart/model
```

#### Key Options
- `--model-path`: Path to HART model (required for generation)
- `--detector-checkpoint`: Path to detection model checkpoint
- `--max-prompts`: Limit prompts for testing
- `--evaluate-only`: Only evaluate existing images
- `--skip-generation`: Skip image generation step
- `--images-per-prompt`: Number of images per prompt

#### Output
```
🎯 GenEval Evaluation Results:
🎯 Overall Accuracy: 0.756
📋 Task-specific Results:
  • Single Object  : 0.892 (1250/1401)
  • Two Object     : 0.734 (445/606)
  • Counting       : 0.623 (187/300)
  • Colors         : 0.801 (561/700)
  • Position       : 0.567 (340/600)
  • Color Attr     : 0.689 (206/299)
```

## ⚙️ Configuration

### Script Configuration Sections

Both scripts have configuration sections at the top:

```bash
# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

MODEL_PATH="/path/to/hart/model"  # Update this
# ... other settings
```

### Environment Variables

You can also set paths via environment variables:
```bash
export HART_MODEL_PATH="/path/to/hart/model"
export MJHQ_METADATA="/path/to/MJHQ30K/meta_data.json"
export MJHQ_IMAGES="/path/to/MJHQ30K/mjhq30k_imgs"
```

## 🔧 Advanced Usage

### Batch Processing

**FID Evaluation on Multiple Categories**
```bash
for category in people animals vehicles; do
    ./run_fid_evaluation.sh \
        --category-filter $category \
        --output-dir ./results_$category
done
```

**GenEval with Different Settings**
```bash
for cfg in 3.0 4.5 6.0; do
    ./run_geneval_evaluation.sh \
        --cfg $cfg \
        --output-dir ./results_cfg_$cfg
done
```

### Pipeline Integration

**Sequential Evaluation**
```bash
# Run FID first
./run_fid_evaluation.sh --model-path /path/to/model

# Then run GenEval
./run_geneval_evaluation.sh --model-path /path/to/model
```

**Parallel Evaluation** (if you have multiple GPUs)
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 ./run_fid_evaluation.sh --device cuda:0 &

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 ./run_geneval_evaluation.sh --device cuda:1 &
```

## 📈 Results Interpretation

### FID Scores
- **< 10**: Excellent quality, very similar to real images
- **10-20**: Good quality, minor differences
- **20-50**: Moderate quality, noticeable differences  
- **> 50**: Poor quality, significant differences

### GenEval Scores
- **> 0.8**: Excellent compositional understanding
- **0.6-0.8**: Good performance with room for improvement
- **0.4-0.6**: Moderate performance, significant issues
- **< 0.4**: Poor compositional understanding

### Task-Specific Analysis

**GenEval Tasks**:
- **Single Object**: Basic object recognition capability
- **Two Object**: Object co-occurrence understanding
- **Counting**: Numerical reasoning (1-5 objects)
- **Colors**: Color attribute binding
- **Position**: Spatial relationship understanding
- **Color Attr**: Multi-object color attribute binding

## 🛠️ Troubleshooting

### Common Issues

**1. Model Not Found**
```bash
❌ ERROR: HART model directory not found
```
Solution: Update `MODEL_PATH` in script or use `--model-path`

**2. MJHQ Dataset Missing**
```bash
❌ ERROR: MJHQ metadata file not found
```
Solution: Download MJHQ-30K from https://huggingface.co/datasets/playgroundai/MJHQ-30K

**3. Detection Checkpoint Missing**
```bash
⚠️ Detection checkpoint not found
```
Solution: Let script auto-download or manually download Mask2Former checkpoint

**4. CUDA Out of Memory**
```bash
RuntimeError: CUDA out of memory
```
Solution: Reduce batch sizes in script configuration

### Debug Mode

Add `--help` to any script for detailed usage:
```bash
./run_fid_evaluation.sh --help
./run_geneval_evaluation.sh --help
```

### Logging

Scripts provide detailed logging. For even more verbose output:
```bash
bash -x ./run_fid_evaluation.sh  # Debug mode
```

## 🔄 Migration from Combined Script

### Before (Combined)
```bash
./run_hart_evaluation_corrected.sh
```

### After (Separated)
```bash
# Run individually as needed
./run_fid_evaluation.sh --model-path /path/to/model
./run_geneval_evaluation.sh --model-path /path/to/model
```

## 📁 Output Structure

### FID Evaluation Output
```
fid_evaluation_results/
├── generated/              # Generated images
├── fid_results.json        # FID score and metadata
└── ...
```

### GenEval Evaluation Output
```
geneval_evaluation_results/
├── evaluation_metadata.jsonl   # Downloaded GenEval metadata
├── prompts/                    # Generated images in GenEval format
│   ├── 0/
│   │   ├── metadata.jsonl
│   │   └── samples/
│   ├── 1/
│   └── ...
├── geneval_results.jsonl       # Detailed per-image results
└── geneval_summary.json        # Summary statistics
```

## ✅ Benefits of Separation

1. **🎯 Focused Evaluation**: Run only the evaluation you need
2. **⚡ Faster Development**: Skip time-consuming evaluations during development
3. **🔧 Easier Debugging**: Isolate issues to specific evaluation types
4. **📊 Flexible Workflows**: Mix and match evaluations as needed
5. **💾 Resource Management**: Control GPU memory usage better
6. **🔄 Pipeline Integration**: Easier to integrate into automated workflows

---

**Choose the right script for your needs:**
- **FID only**: Image quality assessment → `run_fid_evaluation.sh`
- **GenEval only**: Compositional reasoning → `run_geneval_evaluation.sh`  
- **Both**: Complete evaluation → `run_hart_evaluation_corrected.sh`