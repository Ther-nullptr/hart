#!/bin/bash

# HART Model GenEval Evaluation Script
# Dedicated script for GenEval compositional reasoning evaluation

set -e  # Exit on any error

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Model paths (REQUIRED for generation mode)
MODEL_PATH="/path/to/hart/model"  # Update this path
TEXT_MODEL_PATH="Qwen/Qwen2-VL-1.5B-Instruct"

# Output directory
OUTPUT_DIR="./geneval_evaluation_results"
DEVICE="cuda"

# Detection model paths (auto-downloaded if not present)
DETECTOR_CONFIG=""  # Leave empty for default config
DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
CLIP_MODEL="ViT-L-14"

# GenEval settings
DOWNLOAD_GENEVAL="--download_geneval"  # Download official GenEval metadata
IMAGES_PER_PROMPT=4
MAX_PROMPTS=""  # Set to limit prompts for testing, empty for full dataset

# Generation settings
CFG_SCALE=4.5
SEED=42
USE_EMA="--use_ema"
MORE_SMOOTH="--more_smooth"

# Evaluation thresholds
THRESHOLD=0.3
COUNTING_THRESHOLD=0.9
MAX_OBJECTS=16
NMS_THRESHOLD=1.0
POSITION_THRESHOLD=0.1

# Technical settings
IMG_SIZE=1024
MAX_TOKEN_LENGTH=300

# Experiment tracking (optional)
EXP_NAME="hart_geneval_evaluation"
REPORT_TO="none"  # Set to "wandb" to enable tracking
TRACKER_PROJECT="hart-evaluation"

# =============================================================================
# Script execution - Don't modify below unless you know what you're doing
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

echo "=== HART Model GenEval Evaluation ==="
echo "Evaluating compositional reasoning using official GenEval dataset"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Dataset: Official GenEval"
echo ""

# Utility functions
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: File not found: $1"
        return 1
    fi
    return 0
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory not found: $1"
        return 1
    fi
    return 0
}

print_section() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

# Parse command line arguments
EVALUATE_ONLY=false
PROMPTS_DIR=""
GENEVAL_METADATA=""
SKIP_GENERATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --evaluate-only)
            EVALUATE_ONLY=true
            PROMPTS_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --geneval-metadata)
            GENEVAL_METADATA="$2"
            DOWNLOAD_GENEVAL=""  # Don't download if metadata provided
            shift 2
            ;;
        --detector-checkpoint)
            DETECTOR_CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-prompts)
            MAX_PROMPTS="$2"
            shift 2
            ;;
        --cfg)
            CFG_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --images-per-prompt)
            IMAGES_PER_PROMPT="$2"
            shift 2
            ;;
        --help|-h)
            echo "HART GenEval Evaluation Script"
            echo ""
            echo "Usage:"
            echo "  ./run_geneval_evaluation.sh [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  Default:      Generate images and evaluate on GenEval"
            echo "  Evaluate-only: ./run_geneval_evaluation.sh --evaluate-only /path/to/prompts"
            echo ""
            echo "Options:"
            echo "  --model-path PATH             Path to HART model directory"
            echo "  --geneval-metadata PATH       Path to GenEval metadata file"
            echo "  --detector-checkpoint PATH    Path to detection model checkpoint"
            echo "  --output-dir PATH             Output directory"
            echo "  --max-prompts N               Limit number of prompts (for testing)"
            echo "  --cfg SCALE                   CFG scale (default: 4.5)"
            echo "  --seed N                      Random seed (default: 42)"
            echo "  --device DEVICE               Device (default: cuda)"
            echo "  --skip-generation             Skip image generation"
            echo "  --images-per-prompt N         Images per prompt (default: 4)"
            echo "  --evaluate-only PATH          Only evaluate existing images"
            echo "  --help, -h                    Show this help"
            echo ""
            echo "Examples:"
            echo "  # Full GenEval evaluation"
            echo "  ./run_geneval_evaluation.sh --model-path /path/to/model"
            echo ""
            echo "  # Quick test with limited prompts"
            echo "  ./run_geneval_evaluation.sh --max-prompts 100"
            echo ""
            echo "  # Evaluate existing images only"
            echo "  ./run_geneval_evaluation.sh --evaluate-only /path/to/prompts --detector-checkpoint /path/to/checkpoint.pth"
            echo ""
            echo "  # Generate images only (skip evaluation)"
            echo "  ./run_geneval_evaluation.sh --skip-generation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Evaluate-only Mode
# =============================================================================

if [ "$EVALUATE_ONLY" = true ]; then
    print_section "Evaluate-only Mode"
    
    echo "Evaluating existing images in: $PROMPTS_DIR"
    echo ""
    
    if ! check_dir "$PROMPTS_DIR"; then
        echo "❌ ERROR: Prompts directory not found: $PROMPTS_DIR"
        exit 1
    fi
    
    if [ -z "$DETECTOR_CHECKPOINT" ] || ! check_file "$DETECTOR_CHECKPOINT"; then
        echo "❌ ERROR: Detection checkpoint required for evaluation"
        echo "Please specify: --detector-checkpoint /path/to/checkpoint.pth"
        exit 1
    fi
    
    echo "Loading GenEval metadata from prompts directory..."
    
    # Find metadata file
    if [ -f "$PROMPTS_DIR/../evaluation_metadata.jsonl" ]; then
        GENEVAL_METADATA="$PROMPTS_DIR/../evaluation_metadata.jsonl"
    elif [ -f "$PROMPTS_DIR/evaluation_metadata.jsonl" ]; then
        GENEVAL_METADATA="$PROMPTS_DIR/evaluation_metadata.jsonl"
    else
        echo "❌ ERROR: Cannot find evaluation_metadata.jsonl"
        echo "Expected locations:"
        echo "  $PROMPTS_DIR/../evaluation_metadata.jsonl"
        echo "  $PROMPTS_DIR/evaluation_metadata.jsonl"
        exit 1
    fi
    
    echo "Running evaluation with existing images..."
    python evaluate_geneval_corrected.py \
        --geneval_metadata_path "$GENEVAL_METADATA" \
        --detector_checkpoint "$DETECTOR_CHECKPOINT" \
        --clip_model "$CLIP_MODEL" \
        --prompts_dir "$PROMPTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --threshold $THRESHOLD \
        --counting_threshold $COUNTING_THRESHOLD \
        --max_objects $MAX_OBJECTS \
        --nms_threshold $NMS_THRESHOLD \
        --position_threshold $POSITION_THRESHOLD \
        --exp_name "$EXP_NAME" \
        --report_to "$REPORT_TO" \
        --tracker_project_name "$TRACKER_PROJECT" \
        --evaluate_only
    
    print_section "Evaluation Results"
    
    if [ -f "$OUTPUT_DIR/geneval_summary.json" ]; then
        echo "🎯 GenEval Evaluation Completed Successfully!"
        echo ""
        python -c "
import json
data = json.load(open('$OUTPUT_DIR/geneval_summary.json'))
print(f'🎯 Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
print()
print('📋 Task-specific Results:')
for key, value in sorted(data.items()):
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').replace('_', ' ').title()
        task_stats = data.get('tag_statistics', {}).get(key.replace('_accuracy', ''), {})
        total = task_stats.get('total', 0)
        correct = task_stats.get('correct', 0)
        print(f'  • {task:<15}: {value:.3f} ({correct}/{total})')
print()
print(f'📊 Total Samples: {data[\"total_samples\"]}')
print(f'✅ Correct Predictions: {data[\"total_correct\"]}')
"
        echo ""
        echo "✅ Evaluation completed successfully!"
    else
        echo "❌ Evaluation failed - no results file found"
        exit 1
    fi
    
    exit 0
fi

# =============================================================================
# Full GenEval Evaluation Mode
# =============================================================================

print_section "Full GenEval Evaluation Mode"

# Validate model path for generation
if [ "$SKIP_GENERATION" = false ]; then
    if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/hart/model" ]; then
        echo "❌ ERROR: Please specify MODEL_PATH for image generation"
        echo "Use: --model-path /path/to/your/hart/model"
        echo "Or update MODEL_PATH in the script configuration section"
        echo "Or use --skip-generation to skip image generation"
        exit 1
    fi
    
    if ! check_dir "$MODEL_PATH"; then
        echo "❌ ERROR: HART model directory not found: $MODEL_PATH"
        exit 1
    fi
    
    echo "✅ HART model path validated: $MODEL_PATH"
fi

# Check and download detection checkpoint
print_section "Detection Model Setup"

SKIP_EVALUATION_FINAL=false

if [ -n "$DETECTOR_CHECKPOINT" ] && [ "$DETECTOR_CHECKPOINT" != "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    if ! check_file "$DETECTOR_CHECKPOINT"; then
        echo "⚠️  Detection checkpoint not found: $DETECTOR_CHECKPOINT"
        SKIP_EVALUATION_FINAL=true
    fi
elif [ -z "$DETECTOR_CHECKPOINT" ] || [ "$DETECTOR_CHECKPOINT" = "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    if [ ! -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
        echo "Detection checkpoint not found. Downloading..."
        mkdir -p checkpoints
        cd checkpoints
        
        # Download Mask2Former checkpoint
        CHECKPOINT_URL="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth"
        
        if command -v wget >/dev/null 2>&1; then
            echo "Downloading with wget..."
            wget "$CHECKPOINT_URL" -O mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
        elif command -v curl >/dev/null 2>&1; then
            echo "Downloading with curl..."
            curl -L "$CHECKPOINT_URL" -o mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coo.pth
        else
            echo "❌ Neither wget nor curl found. Please download the checkpoint manually:"
            echo "  URL: $CHECKPOINT_URL"
            echo "  Save as: ./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
            SKIP_EVALUATION_FINAL=true
        fi
        
        cd ..
        
        if [ -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
            echo "✅ Checkpoint downloaded successfully"
            DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
        else
            echo "❌ Failed to download checkpoint"
            SKIP_EVALUATION_FINAL=true
        fi
    else
        echo "✅ Detection checkpoint found"
        DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    fi
fi

# Display evaluation settings
print_section "Evaluation Settings"

echo "GenEval Configuration:"
echo "  • Dataset: Official GenEval (auto-download)"
echo "  • Max prompts: ${MAX_PROMPTS:-all (~3000 prompts)}"
echo "  • Images per prompt: $IMAGES_PER_PROMPT"
echo "  • CFG scale: $CFG_SCALE"
echo "  • Random seed: $SEED"
echo "  • Image size: ${IMG_SIZE}px"
echo ""

echo "Evaluation Thresholds:"
echo "  • Detection threshold: $THRESHOLD"
echo "  • Counting threshold: $COUNTING_THRESHOLD"
echo "  • Max objects per class: $MAX_OBJECTS"
echo "  • NMS threshold: $NMS_THRESHOLD"
echo "  • Position threshold: $POSITION_THRESHOLD"
echo ""

# Prepare GenEval arguments
GENEVAL_ARGS="--output_dir \"$OUTPUT_DIR\" \
              --device $DEVICE \
              --cfg $CFG_SCALE \
              --seed $SEED \
              --img_size $IMG_SIZE \
              --max_token_length $MAX_TOKEN_LENGTH \
              --images_per_prompt $IMAGES_PER_PROMPT \
              --threshold $THRESHOLD \
              --counting_threshold $COUNTING_THRESHOLD \
              --max_objects $MAX_OBJECTS \
              --nms_threshold $NMS_THRESHOLD \
              --position_threshold $POSITION_THRESHOLD \
              --exp_name \"$EXP_NAME\" \
              --report_to \"$REPORT_TO\" \
              --tracker_project_name \"$TRACKER_PROJECT\" \
              --clip_model \"$CLIP_MODEL\""

# Add model paths for generation
if [ "$SKIP_GENERATION" = false ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --model_path \"$MODEL_PATH\" \
                  --text_model_path \"$TEXT_MODEL_PATH\" \
                  --generate_images \
                  $USE_EMA \
                  $MORE_SMOOTH"
fi

# Add metadata path or download option
if [ -n "$GENEVAL_METADATA" ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --geneval_metadata_path \"$GENEVAL_METADATA\""
else
    GENEVAL_ARGS="$GENEVAL_ARGS $DOWNLOAD_GENEVAL"
fi

# Add detection checkpoint for evaluation
if [ "$SKIP_EVALUATION_FINAL" = false ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --detector_checkpoint \"$DETECTOR_CHECKPOINT\""
    if [ -n "$DETECTOR_CONFIG" ]; then
        GENEVAL_ARGS="$GENEVAL_ARGS --detector_config \"$DETECTOR_CONFIG\""
    fi
fi

# Add max prompts limitation
if [ -n "$MAX_PROMPTS" ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --max_prompts $MAX_PROMPTS"
    echo "⚡ Using limited prompts for testing: $MAX_PROMPTS"
else
    echo "⏳ Using full GenEval dataset (this may take several hours)"
fi

# Run GenEval evaluation
print_section "Running GenEval Evaluation"

if [ "$SKIP_GENERATION" = false ]; then
    echo "Starting complete GenEval evaluation..."
    echo "This will:"
    echo "  1. Download official GenEval evaluation metadata"
    echo "  2. Load HART model and text encoder"
    echo "  3. Generate images for all GenEval prompts"
    if [ "$SKIP_EVALUATION_FINAL" = false ]; then
        echo "  4. Evaluate compositional reasoning capabilities"
    fi
else
    echo "Skipping image generation (--skip-generation specified)"
fi

echo ""

eval "python evaluate_geneval_corrected.py $GENEVAL_ARGS"

# Display results
print_section "GenEval Evaluation Results"

if [ "$SKIP_EVALUATION_FINAL" = true ]; then
    echo "⚠️  Evaluation step was skipped due to missing detection checkpoint"
    echo ""
    echo "To complete the evaluation:"
    echo "  1. Download the detection checkpoint manually or let the script download it"
    echo "  2. Run evaluation on generated images:"
    echo "     ./run_geneval_evaluation.sh --evaluate-only \"$OUTPUT_DIR/prompts\""
    exit 0
fi

if [ -f "$OUTPUT_DIR/geneval_summary.json" ]; then
    echo "🎯 GenEval Evaluation Completed Successfully!"
    echo ""
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/geneval_summary.json'))
print(f'🎯 Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
print()
print('📋 Task-specific Results:')
for key, value in sorted(data.items()):
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').replace('_', ' ').title()
        task_stats = data.get('tag_statistics', {}).get(key.replace('_accuracy', ''), {})
        total = task_stats.get('total', 0)
        correct = task_stats.get('correct', 0)
        print(f'  • {task:<15}: {value:.3f} ({correct}/{total})')
print()
print(f'📊 Total Samples: {data[\"total_samples\"]}')
print(f'✅ Correct Predictions: {data[\"total_correct\"]}')
print()
print('📁 Results Location:')
print(f'  • Generated Images: $OUTPUT_DIR/prompts/')
print(f'  • Detailed Results: $OUTPUT_DIR/geneval_results.jsonl')
print(f'  • Summary: $OUTPUT_DIR/geneval_summary.json')
"
    echo ""
    echo "✅ GenEval evaluation completed successfully!"
else
    echo "❌ GenEval evaluation failed - no results file found"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi

# Performance guidance
echo ""
echo "📈 GenEval Score Interpretation:"
echo "  • > 0.8:  Excellent compositional understanding"
echo "  • 0.6-0.8: Good performance with room for improvement"
echo "  • 0.4-0.6: Moderate performance, significant issues"
echo "  • < 0.4:  Poor compositional understanding"
echo ""

# Task-specific guidance
echo "🧠 GenEval Task Breakdown:"
echo "  • Single Object: Basic object recognition"
echo "  • Two Object: Object co-occurrence"
echo "  • Counting: Numerical understanding"
echo "  • Colors: Color attribute binding"
echo "  • Position: Spatial relationship understanding"
echo "  • Color Attr: Multi-object color binding"
echo ""

# Suggest next steps
echo "🔄 Next Steps:"
echo "  • Compare results with GenEval paper baselines"
echo "  • For image quality assessment, run: ./run_fid_evaluation.sh"
echo "  • Examine failure cases in: $OUTPUT_DIR/geneval_results.jsonl"
echo ""

echo "=== GenEval Evaluation Complete ==="