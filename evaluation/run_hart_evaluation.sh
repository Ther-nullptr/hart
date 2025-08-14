#!/bin/bash

# HART Model Comprehensive Evaluation Script
# Runs both improved FID and GenEval evaluations

set -e  # Exit on any error

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Model paths (REQUIRED)
MODEL_PATH="/path/to/hart/model"  # Update this path
TEXT_MODEL_PATH="Qwen/Qwen2-VL-1.5B-Instruct"

# Output directory
OUTPUT_DIR="./hart_evaluation_results"
DEVICE="cuda"

# Dataset paths for FID evaluation (optional)
MJHQ_METADATA="/path/to/MJHQ30K/meta_data.json"  # Update or leave empty to skip
MJHQ_IMAGES="/path/to/MJHQ30K/mjhq30k_imgs"      # Update or leave empty to skip

# Detection model paths for GenEval (download these first)
DETECTOR_CONFIG=""  # Leave empty for default config
DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

# Generation settings
CFG_SCALE=4.5
SEED=42
USE_EMA="--use_ema"
MORE_SMOOTH="--more_smooth"

# FID evaluation settings
CATEGORY_FILTER="people"  # Set to "" for all categories
MAX_SAMPLES=1000          # Set to "" for all samples

# GenEval settings
IMAGES_PER_PROMPT=4

# Technical settings
BATCH_SIZE=4
FID_BATCH_SIZE=50
IMG_SIZE=1024
MAX_TOKEN_LENGTH=300

# =============================================================================
# Script execution - Don't modify below unless you know what you're doing
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

echo "=== HART Model Comprehensive Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
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

# Validate essential paths
echo "Validating paths..."

if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/hart/model" ]; then
    echo "ERROR: Please update MODEL_PATH in the configuration section"
    echo "Set MODEL_PATH to point to your HART model directory"
    exit 1
fi

if ! check_dir "$MODEL_PATH"; then
    echo "ERROR: HART model directory not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in the configuration section"
    exit 1
fi

echo "✓ HART model path validated"

# =============================================================================
# 1. FID Evaluation
# =============================================================================

print_section "FID Evaluation"

# Check if we can run FID evaluation
if [ -n "$MJHQ_METADATA" ] && [ -n "$MJHQ_IMAGES" ] && [ "$MJHQ_METADATA" != "/path/to/MJHQ30K/meta_data.json" ]; then
    echo "Dataset: MJHQ-30K"
    echo "Category filter: ${CATEGORY_FILTER:-all}"
    echo "Max samples: ${MAX_SAMPLES:-all}"
    echo ""
    
    if check_file "$MJHQ_METADATA" && check_dir "$MJHQ_IMAGES"; then
        echo "Running FID evaluation..."
        
        FID_ARGS="--model_path \"$MODEL_PATH\" \
                  --text_model_path \"$TEXT_MODEL_PATH\" \
                  --mjhq_metadata_path \"$MJHQ_METADATA\" \
                  --mjhq_images_path \"$MJHQ_IMAGES\" \
                  --output_dir \"$OUTPUT_DIR/fid\" \
                  --device $DEVICE \
                  --cfg $CFG_SCALE \
                  --seed $SEED \
                  --batch_size $BATCH_SIZE \
                  --fid_batch_size $FID_BATCH_SIZE \
                  --img_size $IMG_SIZE \
                  --max_token_length $MAX_TOKEN_LENGTH \
                  $USE_EMA"
        
        if [ -n "$CATEGORY_FILTER" ]; then
            FID_ARGS="$FID_ARGS --category_filter \"$CATEGORY_FILTER\""
        fi
        
        if [ -n "$MAX_SAMPLES" ]; then
            FID_ARGS="$FID_ARGS --max_samples $MAX_SAMPLES"
        fi
        
        eval "python compute_fid_improved.py $FID_ARGS"
        
        echo ""
        echo "✓ FID evaluation completed!"
        echo "Results saved to: $OUTPUT_DIR/fid/"
        
        # Display FID result
        if [ -f "$OUTPUT_DIR/fid/fid_results.json" ]; then
            echo ""
            echo "FID Score:"
            python -c "import json; data=json.load(open('$OUTPUT_DIR/fid/fid_results.json')); print(f'  {data[\"fid_score\"]:.4f}')"
        fi
    else
        echo "⚠️  MJHQ dataset paths are invalid, skipping FID evaluation"
        echo "To run FID evaluation, please:"
        echo "  1. Download MJHQ-30K dataset"
        echo "  2. Update MJHQ_METADATA and MJHQ_IMAGES paths in this script"
    fi
else
    echo "Skipping FID evaluation (MJHQ dataset paths not configured)"
    echo ""
    echo "To enable FID evaluation:"
    echo "  1. Download MJHQ-30K dataset"
    echo "  2. Update MJHQ_METADATA and MJHQ_IMAGES in the configuration section"
    echo ""
    echo "Alternatively, you can compute FID between two existing directories:"
    echo "  python compute_fid_improved.py --compute_fid_only /path/to/real /path/to/generated"
fi

# =============================================================================
# 2. GenEval Evaluation
# =============================================================================

print_section "GenEval Evaluation"
echo "This will generate prompts, images, and evaluate compositional capabilities"
echo ""

# Check detection model
SKIP_EVALUATION=false
if [ -n "$DETECTOR_CHECKPOINT" ] && [ "$DETECTOR_CHECKPOINT" != "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    if ! check_file "$DETECTOR_CHECKPOINT"; then
        echo "⚠️  Detection checkpoint not found: $DETECTOR_CHECKPOINT"
        SKIP_EVALUATION=true
    fi
elif [ -z "$DETECTOR_CHECKPOINT" ] || [ "$DETECTOR_CHECKPOINT" = "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    if [ ! -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
        echo "Detection checkpoint not found. Downloading..."
        mkdir -p checkpoints
        cd checkpoints
        
        # Download Mask2Former checkpoint
        CHECKPOINT_URL="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth"
        
        if command -v wget >/dev/null 2>&1; then
            wget "$CHECKPOINT_URL" -O mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
        elif command -v curl >/dev/null 2>&1; then
            curl -L "$CHECKPOINT_URL" -o mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
        else
            echo "❌ Neither wget nor curl found. Please download the checkpoint manually:"
            echo "  URL: $CHECKPOINT_URL"
            echo "  Save as: ./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
            SKIP_EVALUATION=true
        fi
        
        cd ..
        
        if [ -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
            echo "✓ Checkpoint downloaded successfully"
            DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
        else
            echo "❌ Failed to download checkpoint"
            SKIP_EVALUATION=true
        fi
    else
        echo "✓ Detection checkpoint found"
        DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    fi
fi

# Run GenEval evaluation
if [ "$SKIP_EVALUATION" = false ]; then
    GENEVAL_ARGS="--model_path \"$MODEL_PATH\" \
                  --text_model_path \"$TEXT_MODEL_PATH\" \
                  --detector_checkpoint \"$DETECTOR_CHECKPOINT\" \
                  --output_dir \"$OUTPUT_DIR/geneval\" \
                  --device $DEVICE \
                  --cfg $CFG_SCALE \
                  --seed $SEED \
                  --img_size $IMG_SIZE \
                  --max_token_length $MAX_TOKEN_LENGTH \
                  --images_per_prompt $IMAGES_PER_PROMPT \
                  --generate_prompts \
                  --generate_images \
                  $USE_EMA \
                  $MORE_SMOOTH"
    
    if [ -n "$DETECTOR_CONFIG" ]; then
        GENEVAL_ARGS="$GENEVAL_ARGS --detector_config \"$DETECTOR_CONFIG\""
    fi
    
    echo "Running GenEval evaluation..."
    echo "This may take a while as it generates and evaluates many images..."
    echo ""
    
    eval "python evaluate_geneval_improved.py $GENEVAL_ARGS"
    
    echo ""
    echo "✓ GenEval evaluation completed!"
    echo "Results saved to: $OUTPUT_DIR/geneval/"
    
    # Display GenEval results
    if [ -f "$OUTPUT_DIR/geneval/geneval_summary.json" ]; then
        echo ""
        echo "GenEval Results:"
        python -c "
import json
data = json.load(open('$OUTPUT_DIR/geneval/geneval_summary.json'))
print(f'  Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
for key, value in data.items():
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').title()
        print(f'  {task} Accuracy: {value:.3f}')
"
    fi
else
    echo "❌ Skipping GenEval evaluation due to missing detection checkpoint"
    echo ""
    echo "To run GenEval evaluation:"
    echo "  1. Ensure MMDetection is installed: pip install mmdet"
    echo "  2. Download the detection checkpoint:"
    echo "     mkdir -p checkpoints && cd checkpoints"
    echo "     wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth"
    echo "  3. Run this script again"
    echo ""
    echo "Alternatively, you can run evaluation on existing images:"
    echo "  python evaluate_geneval_improved.py --prompts_dir /path/to/prompts --detector_checkpoint /path/to/checkpoint.pth --evaluate_only"
fi

# =============================================================================
# Evaluation Summary
# =============================================================================

print_section "Evaluation Summary"

# FID Summary
if [ -f "$OUTPUT_DIR/fid/fid_results.json" ]; then
    echo "📊 FID Evaluation Results:"
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/fid/fid_results.json'))
print(f'  • FID Score: {data[\"fid_score\"]:.4f}')
print(f'  • Samples: {data[\"num_samples\"]}')
print(f'  • Category: {data.get(\"category_filter\", \"all\")}')
"
    echo ""
fi

# GenEval Summary
if [ -f "$OUTPUT_DIR/geneval/geneval_summary.json" ]; then
    echo "🎯 GenEval Evaluation Results:"
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/geneval/geneval_summary.json'))
print(f'  • Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
print(f'  • Total Samples: {data[\"total_samples\"]}')
print('  • Task-specific Accuracies:')
for key, value in data.items():
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').replace('cooccurrence', 'co-occurrence').title()
        print(f'    ◦ {task}: {value:.3f}')
"
    echo ""
fi

echo "📁 All results saved to: $OUTPUT_DIR"
echo ""

# Check if both evaluations completed
FID_COMPLETED=$( [ -f "$OUTPUT_DIR/fid/fid_results.json" ] && echo "true" || echo "false" )
GENEVAL_COMPLETED=$( [ -f "$OUTPUT_DIR/geneval/geneval_summary.json" ] && echo "true" || echo "false" )

if [ "$FID_COMPLETED" = "true" ] && [ "$GENEVAL_COMPLETED" = "true" ]; then
    echo "🎉 HART evaluation completed successfully!"
    echo "   Both FID and GenEval evaluations finished."
elif [ "$FID_COMPLETED" = "true" ]; then
    echo "✅ FID evaluation completed successfully!"
    echo "⚠️  GenEval evaluation was skipped (see instructions above to enable)"
elif [ "$GENEVAL_COMPLETED" = "true" ]; then
    echo "✅ GenEval evaluation completed successfully!"
    echo "⚠️  FID evaluation was skipped (see instructions above to enable)"
else
    echo "⚠️  No evaluations were completed. Please check the configuration and try again."
    exit 1
fi

echo ""
echo "=== Evaluation Complete ==="