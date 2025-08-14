#!/bin/bash

# HART Model FID Evaluation Script
# Dedicated script for FID evaluation on MJHQ-30K dataset

set -e  # Exit on any error

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Model paths (REQUIRED for generation mode)
MODEL_PATH="/path/to/hart/model"  # Update this path
TEXT_MODEL_PATH="Qwen/Qwen2-VL-1.5B-Instruct"

# Output directory
OUTPUT_DIR="./fid_evaluation_results"
DEVICE="cuda"

# Dataset paths for MJHQ-30K evaluation
MJHQ_METADATA="/path/to/MJHQ30K/meta_data.json"  # Update this path
MJHQ_IMAGES="/path/to/MJHQ30K/mjhq30k_imgs"      # Update this path

# Generation settings
CFG_SCALE=4.5
SEED=42
USE_EMA="--use_ema"
MORE_SMOOTH="--more_smooth"

# FID evaluation settings
CATEGORY_FILTER="people"  # Set to "" for all categories, or specify: people, animals, objects, etc.
MAX_SAMPLES=""            # Set to limit samples for testing, empty for all

# Technical settings
BATCH_SIZE=4
FID_BATCH_SIZE=50
IMG_SIZE=1024
MAX_TOKEN_LENGTH=300
NUM_WORKERS=4

# Experiment tracking (optional)
EXP_NAME="hart_fid_evaluation"
REPORT_TO="none"  # Set to "wandb" to enable tracking
TRACKER_PROJECT="hart-evaluation"

# =============================================================================
# Script execution - Don't modify below unless you know what you're doing
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

echo "=== HART Model FID Evaluation ==="
echo "Evaluating image quality using Fréchet Inception Distance"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Dataset: MJHQ-30K"
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
DIRECT_FID=false
REAL_PATH=""
GEN_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --direct-fid)
            DIRECT_FID=true
            REAL_PATH="$2"
            GEN_PATH="$3"
            shift 3
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mjhq-metadata)
            MJHQ_METADATA="$2"
            shift 2
            ;;
        --mjhq-images)
            MJHQ_IMAGES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --category-filter)
            CATEGORY_FILTER="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
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
        --help|-h)
            echo "HART FID Evaluation Script"
            echo ""
            echo "Usage:"
            echo "  ./run_fid_evaluation.sh [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  Default mode: Generate images and compute FID on MJHQ-30K"
            echo "  Direct FID:   ./run_fid_evaluation.sh --direct-fid /path/to/real /path/to/generated"
            echo ""
            echo "Options:"
            echo "  --model-path PATH         Path to HART model directory"
            echo "  --mjhq-metadata PATH      Path to MJHQ-30K metadata.json"
            echo "  --mjhq-images PATH        Path to MJHQ-30K images directory"
            echo "  --output-dir PATH         Output directory (default: ./fid_evaluation_results)"
            echo "  --category-filter CAT     Filter to specific category (e.g., people)"
            echo "  --max-samples N           Limit number of samples"
            echo "  --cfg SCALE               CFG scale (default: 4.5)"
            echo "  --seed N                  Random seed (default: 42)"
            echo "  --device DEVICE           Device (default: cuda)"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Examples:"
            echo "  # Full MJHQ-30K evaluation"
            echo "  ./run_fid_evaluation.sh --model-path /path/to/model --mjhq-metadata /path/to/meta.json --mjhq-images /path/to/images"
            echo ""
            echo "  # Direct FID between two directories"
            echo "  ./run_fid_evaluation.sh --direct-fid /path/to/real/images /path/to/generated/images"
            echo ""
            echo "  # Quick test with limited samples"
            echo "  ./run_fid_evaluation.sh --max-samples 100 --category-filter people"
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
# Direct FID Mode
# =============================================================================

if [ "$DIRECT_FID" = true ]; then
    print_section "Direct FID Computation Mode"
    
    echo "Computing FID between:"
    echo "  Real images: $REAL_PATH"
    echo "  Generated images: $GEN_PATH"
    echo ""
    
    # Validate paths
    if ! check_dir "$REAL_PATH"; then
        echo "Error: Real images directory not found: $REAL_PATH"
        exit 1
    fi
    
    if ! check_dir "$GEN_PATH"; then
        echo "Error: Generated images directory not found: $GEN_PATH"
        exit 1
    fi
    
    echo "Running direct FID computation..."
    python compute_fid_improved.py \
        --compute_fid_only "$REAL_PATH" "$GEN_PATH" \
        --device "$DEVICE" \
        --fid_batch_size $FID_BATCH_SIZE \
        --img_size $IMG_SIZE \
        --num_workers $NUM_WORKERS \
        --exp_name "$EXP_NAME" \
        --report_to "$REPORT_TO" \
        --tracker_project_name "$TRACKER_PROJECT"
    
    echo ""
    echo "✅ Direct FID computation completed!"
    exit 0
fi

# =============================================================================
# Full MJHQ-30K FID Evaluation Mode
# =============================================================================

print_section "MJHQ-30K FID Evaluation Mode"

# Validate essential paths
if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/hart/model" ]; then
    echo "❌ ERROR: Please specify MODEL_PATH"
    echo "Use: --model-path /path/to/your/hart/model"
    echo "Or update MODEL_PATH in the script configuration section"
    exit 1
fi

if [ -z "$MJHQ_METADATA" ] || [ "$MJHQ_METADATA" = "/path/to/MJHQ30K/meta_data.json" ]; then
    echo "❌ ERROR: Please specify MJHQ metadata path"
    echo "Use: --mjhq-metadata /path/to/MJHQ30K/meta_data.json"
    echo "Or update MJHQ_METADATA in the script configuration section"
    exit 1
fi

if [ -z "$MJHQ_IMAGES" ] || [ "$MJHQ_IMAGES" = "/path/to/MJHQ30K/mjhq30k_imgs" ]; then
    echo "❌ ERROR: Please specify MJHQ images path"
    echo "Use: --mjhq-images /path/to/MJHQ30K/mjhq30k_imgs"
    echo "Or update MJHQ_IMAGES in the script configuration section"
    exit 1
fi

echo "Validating paths..."

if ! check_dir "$MODEL_PATH"; then
    echo "❌ ERROR: HART model directory not found: $MODEL_PATH"
    exit 1
fi

if ! check_file "$MJHQ_METADATA"; then
    echo "❌ ERROR: MJHQ metadata file not found: $MJHQ_METADATA"
    echo ""
    echo "To download MJHQ-30K dataset:"
    echo "  1. Visit: https://huggingface.co/datasets/playgroundai/MJHQ-30K"
    echo "  2. Download the dataset"
    echo "  3. Update the paths in this script"
    exit 1
fi

if ! check_dir "$MJHQ_IMAGES"; then
    echo "❌ ERROR: MJHQ images directory not found: $MJHQ_IMAGES"
    echo ""
    echo "To download MJHQ-30K dataset:"
    echo "  1. Visit: https://huggingface.co/datasets/playgroundai/MJHQ-30K"
    echo "  2. Download the dataset"
    echo "  3. Update the paths in this script"
    exit 1
fi

echo "✅ All paths validated successfully"
echo ""

# Display evaluation settings
echo "Evaluation Settings:"
echo "  • Category filter: ${CATEGORY_FILTER:-all}"
echo "  • Max samples: ${MAX_SAMPLES:-all}"
echo "  • CFG scale: $CFG_SCALE"
echo "  • Random seed: $SEED"
echo "  • Image size: ${IMG_SIZE}px"
echo "  • Batch size: $BATCH_SIZE (generation), $FID_BATCH_SIZE (FID)"
echo ""

# Prepare FID arguments
FID_ARGS="--model_path \"$MODEL_PATH\" \
          --text_model_path \"$TEXT_MODEL_PATH\" \
          --mjhq_metadata_path \"$MJHQ_METADATA\" \
          --mjhq_images_path \"$MJHQ_IMAGES\" \
          --output_dir \"$OUTPUT_DIR\" \
          --device $DEVICE \
          --cfg $CFG_SCALE \
          --seed $SEED \
          --batch_size $BATCH_SIZE \
          --fid_batch_size $FID_BATCH_SIZE \
          --img_size $IMG_SIZE \
          --max_token_length $MAX_TOKEN_LENGTH \
          --num_workers $NUM_WORKERS \
          --exp_name \"$EXP_NAME\" \
          --report_to \"$REPORT_TO\" \
          --tracker_project_name \"$TRACKER_PROJECT\" \
          $USE_EMA"

if [ -n "$CATEGORY_FILTER" ]; then
    FID_ARGS="$FID_ARGS --category_filter \"$CATEGORY_FILTER\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    FID_ARGS="$FID_ARGS --max_samples $MAX_SAMPLES"
fi

# Run FID evaluation
echo "Starting FID evaluation..."
echo "This will:"
echo "  1. Load HART model and text encoder"
echo "  2. Generate images for MJHQ-30K prompts"
echo "  3. Compute FID score against real MJHQ-30K images"
echo ""

eval "python compute_fid_improved.py $FID_ARGS"

# Display results
print_section "FID Evaluation Results"

if [ -f "$OUTPUT_DIR/fid_results.json" ]; then
    echo "📊 FID Evaluation Completed Successfully!"
    echo ""
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/fid_results.json'))
print(f'🎯 FID Score: {data[\"fid_score\"]:.4f}')
print()
print('📋 Evaluation Details:')
print(f'  • Samples Evaluated: {data[\"num_samples\"]}')
print(f'  • Category Filter: {data.get(\"category_filter\", \"all\")}')
print(f'  • Generation Config:')
print(f'    - CFG Scale: {data[\"config\"][\"cfg\"]}')
print(f'    - Image Size: {data[\"config\"][\"img_size\"]}px')
print(f'    - Random Seed: $SEED')
print()
print('📁 Results Location:')
print(f'  • Generated Images: $OUTPUT_DIR/generated/')
print(f'  • Detailed Results: $OUTPUT_DIR/fid_results.json')
"
    echo ""
    echo "✅ FID evaluation completed successfully!"
else
    echo "❌ FID evaluation failed - no results file found"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi

# Performance guidance
echo ""
echo "📈 FID Score Interpretation:"
echo "  • < 10:  Excellent quality, very similar to real images"
echo "  • 10-20: Good quality, minor differences from real images"
echo "  • 20-50: Moderate quality, noticeable differences"
echo "  • > 50:  Poor quality, significant differences"
echo ""

# Suggest next steps
echo "🔄 Next Steps:"
echo "  • For detailed analysis, examine generated images in: $OUTPUT_DIR/generated/"
echo "  • To compare with other models, save this FID score: $(python -c "import json; print(json.load(open('$OUTPUT_DIR/fid_results.json'))['fid_score'])" 2>/dev/null || echo "N/A")"
echo "  • For compositional reasoning evaluation, run: ./run_geneval_evaluation.sh"
echo ""

echo "=== FID Evaluation Complete ==="