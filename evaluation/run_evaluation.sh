#!/bin/bash

# HART Model Comprehensive Evaluation Script
# Adapted from SANA evaluation pipeline with FastVAR insights

set -e

# Default configuration
MODEL_PATH=""
TEXT_MODEL_PATH=""
SHIELD_MODEL_PATH=""
OUTPUT_DIR="./evaluation_output"
PROMPTS_FILE=""
REFERENCE_PATH=""
DEVICE="cuda"
BATCH_SIZE=32
IMG_SIZE=1024
MAX_SAMPLES=10000
EXP_NAME="hart_evaluation"

# Evaluation flags
EVAL_FID=false
EVAL_CLIP=false
EVAL_GENEVAL=false
EVAL_QUALITY=false
EVAL_SPEED=false
GENERATE_SAMPLES=false

# GenEval configuration
GENEVAL_PROMPTS=""
GENEVAL_MODEL_PATH=""
GENEVAL_CONFIG_PATH=""
GENEVAL_CLASS_NAMES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --text_model_path)
            TEXT_MODEL_PATH="$2"
            shift 2
            ;;
        --shield_model_path)
            SHIELD_MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --prompts_file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --reference_path)
            REFERENCE_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --img_size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --eval_fid)
            EVAL_FID=true
            shift
            ;;
        --eval_clip)
            EVAL_CLIP=true
            shift
            ;;
        --eval_geneval)
            EVAL_GENEVAL=true
            shift
            ;;
        --eval_quality)
            EVAL_QUALITY=true
            shift
            ;;
        --eval_speed)
            EVAL_SPEED=true
            shift
            ;;
        --generate_samples)
            GENERATE_SAMPLES=true
            shift
            ;;
        --geneval_prompts)
            GENEVAL_PROMPTS="$2"
            shift 2
            ;;
        --geneval_model_path)
            GENEVAL_MODEL_PATH="$2"
            shift 2
            ;;
        --geneval_config_path)
            GENEVAL_CONFIG_PATH="$2"
            shift 2
            ;;
        --geneval_class_names)
            GENEVAL_CLASS_NAMES="$2"
            shift 2
            ;;
        --help|-h)
            echo "HART Model Evaluation Script"
            echo ""
            echo "Required arguments:"
            echo "  --model_path PATH          Path to HART model"
            echo "  --text_model_path PATH     Path to text model (Qwen2)"
            echo "  --prompts_file PATH        File containing evaluation prompts"
            echo ""
            echo "Optional arguments:"
            echo "  --shield_model_path PATH   Path to shield model (ShieldGemma)"
            echo "  --output_dir DIR           Output directory (default: ./evaluation_output)"
            echo "  --reference_path PATH      Reference dataset path for FID"
            echo "  --exp_name NAME            Experiment name (default: hart_evaluation)"
            echo "  --device DEVICE            Device to use (default: cuda)"
            echo "  --batch_size SIZE          Batch size (default: 32)"
            echo "  --img_size SIZE            Image size (default: 1024)"
            echo "  --max_samples NUM          Max samples to evaluate (default: 10000)"
            echo ""
            echo "Evaluation options:"
            echo "  --generate_samples         Generate samples using HART model"
            echo "  --eval_fid                 Evaluate FID score"
            echo "  --eval_clip                Evaluate CLIP score"
            echo "  --eval_geneval             Evaluate using GenEval benchmark"
            echo "  --eval_quality             Evaluate image quality metrics"
            echo "  --eval_speed               Benchmark inference speed"
            echo ""
            echo "GenEval options:"
            echo "  --geneval_prompts PATH     GenEval prompts file"
            echo "  --geneval_model_path PATH  MMDetection model path"
            echo "  --geneval_config_path PATH MMDetection config path"
            echo "  --geneval_class_names PATH Class names file"
            echo ""
            echo "Examples:"
            echo "  # Full evaluation"
            echo "  $0 --model_path /path/to/model --text_model_path /path/to/qwen2 \\"
            echo "     --prompts_file prompts.txt --generate_samples --eval_fid \\"
            echo "     --eval_clip --reference_path /path/to/reference"
            echo ""
            echo "  # Quick CLIP evaluation"
            echo "  $0 --model_path /path/to/model --text_model_path /path/to/qwen2 \\"
            echo "     --prompts_file prompts.txt --eval_clip"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    exit 1
fi

if [[ -z "$TEXT_MODEL_PATH" ]]; then
    echo "Error: --text_model_path is required"
    exit 1
fi

if [[ -z "$PROMPTS_FILE" ]]; then
    echo "Error: --prompts_file is required"
    exit 1
fi

# Check if any evaluation is requested
if [[ "$EVAL_FID" == false && "$EVAL_CLIP" == false && "$EVAL_GENEVAL" == false && "$EVAL_QUALITY" == false && "$EVAL_SPEED" == false && "$GENERATE_SAMPLES" == false ]]; then
    echo "Warning: No evaluation requested. Use --eval_* flags to run evaluations."
    echo "Adding default evaluations: --eval_clip --eval_quality"
    EVAL_CLIP=true
    EVAL_QUALITY=true
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "HART Model Comprehensive Evaluation"
echo "=============================================="
echo "Model Path: $MODEL_PATH"
echo "Text Model Path: $TEXT_MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Prompts File: $PROMPTS_FILE"
echo "Experiment Name: $EXP_NAME"
echo "Device: $DEVICE"
echo "=============================================="

# Build evaluation command
EVAL_CMD="python evaluate_hart.py"
EVAL_CMD="$EVAL_CMD --model_path '$MODEL_PATH'"
EVAL_CMD="$EVAL_CMD --text_model_path '$TEXT_MODEL_PATH'"
EVAL_CMD="$EVAL_CMD --prompts_file '$PROMPTS_FILE'"
EVAL_CMD="$EVAL_CMD --output_dir '$OUTPUT_DIR'"
EVAL_CMD="$EVAL_CMD --exp_name '$EXP_NAME'"
EVAL_CMD="$EVAL_CMD --device '$DEVICE'"
EVAL_CMD="$EVAL_CMD --batch_size $BATCH_SIZE"
EVAL_CMD="$EVAL_CMD --img_size $IMG_SIZE"
EVAL_CMD="$EVAL_CMD --max_samples $MAX_SAMPLES"

if [[ -n "$SHIELD_MODEL_PATH" ]]; then
    EVAL_CMD="$EVAL_CMD --shield_model_path '$SHIELD_MODEL_PATH'"
fi

if [[ "$GENERATE_SAMPLES" == true ]]; then
    EVAL_CMD="$EVAL_CMD --generate_samples"
fi

if [[ "$EVAL_FID" == true ]]; then
    if [[ -z "$REFERENCE_PATH" ]]; then
        echo "Error: --reference_path is required for FID evaluation"
        exit 1
    fi
    EVAL_CMD="$EVAL_CMD --eval_fid --reference_path '$REFERENCE_PATH'"
fi

if [[ "$EVAL_CLIP" == true ]]; then
    EVAL_CMD="$EVAL_CMD --eval_clip_score"
fi

if [[ "$EVAL_GENEVAL" == true ]]; then
    if [[ -z "$GENEVAL_PROMPTS" || -z "$GENEVAL_MODEL_PATH" || -z "$GENEVAL_CONFIG_PATH" ]]; then
        echo "Error: GenEval evaluation requires --geneval_prompts, --geneval_model_path, and --geneval_config_path"
        exit 1
    fi
    EVAL_CMD="$EVAL_CMD --eval_geneval --geneval_prompts '$GENEVAL_PROMPTS'"
    EVAL_CMD="$EVAL_CMD --geneval_model_path '$GENEVAL_MODEL_PATH'"
    EVAL_CMD="$EVAL_CMD --geneval_config_path '$GENEVAL_CONFIG_PATH'"
    
    if [[ -n "$GENEVAL_CLASS_NAMES" ]]; then
        EVAL_CMD="$EVAL_CMD --geneval_class_names '$GENEVAL_CLASS_NAMES'"
    fi
fi

if [[ "$EVAL_QUALITY" == true ]]; then
    EVAL_CMD="$EVAL_CMD --eval_quality_metrics"
fi

if [[ "$EVAL_SPEED" == true ]]; then
    EVAL_CMD="$EVAL_CMD --eval_speed"
fi

# Print command for debugging
echo "Running evaluation command:"
echo "$EVAL_CMD"
echo "=============================================="

# Run evaluation
eval "$EVAL_CMD"

# Check if evaluation completed successfully
if [[ $? -eq 0 ]]; then
    echo "=============================================="
    echo "Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    echo "=============================================="
    
    # Display results if available
    RESULTS_FILE="${EXP_NAME}_evaluation_results.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        echo "Evaluation Summary:"
        if command -v jq &> /dev/null; then
            # Use jq if available for pretty printing
            echo "FID Score: $(jq -r '.FID // "N/A"' "$RESULTS_FILE")"
            echo "CLIP Score: $(jq -r '.CLIP_Score.mean // "N/A"' "$RESULTS_FILE")"
            echo "GenEval Accuracy: $(jq -r '.GenEval.overall_accuracy // "N/A"' "$RESULTS_FILE")"
        else
            # Fallback: just show the file exists
            echo "Results available in: $RESULTS_FILE"
        fi
    fi
else
    echo "=============================================="
    echo "Evaluation failed with exit code: $?"
    echo "=============================================="
    exit 1
fi