#!/bin/bash

# HART Cache GenEval Evaluation Script
# Evaluates the impact of VAR-style caching on HART performance using GenEval dataset

# Default parameters
MODEL_PATH=""
GENEVAL_PATH=""
OUTPUT_DIR="./hart_cache_geneval_results"
DETECTOR_CONFIG=""
DETECTOR_MODEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --geneval_path)
            GENEVAL_PATH="$2" 
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --detector_config)
            DETECTOR_CONFIG="$2"
            shift 2
            ;;
        --detector_model)
            DETECTOR_MODEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model_path MODEL_PATH --geneval_path GENEVAL_PATH [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --model_path MODEL_PATH     Path to HART model"
            echo "  --geneval_path GENEVAL_PATH Path to GenEval prompts file"
            echo ""
            echo "Optional arguments:"
            echo "  --output_dir OUTPUT_DIR     Output directory (default: ./hart_cache_geneval_results)"
            echo "  --detector_config CONFIG    Path to MMDetection config file"
            echo "  --detector_model MODEL      Path to MMDetection model file"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Compare all cache presets"
            echo "  $0 --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --compare_presets"
            echo ""
            echo "  # Evaluate specific preset"
            echo "  $0 --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --cache_preset aggressive"
            echo ""
            echo "  # Custom cache configuration"
            echo "  $0 --model_path /path/to/hart --geneval_path /path/to/geneval.jsonl --skip_stages 144 256 --cache_stages 81 144"
            exit 0
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$GENEVAL_PATH" ]; then
    echo "Error: --geneval_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$GENEVAL_PATH" ]; then
    echo "Warning: GenEval path does not exist: $GENEVAL_PATH"
    echo "Will use sample prompts for testing"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "HART Cache GenEval Evaluation"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "GenEval path: $GENEVAL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Extra arguments: $EXTRA_ARGS"
echo "=========================================="

# Construct the python command
PYTHON_CMD="python3 $(dirname "$0")/evaluate_hart_cache_geneval.py"
PYTHON_CMD="$PYTHON_CMD --model_path \"$MODEL_PATH\""
PYTHON_CMD="$PYTHON_CMD --geneval_path \"$GENEVAL_PATH\""
PYTHON_CMD="$PYTHON_CMD --output_dir \"$OUTPUT_DIR\""

if [ ! -z "$DETECTOR_CONFIG" ]; then
    PYTHON_CMD="$PYTHON_CMD --detector_config \"$DETECTOR_CONFIG\""
fi

if [ ! -z "$DETECTOR_MODEL" ]; then
    PYTHON_CMD="$PYTHON_CMD --detector_model \"$DETECTOR_MODEL\""
fi

# Add extra arguments
PYTHON_CMD="$PYTHON_CMD $EXTRA_ARGS"

echo "Running command:"
echo "$PYTHON_CMD"
echo ""

# Execute the command
eval $PYTHON_CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
    
    # Show summary if available
    if [ -f "$OUTPUT_DIR/comparison_report.md" ]; then
        echo ""
        echo "Comparison Report Summary:"
        echo "----------------------------------------"
        head -20 "$OUTPUT_DIR/comparison_report.md"
        echo "..."
        echo "Full report: $OUTPUT_DIR/comparison_report.md"
    fi
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed!"
    echo "Check the error messages above"
    echo "=========================================="
    exit 1
fi