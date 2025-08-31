#!/bin/bash

# MVGFormer Standard Validation Script
# This script runs validation on the standard CMU Panoptic dataset

echo "MVGFormer Standard Dataset Validation"
echo "===================================="

# Default parameters
FRAME_ID=""
CONFIG_FILE="configs/panoptic/knn5-lr4-q1024-memory-optimized.yaml"
MODEL_PATH="models/mvgformer_q1024_model.pth.tar"
BATCH_SIZE=1
CONFIDENCE_THRESHOLD=0.1
DEVICE="cuda"
DEBUG_LOG=false
SEQUENCE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frame_id)
            FRAME_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --confidence_threshold)
            CONFIDENCE_THRESHOLD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_LOG=true
            shift
            ;;
        --sequence)
            SEQUENCE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --frame_id FRAME_ID            Specific frame to validate (optional)"
            echo "  --config CONFIG_FILE           Config file to use (default: knn5-lr4-q1024-memory-optimized.yaml)"
            echo "  --model_path MODEL_PATH        Path to model weights (default: models/mvgformer_q1024_model.pth.tar)"
            echo "  --batch_size BATCH_SIZE        Batch size for validation (default: 1)"
            echo "  --confidence_threshold THRESH  Confidence threshold (default: 0.1)"
            echo "  --device DEVICE                Device to use: cuda/cpu (default: cuda)"
            echo "  --debug                        Enable debug logging"
            echo "  --sequence SEQUENCE            Test sequence (CMU0, CMU1, etc.)"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Validate all frames"
            echo "  $0 --frame_id 779                     # Validate specific frame"
            echo "  $0 --frame_id 0 --debug               # Validate frame 0 with debug"
            echo "  $0 --sequence CMU1                    # Test on CMU1 sequence"
            echo "  $0 --confidence_threshold 0.3         # Use higher confidence threshold"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Memory optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Model path: $MODEL_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  Device: $DEVICE"
if [ ! -z "$FRAME_ID" ]; then
    echo "  Frame ID: $FRAME_ID"
fi
if [ ! -z "$SEQUENCE" ]; then
    echo "  Test sequence: $SEQUENCE"
fi
if [ "$DEBUG_LOG" = true ]; then
    echo "  Debug logging: enabled"
fi
echo ""

# Check if files exist
echo "Checking required files..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la configs/panoptic/
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please download the model file:"
    echo "  https://drive.google.com/file/d/1iPLyUzatBtm7iIoWoErgRXn7sS2OMTxC/view?usp=sharing"
    echo "And place it at: $MODEL_PATH"
    exit 1
fi

# Check if dataset exists
DATASET_DIR="data/panoptic"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Please download the CMU Panoptic dataset following the installation guide:"
    echo "  https://github.com/XunshanMan/MVGFormer/blob/master/docs/detail_install.md"
    exit 1
fi

# Check for some expected sequences
EXPECTED_SEQUENCES=("160422_haggling1" "160906_pizza1" "160906_ian5" "160906_band4")
FOUND_SEQUENCES=()

for seq in "${EXPECTED_SEQUENCES[@]}"; do
    if [ -d "$DATASET_DIR/$seq" ]; then
        FOUND_SEQUENCES+=("$seq")
    fi
done

if [ ${#FOUND_SEQUENCES[@]} -eq 0 ]; then
    echo "Error: No valid sequences found in $DATASET_DIR"
    echo "Expected sequences: ${EXPECTED_SEQUENCES[*]}"
    echo "Please download the dataset following the installation guide"
    exit 1
fi

echo "Found sequences: ${FOUND_SEQUENCES[*]}"

echo "All required files found successfully!"
echo ""

# Build the validation command
CMD="python run/validate_3d.py \
    --cfg $CONFIG_FILE \
    --model_path $MODEL_PATH \
    --device $DEVICE \
    TEST.BATCH_SIZE=$BATCH_SIZE \
    DECODER.pred_conf_threshold=$CONFIDENCE_THRESHOLD"

# Add optional parameters
if [ ! -z "$FRAME_ID" ]; then
    CMD="$CMD --frame_id $FRAME_ID"
fi

if [ "$DEBUG_LOG" = true ]; then
    CMD="$CMD DEBUG.LOG_VAL_LOSS=True DEBUG.PRINT_TO_FILE=True"
fi

if [ ! -z "$SEQUENCE" ]; then
    CMD="$CMD DATASET.TEST_CAM_SEQ=$SEQUENCE"
fi

echo "Starting validation..."
echo "Command to execute:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

# Check if validation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Validation completed successfully!"

    if [ ! -z "$FRAME_ID" ]; then
        echo "Frame $FRAME_ID processed successfully"
    else
        echo "All frames processed successfully"
    fi

    echo ""
    echo "Check the output above for validation metrics (AP25, MPJPE, etc.)"

    # Show log files if debug was enabled
    if [ "$DEBUG_LOG" = true ]; then
        echo ""
        echo "Debug logs available in:"
        ls -la output/ log/ 2>/dev/null || echo "  (No log files found)"
    fi
else
    echo ""
    echo "Error: Validation failed!"
    echo ""
    echo "Common troubleshooting steps:"
    echo "1. Check GPU memory: nvidia-smi"
    echo "2. Try CPU mode: --device cpu"
    echo "3. Reduce batch size: --batch_size 1"
    echo "4. Test with single frame: --frame_id 0"
    echo "5. Enable debug mode: --debug"
    echo ""
    echo "If the problem persists, please check the error messages above"
    exit 1
fi
