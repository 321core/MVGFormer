#!/bin/bash

# MVGFormer Custom 3D Pose Inference Script
# This script demonstrates how to run the custom inference script with your own camera setup and images

echo "MVGFormer Custom 3D Pose Inference"
echo "=================================="

# Default frame number (can be overridden with command line argument)
FRAME_NUMBER=779

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frame_number)
            FRAME_NUMBER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--frame_number FRAME_NUMBER]"
            echo "  --frame_number: Frame number to process (default: 779)"
            exit 1
            ;;
    esac
done

# Configuration parameters - Memory Optimized Version
CONFIG_FILE="configs/panoptic/knn5-lr4-q1024-memory-optimized.yaml"
MODEL_PATH="models/mvgformer_q1024_model.pth.tar"
CAMERA_FILE="real_camera_calibration.json"
OUTPUT_DIR="./custom_inference_output"
CONFIDENCE_THRESHOLD=0.5
DEVICE="cuda"  # Use "cpu" if no GPU available
OUTPUT_FORMAT="both"  # Options: json, npy, both

# Memory optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Format frame number with zero padding (8 digits)
PADDED_FRAME=$(printf "%08d" $FRAME_NUMBER)

# Image files - Real images from panoptic dataset
# Order matches the camera calibration file (00_03, 00_06, 00_12, 00_13, 00_23)
IMAGE_FILES=(
    "data/panoptic/160422_haggling1/hdImgs/00_03/00_03_${PADDED_FRAME}.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_06/00_06_${PADDED_FRAME}.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_12/00_12_${PADDED_FRAME}.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_13/00_13_${PADDED_FRAME}.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_23/00_23_${PADDED_FRAME}.jpg"
)

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Model path: $MODEL_PATH"
echo "  Camera file: $CAMERA_FILE"
echo "  Frame number: $FRAME_NUMBER"
echo "  Number of cameras: ${#IMAGE_FILES[@]}"
echo "  Output directory: $OUTPUT_DIR"
echo "  Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "  Device: $DEVICE"
echo "  Output format: $OUTPUT_FORMAT"
echo ""

# Check if files exist
echo "Checking required files..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please ensure the config file exists or update CONFIG_FILE variable"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please download the model file or update MODEL_PATH variable"
    exit 1
fi

if [ ! -f "$CAMERA_FILE" ]; then
    echo "Error: Camera calibration file not found: $CAMERA_FILE"
    echo "Please create a camera calibration file or update CAMERA_FILE variable"
    echo "You can use the example_camera_calibration.json as a template"
    exit 1
fi

# Check image files
for img in "${IMAGE_FILES[@]}"; do
    if [ ! -f "$img" ]; then
        echo "Error: Image file not found: $img"
        echo "Please update the IMAGE_FILES array with correct paths"
        exit 1
    fi
done

echo "All files found successfully!"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting inference..."
echo "Command to execute:"
echo "python run/custom_inference_3d.py \\"

# Build the command
CMD="python run/custom_inference_3d.py \
    --cfg $CONFIG_FILE \
    --model_path $MODEL_PATH \
    --camera_file $CAMERA_FILE \
    --output_dir $OUTPUT_DIR \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --device $DEVICE \
    --output_format $OUTPUT_FORMAT \
    --num_person 5 \
    --image_files"

# Add image files to command
for img in "${IMAGE_FILES[@]}"; do
    CMD="$CMD $img"
done

echo "$CMD"
echo ""

# Execute the command
eval $CMD

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -la "$OUTPUT_DIR"/poses_3d_*
else
    echo ""
    echo "Error: Inference failed!"
    echo "Please check the error messages above and verify your setup"
    exit 1
fi
