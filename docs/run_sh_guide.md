# run.sh Script Documentation

## Overview

The `run.sh` script is a convenient bash script that demonstrates how to run custom 3D pose estimation inference using the MVGFormer model with arbitrary camera setups and images. It serves as a wrapper script that configures and executes the `run/custom_inference_3d.py` Python script with predefined parameters.

## Purpose

This script was created to provide an easy-to-use interface for running 3D pose estimation on custom images without needing to work with structured dataset formats. Unlike the original `run/validate_3d.py` which works with predefined test datasets, this script allows users to:

- Use arbitrary camera calibration files
- Process any set of input images
- Configure inference parameters easily
- Run inference with memory optimization settings

## Script Structure

### Configuration Section (Lines 9-16)
```bash
CONFIG_FILE="configs/panoptic/knn5-lr4-q1024-memory-optimized.yaml"
MODEL_PATH="models/mvgformer_q1024_model.pth.tar"
CAMERA_FILE="real_camera_calibration.json"
OUTPUT_DIR="./custom_inference_output"
CONFIDENCE_THRESHOLD=0.5
DEVICE="cuda"  # Use "cpu" if no GPU available
OUTPUT_FORMAT="both"  # Options: json, npy, both
```

These variables control the main inference parameters:
- **CONFIG_FILE**: YAML configuration file for the model
- **MODEL_PATH**: Path to the pre-trained model weights
- **CAMERA_FILE**: JSON file containing camera calibration data
- **OUTPUT_DIR**: Directory where results will be saved
- **CONFIDENCE_THRESHOLD**: Minimum confidence score for pose detection
- **DEVICE**: Computing device ("cuda" for GPU, "cpu" for CPU)
- **OUTPUT_FORMAT**: Output file format (json, npy, or both)

### Memory Optimization (Lines 18-20)
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
These settings optimize GPU memory usage to prevent out-of-memory errors.

### Image Files Configuration (Lines 22-30)
```bash
IMAGE_FILES=(
    "data/panoptic/160422_haggling1/hdImgs/00_03/00_03_00000779.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_06/00_06_00000779.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_12/00_12_00000779.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_13/00_13_00000779.jpg"
    "data/panoptic/160422_haggling1/hdImgs/00_23/00_23_00000779.jpg"
)
```
This array contains the paths to input images. **Important**: The order must match the camera order in the calibration file.

### File Validation (Lines 43-72)
The script validates that all required files exist before starting inference:
- Configuration file
- Model weights file
- Camera calibration file
- All input image files

### Command Execution (Lines 84-104)
The script builds and executes the Python command with all necessary arguments.

## Usage

### Basic Usage
```bash
./run.sh
```

### Customizing for Your Data

1. **Update Camera Calibration**: Modify `CAMERA_FILE` to point to your camera calibration JSON file
2. **Update Image Files**: Modify the `IMAGE_FILES` array to include your input images
3. **Adjust Parameters**: Modify confidence threshold, device, or output format as needed

### Camera Calibration File Format
The camera calibration file should be in JSON format with the following structure:
```json
{
  "calibDataSource": "source_name",
  "cameras": [
    {
      "name": "camera_name",
      "type": "hd",
      "resolution": [1920, 1080],
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "distCoef": [k1, k2, p1, p2, k3],
      "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
      "t": [[tx], [ty], [tz]]
    }
  ]
}
```

## Output

The script generates:
- **JSON files**: Human-readable pose data with metadata
- **NPY files**: Raw NumPy arrays for further processing
- **Log files**: Detailed execution logs in the output directory

### Output Structure
```
custom_inference_output/
├── poses_3d_YYYYMMDD_HHMMSS.json
├── poses_3d_YYYYMMDD_HHMMSS.npy
└── (log files in output/panoptic/...)
```

## Requirements

- CUDA-capable GPU (recommended) or CPU
- Pre-trained MVGFormer model weights
- Camera calibration data in JSON format
- Input images from multiple camera views

## Error Handling

The script includes comprehensive error checking:
- Validates all required files exist
- Provides clear error messages
- Exits gracefully on errors
- Reports successful completion with file listings

## Relationship to Other Scripts

- **validate_3d.py**: Original validation script for structured datasets
- **custom_inference_3d.py**: The underlying Python script that performs the actual inference
- **run.sh**: This convenience wrapper script for easy execution

## Example Workflow

1. Prepare your multi-view images
2. Create camera calibration JSON file
3. Update the script variables if needed
4. Run: `./run.sh`
5. Check results in the output directory

## Troubleshooting

- **CUDA errors**: Switch `DEVICE="cpu"` for CPU-only inference
- **Memory errors**: Reduce batch size or use smaller images
- **File not found**: Check all file paths in the configuration section
- **No poses detected**: Try lowering the confidence threshold or check camera calibration accuracy
