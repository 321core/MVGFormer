#!/bin/bash

# Script to reproduce the frame_id validation issue
echo "Reproducing frame_id validation issue..."
echo "======================================="

# Test with frame_id (should fail with assertion error)
echo "Testing with --frame_id 100 (should fail):"
bash run-validate.sh --frame_id 100 --config configs/panoptic/knn5-lr4-q1024-memory-optimized.yaml --model_path models/mvgformer_q1024_model.pth.tar

echo ""
echo "If the above failed with 'AssertionError: number mismatch', the issue is confirmed."
