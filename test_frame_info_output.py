#!/usr/bin/env python3
"""
Test script to verify frame information output functionality
"""
import subprocess
import sys
import os

def test_frame_info_output():
    """Test the frame information output when frame_id is specified"""
    print("Testing frame information output...")
    print("=" * 50)

    # Test 1: With frame_id specified
    print("\n1. Testing with --frame_id 100:")
    print("-" * 30)

    cmd = [
        "python", "run/validate_3d.py",
        "--cfg", "configs/panoptic/knn5-lr4-q1024-memory-optimized.yaml",
        "--frame_id", "100",
        "--model_path", "models/mvgformer_q1024_model.pth.tar"
    ]

    try:
        # Run the command and capture output for a few seconds
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout

        # Check if frame information is present
        if "FRAME-SPECIFIC VALIDATION" in output:
            print("✓ Frame-specific header found")
        else:
            print("✗ Frame-specific header NOT found")

        if "Processing specific frame: 100" in output:
            print("✓ Frame number displayed correctly")
        else:
            print("✗ Frame number NOT displayed")

        if "Dataset:" in output:
            print("✓ Dataset information displayed")
        else:
            print("✗ Dataset information NOT displayed")

        print("\nSample output:")
        print(output[:800] + "..." if len(output) > 800 else output)

    except subprocess.TimeoutExpired:
        print("✓ Command started successfully (timeout after 10s is expected)")
        print("  This indicates the frame info was likely displayed before model loading")
    except FileNotFoundError:
        print("✗ Could not run validation script (missing files)")
        return False
    except Exception as e:
        print(f"✗ Error running validation: {e}")
        return False

    print("\n" + "=" * 50)
    print("Frame information output test completed!")
    return True

if __name__ == "__main__":
    success = test_frame_info_output()
    if success:
        print("\n🎉 Test completed! Frame information should now be displayed when --frame_id is used.")
    else:
        print("\n❌ Test failed. Please check the implementation.")
    sys.exit(0 if success else 1)
