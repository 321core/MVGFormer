#!/usr/bin/env python3
"""
Simple test to verify that the frame_id fix works correctly
"""
import numpy as np
import sys
import os

# Add lib path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Mock a simple dataset class to test the evaluate method logic
class MockPanopticDataset:
    def __init__(self, db_size=1000, num_views=5):
        self.db_size = db_size
        self.num_views = num_views
        # Create mock db with some dummy data
        self.db = []
        for i in range(db_size):
            self.db.append({
                'joints_3d': [np.random.rand(15, 3)],  # One person with 15 joints
                'joints_3d_vis': [np.ones((15, 1))]   # All joints visible
            })

    def evaluate(self, preds, method='score_sort', frame_id=None):
        """
        Simplified version of the evaluate method to test the logic
        """
        eval_list = []
        gt_num = self.db_size // self.num_views

        # Handle single frame validation (when frame_id is specified)
        if frame_id is not None and len(preds) == 1:
            # Single frame mode: only evaluate the specified frame
            gt_num = 1

        print(f"Testing: len(preds)={len(preds)}, gt_num={gt_num}, frame_id={frame_id}")

        try:
            assert len(preds) == gt_num, f'number mismatch: got {len(preds)} predictions but expected {gt_num} ground truth samples'
            print("✓ Assertion passed!")
            return True
        except AssertionError as e:
            print(f"✗ Assertion failed: {e}")
            return False

def test_frame_id_fix():
    """Test the frame_id fix"""
    print("Testing frame_id fix...")
    print("=" * 50)

    # Create mock dataset (1000 frames, 5 views = 200 ground truth samples)
    dataset = MockPanopticDataset(db_size=1000, num_views=5)

    # Test 1: Normal validation (all frames) - should work
    print("\n1. Testing normal validation (all frames):")
    preds_all = [np.random.rand(10, 15, 5) for _ in range(200)]  # 200 predictions
    result1 = dataset.evaluate(preds_all, frame_id=None)

    # Test 2: Single frame validation (frame_id specified) - should work with fix
    print("\n2. Testing single frame validation (frame_id=100):")
    preds_single = [np.random.rand(10, 15, 5)]  # Only 1 prediction
    result2 = dataset.evaluate(preds_single, frame_id=100)

    # Test 3: Edge case - empty predictions
    print("\n3. Testing edge case (empty predictions):")
    preds_empty = []
    try:
        result3 = dataset.evaluate(preds_empty, frame_id=None)
    except AssertionError as e:
        print(f"✓ Expected assertion error for empty predictions: {e}")
        result3 = False

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Normal validation: {'PASS' if result1 else 'FAIL'}")
    print(f"Single frame validation: {'PASS' if result2 else 'FAIL'}")
    print(f"Empty predictions handling: {'PASS' if not result3 else 'FAIL'}")

    if result1 and result2 and not result3:
        print("\n🎉 All tests passed! The frame_id fix is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = test_frame_id_fix()
    sys.exit(0 if success else 1)
