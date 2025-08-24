#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom 3D Pose Inference Script for MVGFormer

This script performs multi-view 3D pose estimation using custom camera calibration files
and individual image files, instead of structured dataset formats.

Usage:
python run/custom_inference_3d.py \
    --cfg configs/panoptic/knn5-1r4-q1024.yaml \
    --model_path models/mvgformer_q1024_model.pth.tar \
    --camera_file /path/to/camera_calibration.json \
    --image_files /path/to/cam1.jpg /path/to/cam2.jpg ... \
    --output_dir ./output \
    --confidence_threshold 0.5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import os
import pprint
import json
import numpy as np
from PIL import Image
from datetime import datetime
import gc

import _init_paths

from core.config import config
from core.config import update_config, update_config_dynamic_input
from utils.utils import create_logger
import lib.utils.misc as utils
from models.util.misc import is_main_process
from mvn.utils.multiview import Camera

import dataset
import models
import models.dq_transformer


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for loading individual image files with camera calibration"""

    def __init__(self, camera_file, image_files, transform=None):
        self.camera_file = camera_file
        self.image_files = image_files
        self.transform = transform
        self.num_views = len(image_files)

        # Load camera calibration data
        self.cameras = self._load_cameras()

        # Validate that we have matching number of cameras and images
        if len(self.cameras) != len(self.image_files):
            raise ValueError(f"Number of cameras ({len(self.cameras)}) doesn't match number of images ({len(self.image_files)})")

    def _load_cameras(self):
        """Load camera calibration from JSON file"""
        with open(self.camera_file, 'r') as f:
            calib_data = json.load(f)

        cameras = []

        # Handle different possible JSON formats
        if 'cameras' in calib_data:
            # Panoptic-style format
            for cam_data in calib_data['cameras']:
                camera = Camera(
                    R=np.array(cam_data['R']),
                    t=np.array(cam_data['t']).reshape((3, 1)),
                    K=np.array(cam_data['K']),
                    dist=np.array(cam_data.get('distCoef', [])) if cam_data.get('distCoef') else None,
                    name=f"cam_{cam_data.get('panel', 0)}_{cam_data.get('node', 0)}"
                )
                cameras.append(camera)
        else:
            # Simple format: list of camera dictionaries
            for i, cam_data in enumerate(calib_data):
                camera = Camera(
                    R=np.array(cam_data['R']),
                    t=np.array(cam_data['t']).reshape((3, 1)) if np.array(cam_data['t']).ndim == 1 else np.array(cam_data['t']),
                    K=np.array(cam_data['K']),
                    dist=np.array(cam_data.get('distCoef', [])) if cam_data.get('distCoef') else None,
                    name=f"cam_{i}"
                )
                cameras.append(camera)

        return cameras

    def __len__(self):
        return 1  # Single frame inference

    def __getitem__(self, idx):
        inputs = []
        metas = []

        for i, image_path in enumerate(self.image_files):
            # Load and preprocess image
            image_pil = Image.open(image_path).convert('RGB')
            original_size = image_pil.size  # (width, height)

            # Convert to numpy for processing
            data_numpy = np.array(image_pil)
            height, width, _ = data_numpy.shape

            # Calculate center and scale
            c = np.array([width / 2.0, height / 2.0])
            from utils.transforms import get_scale
            image_size = np.array([256, 256])  # Default image size
            s = get_scale((width, height), image_size)
            r = 0  # No rotation

            # Apply affine transformation
            from utils.transforms import get_affine_transform
            import cv2
            trans = get_affine_transform(c, s, r, image_size, inv=0)
            input_img = cv2.warpAffine(
                data_numpy,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)

            # Apply transform to tensor
            if self.transform:
                input_tensor = self.transform(input_img)
            else:
                input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

            inputs.append(input_tensor)

            # Create metadata matching JointsDataset format
            camera = self.cameras[i]

            # Extract camera parameters from Camera object
            K = camera.K
            R = camera.R
            t = camera.t.reshape(-1)  # flatten to 1D

            # Create camera dict in expected format
            # Extract distortion coefficients from camera
            distCoef = camera.dist if camera.dist is not None else np.zeros(5)

            # Ensure we have at least 5 distortion coefficients (pad with zeros if needed)
            if len(distCoef) < 5:
                distCoef = np.pad(distCoef, (0, 5 - len(distCoef)))

            # Split distortion coefficients into k (radial) and p (tangential)
            # Standard format: [k1, k2, p1, p2, k3]
            k = np.array([distCoef[0], distCoef[1], distCoef[4]])  # k1, k2, k3
            p = np.array([distCoef[2], distCoef[3]])  # p1, p2

            cam_dict = {
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
                'R': R,
                'T': t,
                'standard_T': t,  # Use same as T for simplicity
                'k': k,  # Radial distortion coefficients [k1, k2, k3]
                'p': p,  # Tangential distortion coefficients [p1, p2]
            }

            # Create camera intrinsic matrix
            cam_intri = np.eye(3, 3)
            cam_intri[0, 0] = float(K[0, 0])
            cam_intri[1, 1] = float(K[1, 1])
            cam_intri[0, 2] = float(K[0, 2])
            cam_intri[1, 2] = float(K[1, 2])

            # Create affine transform matrices
            aff_trans = np.eye(3, 3)
            aff_trans[0:2] = trans
            inv_trans = get_affine_transform(c, s, r, image_size, inv=1)
            inv_aff_trans = np.eye(3, 3)
            inv_aff_trans[0:2] = inv_trans

            # Create augmentation transform
            aug_trans = np.eye(3, 3)
            aug_trans[0:2] = trans
            heatmap_size = np.array([64, 64])  # Default heatmap size
            hm_scale = heatmap_size / image_size
            scale_trans = np.eye(3, 3)
            scale_trans[0, 0] = hm_scale[1]
            scale_trans[1, 1] = hm_scale[0]
            aug_trans = scale_trans @ aug_trans

            # Create dummy joint data with correct shapes
            maximum_person = 10  # Default maximum persons
            num_joints = 19  # COCO format

            joints_3d_u = np.zeros((maximum_person, num_joints, 3))
            joints_3d_vis_u = np.zeros((maximum_person, num_joints, 3))
            joints_3d_voxelpose_pred_u = np.zeros((maximum_person, num_joints, 5))
            joints_u = np.zeros((maximum_person, num_joints, 2))
            joints_vis_u = np.zeros((maximum_person, num_joints, 2))
            roots_3d = np.zeros((maximum_person, 3))

            meta = {
                'image': image_path,
                'num_person': 0,  # No ground truth persons for inference
                'joints_3d': joints_3d_u,
                'joints_3d_vis': joints_3d_vis_u,
                'joints_3d_voxelpose_pred': joints_3d_voxelpose_pred_u,
                'roots_3d': roots_3d,
                'joints': joints_u,
                'joints_vis': joints_vis_u,
                'center': c,
                'scale': s,
                'rotation': r,
                'camera': cam_dict,
                'camera_Intri': cam_intri,
                'camera_R': R,
                'camera_focal': np.stack([cam_dict['fx'], cam_dict['fy'], np.ones_like(cam_dict['fy'])]),
                'camera_T': t,
                'camera_standard_T': t,
                'affine_trans': aff_trans,
                'inv_affine_trans': inv_aff_trans,
                'aug_trans': aug_trans,
            }
            metas.append(meta)

        return inputs, metas


def parse_args():
    parser = argparse.ArgumentParser(description='Custom 3D pose inference with MVGFormer')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    parser.add_argument('--model_path', help='path to model file',
                        required=True, type=str)
    parser.add_argument('--camera_file', help='path to camera calibration JSON file',
                        required=True, type=str)
    parser.add_argument('--image_files', help='paths to input image files (one per camera)',
                        required=True, nargs='+', type=str)
    parser.add_argument('--output_dir', help='output directory for results',
                        default='./custom_inference_output', type=str)
    parser.add_argument('--confidence_threshold', help='confidence threshold for pose detection',
                        default=0.5, type=float)
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    parser.add_argument('--output_format', choices=['json', 'npy', 'both'],
                        default='json', help='output format for pose data')

    args, unknown = parser.parse_known_args()
    update_config(args.cfg)
    update_config_dynamic_input(unknown)
    return args


def custom_inference_3d(config, model, dataset, output_dir, threshold, device="cpu"):
    """Perform 3D pose inference on custom data with memory optimization"""
    model.eval()

    # Create data loader with reduced memory usage
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,  # Disabled to save memory
        num_workers=0
    )

    results = []

    # Enable mixed precision if CUDA is available
    use_amp = device.type == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    with torch.no_grad():
        for i, (inputs, meta) in enumerate(data_loader):
            # Clear GPU cache before processing each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Move to device with non-blocking transfer
            inputs = [img.to(device, non_blocking=True) for img in inputs]
            meta = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in m.items()} for m in meta]

            # Run inference with mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(views=inputs, meta=meta, output_dir=output_dir, frame_id=i, threshold=threshold)
            else:
                output = model(views=inputs, meta=meta, output_dir=output_dir, frame_id=i, threshold=threshold)

            # Extract predictions
            bs, num_queries = output["pred_logits"].shape[:2]
            num_joints = 19  # COCO format

            src_poses = output['pred_poses']['outputs_coord'].view(bs, num_queries, num_joints, 3)
            scores = output['pred_logits'][:, :, 1:2].sigmoid()  # Use 2nd dim for positive class
            scores = scores.unsqueeze(2).expand(-1, -1, num_joints, -1)

            # Filter by confidence threshold
            valid_mask = (scores > threshold).float()

            # Combine poses with scores
            pred = torch.cat([src_poses, valid_mask - 1, scores], dim=-1)  # [x,y,z,valid,confidence]
            pred = pred.detach().cpu().numpy()

            # Process predictions for each batch item
            for b in range(pred.shape[0]):
                batch_preds = pred[b]
                # Filter valid predictions
                valid_preds = batch_preds[batch_preds[:, 0, 3] >= 0]  # Keep only valid poses

                frame_result = {
                    'frame_id': i,
                    'image_files': [m['image'] for m in meta],
                    'num_detected_people': len(valid_preds),
                    'poses_3d': []
                }

                for person_idx, pose in enumerate(valid_preds):
                    person_data = {
                        'person_id': person_idx,
                        'joints_3d': pose[:, :3].tolist(),  # [joint_idx][xyz]
                        'confidence_scores': pose[:, 4].tolist(),  # [joint_idx]
                        'visibility': (pose[:, 3] >= 0).tolist()  # [joint_idx]
                    }
                    frame_result['poses_3d'].append(person_data)

                results.append(frame_result)

            # Clear memory after processing each batch
            del inputs, meta, output, pred, batch_preds
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            # Print memory usage for monitoring
            if device.type == 'cuda' and i % 1 == 0:  # Print every batch for debugging
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(device) / 1024**3      # GB
                print(f"Batch {i}: GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    return results


def save_results(results, output_dir, output_format):
    """Save inference results in specified format"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_format in ['json', 'both']:
        json_file = os.path.join(output_dir, f'poses_3d_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_file}")

    if output_format in ['npy', 'both']:
        npy_file = os.path.join(output_dir, f'poses_3d_{timestamp}.npy')
        # Convert to numpy format for compatibility with existing evaluation code
        np_results = []
        for result in results:
            for pose_data in result['poses_3d']:
                joints = np.array(pose_data['joints_3d'])  # [19, 3]
                confidence = np.array(pose_data['confidence_scores'])  # [19]
                visibility = np.array(pose_data['visibility'])  # [19]
                # Format: [x, y, z, visibility_flag, confidence]
                pose_array = np.concatenate([
                    joints,
                    visibility.reshape(-1, 1) * 2 - 1,  # Convert to -1/1
                    confidence.reshape(-1, 1)
                ], axis=1)
                np_results.append(pose_array)

        if np_results:
            np.save(npy_file, np_results)
            print(f"Results saved to {npy_file}")


def main():
    args = parse_args()

    # Create logger
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'custom_inference')

    device = torch.device(args.device)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # Validate input files
    if not os.path.exists(args.camera_file):
        raise FileNotFoundError(f"Camera file not found: {args.camera_file}")

    for img_file in args.image_files:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file not found: {img_file}")

    print(f'=> Loading camera calibration from {args.camera_file}')
    print(f'=> Processing {len(args.image_files)} images')

    # Setup data transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Create custom dataset
    test_dataset = CustomDataset(
        camera_file=args.camera_file,
        image_files=args.image_files,
        transform=transform
    )

    print(f'=> Loaded {test_dataset.num_views} camera views')

    # Setup CUDNN
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing model...')
    model = eval('models.dq_transformer.get_mvp')(config, is_train=False)
    model.to(device)

    # Load model weights
    if os.path.isfile(args.model_path):
        logger.info(f'=> Loading model state from {args.model_path}')
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    else:
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    print('=> Running inference...')
    results = custom_inference_3d(
        config=config,
        model=model,
        dataset=test_dataset,
        output_dir=args.output_dir,
        threshold=args.confidence_threshold,
        device=device
    )

    print('=> Saving results...')
    save_results(results, args.output_dir, args.output_format)

    print('=> Inference completed successfully!')
    print(f'Total detected people: {sum(len(r["poses_3d"]) for r in results)}')

    # Print summary
    for result in results:
        print(f"Frame {result['frame_id']}: {result['num_detected_people']} people detected")


if __name__ == '__main__':
    main()
