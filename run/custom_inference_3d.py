#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom 3D Pose Inference Script for MVGFormer

This script performs multi-view 3D pose estimation using custom camera calibration files
and individual image files, instead of structured dataset formats.

Usage:
python run/custom_inference_3d.py \
    --cfg configs/panoptic/knn5-lr4-q1024.yaml \
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
import numpy as np
import json
import cv2
from PIL import Image
from datetime import datetime

import _init_paths

from core.config import config, update_config, update_config_dynamic_input
from utils.utils import create_logger
import lib.utils.misc as utils
from models.util.misc import is_main_process
import models
import models.dq_transformer
from utils.transforms import get_affine_transform, get_scale, affine_transform


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for arbitrary images and camera calibrations"""

    def __init__(self, image_files, camera_file, transform=None, num_person=5):
        self.image_files = image_files
        self.transform = transform
        self.cameras = self._load_cameras(camera_file)
        self.num_views = len(image_files)
        self.num_person = num_person
        # Use same image size as original dataset (matching JointsDataset)
        self.image_size = np.array([256, 256])  # Model's expected input size
        self.heatmap_size = np.array([64, 64])  # For aug_trans calculation

    def _load_cameras(self, camera_file):
        """Load camera calibration from JSON file"""
        with open(camera_file, 'r') as f:
            calib_data = json.load(f)

        cameras = {}
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])

        for i, cam in enumerate(calib_data['cameras']):
            if i < len(self.image_files):
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[i] = sel_cam

        return cameras

    def __len__(self):
        return 1  # Single sample with multiple views

    def __getitem__(self, idx):
        inputs = []
        metas = []

        for i, image_file in enumerate(self.image_files):
            # Load image using cv2 (same as JointsDataset)
            data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if data_numpy is None:
                raise ValueError(f"Could not load image: {image_file}")

            # Convert BGR to RGB (same as JointsDataset when color_rgb=True)
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

            # Get image dimensions and calculate center/scale (same as JointsDataset)
            height, width, _ = data_numpy.shape
            c = np.array([width / 2.0, height / 2.0])
            s = get_scale((width, height), self.image_size)
            r = 0  # No rotation augmentation for inference

            # Apply affine transformation (same as JointsDataset)
            trans = get_affine_transform(c, s, r, self.image_size, inv=0)
            input_image = cv2.warpAffine(
                data_numpy,
                trans, (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

            # Apply transforms (same as JointsDataset)
            if self.transform:
                image_tensor = self.transform(input_image)
            else:
                image_tensor = transforms.ToTensor()(input_image)

            # Get camera parameters
            cam_data = self.cameras[i] if i in self.cameras else {
                'K': np.eye(3), 'R': np.eye(3), 't': np.zeros((3, 1)), 'distCoef': np.zeros(5)
            }

            # Extract camera intrinsics
            K = cam_data['K']
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Extract distortion coefficients
            # distCoef format: [k1, k2, p1, p2, k3] (radial: k1,k2,k3; tangential: p1,p2)
            distCoef = cam_data['distCoef']
            k1, k2, p1, p2, k3 = distCoef[0], distCoef[1], distCoef[2], distCoef[3], distCoef[4]

            # Create camera parameters with batch dimensions for dq_decoder stacking
            cam = {
                'fx': torch.tensor([fx], dtype=torch.float32),  # [1]
                'fy': torch.tensor([fy], dtype=torch.float32),  # [1]
                'cx': torch.tensor([cx], dtype=torch.float32),  # [1]
                'cy': torch.tensor([cy], dtype=torch.float32),  # [1]
                'R': torch.from_numpy(cam_data['R']).float().unsqueeze(0),  # [1, 3, 3]
                'T': torch.from_numpy(cam_data['t']).float().unsqueeze(0),  # [1, 3, 1]
                'standard_T': torch.from_numpy(cam_data['t'].flatten()).float().unsqueeze(0),  # [1, 3]
                'k': torch.tensor([k1, k2, k3], dtype=torch.float32).unsqueeze(0).unsqueeze(-1),  # [1, 3, 1]
                'p': torch.tensor([p1, p2], dtype=torch.float32).unsqueeze(0).unsqueeze(-1),  # [1, 2, 1]
            }

            # Create proper transform matrices (same as JointsDataset)
            aff_trans = np.eye(3, 3)
            aff_trans[0:2] = trans  # full img -> cropped img
            inv_aff_trans = np.eye(3, 3)
            inv_trans = get_affine_transform(c, s, r, self.image_size, inv=1)
            inv_aff_trans[0:2] = inv_trans

            # 3x3 data augmentation affine trans (same as JointsDataset)
            aug_trans = np.eye(3, 3)
            aug_trans[0:2] = trans  # full img -> cropped img
            hm_scale = self.heatmap_size / self.image_size
            scale_trans = np.eye(3, 3)  # cropped img -> heatmap
            scale_trans[0, 0] = hm_scale[1]
            scale_trans[1, 1] = hm_scale[0]
            aug_trans = scale_trans @ aug_trans

            # Create dummy joint data with correct shapes (maximum_person=10, num_joints=15)
            maximum_person = 10
            num_joints = 15

            # Create metadata matching JointsDataset format
            meta = {
                'image': image_file,
                'num_person': torch.tensor([self.num_person]),  # Tensor with batch dimension
                'joints_3d': torch.randn((1, maximum_person, num_joints, 3)) * 500,  # Random values instead of zeros
                'joints_3d_vis': torch.ones((1, maximum_person, num_joints, 3)),
                'joints_3d_voxelpose_pred': torch.zeros((1, maximum_person, num_joints, 5)),
                'roots_3d': torch.zeros((1, maximum_person, 3)),
                'joints': torch.zeros((1, maximum_person, num_joints, 2)),
                'joints_vis': torch.zeros((1, maximum_person, num_joints, 2)),
                'center': torch.from_numpy(c).unsqueeze(0),  # Add batch dimension
                'scale': torch.from_numpy(s).unsqueeze(0),
                'rotation': torch.tensor([0.0]),  # Add batch dimension
                'camera': cam,
                'camera_Intri': torch.from_numpy(K).unsqueeze(0),
                'camera_R': torch.from_numpy(cam_data['R']).unsqueeze(0),
                'camera_focal': torch.from_numpy(np.array([fx, fy, 1.0])).unsqueeze(0),
                'camera_T': torch.from_numpy(cam_data['t'].flatten()).unsqueeze(0),
                'camera_standard_T': torch.from_numpy(cam_data['t'].flatten()).unsqueeze(0),
                'affine_trans': torch.from_numpy(aff_trans).unsqueeze(0),
                'inv_affine_trans': torch.from_numpy(inv_aff_trans).unsqueeze(0),
                'aug_trans': torch.from_numpy(aug_trans).unsqueeze(0),
            }

            inputs.append(image_tensor)
            metas.append(meta)

        return inputs, metas


def parse_args():
    parser = argparse.ArgumentParser(description='Custom 3D Pose Inference')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    parser.add_argument('--model_path', help='path to model weights',
                        required=True, type=str)
    parser.add_argument('--camera_file', help='path to camera calibration JSON',
                        required=True, type=str)
    parser.add_argument('--image_files', nargs='+', help='paths to input images',
                        required=True)
    parser.add_argument('--output_dir', help='output directory',
                        default='./custom_inference_output', type=str)
    parser.add_argument('--confidence_threshold', help='confidence threshold',
                        default=0.5, type=float)
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    parser.add_argument('--output_format', default='both',
                        help='output format: json, npy, or both')
    parser.add_argument('--num_person', default=5, type=int,
                        help='number of expected persons in the scene')

    args, unknown = parser.parse_known_args()
    update_config(args.cfg)
    update_config_dynamic_input(unknown)
    return args


def custom_inference_3d(config, model, inputs, metas, threshold, device="cpu"):
    """Perform 3D pose inference on custom input"""
    model.eval()

    preds = []
    with torch.no_grad():
        # Add batch dimension and move inputs to device
        inputs = [i.unsqueeze(0).to(device) for i in inputs]  # Add batch dimension

        # Move metadata to device - the dq_decoder will handle camera parameter combining
        metas_device = []
        for meta in metas:
            meta_device = {}
            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    meta_device[k] = v.to(device)
                elif k == 'camera':
                    # Move camera parameters to device
                    cam_device = {}
                    for cam_k, cam_v in v.items():
                        cam_device[cam_k] = cam_v.to(device) if isinstance(cam_v, torch.Tensor) else cam_v
                    meta_device[k] = cam_device
                else:
                    meta_device[k] = v
            metas_device.append(meta_device)

        # Run inference - pass all metadata, model expects list of meta per view
        output = model(views=inputs, meta=metas_device, threshold=threshold)

        # Extract predictions
        gt_3d = metas_device[0]['joints_3d'].float()  # Access first metadata
        num_joints = gt_3d.shape[2]
        bs, num_queries = output["pred_logits"].shape[:2]

        src_poses = output['pred_poses']['outputs_coord'].view(bs, num_queries, num_joints, 3)
        score = output['pred_logits'][:, :, 1:2].sigmoid()
        score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
        temp = (score > threshold).float() - 1

        pred = torch.cat([src_poses, temp, score], dim=-1)
        pred = pred.detach().cpu().numpy()

        for b in range(pred.shape[0]):
            preds.append(pred[b])

    return preds


def save_results(preds, output_dir, output_format, image_files, confidence_threshold):
    """Save inference results in specified format"""
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Filter predictions by confidence
    valid_preds = []
    for pred in preds:
        # Keep only predictions above confidence threshold
        valid_pred = pred[pred[:, 0, 3] >= 0]  # confidence flag check
        if len(valid_pred) > 0:
            valid_preds.extend(valid_pred)

    results = {
        'timestamp': timestamp,
        'image_files': image_files,
        'confidence_threshold': confidence_threshold,
        'num_detected_poses': len(valid_preds),
        'poses_3d': []
    }

    for i, pose in enumerate(valid_preds):
        pose_data = {
            'pose_id': i,
            'joints_3d': pose[:, :3].tolist(),  # x, y, z coordinates
            'confidence_scores': pose[:, 4].tolist()  # confidence scores
        }
        results['poses_3d'].append(pose_data)

    # Save in requested format(s)
    if output_format in ['json', 'both']:
        json_file = os.path.join(output_dir, f'poses_3d_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_file}")

    if output_format in ['npy', 'both']:
        npy_file = os.path.join(output_dir, f'poses_3d_{timestamp}.npy')
        np.save(npy_file, np.array(valid_preds))
        print(f"Raw predictions saved to: {npy_file}")

    return results


def main():
    args = parse_args()

    # Setup logging
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'custom_inference')
    device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"Number of input images: {len(args.image_files)}")
    logger.info(f"Camera calibration file: {args.camera_file}")
    logger.info(f"Confidence threshold: {args.confidence_threshold}")

    # Check input files
    if not os.path.exists(args.camera_file):
        raise FileNotFoundError(f"Camera file not found: {args.camera_file}")

    for img_file in args.image_files:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file not found: {img_file}")

    # Setup transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Create custom dataset
    dataset = CustomDataset(args.image_files, args.camera_file, transform, args.num_person)
    num_views = dataset.num_views

    # Setup CUDA
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # Load model
    logger.info('=> Constructing model ...')
    model = eval('models.dq_transformer.get_mvp')(config, is_train=False)
    model.to(device)

    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    logger.info(f'=> Loading model weights from {args.model_path}')
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    # Get data
    inputs, metas = dataset[0]

    logger.info('=> Starting inference...')
    # Run inference
    preds = custom_inference_3d(
        config, model, inputs, metas,
        args.confidence_threshold, device)

    # Save results
    results = save_results(preds, args.output_dir, args.output_format,
                          args.image_files, args.confidence_threshold)

    logger.info(f'=> Inference completed!')
    logger.info(f'=> Detected {results["num_detected_poses"]} poses')
    logger.info(f'=> Results saved to {args.output_dir}')

    return results


if __name__ == '__main__':
    main()
