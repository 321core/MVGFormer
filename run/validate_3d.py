# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Multi-view Pose transformer
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import pprint

import _init_paths

from core.config import config
from core.config import update_config, update_config_dynamic_input
from core.function import validate_3d
from core.nms import nearby_joints_nms
from utils.utils import create_logger
import lib.utils.misc as utils
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler
from models.util.misc import is_main_process, collect_results

import dataset
import models

import models.dq_transformer

import numpy as np
from datetime import datetime
from prettytable import PrettyTable

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--model_path', default=None, type=str,
                        help='pass model path for evaluation')
    parser.add_argument('--exp_name', '-n', default='exp', type=str)
    parser.add_argument('--frame_id', default=None, type=int,
                        help='which frame to process')

    # args, rest = parser.parse_known_args()
    # update_config(args.cfg)

    # if set then update the config.
    args, unknown = parser.parse_known_args()
    update_config(args.cfg)

    update_config_dynamic_input(unknown)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'validate')
    device = torch.device(args.device)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if is_main_process():
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

        if config.DEBUG.WANDB_KEY:
            wandb.login(key=config.DEBUG.WANDB_KEY)
        if config.DEBUG.WANDB_NAME:
            wandb.init(project="mvp-val",name=config.DEBUG.WANDB_NAME)
        else:
            # close wandb
            pass

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        rank, world_size = get_dist_info()
        sampler_val = DistributedSampler(test_dataset, world_size, rank,
                                         shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
        # print('Notice : random sampler is used.')
        # sampler_val = torch.utils.data.RandomSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)

    num_views = test_dataset.num_views

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + 'dq_transformer' + '.get_mvp')(
        config, is_train=True)
    model.to(device)
    # with torch.no_grad():
    #     model = torch.nn.DataParallel(model, device_ids=0).cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.model_path is not None:
        logger.info('=> load models state {}'.format(args.model_path))
        if args.distributed:
            model.module.load_state_dict(torch.load(args.model_path), strict=False)
        else:
            model.load_state_dict(torch.load(args.model_path), strict=False)
    elif os.path.isfile(
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)):
        test_model_file = \
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file),strict=False)
    else:
        test_model_file = \
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)
        raise ValueError(f'Check the model file for test file:{test_model_file}!')

    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_name

    os.makedirs(final_output_dir+'/validate-{}-{}/'.format(exp_name,now_str), exist_ok=True)

    conf_thr_tb = PrettyTable()
    mpjpe_threshold = np.arange(25, 155, 25)
    conf_thr_tb.field_names = \
        ["inference_conf_thr"] + \
        [f'AP{i}' for i in mpjpe_threshold] + \
        [f'Recall{i}' for i in mpjpe_threshold] + \
        ['Recall500','MPJPE']

    process_frame_id = args.frame_id
    max_gt_frames = len(test_dataset) // num_views
    is_out_of_range = process_frame_id is not None and process_frame_id >= max_gt_frames

    # Output frame information when frame_id is specified
    if process_frame_id is not None:
        print("=" * 60)
        if is_out_of_range:
            print("INFERENCE-ONLY MODE")
        else:
            print("FRAME-SPECIFIC VALIDATION")
        print("=" * 60)
        print(f"Processing specific frame: {process_frame_id}")
        print(f"Dataset: {config.DATASET.TEST_DATASET}")
        print(f"Test subset: {config.DATASET.TEST_SUBSET}")
        print(f"Number of views: {num_views}")
        print(f"Total dataset size: {len(test_dataset)} samples")
        print(f"Ground truth samples: {max_gt_frames}")

        if is_out_of_range:
            print(f"WARNING: Frame {process_frame_id} is beyond GT range (max: {max_gt_frames - 1})")
            print("Ground Truth not available - performing inference only")
        else:
            print(f"Frame {process_frame_id} is valid (0-{max_gt_frames - 1})")
        print("=" * 60)
        print()
    else:
        print(f"Processing all frames ({max_gt_frames} total)")
        print()

    for thr in config.DECODER.inference_conf_thr:
        pred_path = os.path.join(final_output_dir, f"{config.TEST.PRED_FILE}-{thr}.npy")
        if config.TEST.PRED_FILE and os.path.isfile(pred_path):
            preds = np.load(pred_path)
            logger.info(f"=> load pred_file from {pred_path}")
        else:
            preds_single, meta_image_files_single = validate_3d(
                config, model, test_loader, final_output_dir, thr,
                num_views=num_views, device=device, frame_id = process_frame_id)
            preds = collect_results(preds_single, len(test_dataset))
            np.save(final_output_dir+'/{}-{}-{}.npy'.format(exp_name,now_str,thr),preds)
            logger.info(f"=> save pred_file with TEST.PRED_FILE={exp_name}-{now_str}")

        if is_main_process():
            precision = None

            if is_out_of_range:
                # Inference-only mode: skip evaluation, show predictions only
                print("\n" + "=" * 60)
                print("INFERENCE RESULTS")
                print("=" * 60)

                # Apply NMS to predictions
                preds_nms = []
                total_predictions = 0

                for pred in preds:
                    pred = pred[pred[:, 0, 3] >= 0]
                    indices = nearby_joints_nms(pred, 0.3, 7)  # Default NMS parameters
                    pred_nms = pred[indices]
                    total_predictions += len(pred_nms)
                    preds_nms.append(pred_nms.copy())

                print(f"Confidence threshold: {thr}")
                print(f"Total predictions generated: {total_predictions}")

                if total_predictions > 0:
                    for i, pred in enumerate(preds_nms):
                        if len(pred) > 0:
                            print(f"Frame {process_frame_id}: {len(pred)} persons detected")
                            for person_idx, person_pred in enumerate(pred):
                                joints_3d = person_pred[:, :3]
                                confidence = np.mean(person_pred[:, 4])
                                print(f"  Person {person_idx + 1}: {joints_3d.shape[0]} joints, confidence: {confidence:.3f}")
                else:
                    print("No predictions generated (all below confidence threshold)")

                print("=" * 60)
                print("NOTE: Accuracy metrics (AP, MPJPE, Recall) cannot be computed")
                print("      because Ground Truth is not available for this frame.")
                print("=" * 60)

            elif 'panoptic' in config.DATASET.TEST_DATASET \
                    or 'h36m' in config.DATASET.TEST_DATASET:
                if config.DATASET.NMS_DETAIL:

                    nms_tb = PrettyTable()
                    nms_tb.field_names = \
                        ["dist_thr","num_nearby_joints_thr"] + \
                        [f'AP{i}' for i in mpjpe_threshold] + \
                        [f'Recall{i}' for i in mpjpe_threshold] + \
                        ['Recall500','MPJPE']

                    if config.DATASET.NMS_DETAIL_ALL:
                        dist_thrs = [0.01,0.03,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.8]
                        num_nearby_joints_thrs = [3,4,5,6,7,8,9,10,13]
                    else: # default
                        dist_thrs = [0.3]
                        num_nearby_joints_thrs = [7]

                    for dist_thr in dist_thrs:
                        for num_nearby_joints_thr in num_nearby_joints_thrs:
                            preds_nms = []
                            preds_nms_num = 0
                            for pred in preds:
                                pred = pred[pred[:, 0, 3] >= 0]
                                indices = nearby_joints_nms(pred,dist_thr,num_nearby_joints_thr)
                                pred_nms = pred[indices]
                                preds_nms_num = preds_nms_num + len(indices)
                                preds_nms.append(pred_nms.copy())
                            aps, recs, mpjpe, recall500 = test_loader.dataset.evaluate(preds_nms, frame_id=process_frame_id)
                            nms_tb.add_row(
                                [dist_thr , num_nearby_joints_thr] +
                                [f'{ap * 100:.2f}' for ap in aps] +
                                [f'{re * 100:.2f}' for re in recs] +
                                [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
                            )
                            # logger.info(nms_tb) #! DEBUG

                    # Display prediction results when frame_id is specified
                    if process_frame_id is not None:
                        print("\n" + "=" * 60)
                        print("PREDICTION RESULTS (NPY DATA)")
                        print("=" * 60)
                        print(f"Confidence threshold: {thr}")
                        print(f"NMS parameters: dist_thr=0.3, nearby_joints_thr=7")

                        # Use the last processed preds_nms (with default parameters)
                        total_predictions = 0
                        for pred in preds_nms:
                            total_predictions += len(pred)

                        print(f"Total predictions generated: {total_predictions}")

                        if total_predictions > 0:
                            # Display the actual NPY data structure
                            print(f"\nNPY Prediction Data for Frame {process_frame_id}:")
                            print("-" * 40)

                            for i, pred in enumerate(preds_nms):
                                if len(pred) > 0:
                                    print(f"\nFrame {process_frame_id}: {len(pred)} persons detected")
                                    print(f"Prediction array shape: {pred.shape}")
                                    print(f"Data format: [x, y, z, visibility_flag, confidence_score] per joint")
                                    print()

                                    for person_idx, person_pred in enumerate(pred):
                                        print(f"Person {person_idx + 1}:")
                                        print(f"  Shape: {person_pred.shape}")
                                        print(f"  NPY Data:")

                                        # Display all joint data in NPY format
                                        joint_names = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder',
                                                      'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip',
                                                      'LKnee', 'LAnkle', 'Head']

                                        for j_idx, joint_data in enumerate(person_pred):
                                            joint_name = joint_names[j_idx] if j_idx < len(joint_names) else f"Joint_{j_idx}"
                                            x, y, z, vis_flag, conf = joint_data[0], joint_data[1], joint_data[2], joint_data[3], joint_data[4]
                                            print(f"    {j_idx:2d} {joint_name:10s}: [{x:8.3f}, {y:8.3f}, {z:8.3f}, {vis_flag:6.1f}, {conf:6.3f}]")

                                        print()  # Empty line between persons
                        else:
                            print("No predictions generated (all below confidence threshold)")

                        print("=" * 60)
                        print("NOTE: This is the same data structure saved to the NPY file")
                        print("=" * 60)
                        print()

                    # logger.info(nms_tb)

                # nomral evo w/o NMS
                # aps, recs, mpjpe, recall500 = \
                #     test_loader.dataset.evaluate(preds)
                # conf_thr_tb.add_row(
                #     [thr] +
                #     [f'{ap * 100:.2f}' for ap in aps] +
                #     [f'{re * 100:.2f}' for re in recs] +
                #     [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
                # )

                # upper bound
                output_upper_bound = False
                if output_upper_bound:
                    aps_upper, recs_upper, mpjpe_upper, recall500_upper = \
                        test_loader.dataset.evaluate(preds, method='mpjpe_sort', frame_id=process_frame_id)
                    conf_thr_tb.add_row(
                        ['upper_bound (debug)'] +
                        [f'{ap * 100:.2f}' for ap in aps_upper] +
                        [f'{re * 100:.2f}' for re in recs_upper] +
                        [f'{recall500_upper * 100:.2f}',f'{mpjpe_upper:.2f}']
                    )

                # print table values
                logger.info(nms_tb.get_string(fields=nms_tb.field_names[2:]))

            elif 'campus' in config.DATASET.TEST_DATASET \
                    or 'shelf' in config.DATASET.TEST_DATASET:
                actor_pcp, avg_pcp, _, recall = \
                    test_loader.dataset.evaluate(preds, frame_id=process_frame_id)
                msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
                    ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  ' \
                    '|  {pcp_3:.2f}  |  {pcp_avg:.2f}  |' \
                    '\t Recall@500mm: {recall:.4f}'\
                    .format(pcp_1=actor_pcp[0] * 100,
                            pcp_2=actor_pcp[1] * 100,
                            pcp_3=actor_pcp[2] * 100,
                            pcp_avg=avg_pcp * 100,
                            recall=recall)
                logger.info(msg)
                precision = np.mean(avg_pcp)

if __name__ == '__main__':
    main()
