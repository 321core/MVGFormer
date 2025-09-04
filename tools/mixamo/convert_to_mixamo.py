#!/usr/bin/env python3
"""
CMU Panoptic 3D Pose Data to Mixamo Compatible Format Converter

This script converts CMU Panoptic hdPose3d_stage1_coco19 JSON files to Mixamo compatible BVH format.
CMU Panoptic uses 3D position coordinates, while Mixamo uses hierarchical joint rotations.

Author: AI Assistant
Date: 2025-09-04
"""

import json
import numpy as np
import os
import glob
import argparse
from pathlib import Path
import math

class CMUPanopticToMixamo:
    def __init__(self):
        # CMU Panoptic 15 joints definition (from panoptic.py)
        self.cmu_joints = {
            'neck': 0,
            'nose': 1,
            'mid-hip': 2,
            'l-shoulder': 3,
            'l-elbow': 4,
            'l-wrist': 5,
            'l-hip': 6,
            'l-knee': 7,
            'l-ankle': 8,
            'r-shoulder': 9,
            'r-elbow': 10,
            'r-wrist': 11,
            'r-hip': 12,
            'r-knee': 13,
            'r-ankle': 14,
        }

        # Mixamo skeleton hierarchy (standard humanoid)
        self.mixamo_hierarchy = {
            'Hips': {'parent': None, 'cmu_joint': 'mid-hip'},
            'Spine': {'parent': 'Hips', 'cmu_joint': None},  # interpolated
            'Spine1': {'parent': 'Spine', 'cmu_joint': None},  # interpolated
            'Spine2': {'parent': 'Spine1', 'cmu_joint': None},  # interpolated
            'Neck': {'parent': 'Spine2', 'cmu_joint': 'neck'},
            'Head': {'parent': 'Neck', 'cmu_joint': 'nose'},

            # Left arm
            'LeftShoulder': {'parent': 'Spine2', 'cmu_joint': 'l-shoulder'},
            'LeftArm': {'parent': 'LeftShoulder', 'cmu_joint': 'l-elbow'},
            'LeftForeArm': {'parent': 'LeftArm', 'cmu_joint': 'l-wrist'},
            'LeftHand': {'parent': 'LeftForeArm', 'cmu_joint': None},  # end effector

            # Right arm
            'RightShoulder': {'parent': 'Spine2', 'cmu_joint': 'r-shoulder'},
            'RightArm': {'parent': 'RightShoulder', 'cmu_joint': 'r-elbow'},
            'RightForeArm': {'parent': 'RightArm', 'cmu_joint': 'r-wrist'},
            'RightHand': {'parent': 'RightForeArm', 'cmu_joint': None},  # end effector

            # Left leg
            'LeftUpLeg': {'parent': 'Hips', 'cmu_joint': 'l-hip'},
            'LeftLeg': {'parent': 'LeftUpLeg', 'cmu_joint': 'l-knee'},
            'LeftFoot': {'parent': 'LeftLeg', 'cmu_joint': 'l-ankle'},
            'LeftToeBase': {'parent': 'LeftFoot', 'cmu_joint': None},  # end effector

            # Right leg
            'RightUpLeg': {'parent': 'Hips', 'cmu_joint': 'r-hip'},
            'RightLeg': {'parent': 'RightUpLeg', 'cmu_joint': 'r-knee'},
            'RightFoot': {'parent': 'RightLeg', 'cmu_joint': 'r-ankle'},
            'RightToeBase': {'parent': 'RightFoot', 'cmu_joint': None},  # end effector
        }

        # CMU Panoptic coordinate transformation matrix (from panoptic.py)
        self.coord_transform = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [0.0, 1.0, 0.0]])

    def load_cmu_json(self, json_file):
        """Load CMU Panoptic JSON file and extract 3D pose data"""
        with open(json_file, 'r') as f:
            data = json.load(f)

        poses_3d = []
        frame_info = {
            'version': data.get('version', 0.7),
            'univTime': data.get('univTime', 0),
            'fpsType': data.get('fpsType', 'hd_29_97')
        }

        for body in data['bodies']:
            joints19 = np.array(body['joints19']).reshape(-1, 4)
            # Take only first 15 joints (as per panoptic.py)
            joints_3d = joints19[:15, :3]  # x, y, z coordinates
            confidence = joints19[:15, 3]   # confidence values

            # Apply coordinate transformation (as per panoptic.py)
            joints_3d = joints_3d.dot(self.coord_transform)

            # Convert cm to mm (as per panoptic.py)
            joints_3d *= 10.0

            pose_data = {
                'id': body['id'],
                'joints_3d': joints_3d,
                'confidence': confidence
            }
            poses_3d.append(pose_data)

        return poses_3d, frame_info

    def calculate_joint_rotations(self, joints_3d):
        """Calculate joint rotations from 3D positions"""
        rotations = {}

        # For simplicity, we'll calculate basic rotations based on bone directions
        # This is a simplified approach - more sophisticated IK could be implemented

        for joint_name, joint_info in self.mixamo_hierarchy.items():
            if joint_info['cmu_joint'] is None:
                # For interpolated joints, use identity rotation
                rotations[joint_name] = [0.0, 0.0, 0.0]  # Euler angles in degrees
            else:
                cmu_idx = self.cmu_joints[joint_info['cmu_joint']]
                joint_pos = joints_3d[cmu_idx]

                # Calculate rotation based on parent-child relationship
                if joint_info['parent'] and joint_info['parent'] in self.mixamo_hierarchy:
                    parent_info = self.mixamo_hierarchy[joint_info['parent']]
                    if parent_info['cmu_joint']:
                        parent_idx = self.cmu_joints[parent_info['cmu_joint']]
                        parent_pos = joints_3d[parent_idx]

                        # Calculate bone vector and convert to rotation
                        bone_vector = joint_pos - parent_pos
                        bone_vector = bone_vector / np.linalg.norm(bone_vector)

                        # Convert direction vector to Euler angles (simplified)
                        # This is a basic implementation - could be improved with proper IK
                        rx = math.degrees(math.atan2(bone_vector[1], bone_vector[2]))
                        ry = math.degrees(math.atan2(-bone_vector[0],
                                         math.sqrt(bone_vector[1]**2 + bone_vector[2]**2)))
                        rz = 0.0  # Simplified - could calculate twist

                        rotations[joint_name] = [rx, ry, rz]
                    else:
                        rotations[joint_name] = [0.0, 0.0, 0.0]
                else:
                    # Root joint - use position as translation
                    rotations[joint_name] = [0.0, 0.0, 0.0]

        return rotations

    def generate_bvh_header(self):
        """Generate BVH file header with skeleton hierarchy"""
        bvh_header = "HIERARCHY\n"

        def write_joint(joint_name, indent=0):
            joint_info = self.mixamo_hierarchy[joint_name]
            indent_str = "  " * indent

            if joint_info['parent'] is None:
                bvh_header = f"{indent_str}ROOT {joint_name}\n"
            else:
                bvh_header = f"{indent_str}JOINT {joint_name}\n"

            bvh_header += f"{indent_str}{{\n"
            bvh_header += f"{indent_str}  OFFSET 0.0 0.0 0.0\n"

            if joint_info['parent'] is None:
                bvh_header += f"{indent_str}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
            else:
                bvh_header += f"{indent_str}  CHANNELS 3 Zrotation Xrotation Yrotation\n"

            # Find children
            children = [name for name, info in self.mixamo_hierarchy.items()
                       if info['parent'] == joint_name]

            for child in children:
                bvh_header += write_joint(child, indent + 1)

            if not children:
                bvh_header += f"{indent_str}  End Site\n"
                bvh_header += f"{indent_str}  {{\n"
                bvh_header += f"{indent_str}    OFFSET 0.0 0.0 0.0\n"
                bvh_header += f"{indent_str}  }}\n"

            bvh_header += f"{indent_str}}}\n"
            return bvh_header

        # Start with root joint (Hips)
        bvh_header += write_joint('Hips')

        return bvh_header

    def convert_sequence_to_bvh(self, json_files, output_file, person_id=0):
        """Convert a sequence of JSON files to BVH format"""
        print(f"Converting {len(json_files)} frames to BVH format...")

        all_frames_data = []
        frame_time = 1.0 / 30.0  # 30 FPS default

        for json_file in sorted(json_files):
            poses_3d, frame_info = self.load_cmu_json(json_file)

            # Use the specified person_id (default 0)
            if person_id < len(poses_3d):
                pose_data = poses_3d[person_id]
                rotations = self.calculate_joint_rotations(pose_data['joints_3d'])

                # Extract root position (Hips)
                root_pos = pose_data['joints_3d'][self.cmu_joints['mid-hip']]

                frame_data = {
                    'root_pos': root_pos,
                    'rotations': rotations
                }
                all_frames_data.append(frame_data)

        # Generate BVH file
        bvh_content = self.generate_bvh_header()
        bvh_content += f"MOTION\n"
        bvh_content += f"Frames: {len(all_frames_data)}\n"
        bvh_content += f"Frame Time: {frame_time}\n"

        # Write frame data
        for frame_data in all_frames_data:
            root_pos = frame_data['root_pos']
            rotations = frame_data['rotations']

            # Start with root position and rotation
            frame_line = f"{root_pos[0]:.6f} {root_pos[1]:.6f} {root_pos[2]:.6f} "
            frame_line += f"{rotations['Hips'][2]:.6f} {rotations['Hips'][0]:.6f} {rotations['Hips'][1]:.6f} "

            # Add rotations for all other joints in hierarchy order
            joint_order = ['Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
                          'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                          'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                          'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                          'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

            for joint in joint_order:
                rot = rotations[joint]
                frame_line += f"{rot[2]:.6f} {rot[0]:.6f} {rot[1]:.6f} "

            bvh_content += frame_line.strip() + "\n"

        # Save BVH file
        with open(output_file, 'w') as f:
            f.write(bvh_content)

        print(f"BVH file saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert CMU Panoptic JSON to Mixamo BVH')
    parser.add_argument('input_dir', help='Directory containing hdPose3d_stage1_coco19 JSON files')
    parser.add_argument('output_dir', help='Output directory for BVH files')
    parser.add_argument('--person_id', type=int, default=0, help='Person ID to extract (default: 0)')
    parser.add_argument('--sequence_name', help='Sequence name (auto-detected if not provided)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    converter = CMUPanopticToMixamo()

    # Find all JSON files in input directory
    json_pattern = os.path.join(args.input_dir, 'body3DScene_*.json')
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Determine sequence name
    if args.sequence_name:
        seq_name = args.sequence_name
    else:
        # Auto-detect from directory path
        seq_name = os.path.basename(os.path.dirname(args.input_dir))

    # Output BVH file
    output_file = os.path.join(args.output_dir, f"{seq_name}_person{args.person_id}.bvh")

    # Convert to BVH
    converter.convert_sequence_to_bvh(json_files, output_file, args.person_id)

if __name__ == '__main__':
    main()
