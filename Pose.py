#############################################
## Using posenet to get the poses, key points
## as well as their scores.
#############################################
import torch
import numpy as np
import posenet

class Pose:

    model = posenet.load_model(101).cuda()
    output_stride = model.output_stride
    scale_factor = 0.7125

    num_pose = 4
    min_pose_score = 0.15
    min_part_score = 0.10

    @staticmethod
    def get_pose(img, num_pose=4, min_pose_score=0.15, min_part_score=0.10):
        input_image, display_image, output_scale = \
            posenet.utils._process_input(img, scale_factor=Pose.scale_factor, output_stride=Pose.output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=num_pose,
                min_pose_score=min_pose_score)
        
        keypoint_coords *= output_scale

        return display_image, keypoint_coords, keypoint_scores, pose_scores

    
    def __init__(self, num_pose=4, min_part_score=0.15, min_pose_score=0.10, model_id=101, scale_factor=0.7125):
        self.model = posenet.load_model(model_id).cuda()
        self.output_stride = self.model.output_stride
        self.scale_factor = scale_factor
        
        self.num_pose = num_pose
        self.min_part_score = min_part_score
        self.min_pose_score = min_pose_score

    
    def get_pose(self):
        input_image, display_image, output_scale = \
            posenet.utils._process_input(img, scale_factor=self.scale_factor, output_stride=self.output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=self.output_stride,
                max_pose_detections=self.num_pose,
                min_pose_score=self.min_pose_score)
        
        keypoint_coords *= output_scale

        return display_image, keypoint_coords, keypoint_scores, pose_scores

