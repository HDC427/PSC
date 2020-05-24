import os
import torch
import cv2
import time
import numpy as np

import posenet
from Skeleton import draw_skeleton as draw
import util
from Person import Person
from confiance import conf
from confiance import cdf

from Track import Track

## Load the posenet model
model = posenet.load_model(101)
model = model.cuda()
output_stride = model.output_stride
scale_factor = 0.7125

## Function used to get all skeletons in one img, using posenet.
##
##  
def process_posenet(img, num_pose=2, min_pose_score=0.15, min_part_score=0.10):

    input_image, display_image, output_scale = posenet.utils._process_input(img, scale_factor=scale_factor, output_stride=output_stride)
    
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

    keypoint_coords = np.array(keypoint_coords)
    keypoint_scores = np.array(keypoint_scores)
    pose_scores = np.array(pose_scores)
    display_img = np.array(display_image)
    
    return display_img, keypoint_coords, keypoint_scores, pose_scores


## Params:av
SAVE_VIDEO = True
video_name = os.path.join('.', 'videos', 'Duo_1.wmv')
result_name = os.path.join('.', 'results', 'Duo_1_cut.avi')

# Video reader.
cap = cv2.VideoCapture(video_name)
colors = [(0,255,255), (255,255,0), (255,0,0), (255,0,255)]
# Video size.
video_width  = int(cap.get(3))  
video_height = int(cap.get(4)) 

print('width :', video_width, ', height :', video_height)

# If we store the video.
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(result_name,fourcc, 20.0, (video_width,video_height))

num_pose = 2
min_score = 0.15

###############
# The main loop
###############

tracker = Track(2)

frame_total = 0

while(True):
    res, img = cap.read()
    if not res:
        break

    # The process of posenet.
    overlay_img, keypoint_coords, keypoint_scores, pose_scores = process_posenet(img, num_pose=num_pose)

    text_height = 20 ## Used to put text

    ## discard the first frame
    if frame_total == 0:
        frame_total += 1
        continue
    ## initialization during the first frames
    if frame_total <= tracker.sample_size:
        tracker.initialize(keypoint_coords, keypoint_scores, img)
    ## The part of showing img.
    else:
        tracker.update(keypoint_coords, keypoint_scores, pose_scores, img)
        cv2.putText(overlay_img, 'frame_direct: '+str(tracker.frame_color), (0, text_height+10), cv2.FONT_HERSHEY_PLAIN, 1, colors[0], 1)
        #video_out.write(overlay_img)
    
    for person in tracker.person_list:
        #print(person.name, person.validity)
        person.draw(overlay_img)
    cv2.putText(overlay_img, 'frame_total: '+str(frame_total), (0, text_height), cv2.FONT_HERSHEY_PLAIN, 1, colors[0], 1)
    cv2.putText(overlay_img, 'percentage: '+str(tracker.frame_color/frame_total), (0, text_height+20), cv2.FONT_HERSHEY_PLAIN, 1, colors[1], 1)
    cv2.imshow('cover_test', overlay_img)
    video_out.write(overlay_img)

    frame_total += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

