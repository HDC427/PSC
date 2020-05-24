import torch
import cv2
import time
import argparse
from psc_funcs import SkltDrawer
import numpy as np

import posenet


MAX_POSE_DETECTION = 1  ##Only 1 persons
MIN_POSE_SCORE = 0.05
MIN_PART_SCORE = 0.10

DEBUG = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--video', type=str, default=None)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--sample_size', type=int, default=8)
parser.add_argument('--out_file', type=str, default=None)
args = parser.parse_args()



def main():


    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    ## The source of input.
    ## If not given the video, use camera.
    if args.video != None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)


    ## The output file.
    ## If not given the output file name, 
    ## use '$video_name$_dbY_HMS.data'
    ## or  'camera_dbY_MHS.data'
    ## e.g.: 'demo.mp4_24Sep2019_193957.data'
    if args.out_file != None:
        out_file = open(out_file, 'a+')
    elif args.video != None:
        out_file_name = args.video + \
            time.strftime('%d%b%Y_%H%M%S', time.gmtime()) + \
            '.data'

        out_file = open(out_file_name, 'a+')
    else:
        out_file_name = 'camera' + \
            time.strftime('%d%b%Y_%H%M%S', time.gmtime()) + \
            '.data'

        out_file = open(out_file_name, 'a+')



    start = time.time()
    frame_count = 0

    ## init the sample
    sample = np.zeros((0,2))
    target_size = args.sample_size

    while True:
        ## Read the image.
        ## if empty, end the loop.
        res, img = cap.read()
        if not res:
            break

        input_image, display_image, output_scale = posenet.utils._process_input(img, \
            scale_factor=args.scale_factor, output_stride=output_stride)

        #input_image, display_image, output_scale = posenet.read_cap(
        #    cap, scale_factor=args.scale_factor, output_stride=output_stride)

        ## Detect the key points.
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=MAX_POSE_DETECTION,
                min_pose_score=MIN_POSE_SCORE)

        keypoint_coords *= output_scale

        ###############################
        ## part of image.
        ## draw the circle of nose.
        overlay_image = display_image
        overlay_image = cv2.circle(display_image, \
                SkltDrawer.to_tuple_coord(keypoint_coords[0][0]), \
                3, color=(0,255,0), thickness=3)
        frame_count += 1

        ## flip the image, and show it.
        overlay_image = cv2.flip(overlay_image, 1)
        cv2.imshow('posenet', overlay_image)
        

        ##############################
        ## part the datas.
        ## the new coord, convert to np.ndarray.
        new_coord = np.array(keypoint_coords[0][0])
        np.savetxt(out_file, new_coord[None])


        ##############################
        # For debug.
        if DEBUG:
            if frame_count%20 == 0: 
                end = time.time()
                print('Frame number:', frame_count)
                print('FPS:', 20/(end-start))
                start = end
    


        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            while(True):
                if cv2.waitKey(1)&0xFF == ord('c'):
                    break
        elif key == ord('q'):
            break  

    
    print('#Num of datas', frame_count+1 ,file=out_file)
    out_file.close()

if __name__ == "__main__":
    main()