import sys
import os
import json

from collections import defaultdict

import cv2

from yoloface.utils import *


'''
====== 1. YOLO face detection ======
'''

def yoloface_detection(original_video_path, frame_subtitle_id, yoloface_result_json_path, model_cfg, model_weights):
    
    # Give the configuration and weight files for the model and load the network
    # using them.
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Read input video
    if not os.path.isfile(original_video_path):
        print("[!] ==> Input video file {} doesn't exist".format(original_video_path))
        sys.exit(1)
    cap = cv2.VideoCapture(original_video_path)
    n_frames = int(cap.get(7))

    # Dictionary to store all bounding boxes from each frame
    # Bounding box is formatted as [left, top, width, height]
    subtitleId2frame2faces = defaultdict(dict)
    frame_id = 0

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        # Skip when the frame does not have any subtitle
        subtitleId = frame_subtitle_id[frame_id]

        if subtitleId == -1:
            frame_id += 1
            continue

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print(f'[i]frame({frame_id}/{n_frames}) ==> # detected faces: {len(faces)}')

        # Write the results        
        subtitleId2frame2faces[subtitleId][frame_id] = faces

        frame_id += 1

        # Break by user interupt
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    
    json.dump(subtitleId2frame2faces, open(yoloface_result_json_path, 'w'), indent=4)

    print('[i] ==> Output json  stored at ', yoloface_result_json_path)
    print('==> File is formatted as {subtitle_id: {frame_id: [face, ...]}, ...}')
    print('==> YOLO-face detection done!')
    print('***********************************************************')
    print()
