import argparse
import sys
import os
import json

import cv2

from utils import *

from yoloface_detection import yoloface_detection
from generate_cropped_videos import generate_cropped_videos

#####################################################################
parser = argparse.ArgumentParser()

# input argument
parser.add_argument('--original_input_video', type=str, default='TED_sample.mp4',
                    help='path to original input video file')
parser.add_argument('--subtitle_csv', type=str, default='TED_subtitle.csv',
                    help='path to subtitle csv file with columns [subtitle_id, start_time, end_time, speaker_id, subtitle]')

# output argument
parser.add_argument('--temp_outputs_dir', type=str, default='./temp/',
                    help='store intermediate outputs of the pipeline')
parser.add_argument('--cropped_videos_output_dir', type=str, default='./cropped_videos/',
                    help='path to the dirtectory where the cropped videos will be stored')
parser.add_argument('--cropped_bbox_info_json', type=str, default='faces.json',
                    help='path to the file containing outputing bounding boxes info, will be stored under the output_dir')

################

# yoloface detection arguments
parser.add_argument('--model-cfg', type=str, default='./yoloface/cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./yoloface/model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')


args = parser.parse_args()
#####################################################################





def pipeline():

    # check outputs directory
    if not os.path.exists(args.cropped_videos_output_dir):
        print('==> Creating the {} directory...'.format(args.cropped_videos_output_dir))
        os.makedirs(args.cropped_videos_output_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.cropped_videos_output_dir))

    # check temp directory
    if not os.path.exists(args.temp_outputs_dir):
        print('==> Creating the {} directory...'.format(args.temp_outputs_dir))
        os.makedirs(args.temp_outputs_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.temp_outputs_dir))

    # Read input video
    if not os.path.isfile(args.original_input_video):
        print("[!] ==> Input video file {} doesn't exist".format(args.original_input_video))
        sys.exit(1)
    cap = cv2.VideoCapture(args.original_input_video)

    # Read the video
    cap = cv2.VideoCapture(args.original_input_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(7))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    print('==> Total frames: {}'.format(total_frames))
    print('==> FPS: {}'.format(FPS))
    print('==> Frame size: {}x{}'.format(frame_width, frame_height))
    print()

    # Read the subtitle csv file
    # And cut the original video as the subtitle id
    frameId2subtitleId = get_frameId2subtitleId(args.subtitle_csv, FPS)

    # Blindly detect faces from original video
    # By using yolo-face
    yoloface_result_json = os.path.join(args.temp_outputs_dir, 'yoloface_result.json')
    yoloface_detection(original_video_path = args.original_input_video,
                    frame_subtitle_id = frameId2subtitleId,
                    yoloface_result_json_path = yoloface_result_json,
                    model_cfg = args.model_cfg,
                    model_weights = args.model_weights)

    # Read the json file with faces results
    with open(yoloface_result_json, 'r') as f:
        whole_faces_results = json.load(f)

    # whole_faces_results = {subtitle_id(int): {face_id: [face, ...], ...}, ...}

    # crop the video and store those under the directory as
    # {$args.cropped_videos_output_dir}/{$subtitle_id}/{$speaker_id}.mp4
    for subtitle_id, faces_results in whole_faces_results.items():
        
        cropped_videos_output_dir = os.path.join(args.cropped_videos_output_dir, f'{subtitle_id}/')
        if not os.path.exists(cropped_videos_output_dir):
            os.makedirs(cropped_videos_output_dir)

        generate_cropped_videos(original_video_path = args.original_input_video,
                            faces_results = faces_results,
                            cropped_bbox_info_json_filename = args.cropped_bbox_info_json,
                            cropped_videos_output_dir = cropped_videos_output_dir)
    

if __name__ == '__main__':
    pipeline()
    