import cv2
import json
import os
import csv
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser()

# input arguments
parser.add_argument('--original_input_video', type=str, default='TED_sample.mp4',
                    help='path to original input video file')
parser.add_argument('--cropped_videos_dir', type=str, default='./cropped_videos/',
                    help='path to the directory that contains cropped videos')
parser.add_argument('--matched_subtitle_path', type=str, default='matched_TED_subtitle.csv',
                help='path to subtitle csv file formatted as "[subtitle_id, start_time, end_time, speaker_id, subtitle]"')

# output arguments
parser.add_argument('--output_video_path', type=str, default='final_output.mp4',)

# subtitle control arguments
parser.add_argument('--fix_subtitle', action='store_true')

args = parser.parse_args()


def time_string_to_seconds(time_string):
    '''
    time_string: string in the format of '00:00:00.000'
    ===
    Return: time in seconds
    '''
    time_string = time_string.split(':')
    seconds = int(time_string[0]) * 3600 + int(time_string[1]) * 60 + float(time_string[2].replace(',','.'))
    return seconds


def time_to_frame_ids(start_time, end_time, fps):
    strat_frame_id = int(start_time * fps)
    end_frame_id = int(end_time * fps)
    return list(range(strat_frame_id, end_frame_id))


def convert_csv_to_frame_data(csv_path, FPS):
    '''
    csv_path: path to csv file
    ===
    The file should be formatted as:
    [scene_id, start_time, end_time, speaker_id, subtitle]
    '''

    data = defaultdict(dict)
    # {frame_id: {speaker_id: subtitle1, speaker_id: subtitle2, ...}, ...}
    frame2subtitleId = {}

    with open(csv_path, 'r') as f:
        csvReader = csv.DictReader(f, delimiter=';')

        # convert each row into dictionary
        for rows in csvReader:

            start_time = time_string_to_seconds( rows['start_time'] )
            end_time = time_string_to_seconds( rows['end_time'] )

            frame_ids = time_to_frame_ids(start_time, end_time, FPS)

            for frame_id in frame_ids:
                data[frame_id][rows['speaker_id']] = rows['subtitle']
                frame2subtitleId[frame_id] = rows['subtitle_id']

    return data, frame2subtitleId

        
def main():
    
    # Check if output directory exists
    # if not os.path.exists(args.output_video_dir):
    #     print('==> Creating the {} directory...'.format(args.output_video_dir))
    #     os.makedirs(args.output_video_dir)
    #     print()

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
    subtitle_data, frame2subtitleId = convert_csv_to_frame_data(args.matched_subtitle_path, FPS)
    # subtitle data = {frame_id: {speaker_id: subtitle1, speaker_id: subtitle2, ...}, ...}
    # frame2subtitleId = {frame_id: subtitle_id, ...}

    # Add subtitle to the video frame by frame
    video_writer = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))

    frame_id = 0
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    init_pos = {}
    while len(init_pos) < 2:
        has_frame, frame = cap.read()

        if frame_id in subtitle_data:

            # Read the json file with faces results
            subtitle_id = frame2subtitleId[frame_id]
            face_result_path = os.path.join(args.cropped_videos_dir, f'{subtitle_id}/faces.json')
            with open(face_result_path, 'r') as f:
                faces_loc_results = json.load(f)

            faces_in_frame = faces_loc_results[str(frame_id)]

            face2loc = {str(face_id): (x_min + x_max, y_max) for face_id, x_min, y_min, x_max, y_max in faces_in_frame}

            for speaker_id, subtitle in subtitle_data[frame_id].items():
                    textsize = cv2.getTextSize(subtitle, font, 1, 2)[0]
                    init_pos[int(speaker_id)] = (int(face2loc[speaker_id][0]/2 - textsize[0]/2), int(face2loc[speaker_id][1]))
                    

        frame_id += 1

    frame_id = 0
    while True:
        has_frame, frame = cap.read()
        
        # Break the loop when the video is finished
        if not has_frame:
            print('==> Done!')
            # print('==> The final output video is saved in {}'.format(args.output_video_dir))
            cv2.waitKey(1000)
            break

        # Add subtitle to the frame
        if frame_id in subtitle_data:

            # Read the json file with faces results
            subtitle_id = frame2subtitleId[frame_id]
            face_result_path = os.path.join(args.cropped_videos_dir, f'{subtitle_id}/faces.json')
            with open(face_result_path, 'r') as f:
                faces_loc_results = json.load(f)

            faces_in_frame = faces_loc_results[str(frame_id)]
            # print("what?", faces_in_frame)
            # locate subtitle
            face2loc = {str(face_id): (x_min + x_max, y_max) for face_id, x_min, y_min, x_max, y_max in faces_in_frame}
            for speaker_id, subtitle in subtitle_data[frame_id].items():
                if args.fix_subtitle and len(face2loc) != 2:
                    print("처음임...ㅜ")
                    textsize = cv2.getTextSize(subtitle, font, 1, 2)[0]
                    init_pos[int(speaker_id)] = (int(face2loc[speaker_id][0]/2 - textsize[0]/2), int(face2loc[speaker_id][1]))
                    cv2.putText(frame, subtitle, init_pos[int(speaker_id)], font, 1, (255,255,255), 2)
                    
                else:
                    if args.fix_subtitle:
                        print("fix_subtitle! working!")
                        textsize = cv2.getTextSize(subtitle, font, 1, 2)[0]
                        cv2.putText(frame, subtitle, init_pos[int(speaker_id)], font, 1, (255,255,255), 2)
                    else:
                        print("fix_subtitle! 아님!")
                        textsize = cv2.getTextSize(subtitle, font, 1, 2)[0]
                        cv2.putText(frame, subtitle, (int(face2loc[speaker_id][0]/2 - textsize[0]/2), int(face2loc[speaker_id][1])), font, 1, (255, 255, 255), 2)

        # Write the frame to the output video
        video_writer.write(frame.astype('uint8'))

        frame_id += 1

    cap.release()
    video_writer.release()
                
if __name__ == '__main__':
    main()