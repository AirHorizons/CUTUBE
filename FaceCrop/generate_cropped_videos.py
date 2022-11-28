import sys
import os
import json

from collections import defaultdict

import cv2


def ltwh2ltrb(box):
    # Return left, top, right, bottom
    # In other words, return x_min, y_min, x_max, y_max
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

def closest_box(face, prev_faces, threshold=100):
    # face = [x_min, y_min, x_max, y_max]
    # prev_faces = {face_id: [x_min, y_min, x_max, y_max], ...}

    close_faces_dist = {}

    for face_id, prev_face in prev_faces.items():
        dist = abs(face[0] - prev_face[0]) + abs(face[1] - prev_face[1]) + abs(face[2] - prev_face[2]) + abs(face[3] - prev_face[3])
        if dist < threshold:
            close_faces_dist[face_id] = dist

    if not close_faces_dist:
        return None
    else:
        return min(close_faces_dist, key=close_faces_dist.get)

def identify_faces(faces_results):

    result_output = defaultdict(list)
    # {frame_id: [face1, face2, ...], ...}
    # face1 = [face_id, x_min, y_min, x_max, y_max]

    bbox_max_sizes = defaultdict(lambda: (0,0))
    # {face_id1: (max_width, max_height), face_id2: (max_width, max_height), ...}
    

    prev_frame_boxes = {} # {face_id1: [x1, y1, x2, y2], face_id2: [x1, y1, x2, y2]....}
    n_unique_faces = 0
    
    for frame_id, faces in faces_results.items():

        if not prev_frame_boxes:
            prev_frame_boxes = {i: ltwh2ltrb(face) for i, face in enumerate(faces)}
            n_unique_faces += len(faces)
            result_output[frame_id] = [[i, *ltwh2ltrb(face)] for i, face in enumerate(faces)]
            continue

        cur_frame_boxes = {}

        for face in faces:
            left, top, right, bottom = ltwh2ltrb(face)
            face_id = closest_box([left, top, right, bottom], prev_frame_boxes)

            if face_id is None:
                cur_frame_boxes[n_unique_faces] = [left, top, right, bottom]
                result_output[frame_id].append([n_unique_faces, left, top, right, bottom])
                bbox_max_sizes[n_unique_faces] = (face[2], face[3]) # bbox width, height
                n_unique_faces += 1
            else:
                cur_frame_boxes[face_id] = [left, top, right, bottom]
                result_output[frame_id].append([face_id, left, top, right, bottom])
                bbox_max_sizes[face_id] = (max(bbox_max_sizes[face_id][0], face[2]), max(bbox_max_sizes[face_id][1], face[3]))

        prev_frame_boxes = cur_frame_boxes

    return result_output, n_unique_faces, bbox_max_sizes


def expand_bbox(face, bbox_size):
    # face = [x_min, y_min, x_max, y_max]
    # bbox_size = (width, height)

    x_min, y_min, x_max, y_max = face
    width, height = bbox_size
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    new_x_min = cx - ( width // 2 )
    new_y_min = cy - ( height // 2 )
    new_x_max = cx + ( width - width // 2 )
    new_y_max = cy + ( height - height // 2 )

    return [new_x_min, new_y_min, new_x_max, new_y_max]


'''
====== 2. Generating face-cropped videos ======
'''

def generate_cropped_videos(original_video_path, faces_results, cropped_bbox_info_json_filename, cropped_videos_output_dir):

    # Preprocess the faces results
    faces_results, n_unique_faces, bbox_max_sizes = identify_faces(faces_results)
    
    json.dump(faces_results, open(os.path.join(cropped_videos_output_dir, cropped_bbox_info_json_filename), 'w'), indent=4)

    print('==> {} unique faces are found in the video'.format(n_unique_faces))
    print()

    # Read the video
    cap = cv2.VideoCapture(original_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(7))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    print('==> Total frames: {}'.format(total_frames))
    print('==> FPS: {}'.format(FPS))
    print('==> Frame size: {}x{}'.format(frame_width, frame_height))
    print()

    # Crop the video

    # Video writer and video path for each face
    video_writer = {}
    for face_id, (max_width, max_height) in bbox_max_sizes.items():
        video_name = os.path.join(cropped_videos_output_dir, f'{face_id}.mp4')
        video_writer[face_id] = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (max_width, max_height))

    frame_id = 0
    while True:
        has_frame, frame = cap.read()

        # Break the loop when the video is finished
        if not has_frame:
            print('==> Done!')
            print('==> The cropped videos are saved in {}'.format(cropped_videos_output_dir))
            cv2.waitKey(1000)
            break

        # Skip if the frame id is not in faces_results
        if str(frame_id) not in faces_results:
            frame_id += 1
            continue

        # Crop the faces
        for face in faces_results[str(frame_id)]:
            face_id, left, top, right, bottom = face
            left, top, right, bottom = expand_bbox([left, top, right, bottom], bbox_max_sizes[face_id])

            face_img = frame[top:bottom, left:right]

            video_writer[face_id].write(face_img)
        
        frame_id += 1

    for face_id in video_writer:
        video_writer[face_id].release()
    
    cap.release()