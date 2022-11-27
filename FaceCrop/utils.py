import csv
from collections import defaultdict

########################################################
# Utils for frame splitting according to the subtitle id
########################################################

def time_string_to_seconds(time_string):
    '''
    time_string: string in the format of '00:00:00.000'
    ===
    Return: time in seconds
    '''
    time_string = time_string.split(':')
    seconds = int(time_string[0]) * 3600 + int(time_string[1]) * 60 + float(time_string[2])
    return seconds


def time_to_frame_ids(start_time, end_time, fps):
    strat_frame_id = int(start_time * fps)
    end_frame_id = int(end_time * fps)
    return list(range(strat_frame_id, end_frame_id))


def get_frameId2subtitleId(csv_path, FPS):
    '''
    csv_path: path to csv file
    ===
    The file should be formatted as:
    [subtitle_id, start_time, end_time, speaker_id, subtitle]
    '''

    frameId2subtitleId = defaultdict(lambda: -1)
    # {frame_id: subtitle_id}
    
    with open(csv_path, 'r') as f:
        csvReader = csv.DictReader(f)

        # convert each row into dictionary
        for rows in csvReader:

            start_time = time_string_to_seconds( rows['start_time'] )
            end_time = time_string_to_seconds( rows['end_time'] )

            frame_ids = time_to_frame_ids(start_time, end_time, FPS)

            for frame_id in frame_ids:
                frameId2subtitleId[frame_id] = int(rows['subtitle_id'])

    return frameId2subtitleId
