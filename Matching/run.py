from phoneme_edit_distance import match_subtitles
import sys
'''
Pseudocode:

positions: coordiate of the person, face usually
subtitles: size(dynamic), color and content of the subtitles
assume subtitles are aligned with their positions

def allocate(subtitle_path, generated_path):

'''

import os, csv, json

def allocate(subtitle_path = 'subtitle.csv', generated_path = 'generated_subtitles.csv'):
    '''
    subtitle_path : string, path of csv file
    generated_path: string, path of csv file
    '''
    with open('matched_subtitles.csv', 'w') as fw:
        writer = csv.writer(fw, delimiter='/')
        with open(generated_path) as f_gen:
            gen_reader = csv.reader(f_gen, delimiter='/')
            with open(subtitle_path) as f_sub:
                sub_reader = csv.reader(f_sub, delimiter='/')
                # skip the first row
                speaker_dict = {}
                gen_data = list(gen_reader)
                sub_data = list(sub_reader)
                sentences, lines = [], []
                latest_sub_id = -1
                for i, (sub_line, gen_line) in enumerate(zip(sub_data, gen_data)):
                    sub_id, speaker_id, gen_text = map(lambda x:x.strip(), gen_line)
                    print(sub_id, speaker_id, gen_text)
                    speaker_dict[gen_text] = speaker_id
                    _, start_time, end_time, sub_text = map(lambda x:x.strip(), sub_line)
                    print(start_time, end_time, sub_text)

                    # if new sub_id is met, process current 
                    if int(sub_id) != latest_sub_id and len(lines) != 0:
                        result = match_subtitles(sentences)
                        for sub_id_, start_time_, end_time_, sub_text_ in lines:
                            row = [sub_id_, start_time_, end_time_, speaker_dict[result[sub_text_]], sub_text_]
                            writer.writerow(row)
                            sentences, lines = [], []

                    sentences.append((sub_text, gen_text))
                    lines.append((sub_id, start_time, end_time, sub_text))

                    latest_sub_id = int(sub_id)

                if len(lines) != 0:
                    result = match_subtitles(sentences)
                    for sub_id, start_time, end_time, sub_text in lines:
                        row = [sub_id, start_time, end_time, speaker_dict[result[sub_text]], sub_text]
                        writer.writerow(row)
                        sentences, lines = [], []

if __name__ == '__main__':
    if len(sys.argv) > 2:
        allocate(sys.argv[1], sys.argv[3])
    else:
        allocate()