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
        writer = csv.writer(fw, delimiter=';')
        writer.writerow(['subtitle_id', 'start_time', 'end_time', 'speaker_id', 'subtitle'])
        with open(generated_path) as f_gen:
            gen_reader = csv.reader(f_gen, delimiter=';')
            with open(subtitle_path) as f_sub:
                sub_reader = csv.reader(f_sub, delimiter=';')
                # skip the first row
                gen_data = list(gen_reader)
                gen_data_dict = {}
                for sub_id, speaker_id, gen_text in gen_data:
                    if sub_id not in gen_data_dict:
                        gen_data_dict[sub_id] = [(speaker_id, gen_text)]
                    else:
                        gen_data_dict[sub_id].append((speaker_id, gen_text))

                sub_data = list(sub_reader)[1:]
                sub_data_dict = {}
                for sub_id, start_time, end_time, sub_text in sub_data:
                    if sub_id not in sub_data_dict:
                        sub_data_dict[sub_id] = [(start_time, end_time, sub_text)]
                    else:
                        sub_data_dict[sub_id].append((start_time, end_time, sub_text))

                for sub_id in sub_data_dict:
                    subs = sub_data_dict[sub_id]
                    gens = gen_data_dict[sub_id]
                    if len(subs) == 1 and len(gens) == 1:
                        start_time, end_time, sub_text = subs[0]
                        speaker_id, gen_text = gens[0]
                        row = [sub_id, start_time, end_time, speaker_id, sub_text]
                        writer.writerow(row)
                    elif len(subs) == len(gens):
                        speaker_dict = {}
                        sentences = []
                        for i, (speaker_id, gen_text) in enumerate(gens):
                            speaker_dict[gen_text] = speaker_id
                            sentences.append((subs[i][2], gen_text))
                        result = match_subtitles(sentences)
                        for i, (start_time, end_time, sub_text) in enumerate(subs):
                            row = [sub_id, start_time, end_time, speaker_dict[result[sub_text]], sub_text]
                            writer.writerow(row)
                    elif len(subs) > len(gens):
                        speaker_dict = {'': -1}
                        sentences = []
                        for i, (speaker_id, gen_text) in enumerate(gens):
                            speaker_dict[gen_text] = speaker_id
                            sentences.append((subs[i][2], gen_text))
                        for i in range(len(subs) - len(gens)):
                            sentences.append(subs[i + len(gens)][2], '')
                        result = match_subtitles(sentences)
                        for i, (start_time, end_time, sub_text) in enumerate(subs):
                            row = [sub_id, start_time, end_time, speaker_dict[result[sub_text]], sub_text]
                            writer.writerow(row)
                    else: # len(subs) < len(gens)
                        speaker_dict = {'': -1}
                        sentences = []
                        for i, (speaker_id, gen_text) in enumerate(gens):
                            speaker_dict[gen_text] = speaker_id
                            sub_text = subs[i][2] if i < len(subs) else ''
                            sentences.append((sub_text, gen_text))
                        result = match_subtitles(sentences)
                        for i, (start_time, end_time, sub_text) in enumerate(subs):
                            row = [sub_id, start_time, end_time, speaker_dict[result[sub_text]], sub_text]
                            writer.writerow(row)

                    # print(sub_id)
                    # print(subs)
                    # print(gens)

                '''sentences, lines = [], []
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
                        sentences, lines = [], []'''

if __name__ == '__main__':
    if len(sys.argv) > 2:
        allocate(sys.argv[1], sys.argv[2])
    else:
        allocate()