#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import glob

sys.path.append("Visual_Speech_Recognition_for_Multiple_Languages")


import csv
import os
import torch
import argparse
from metrics.measures import get_wer
from metrics.measures import get_cer
from lipreading.utils import save2npz
from lipreading.utils import save2avi
from lipreading.utils import AverageMeter
from lipreading.subroutines import LipreadingPipeline


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Project')
    parser.add_argument(
        "--config-filename",
        type=str,
        default=None,
        help="Model configuration with ini format",
    )
    # -- 
    parser.add_argument("--data-filename",
        default=None,
        type=str,
        help="The filename for sequence.",
    )
    parser.add_argument(
        "--landmarks-filename",
        default="",
        type=str,
        help="The filename for tracked landmarks.",
    )
    parser.add_argument(
        "--dst-filename",
        type=str,
        default=None,
        help="The filename of the saved mouth patches or embedding.",
    )
    # -- for benchmark evaluation
    parser.add_argument("--data-dir",
        default=None,
        type=str,
        help="The directory for sequence.",
    )
    parser.add_argument(
        "--landmarks-dir",
        default="",
        type=str,
        help="The directory for tracked landmarks.",
    )
    parser.add_argument("--labels-filename",
        default=None,
        type=str,
        help="The filename for labels.",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        default=None,
        help="The directory of saved mouth patches or embeddings.",
    )
    # -- feature extraction
    parser.add_argument(
        "--feats-position",
        default="",
        choices=["", "mouth", "resnet", "conformer"],
        help="Specify the position for feature extraction.",
    )
    # -- utils
    parser.add_argument(
        "--video-ext",
        default=".mp4",
        type=str,
        help="The extension for video files.",
    )
    parser.add_argument(
        "--landmarks-ext",
        default=".pkl",
        type=str,
        help="The extension for landmarks file.",
    )
    parser.add_argument(
        "--gpu-idx",
        default=-1,
        type=int,
        help="Inference in GPU when gpu_idx >= 0 or in CPU when gpu_idx < 0.",
    )

    args = parser.parse_args()
    return args

args = load_args()

def one_step_inference(lipreader, data_filename, landmarks_filename, dst_filename=""):
    """one_step_inference.

    :param lipreader: LipreadingPipeline object, contains the function for \
        facial tracking[option], facial pre-processing and lipreading inference.
    :param data_filename: str, the filename for tracked landmarks.
    :param landmarks_filename: str, the filename for tracked landmarks.
    :param dst_filename: str, the filename of the saved mouth patch or embedding.
    """
    output = lipreader(data_filename, landmarks_filename)
    if isinstance(output, str):
        print(f"hyp: {output}")
        return output
    elif isinstance(output, tuple):
        assert dst_filename[-4:] == ".avi", f"the ext of {dst_filename} should be .avi"
        print(f"mouth patch is saved to {dst_filename}.")
        save2avi(dst_filename, data=output[0], fps=output[1])
    else:
        assert dst_filename[-4:] == ".npz", f"the ext of {dst_filename} should be .npz"
        print(f"embedding is saved to {dst_filename}.")
        save2npz(dst_filename, data=output.cpu().detach().numpy())
    return


def main():

    # -- pick device for inference.
    if torch.cuda.is_available() and args.gpu_idx >= 0:
        device = torch.device(f"cuda:{args.gpu_idx}")
    else:
        device = "cpu"

    lipreader = LipreadingPipeline(
        config_filename=args.config_filename,
        feats_position=args.feats_position,
        device=device,
        face_track=not args.landmarks_filename and not args.landmarks_dir,
    )
    
    if args.data_filename is not None:
        print("Let's start one_step_inference")

        if args.data_filename[-4:] != ".mp4" :
            with open("generated_subtitles.csv", 'w') as fw :
                writer = csv.writer(fw, delimiter = '/')
                for subt_id in os.listdir(args.data_filename) :
                
                    path = os.path.join(args.data_filename,subt_id,'*.mp4')
                    #breakpoint()
                    filelist = glob.glob(path)
					#st_id = args.data_filename
                    for fl in filelist :
						#fl은 speaker id.
                        print(f"{subt_id}/{fl} inference")
                        res = one_step_inference(
						lipreader,
						fl,
						args.landmarks_filename,
						args.dst_filename,
						)
                        spk_id = fl.split('/')[-1][:-4]
                        row = [subt_id, spk_id, res]
                        writer.writerow(row)
        
    else:
        print("There is no data file")


if __name__ == '__main__':
    main()
