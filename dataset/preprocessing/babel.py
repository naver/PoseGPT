# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import sys
import os
#import glob
from tqdm import tqdm
# Paths need to be fixed.
#from utils.constants import BABEL_DIR, AMASS_DIR, MOCAP_DIR
#import numpy as np
import torch
#import roma
import json
from threed.skeleton import *
#import ipdb
#from PIL import Image
from dataset.preprocessing.amass import AMASS, create_subsequences
from torch.utils.data import DataLoader
from collections import Counter

try:
    import _pickle as pickle
except:
    import pickle

os.umask(0x0002)  # give write right to the team for all created files

ACTION2ID = {
    "walk": 0,
    "stand": 1,
    "hand movements": 2,
    "turn": 3,
    "interact with/use object": 4,
    "arm movements": 5,
    "t pose": 6,
    "step": 7,
    "backwards movement": 8,
    "raising body part": 9,
    "look": 10,
    "touch object": 11,
    "leg movements": 12,
    "forward movement": 13,
    "circular movement": 14,
    "stretch": 15,
    "jump": 16,
    "touching body part": 17,
    "sit": 18,
    "place something": 19,
    "take/pick something up": 20,
    "run": 21,
    "bend": 22,
    "throw": 23,
    "foot movements": 24,
    "a pose": 25,
    "stand up": 26,
    "lowering body part": 27,
    "sideways movement": 28,
    "move up/down incline": 29,
    "action with ball": 30,
    "kick": 31,
    "gesture": 32,
    "head movements": 33,
    "jog": 34,
    "grasp object": 35,
    "waist movements": 36,
    "lift something": 37,
    "knee movement": 38,
    "wave": 39,
    "move something": 40,
    "swing body part": 41,
    "catch": 42,
    "dance": 43,
    "lean": 44,
    "greet": 45,
    "poses": 46,
    "touching face": 47,
    "sports move": 48,
    "exercise/training": 49,
    "clean something": 50,
    "punch": 51,
    "squat": 52,
    "scratch": 53,
    "hop": 54,
    "play sport": 55,
    "stumble": 56,
    "crossing limbs": 57,
    "perform": 58,
    "martial art": 59,  # last BABEL_60 action
    "balance": 60,
    "kneel": 61,
    "shake": 62,
    "grab body part": 63,
    "clap": 64,
    "crouch": 65,
    "spin": 66,
    "upper body movements": 67,
    "knock": 68,
    "adjust": 69,
    "crawl": 70,
    "twist": 71,
    "move back to original position": 72,
    "bow": 73,
    "hit": 74,
    "touch ground": 75,
    "shoulder movements": 76,
    "telephone call": 77,
    "grab person": 78,
    "play instrument": 79,
    "tap": 80,
    "spread": 81,
    "skip": 82,
    "rolling movement": 83,
    "jump rope": 84,
    "play catch": 85,
    "drink": 86,
    "evade": 87,
    "support": 88,
    "point": 89,
    "side to side movement": 90,
    "stop": 91,
    "protect": 92,
    "wrist movements": 93,
    "stances": 94,
    "wait": 95,
    "shuffle": 96,
    "lunge": 97,
    "communicate (vocalise)": 98,
    "jumping jacks": 99,
    "rub": 100,
    "dribble": 101,
    "swim": 102,
    "sneak": 103,
    "to lower a body part": 104,
    "misc. abstract action": 105,
    "mix": 106,
    "limp": 107,
    "sway": 108,
    "slide": 109,
    "cartwheel": 110,
    "press something": 111,
    "shrug": 112,
    "open something": 113,
    "leap": 114,
    "trip": 115,
    "golf": 116,
    "move misc. body part": 117,
    "get injured": 118,
    "sudden movement": 119,  # last BABEL_60 action
    "duck": 120,
    "flap": 121,
    "salute": 122,
    "stagger": 123,
    "draw": 124,
    "tie": 125,
    "eat": 126,
    "style hair": 127,
    "relax": 128,
    "pray": 129,
    "flip": 130,
    "shivering": 131,
    "interact with rope": 132,
    "march": 133,
    "zombie": 134,
    "check": 135,
    "wiggle": 136,
    "bump": 137,
    "give something": 138,
    "yoga": 139,
    "mime": 140,
    "wobble": 141,
    "release": 142,
    "wash": 143,
    "stroke": 144,
    "rocking movement": 145,
    "swipe": 146,
    "strafe": 147,
    "hang": 148,
    "flail arms": 149
}
ID2ACTION = {v: k for k, v in ACTION2ID.items()}


def create_fns_for_babel(annot, type, amass_dir=''):
    fns = []
    ids = []
    type_ = 'smplh' if type == 'smpl' else type
    for k, v in tqdm(annot.items()):
        feat_p = '/'.join(v['feat_p'].split('/')[1:])

        if type == 'smplx':
            # Some annots are missing in smplx format
            feat_p = feat_p.replace('poses.npz', 'stageii.npz')
            feat_p = feat_p.replace(' ', '_')
            feat_p = feat_p.replace('DFaust_67', 'DFaust')
            feat_p = feat_p.replace('Transitions_mocap', 'Transitions')
            feat_p = feat_p.replace('TCD_handMocap', 'TCDHands')
            feat_p = feat_p.replace('SSM_synced', 'SSM')
            feat_p = feat_p.replace('MPI_mosh', 'MoSh')
            feat_p = feat_p.replace('MPI_HDM05', 'HDM05')
            feat_p = feat_p.replace('MPI_Limits', 'PosePrior')
            feat_p = feat_p.replace('BioMotionLab_NTroje', 'BMLrub')
        
        fn = os.path.join(amass_dir, type_, feat_p)
        
        if os.path.isfile(fn):
            fns.append(fn)
            ids.append(k)

    print(
        f"{len(fns)} files exist from the existing {len(annot)} files given by BABEL - ({100. * len(fns) / len(annot):.2f} %)")
    return fns, ids


def prepare_annots_trimmed(
        type='smplx',
        debug=False,
        fps=30, seq_len=64, min_seq_len=16, overlap_len=0,
        num_workers=0,
        split='train', n_actions=60, n_max=None,
        save_into_array=False,
        babel_dir='./babel/babel_v1.0_release',
        amass_dir='./amass/',
        mocap_dir='/scratch/1/user/fbaradel/posegpt/preprocesed_data',
        ):
    assert n_actions in [60, 120]
    assert split in ['train', 'val']

    id2action = {k: v for k, v in ID2ACTION.items() if k < n_actions}
    action2id = {v: k for k, v in id2action.items()}

    out_dir = os.path.join(mocap_dir, type, 'babel_trimmed')
    os.makedirs(out_dir, exist_ok=True)

    # BABEL annots
    annot = json.load(open(os.path.join(babel_dir, split + '.json')))

    # Create fns
    fns, ids = create_fns_for_babel(annot, type, amass_dir)

    # Init list
    list_x, list_v, list_a, list_k = [], [], [], []

    # Iterate over files
    error = 0
    loader = DataLoader(AMASS(fns, type, fps),
                        batch_size=1,
                        num_workers=num_workers,
                        shuffle=False)
    
    k = 0
    for i, fullpose in enumerate(tqdm(loader)):

        sys.stdout.flush()
        if k % 10 == 0:
            print(
                f"{len(list_x)} subseq extracted from {k + 1} trimmed" + \
                        " seq and {i + 1} long videos - nb error: {error}\n")

        if fullpose.size(1) > 0:
            fullpose = fullpose[0]  # [seq_len,pose_dim]
            try:
                # Annot
                if annot[ids[i]]['frame_ann'] is not None:
                    # frame annot
                    labels = annot[ids[i]]['frame_ann']['labels']
                    for lab in labels:
                        # create the k-hot vector
                        actions = torch.zeros(n_actions).int()
                        for y in lab['act_cat']:
                            if y in id2action.values():
                                actions[action2id[y]] = 1

                        if sum(actions) > 0:
                            start = int(lab['start_t'] * fps)
                            end = int(lab['end_t'] * fps)
                            start = min([start, end])  # mismatch sometimes
                            end = max([start, end])
                            pose = fullpose[start:end + 1].clone()
                            pose[:, -3:] = pose[:, -3:] - pose[[0], -3:]
                            list_x_i, list_v_i = create_subsequences(pose, seq_len, min_seq_len, overlap_len,
                                                                     debug=debug, pad=save_into_array)
                            # print(len(list_x_i))
                            for j in range(len(list_x_i)):
                                list_x.append(list_x_i[j].clone())
                                list_v.append(list_v_i[j].clone())
                                list_a.append(actions.clone())
                                list_k.append(k)
                            del list_x_i, list_v_i, actions
                            k += 1
                else:
                    # sequence annot because this is a short sequence
                    labels = annot[ids[i]]['seq_ann']['labels']
                    for lab in labels:
                        # create the k-hot vector
                        actions = torch.zeros(n_actions).int()
                        for y in lab['act_cat']:
                            if y in id2action.values():
                                actions[action2id[y]] = 1

                        if sum(actions) > 0:
                            pose = fullpose.clone()
                            list_x_i, list_v_i = create_subsequences(pose, seq_len, min_seq_len, overlap_len,
                                                                     debug=debug)
                            # print(len(list_x_i))
                            for j in range(len(list_x_i)):
                                list_x.append(list_x_i[j].clone())
                                list_v.append(list_v_i[j].clone())
                                list_a.append(actions.clone())
                                list_k.append(k)
                            del list_x_i, list_v_i, actions
                            k += 1
            except Exception as e:
                print(i, k, e)
                error += 1

        if n_max is not None and len(list_x) > n_max:
            break

    # Save in numpy array format
    out_dir = os.path.join(out_dir, f"{split}_{n_actions}",
                           f"seqLen{seq_len}_fps{fps}_overlap{overlap_len}_minSeqLen{min_seq_len}")
    if n_max is not None:
        out_dir = f"{out_dir}_nMax{n_max}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving preprocessed_data into {out_dir}")

    if save_into_array:
        torch.save(torch.stack(list_v), os.path.join(out_dir, 'valid.pt'))
        torch.save(torch.stack(list_x), os.path.join(out_dir, 'pose.pt'))
        torch.save(torch.stack(list_a), os.path.join(out_dir, 'action.pt'))
        torch.save(torch.Tensor(list_k).long(), os.path.join(out_dir, 'idx.pt'))
    else:
        # save list_pose, list_action
        torch.save(torch.stack(list_a), os.path.join(out_dir, 'action.pt'))

        list_pose = []
        for i in range(len(list_x)):
            pose = list_x[i][:list_v[i].sum().item()]  # [seq_len,168]
            list_pose.append(pose)

        with open(os.path.join(out_dir, f"pose.pkl"), 'wb') as f:
            pickle.dump(list_pose, f)


if __name__ == "__main__":
    exec(sys.argv[1])
