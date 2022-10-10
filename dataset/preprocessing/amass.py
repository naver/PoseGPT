# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import numpy as np
import roma
import sys
import os
#import glob
from tqdm import tqdm
#import smplx
#import ipdb
from threed.renderer import PyTorch3DRenderer
from pytorch3d.renderer import look_at_view_transform
import smplx
from PIL import Image
#import ipdb
try:
    import _pickle as pickle
except:
    import pickle

os.umask(0x0002)  # give write right to the team for all created files

AMASS_SPLITS = {
    'smplh': {
        'val': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
                  # 'BML', replaced by BioMotionLab_NTroje
                  'BioMotionLab_NTroje',
                  'EKUT', 'TCD_handMocap']
        # ACCAD
    },
    'smplx': {
        'val': ['HumanEva',
                'HDM05',  # initially 'MPI_HDM05',
                'SFU',
                'MoSh'  # initially 'MPI_mosh'
                ],
        'test': ['Transitions',  # initi 'Transitions_mocap',
                 'SSM'  # init 'SSM_synced'
                 ],
        'train': ['CMU',
                  'PosePrior',  # init 'MPI_Limits',
                  'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
                  'BMLrub',  # init 'BML', replaced by BioMotionLab_NTroje
                  'EKUT',
                  'TCDHands'  # init 'TCD_handMocap'
                  ]
        # ACCAD
    }
}


class AMASS(object):
    def __init__(self, fns, type, fps=30):
        super().__init__()
        self.fns = fns
        self.fps = fps
        self.type = type

        if type == 'smplh' or type == 'smpl':
            self.fps_ = 'mocap_framerate'
            self.pose_ = 'poses'
            self.trans_ = 'trans'
        elif type == 'smplx':
            self.fps_ = 'mocap_frame_rate'
            self.pose_ = 'poses'
            self.trans_ = 'trans'
        else:
            raise NameError

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, item):
        try:

            fn = self.fns[item]
            data = np.load(fn, allow_pickle=True)

            # Select the full pose
            pose = data[self.pose_]

            if pose.shape[0] > 1:
                # Subsample at a certain fps
                bin_len = int(data[self.fps_]) / float(self.fps)
                length_up = int(pose.shape[0] / bin_len)
                tt = (bin_len * np.arange(0, length_up)).astype(np.int32).tolist()

                pose = torch.from_numpy(pose).float()[tt]  # root_orient+pose_body+pose_jaw+pose_eye+pose_hand
                trans = torch.from_numpy(data[self.trans_]).float()[tt]
                root_orient = pose[:, :3]
                body = pose[:, 3:]
                seq_len = root_orient.shape[0]

                # Rotate poses and transl (up side down)
                rot1 = torch.Tensor([[np.pi / 2., 0., 0.]]).repeat(seq_len, 1).float()
                rot2 = torch.Tensor([[0., 0., 0.]]).repeat(seq_len, 1).float()
                rot3 = torch.Tensor([[0., 0., 0.]]).repeat(seq_len, 1).float()
                rot = roma.rotvec_composition([rot1, rot2, rot3])
                root_orient = roma.rotvec_composition([rot, root_orient])
                trans = torch.matmul(roma.rotvec_to_rotmat(rot), trans.unsqueeze(-1))[..., 0]
                trans = trans - trans[[0]]

                # Full body pose
                if self.type == 'smpl':
                    body = body.reshape(seq_len, -1, 3)
                    body_ = body[:, :21].flatten(1)
                    hand = body[:, 21:][:, [0, 15]].flatten(1)
                    body = torch.cat([body_, hand], 1)
                full_pose = torch.cat([root_orient, body, trans], 1)

                return full_pose
            else:
                print(self.fns[item])
                return torch.zeros((0, 1)).float()
        except Exception as e:
            print(self.fns[item], e)
            return torch.zeros((0, 1)).float()


def create_subsequences(pose, seq_len=64, min_seq_len=16, overlap_len=0, debug=False, pad=True):
    """
    Args:
        - pose: [seq_len_tot,pose_dim]
    Return:
        - list of pose
    """
    list_x, list_v = [], []

    # Split into subseq
    real_seq_len = pose.shape[0]
    start = 0
    end = 0
    # Cut the seq into subseq
    while end < real_seq_len:
        # Subseq
        end = min([start + seq_len, real_seq_len])
        real_subseq_len = end - start

        # If subseq is big enough then append
        if real_subseq_len >= min_seq_len:
            tt = range(start, end)

            # Valid
            t_missing = 0
            valid = torch.ones(len(tt)).long()
            if pad:
                t_missing = seq_len - (end - start)
                valid = torch.cat([valid, torch.zeros(t_missing)]).long()

            # Pose
            x = pose[tt]
            x = torch.cat([x, x[-1:].repeat(t_missing, 1)])

            # Make sure that the trans starts from zero
            x[:, -3:] = x[:, -3:] - x[[0], -3:]

            list_x.append(x)
            list_v.append(valid)

            if debug:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

                rotation, camera = look_at_view_transform(dist=7, elev=45, azim=180)
                if pose.size(-1) == 75:
                    type = 'smpl'
                elif pose.size(-1) == 168:
                    type = 'smplx'
                else:
                    type = 'smplh'
                get_pose_fn = eval(f"get_{type}_pose")
                bodymodel = smplx.create(SMPLX_DIR, type, use_pca=False)
                renderer = PyTorch3DRenderer(500).to(device)
                faces = torch.as_tensor(np.array(bodymodel.faces, dtype=np.int32))[None, :, :]

                tmp_dir = '/tmp/amass_processing/vid'
                os.makedirs(tmp_dir, exist_ok=True)
                os.system(f"rm {tmp_dir}/*.jpg")
                t_max = min([1000, x.shape[0]])
                for t in tqdm(range(t_max)):
                    with torch.no_grad():
                        body = bodymodel(**get_pose_fn(x[[t]]))
                        img = \
                            renderer.renderPerspective(vertices=body.vertices.to(device),
                                                       faces=faces.to(device),
                                                       camera_translation=camera.to(device))[0].cpu().numpy()
                        Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")
                os.system(
                    f"ffmpeg -framerate 30 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y")
                os.system(f"rm {tmp_dir}/*.jpg")

                import ipdb
                ipdb.set_trace()

        start = end - overlap_len
    return list_x, list_v

if __name__ == "__main__":
    exec(sys.argv[1])
