# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from torch.utils.data import Dataset
import sys
import torch
import numpy as np
import glob
import os
import roma
# import time
# import ipdb
from threed.skeleton import get_smplx_flip_params, get_smpl_flip_params

try:
    import _pickle as pickle
except:
    import pickle


class MocapDataset(Dataset):
    def __init__(self, data_dir='./', seq_len=32, training=False, n_iter=1000, n=-1, data_augment=True, dummy=False, sample_start=True):
        super().__init__()

        self.dummy = dummy
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.training = training
        self.n_iter = n_iter
        self.data_augment = data_augment
        self.sample_start = sample_start
        
        if self.dummy:
            # creating dummy data - 100 sequences with sequence length=128
            if 'smplx' in self.data_dir:
                self.pose = [torch.zeros(128, 168).float() for _ in range(100)]
            else:
                self.pose = [torch.zeros(128, 75).float() for _ in range(100)]
        else:
            with open(os.path.join(self.data_dir, 'pose.pkl'), 'rb') as f:
                self.pose = pickle.load(f)
        
        assert len(self.pose) > 0
        if n > 0:
            self.pose = self.pose[:n]

        self.dim2type = {168: 'smplx', 75: 'smpl'}
        assert self.pose[0].size(1) in self.dim2type.keys()
        self.type = self.dim2type[self.pose[0].size(1)]

        self.action = torch.load(os.path.join(self.data_dir, 'action.pt'))

        # self.pose - list - each element is a torch.Tensor of shape [seq_len, 168] - 56 axis-angle representation
        # self.action - list - each elemnt is a torch.Tensor of shape [n_action] - 0/1

        # Data augmentation by applying a random rotation
        max_tilt_angle, max_roll_angle, max_vertical_angle = 15, 15, 180
        self.tilt = torch.arange(-max_tilt_angle, max_tilt_angle + 0.01) * np.pi / 180
        self.roll = torch.arange(-max_roll_angle, max_roll_angle + 0.01) * np.pi / 180
        self.ry = torch.arange(-max_vertical_angle, max_vertical_angle + 0.01) * np.pi / 180

        # Data augmentation by flipping
        # TODO do the same for smpl
        if self.type == 'smplx':
            self.flip_perm, self.flip_inv = get_smplx_flip_params()
        elif self.type == 'smpl':
            self.flip_perm, self.flip_inv = get_smpl_flip_params()

    def __len__(self):
        #if self.training:
        if self.n_iter is not None:
            if not self.training:
                print("WARNING: not using the full validation set.")
            return self.n_iter
        else:
            return len(self.pose)

    def __repr__(self):
        return "MOCAP: Dirname: {} - Size: {}".format(self.data_dir, self.__len__())

    def __getitem__(self, idx):
        # Retrieve info
        if self.training:
            i = np.random.choice(range(len(self.pose)))
        else:
            i = idx
        pose = self.pose[i]
        action = self.action[i]

        # Sample a subseq
        if self.training and self.sample_start:
            start = np.random.choice(range(0, max([1, pose.shape[0] - self.seq_len])))
        else:
            start = max([0, pose.shape[0] // 2 - self.seq_len // 2])
        # print(start)
        pose = pose[start:start + self.seq_len]
        if len(action.shape) == 2:
            action = action[start:start + pose.shape[0]]

        # Fill in missing timesteps
        t_missing = self.seq_len - pose.shape[0]
        if t_missing > 0:
            pose = torch.cat([pose, pose[-1:].repeat(t_missing, 1)])
            if len(action.shape) == 2:
                action = torch.cat([action, action[-1:].repeat(t_missing, 1)])
        valid = torch.cat([torch.ones(self.seq_len - t_missing), torch.zeros(t_missing)]).long()

        # Start trans from (0,0,0)
        trans = pose[:, -3:]
        trans = trans - trans[[0]]

        # Root orient and body
        root_orient = pose[:, :3]

        # Data augment on the root orientation and trans
        if self.training and self.data_augment:
            # Random rotation of the root_orient and translation vector
            tilt, ry, roll = np.random.choice(self.tilt), np.random.choice(self.ry), np.random.choice(self.roll)
            rot = roma.rotvec_composition(
                [torch.Tensor([tilt, 0., 0.]), torch.Tensor([0., ry, 0.]), torch.Tensor([0., 0., roll])])
            rot = rot.unsqueeze(0).repeat(pose.shape[0], 1)
            root_orient = roma.rotvec_composition([rot, root_orient])
            rot = roma.rotvec_to_rotmat(rot)
            trans = torch.matmul(rot, trans.unsqueeze(-1))[..., 0]

        # Assemble
        pose = torch.cat([root_orient, pose[:, 3:-3], trans], -1)

        # Flip right and left joints
        if self.training and np.random.choice([True, False]) and self.data_augment:
        # if True:
            pose = pose[:, self.flip_perm] * self.flip_inv
        
        # pose - torch.Tensor torch.float32 - [seq_len,168] - concat of axis the 56 axis-angle representation
        # valid - torch.Tensor torch.int64 - [seq_len] - 1/0 indicating if a pose if present
        # action - torch.Tensor - torch.int32 - [n_action] or [seq_len,n_action] - 1/0 indicating if an action is present in the sequence
        # print(pose.shape, valid.shape, action.shape)
        return pose, valid, action


def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1)
    print("Worker id: {} - Seed: {}".format(worker_id, seed))
    np.random.seed(seed)


def test_mocap_dataset(
        # data_dir='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16_nMax1000',
        data_dir='data/smplx/babel_untrimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16',
        # data_dir='data/smpl/humanact12/train',
        training=False, seq_len=32, batch_size=16, num_workers=0, n_iter=10, dummy=False,
        start_rendering = -1,
        end_rendering=10000,
        n_max=5, ry=np.pi/2.,
        color_start=[173, 216, 230],
        color_end=[230, 187, 173],
        freq=10,
        ):
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = MocapDataset(data_dir=data_dir, seq_len=seq_len, training=training, n_iter=n_iter * batch_size,
                           dummy=dummy)
    print(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=False)

    # Looping
    n_ = 0
    for _, item in tqdm(enumerate(loader), total=len(loader)):
        if 'babel' in data_dir:
            idx = torch.where(item[-1][:,43] == 1)[0]
            if len(idx) > 0:
                print("found a person dancing")
                i = idx[0].item()
                n_ +=1
                if n_ == n_max:
                    break
            else:
                i = 0

    # Visu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    from threed.renderer import PyTorch3DRenderer
    from pytorch3d.renderer import look_at_view_transform
    import smplx
    from PIL import Image
    from threed.skeleton import get_smplx_pose, get_smpl_pose
    get_pose = get_smplx_pose if dataset.type == 'smplx' else get_smpl_pose 

    rotation, camera = look_at_view_transform(dist=8, elev=0, azim=180)
    bodymodel = smplx.create('/gfs/project/humans/SMPLX/models', dataset.type, use_pca=False)
    renderer = PyTorch3DRenderer(1024, background_color=(1, 1, 1)).to(device)
    faces = torch.as_tensor(np.array(bodymodel.faces, dtype=np.int32))[None, :, :]

    tmp_dir = '/tmp/fbaradel/vid'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm {tmp_dir}/*.jpg")
    
    t_max = item[1][i].sum().item()
    x = item[0][i]
    print(f"SeqLen={t_max}")
    if 'babel' in data_dir:
        from dataset.preprocessing.babel import ID2ACTION
        for actId in torch.where(item[-1][i] == 1)[0]:
            print(ID2ACTION[actId.item()])

    color_start = [x/255. for x in color_start]
    color_end = [x/255. for x in color_end]
    range_x = [color_start[0] for _ in range(t_max)] if color_start[0] == color_end [0] else np.arange(color_start[0], color_end[0], (color_end[0] - color_start[0])/t_max).tolist()
    range_y = [color_start[1] for _ in range(t_max)] if color_start[1] == color_end [1] else np.arange(color_start[1], color_end[1], (color_end[1] - color_start[1])/t_max).tolist()
    range_z = [color_start[2] for _ in range(t_max)] if color_start[2] == color_end [2] else np.arange(color_start[2], color_end[2], (color_end[2] - color_start[2])/t_max).tolist()
    img_video = None
    x[..., -3:] = x[..., -3:] - torch.median(x[..., -3:], 0).values.unsqueeze(0)
    
    # rotate by pi/2
    root, rel, trans = x[:,:3], x[:,3:-3], x[:,-3:]
    rot = torch.Tensor([0., ry, 0.])
    rot = rot.unsqueeze(0).repeat(root.shape[0], 1)
    root = roma.rotvec_composition([rot, root])
    rot = roma.rotvec_to_rotmat(rot)
    trans = torch.matmul(rot, trans.unsqueeze(-1))[..., 0]
    x = torch.cat([root, rel, trans], 1)

    for t in tqdm(range(t_max)):
        with torch.no_grad():
            body = bodymodel(**get_pose(x[[t]]))
            img = \
                renderer.renderPerspective(vertices=body.vertices.to(device),
                                           faces=faces.to(device),
                                           rotation=rotation.to(device),
                                           camera_translation=camera.to(device),
                                           color=torch.Tensor([[[range_x[t], range_y[t], range_z[t]]]]).to(device),
                                        #    color=torch.Tensor([[[0., 0.7, 1.]]]).to(device),
                                           )[0].cpu().numpy()
            Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")

            # video-image
            if t % freq == 0 and t > start_rendering and t < end_rendering:
                if img_video is None:
                    img_video = img
                else:
                    alpha = 1.
                    fg_mask = (np.sum(img, axis=-1) != 3*255)
                    fg_mask = np.concatenate((fg_mask[:,:,None], fg_mask[:,:,None], fg_mask[:,:,None]), axis=2)
                    img_video = (fg_mask * (alpha * img + (1.0-alpha) * img_video) + (1-fg_mask) * img_video)

    Image.fromarray(img_video.astype(np.uint8)).save(f"img.jpg")

    os.system(
        f"ffmpeg -framerate 30 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y")
    os.system(f"rm {tmp_dir}/*.jpg")


if __name__ == "__main__":
    exec(sys.argv[1])
