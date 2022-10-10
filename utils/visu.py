# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from threed.renderer import render_video

def visu_gt_rec(err, valid, verts_hat, verts, faces,
                device, visu_dir, idx, *, sample=None,
                save_to_disk=False):

    rotation, camera = look_at_view_transform(dist=8, elev=20, azim=180)
    cameras = camera.repeat(verts.size(1), 1)

    # show errors on vertices
    n_bins = 1000
    pve = (err.mean(-1) * valid).sum() / valid.sum()
    err_i = ((n_bins - 1) * err / (torch.max(err))).cpu().int().numpy()  # [t,v] 0-n_bins
    COLORS = plt.get_cmap('viridis', lut=n_bins)(range(n_bins))[:, :3]
    color = COLORS[err_i.reshape(-1)].reshape(err_i.shape[0], err_i.shape[1], 3)
    color = torch.from_numpy(color).float().to(device)

    rec = render_video(verts_hat, cameras, faces, rotation=rotation,
                       text=f"Rec - pve={pve.item():.1f}",
                       add_border=False, color=color)
    gt = render_video(verts, cameras, faces, rotation=rotation,
                      text="GT", add_border=False)
    vsample = []
    if sample is not None:
        vsample = [render_video(sample, cameras, faces, rotation=rotation, text="Samples", add_border=False)]
    list_video = [rec, gt] + vsample


    if save_to_disk:
        for t in range(valid.sum().item()):
            img = np.concatenate([x[t] for x in list_video], 1)
            Image.fromarray(img).save(f"{visu_dir}/{t:06d}.jpg")
        cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{visu_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {visu_dir}/{idx:06d}.mp4 -y"
        os.system(cmd)
        os.system(f"rm {visu_dir}/*.jpg")
    return list_video




def visu_sample_gt_rec(verts_sample, verts_half_sample, verts_upper_bound, err, valid, verts, faces, device, visu_dir, idx):
    rotation, camera = look_at_view_transform(dist=5, elev=0, azim=180)
    cameras = camera.repeat(verts.size(1), 1)

    # show errors on vertices
    n_bins = 1000
    pve = (err.mean(-1) * valid).sum() / valid.sum()
    err_i = ((n_bins - 1) * err / (torch.max(err))).cpu().int().numpy()  # [t,v] 0-n_bins
    COLORS = plt.get_cmap('viridis', lut=n_bins)(range(n_bins))[:, :3]
    color = COLORS[err_i.reshape(-1)].reshape(err_i.shape[0], err_i.shape[1], 3)
    color = torch.from_numpy(color).float().to(device)

    list_video = []
    list_video.append(
        render_video(verts_sample, cameras, faces, rotation=rotation, text=f"Sample",
                     add_border=False)
                     )
    list_video.append(
        render_video(verts_half_sample, cameras, faces, rotation=rotation, text=f"Half_sample",
                     add_border=False))
    list_video.append(
        render_video(verts_upper_bound, cameras, faces, rotation=rotation,
                     text=f"GT upper bound - pve={pve.item():.1f}",
                     add_border=False, color=color))
    list_video.append(
        render_video(verts, cameras, faces, rotation=rotation, text=f"GT", add_border=False))

    for t in range(valid.sum().item()):
        img = np.concatenate([x[t] for x in list_video], 1)
        Image.fromarray(img).save(f"{visu_dir}/{t:06d}.jpg")

    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{visu_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {visu_dir}/{idx:06d}.mp4 -y"
    os.system(cmd)
    os.system(f"rm {visu_dir}/*.jpg")
    return list_video

