# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.renderer
import numpy as np
import pickle
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from utils.constants import FOCAL_LENGTH, IMG_SIZE
from threed.geometry import *
import sys
import ipdb
from threed.geometry import find_best_camera_for_video


class PyTorch3DRenderer(torch.nn.Module):
    """
    Thin wrapper around pytorch3d threed.
    Only square renderings are supported.
    Remark: PyTorch3D uses a camera convention with z going out of the camera and x pointing left.
    """

    def __init__(self,
                 image_size,
                 background_color=(0, 0, 0),
                 convention='opencv',
                 blur_radius=0,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                 ):
        super().__init__()
        self.image_size = image_size

        raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel)
        rasterizer = pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings_soft)

        materials = pytorch3d.renderer.materials.Materials(shininess=1.0)
        blend_params = pytorch3d.renderer.BlendParams(background_color=background_color)

        # One need to attribute a camera to the shader, otherwise the method "to" does not work.
        dummy_cameras = pytorch3d.renderer.OrthographicCameras()
        shader = pytorch3d.renderer.SoftPhongShader(cameras=dummy_cameras,
                                                    materials=materials,
                                                    blend_params=blend_params)

        # Differentiable soft threed using per vertex RGB colors for texture
        self.renderer = pytorch3d.renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

        self.convention = convention
        if convention == 'opencv':
            # Base camera rotation
            base_rotation = torch.as_tensor([[[-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, 1]]], dtype=torch.float)
            self.register_buffer("base_rotation", base_rotation)
            self.register_buffer("base_rotation2d", base_rotation[:, 0:2, 0:2])

        # Light Color
        self.ambient_color = 0.5
        self.diffuse_color = 0.3
        self.specular_color = 0.2

        self.bg_blending_radius = bg_blending_radius
        if bg_blending_radius > 0:
            self.register_buffer("bg_blending_kernel",
                                 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                         2 * bg_blending_radius + 1) ** 2)
            self.register_buffer("bg_blending_bias", -torch.ones(1))
        else:
            self.blending_kernel = None
            self.blending_bias = None

    def to(self, device):
        # Transfer to device is a bit bugged in pytorch3d, one needs to do this manually
        self.renderer.shader.to(device)
        return super().to(device)

    def render(self, vertices, faces, cameras, color=None):
        """
        Args:
            - vertices: [B,N,V,3]
            - faces: [B,F,3]
            - maps: [B,N,W,H,3] in 0-1 range - if None the texture will be metallic
            - cameras: PerspectiveCamera or OrthographicCamera object
            - color: [B,N,V,3]
        Return:
            - img: [B,W,H,C]
        """

        if isinstance(vertices, torch.Tensor):
            _, N, V, _ = vertices.size()
            list_faces = []
            list_vertices = []
            for i in range(N):
                list_faces.append(faces + V * i)
                list_vertices.append(vertices[:, i])
            faces = torch.cat(list_faces, 1)  # [B,N*F,3]
            vertices = torch.cat(list_vertices, 1)  # [B,N*V,3]

            # Metallic texture
            verts_rgb = torch.ones_like(vertices).reshape(-1, N, V, 3)  # [1,N,V,3]
            if color is not None:
                verts_rgb = color * verts_rgb
            verts_rgb = verts_rgb.flatten(1, 2)
            
            textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)
            # Create meshes
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces , textures=textures)
        else:
            tex = [torch.ones_like(vertices[i]) * color[i] for i in range(len(vertices))]
            tex = torch.cat(tex)[None]
            textures = pytorch3d.renderer.Textures(verts_rgb=tex)
            
            verts = torch.cat(vertices)

            faces_up = []
            n = 0
            for i in range(len(faces)):
                faces_i = faces[i] + n
                faces_up.append(faces_i)
                n += vertices[i].shape[0]
            faces = torch.cat(faces_up)
            # ipdb.set_trace()
            meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures)

        # Create light
        lights = pytorch3d.renderer.DirectionalLights(
            ambient_color=((self.ambient_color, self.ambient_color, self.ambient_color),),
            diffuse_color=((self.diffuse_color, self.diffuse_color, self.diffuse_color),),
            specular_color=(
                (self.specular_color, self.specular_color, self.specular_color),),
            direction=((0, 0, -1.0),),
            device=vertices[0].device)
        images = self.renderer(meshes, cameras=cameras, lights=lights)

        rgb_images = images[..., :3]
        rgb_images = torch.clamp(rgb_images, 0., 1.)
        rgb_images = rgb_images * 255
        rgb_images = rgb_images.to(torch.uint8)

        return rgb_images

    def renderPerspective(self, vertices, faces, camera_translation, principal_point=None, color=None, rotation=None,
                          focal_length=2 * FOCAL_LENGTH / IMG_SIZE):
        """
        Args:
            - vertices: [B,V,3] or [B,N,V,3] where N is the number of persons
            - faces: [B,13776,3]
            - focal_length: float
            - principal_point: [B,2]
            - T: [B,3]
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C] in range 0-1
        """

        device = vertices[0].device
        # device = vertices.device

        if principal_point is None:
            principal_point = torch.zeros_like(camera_translation[:, :2])

        if isinstance(vertices, torch.Tensor) and vertices.dim() == 3:
            vertices = vertices.unsqueeze(1)

        # Create cameras
        if rotation is None:
            R = self.base_rotation
        else:
            R = torch.bmm(self.base_rotation, rotation)
        camera_translation = torch.einsum('bik, bk -> bi', self.base_rotation.repeat(camera_translation.size(0), 1, 1),
                                          camera_translation)
        if self.convention == 'opencv':
            principal_point = -torch.as_tensor(principal_point)
        cameras = pytorch3d.renderer.PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                                        R=R, T=camera_translation, device=device)

        rgb_images = self.render(vertices, faces, cameras, color)

        return rgb_images


def render_video(vertices, cameras, faces, image_size=400, device=None, add_border=True, last_t_green_border=10000,
                 text=None,
                 focal_length=2 * FOCAL_LENGTH / IMG_SIZE, pad_width=None, rotation=None, color=None,
                 create_video=True,
                 create_video_image=False,
                 color_start=[173, 216, 230],
                 color_end=[230, 187, 173],
                 freq=10,
                 background_color=(1, 1, 1),
                 adapt_camera=False
                 ):
    """
    Rendering human 3d mesh into RGB images
    :param verts: [seq_len,V,3] or [seq_len,N,V,3] or list of length 'seq_len' with tensor shape [seq_len,N,V,3] where N is the number of persons
    :param faces: [1,13776,3]
    :param camera_translation: [seq_len,3]
    :param image_size: int
    :param device: cpu or cuda
    :param color: [seq_len,N,V,3] or list of [N,V,3] of length 'seq_len'
    :return: video: [seq_len,image_size,image_size,3]
    """
    if cameras is None:
        from pytorch3d.renderer import look_at_view_transform
        rotation, camera = look_at_view_transform(dist=5, elev=0, azim=180)
        cameras = camera.repeat(vertices.size(1), 1)

    if adapt_camera:
        cam = find_best_camera_for_video(vertices.cpu(), factor=1.2, n_jts_to_take=256)
        cameras[:] = cam

    seq_len = len(vertices)

    color_start = [x/255. for x in color_start]
    color_end = [x/255. for x in color_end]
    range_x = [color_start[0] for _ in range(seq_len)] if color_start[0] == color_end [0] else np.arange(color_start[0], color_end[0], (color_end[0] - color_start[0])/seq_len).tolist()
    range_y = [color_start[1] for _ in range(seq_len)] if color_start[1] == color_end [1] else np.arange(color_start[1], color_end[1], (color_end[1] - color_start[1])/seq_len).tolist()
    range_z = [color_start[2] for _ in range(seq_len)] if color_start[2] == color_end [2] else np.arange(color_start[2], color_end[2], (color_end[2] - color_start[2])/seq_len).tolist()
    img_video = None
    
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    renderer = PyTorch3DRenderer(image_size, background_color=background_color).to(device)
    list_image = []
    r_cons, g_cons, b_cons = 0, 255, 0
    pad_width = 0 if pad_width is not None else int(0.025 * image_size)
    for t in range(seq_len):
        if t == last_t_green_border:
            r_cons, g_cons, b_cons = 255, 0, 0

        color_t = None if color is None else color[[t]].to(device)
        rotation_t = None if rotation is None else rotation.to(device)

        if create_video:
            with torch.no_grad():
                image = renderer.renderPerspective(vertices=vertices[[t]].to(device),
                                                camera_translation=cameras[[t]].to(device),
                                                faces=faces.to(device),
                                                focal_length=focal_length,
                                                rotation=rotation_t,
                                                color=color_t,
                                                ).cpu().numpy()[0]
            # add border
            if add_border:
                rb = np.pad(array=image[..., 0], pad_width=pad_width, mode='constant', constant_values=r_cons)
                gb = np.pad(array=image[..., 1], pad_width=pad_width, mode='constant', constant_values=g_cons)
                bb = np.pad(array=image[..., 2], pad_width=pad_width, mode='constant', constant_values=b_cons)
                image = np.dstack(tup=(rb, gb, bb))

            # add text
            if text is not None:
                img = Image.fromarray(image)
                draw = ImageDraw.Draw(img)
                draw.text((1.5 * pad_width, 1.5 * pad_width), text, fill=(128, 128, 128), font=font)
                image = np.asarray(img)

            list_image.append(image)

        # video-image
        if create_video_image and t % freq == 0:
            with torch.no_grad():
                img = renderer.renderPerspective(vertices=vertices[[t]].to(device),
                                                camera_translation=cameras[[t]].to(device),
                                                faces=faces.to(device),
                                                focal_length=focal_length,
                                                rotation=rotation_t,
                                                color=torch.Tensor([[[range_x[t], range_y[t], range_z[t]]]]).to(device)
                                                ).cpu().numpy()[0]

            if img_video is None:
                img_video = img
            else:
                alpha = 1.
                fg_mask = (np.sum(img, axis=-1) != 3*255)
                fg_mask = np.concatenate((fg_mask[:,:,None], fg_mask[:,:,None], fg_mask[:,:,None]), axis=2)
                img_video = (fg_mask * (alpha * img + (1.0-alpha) * img_video) + (1-fg_mask) * img_video)

        
    del renderer

    video = 0
    if create_video:
        video = np.stack(list_image)

    if create_video_image and create_video:
        return video, img_video.astype(np.uint8)
    elif create_video_image and not create_video:
        return img_video.astype(np.uint8)
    return video


def test():
    import roma
    import matplotlib.pyplot as plt

    img_size = 500
    f_x = f_y = FOCAL_LENGTH

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/smplh/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    t = int(data['root_orient'].size(0) / 2.)
    root_orient = data['root_orient'][[t]]
    pose_body = data['pose_body'][[t]]
    pose_hand = data['pose_hand'][[t]]
    trans = data['trans'][[t]]
    trans = torch.zeros_like(trans)

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    camera_translation = find_best_camera(joints, factor=1.3, f_x=f_x, f_y=f_y)

    # Rendering
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rend = PyTorch3DRenderer(img_size).to(device)
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    vertices = torch.stack([vertices], 1)  # [t,n,v,3]

    V = vertices.size(2)
    color = plt.get_cmap('viridis', lut=V)(range(V))[:,:3]
    color = torch.from_numpy(color).float().reshape(1, 1, V, 3)

    with torch.no_grad():
        img = rend.renderPerspective(vertices=vertices.to(device),
                                     faces=pyfaces.to(device),
                                     camera_translation=camera_translation.to(device),
                                     color=color.to(device)
                                     )[0].cpu().numpy()
    image = Image.fromarray(img)
    image.save('img.jpg')


def test_renderer():
    import roma
    img_size = 500
    f_x = f_y = FOCAL_LENGTH

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/smplh/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    t = int(data['root_orient'].size(0) / 2.)
    root_orient = data['root_orient'][[t]]
    pose_body = data['pose_body'][[t]]
    pose_hand = data['pose_hand'][[t]]
    trans = data['trans'][[t]]
    trans = torch.zeros_like(trans)
    print(trans)

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    camera_translation = find_best_camera(joints, factor=1.3, f_x=f_x, f_y=f_y)

    # Rendering
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rend = PyTorch3DRenderer(img_size).to(device)
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    vertices = torch.stack([vertices, 0.25 + vertices], 1)  # [t,n,V,3] - add the same person
    with torch.no_grad():
        img = rend.renderPerspective(vertices=vertices.to(device),
                                     faces=pyfaces.to(device),
                                     camera_translation=camera_translation.to(device),
                                     color=torch.randn(1, 2, 3).to(device)
                                     )[0].cpu().numpy()
    image = Image.fromarray(img)

    # Project jts into 2d
    camera_center = torch.as_tensor([[IMG_SIZE / 2., IMG_SIZE / 2.]])
    rotation = torch.eye(3).type_as(joints).unsqueeze(0)
    keypoints = perspective_projection(world2cam(joints, camera_translation, rotation), camera_center, f_x, f_y)
    keypoints /= IMG_SIZE
    draw = ImageDraw.Draw(image)
    r = 2
    for po in keypoints[0]:
        x_, y_ = po * img_size
        leftUpPoint = (x_ - r, y_ - r)
        rightDownPoint = (x_ + r, y_ + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill='red')
    image.save('img.jpg')

    # Estim translation
    camera_translation_hat = estimate_translation_np(joints[0].numpy(), keypoints[0].numpy() * IMG_SIZE,
                                                     f_x=f_x, f_y=f_y)
    print(camera_translation_hat, camera_translation)
    for i in range(3):
        assert abs(camera_translation_hat[i] - camera_translation[0, i]).item() < 0.0001

    # import ipdb
    # ipdb.set_trace()
    out = render_video(vertices, camera_translation, pyfaces, device=None)
    Image.fromarray(out[0]).save(f"img_bis.jpg")


def test_video_rendering():
    from tqdm import tqdm
    import os
    import ipdb

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    # Creating a human mesh
    print("Generating 3d human mesh using SMPL")
    seq_len = 60
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    trans = data['trans'][:seq_len]
    # trans = trans - trans[[int(seq_len/2.)]]
    trans = trans - trans[[0]]
    bm_out = bm(root_orient=data['root_orient'][:seq_len], pose_body=data['pose_body'][:seq_len],
                pose_hand=data['pose_hand'][:seq_len],
                trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    print("Finding the best camera params")
    camera_translation = find_best_camera_for_video(joints, factor=1.3, n_jts_to_take=100)
    print(camera_translation)
    camera_translation = camera_translation.repeat(seq_len, 1)

    # Rendering
    print("2D rendering")
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    print(vertices.shape, camera_translation.shape, pyfaces.shape)
    video = render_video(vertices, camera_translation, pyfaces, last_t_green_border=int(seq_len / 2.), text='Test')

    # Building video
    print("Video creation")
    tmp_dir = 'output'
    for t in tqdm(range(video.shape[0])):
        Image.fromarray(video[t]).save(f"{tmp_dir}/{t:06d}.jpg")
    cmd = f"ffmpeg -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {tmp_dir}/video.mp4 -y"
    os.system(cmd)
    os.system(f"rm {tmp_dir}/*.jpg")


def test_camera_params_estimation():
    import ipdb
    import os

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    seq_len = 30
    factor = 1.1
    ttt = [int(data['root_orient'].size(0) / 2.) + i for i in range(seq_len)]
    root_orient = data['root_orient'][ttt]
    pose_body = data['pose_body'][ttt]
    pose_hand = data['pose_hand'][ttt]
    trans = data['trans'][ttt]
    trans = trans - trans[[0]]

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    faces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32))[None, :, :]
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand)

    # Rendering using trans
    camera = find_best_camera_for_video(bm_out.Jtr + trans.unsqueeze(1), factor=factor)
    print("Camera: ", camera)
    video = render_video(bm_out.v + trans.unsqueeze(1), camera.repeat(seq_len, 1), faces, text='w/ trans')

    # Estim. camera
    camera_bis = estimate_video_camera_params_wo_trans(bm_out.Jtr.unsqueeze(0), trans.unsqueeze(0), factor=factor)[0]
    print("Camera bis t=0: ", camera_bis[0])
    video_bis = render_video(bm_out.v, camera_bis, faces, text='wo trans')

    print(np.abs(video_bis - video).sum())

    # Create video
    visu_dir = './output'
    os.makedirs(visu_dir, exist_ok=True)
    for t in range(seq_len):
        img = np.concatenate([video[t], video_bis[t]], 1)
        Image.fromarray(img).save(f"{visu_dir}/{t:06d}.jpg")
    cmd = f"ffmpeg -framerate 5 -pattern_type glob -i '{visu_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y"
    os.system(cmd)
    os.system(f"rm {visu_dir}/*.jpg")

    # print(camera)
    #
    # # Image with trans
    # cameras = torch.Tensor([[0., 0., 44.]])
    # img = render_video(bm_out.v, cameras, faces)[0]
    #
    # # Image w/o trans but with estimated cam params
    # kps = perspective_projection(bm_out.Jtr, cameras)  # range 0-1
    # jts = bm_out.Jtr - trans
    # cameras_hat = estimate_translation_np(jts[0].numpy(), kps[0].numpy())
    # cameras_hat = torch.from_numpy(cameras_hat).unsqueeze(0).float()
    # img_bis = render_video(bm_out.v - trans, cameras_hat, faces)[0]
    # Image.fromarray(np.concatenate([img, img_bis])).save(f"img.jpg")
    # print(np.abs(img_bis - img).sum())


if __name__ == "__main__":
    exec(sys.argv[1])
