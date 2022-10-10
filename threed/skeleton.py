# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np
import torch
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from threed.geometry import angle_between, rotation_matrix
import roma

""" The function preprocess_skeleton is adapted from https://github.com/lshiwjx/2s-AGCN"""

def get_dict_bodymodel_joint_names():
    """ pelvis at 0 + body joint names """
    return {0: 'pelvis', 1: 'left_hip', 4: 'left_knee', 7: 'left_ankle', 10: 'left_foot', 2: 'right_hip', 5: 'right_knee', 8: 'right_ankle', 11: 'right_foot', 3: 'spine1', 6: 'spine2', 9: 'spine3', 12: 'neck', 15: 'head', 13: 'left_collar', 16: 'left_shoulder', 18: 'left_elbow', 20: 'left_wrist', 14: 'right_collar', 17: 'right_shoulder', 19: 'right_elbow', 21: 'right_wrist', 36: 'left_index', 35: 'left_thumb', 41: 'right_index', 40: 'right_thumb'}


def get_smplh_joint_names():
    return [
        'pelvis',  # 0
        'left_hip',
        'right_hip',
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',
        'spine3',
        'left_foot',  # 10
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'head',  # 15
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',  # 20
        'right_wrist',
        # SMPLH - the rest is wrong! it should only be MANO joints
        # TODO replace by MANO joints
        'left_index1',  # not sure
        'right_index1',  # not sure
        'nose',  # 24
        'right_eye',  # 25
        'left_eye',  # 26
        'right_ear',  # 27
        'left_ear',  # 28
        'left_big_toe',  # 29
        'left_small_toe',  # 30
        'left_heel',  # 31
        'right_big_toe',  # 32
        'right_small_toe',  # 33
        'right_heel',  # 34
        'left_thumb',
        'left_index',
        'left_middle',
        'left_ring',
        'left_pinky',
        'right_thumb',
        'right_index',
        'right_middle',
        'right_ring',
        'right_pinky']


def get_mupots_joint_names():
    return [
        'headtop',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'hip',
        'Spine (H36M)',
        'Head (H36M)'
    ]


def get_dope_joint_names():
    # Missing joints compared to h36m:
    # - hip:
    # - Spine (H36M):
    # - neck:
    # - Head (H36M):
    #
    # Missing extra joints compared to H36M_plus
    # - pelvis:
    # others are not very important since it is more about extrapolating the skeleton
    #
    return [
        'right_ankle',  # 0
        'left_ankle',  # 1
        'right_knee',  # 2
        'left_knee',  # 3
        'right_hip',  # 4
        'left_hip',  # 5
        'right_wrist',
        'left_wrist',
        'right_elbow',
        'left_elbow',
        'right_shoulder',
        'left_shoulder',
        'headtop'
    ]


def get_h36m_joint_names():
    return [
        'hip',  # 0
        'left_hip',  # 1
        'left_knee',  # 2
        'left_ankle',  # 3
        'right_hip',  # 4
        'right_knee',  # 5
        'right_ankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'left_shoulder',  # 11
        'left_elbow',  # 12
        'left_wrist',  # 13
        'right_shoulder',  # 14
        'right_elbow',  # 15
        'right_wrist',  # 16
    ]


def get_ntu_joint_names():
    # from https://arxiv.org/pdf/1905.04757.pdf
    return [
        'pelvis',  # or pelvis?
        'Spine (H36M)',  # Spine (H36M)?
        'neck',
        'headtop',
        'left_shoulder',  # 5
        'left_elbow',
        'left_wrist',
        'left_hand',  # MISSING in H36m_plus
        'right_shoulder',
        'right_elbow',  # 10
        'right_wrist',
        'right_hand',  # MISSING in H36m_plus
        'left_hip',
        'left_knee',
        'left_ankle',  # 15
        'left_foot',
        'right_hip',
        'right_knee',
        'right_ankle',
        'right_foot',  # 20
        'spine ?',
        'left_index',  # 'tip of left hand'
        'left_thumb',
        'right_index',  # 'tip of right hand'
        'right_thumb'
    ]


def get_ntu_skeleton():
    names = get_ntu_joint_names()
    return np.array(
        [
            [
                # right
                [names.index('hip'), names.index('right_hip')],
                [names.index('right_hip'), names.index('right_knee')],
                [names.index('right_knee'), names.index('right_ankle')],
                [names.index('right_foot'), names.index('right_ankle')],
                [names.index('hip'), names.index('Spine (H36M)')],
                [names.index('Spine (H36M)'), names.index('neck')],
                [names.index('neck'), names.index('Head (H36M)')],
                [names.index('neck'), names.index('right_shoulder')],
                [names.index('right_shoulder'), names.index('right_elbow')],
                [names.index('right_elbow'), names.index('right_wrist')],
                [names.index('headtop'), names.index('Head (H36M)')],
                [names.index('right_index'), names.index('right_wrist')],
                [names.index('right_thumb'), names.index('right_wrist')]
            ],
            # left
            [
                [names.index('hip'), names.index('left_hip')],
                [names.index('left_hip'), names.index('left_knee')],
                [names.index('left_knee'), names.index('left_ankle')],
                [names.index('left_foot'), names.index('left_ankle')],
                [names.index('neck'), names.index('left_shoulder')],
                [names.index('left_shoulder'), names.index('left_elbow')],
                [names.index('left_elbow'), names.index('left_wrist')],
                [names.index('left_index'), names.index('left_wrist')],
                [names.index('left_thumb'), names.index('left_wrist')]
            ]
        ]
    )


def get_h36m_plus_skeleton():
    names = get_h36m_plus_joint_names()
    return np.array(
        [
            [
                # right
                [names.index('hip'), names.index('right_hip')],
                [names.index('right_hip'), names.index('right_knee')],
                [names.index('right_knee'), names.index('right_ankle')],
                [names.index('right_foot'), names.index('right_ankle')],
                [names.index('hip'), names.index('Spine (H36M)')],
                [names.index('Spine (H36M)'), names.index('neck')],
                [names.index('neck'), names.index('Head (H36M)')],
                [names.index('neck'), names.index('right_shoulder')],
                [names.index('right_shoulder'), names.index('right_elbow')],
                [names.index('right_elbow'), names.index('right_wrist')],
                [names.index('headtop'), names.index('Head (H36M)')],
                [names.index('right_index'), names.index('right_wrist')],
                [names.index('right_thumb'), names.index('right_wrist')]
            ],
            # left
            [
                [names.index('hip'), names.index('left_hip')],
                [names.index('left_hip'), names.index('left_knee')],
                [names.index('left_knee'), names.index('left_ankle')],
                [names.index('left_foot'), names.index('left_ankle')],
                [names.index('neck'), names.index('left_shoulder')],
                [names.index('left_shoulder'), names.index('left_elbow')],
                [names.index('left_elbow'), names.index('left_wrist')],
                [names.index('left_index'), names.index('left_wrist')],
                [names.index('left_thumb'), names.index('left_wrist')]
            ]
        ]
    )


def update_h36m_with_smplh(h36m, smplh):
    """
    Add pelvis, left/right foot and hans joints
    Args:
        - h36m: [b,17,3]
        - smplh: [b,52,3]
    """
    # smplh of interest
    pelvis = smplh[:, [0]]
    left_foot = smplh[:, [10]]
    right_foot = smplh[:, [11]]
    left_index = smplh[:, [36]]
    left_thumb = smplh[:, [35]]
    right_index = smplh[:, [41]]
    right_thumb = smplh[:, [40]]

    # update h36m
    h36m_plus = torch.cat([h36m, pelvis,
                           left_foot, left_index, left_thumb,
                           right_foot, right_index, right_thumb],
                          1)
    return h36m_plus


def get_h36m_plus_joint_names():
    return [
        'hip',  # 0
        'left_hip',  # 1
        'left_knee',  # 2
        'left_ankle',  # 3
        'right_hip',  # 4
        'right_knee',  # 5
        'right_ankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'left_shoulder',  # 11
        'left_elbow',  # 12
        'left_wrist',  # 13
        'right_shoulder',  # 14
        'right_elbow',  # 15
        'right_wrist',  # 16
        'pelvis',  # 17
        'left_foot',  # 18
        'left_index',  # 19
        'left_thumb',  # 20
        'right_foot',  # 21
        'right_index',  # 22
        'right_thumb'  # 23
    ]


def get_h36m_plus_traversal():
    # bottom left/right
    traversal_bottom_left = ['hip', 'left_hip', 'left_knee', 'left_ankle', 'left_foot']
    parents_bottom_left = ['pelvis', 'hip', 'left_hip', 'left_knee', 'left_ankle']
    traversal_bottom_right = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
    parents_bottom_right = ['hip', 'right_hip', 'right_knee', 'right_ankle']

    # top left/right
    traversal_top_left = ['Spine (H36M)', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_index',
                          'left_thumb', 'Head (H36M)', 'headtop']
    parents_top_left = ['hip', 'Spine (H36M)', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_wrist',
                        'neck', 'Head (H36M)']
    traversal_top_right = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_index', 'right_thumb']
    parents_top_right = ['neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_wrist']

    traversal = traversal_bottom_left + traversal_bottom_right + traversal_top_left + traversal_top_right
    parents = parents_bottom_left + parents_bottom_right + parents_top_left + parents_top_right

    names = get_h36m_plus_joint_names()
    traversal_idx = []
    parents_idx = []
    for i in range(len(traversal)):
        traversal_idx.append(names.index(traversal[i]))
        parents_idx.append(names.index(parents[i]))

    assert len(traversal_idx) == len(parents_idx)

    return traversal_idx, parents_idx


def get_h36m_skeleton():
    return np.array(
        [
            [
                # right
                [0, 4],
                [4, 5],
                [5, 6],
                [0, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [8, 14],
                [14, 15],
                [15, 16]
            ],
            # left
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [8, 11],
                [11, 12],
                [12, 13],
            ]
        ]
    )


def get_dope_skeleton():
    return np.array(
        [
            [
                # right
                [0, 2],
                [2, 4],
                # [4, 10],
                [10, 8],
                [8, 6],
            ],
            # left
            [
                [1, 3],
                [3, 5],
                # [5, 11],
                [11, 9],
                [9, 7],
            ]
        ]
    )


def convert_pose(jts, src, dst):
    """
    Conversion from one format to another one
    :param jts: np.array [n,k,d]
    :return: out: np.array [n,k',d]
    """

    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    list_out = []
    for _, jn in enumerate(dst_names):
        if jn in src_names:
            idx = src_names.index(jn)
            list_out.append(jts[:, idx])
        else:
            if src == 'dope' and dst == 'mupots':
                if jn == 'neck':
                    idx1 = src_names.index('left_shoulder')
                    idx2 = src_names.index('right_shoulder')
                    list_out.append(jts[:, [idx1, idx2]].mean(1))
                elif jn == 'hip':
                    idx1 = src_names.index('left_hip')
                    idx2 = src_names.index('right_hip')
                    list_out.append(jts[:, [idx1, idx2]].mean(1))
                elif jn == 'Spine (H36M)':
                    idx1 = src_names.index('left_hip')
                    idx2 = src_names.index('right_hip')
                    idx3 = src_names.index('left_shoulder')
                    idx4 = src_names.index('right_shoulder')
                    list_out.append(jts[:, [idx1, idx2, idx3, idx4]].mean(1))
                elif jn == 'Head (H36M)':
                    idx1 = src_names.index('left_shoulder')
                    idx2 = src_names.index('right_shoulder')
                    idx3 = src_names.index('headtop')
                    list_out.append(jts[:, [idx1, idx2, idx3]].mean(1))
                else:
                    print(jn)
                    # import ipdb
                    # ipdb.set_trace()
            elif src == 'ntu' and (dst == 'h36m' or dst == 'h36m_plus'):
                if jn == 'Head (H36M)':
                    idx = []
                    idx.append(src_names.index('neck'))
                    idx.append(src_names.index('headtop'))
                    list_out.append(jts[:, idx].mean(1))
                elif jn == 'hip':
                    idx = []
                    idx.append(src_names.index('left_hip'))
                    idx.append(src_names.index('right_hip'))
                    list_out.append(jts[:, idx].mean(1))
                else:
                    print(jn)
            else:
                print(jn)

    out = np.stack(list_out, 1)

    return out


def visu_h36m_pose3d(x, res=1024, lw=2):
    """
    x: tensor of shape [N,13,3]
    """

    bones = get_h36m_skeleton()
    right = bones[0]
    left = bones[1]

    plt.style.use('dark_background')
    fig = Figure((res / 100., res / 100.))
    canvas = FigureCanvas(fig)

    # 3D
    ax = fig.gca(projection='3d')
    for i, pose3d in enumerate(x):
        if isinstance(pose3d, torch.Tensor):
            pose3d = pose3d.detach().cpu().numpy()
        if isinstance(pose3d, np.ndarray):
            pass
        else:
            raise NotImplementedError

        pose3d = np.stack([
            - pose3d[:, 0],
            - pose3d[:, 2],
            - pose3d[:, 1]
        ], 1)

        # draw green lines on the left side
        for i, j in left:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'g', scalex=None, scaley=None, lw=lw)

        # draw blue lines on the right side and center
        for i, j in right:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)

        # red circle for all joints
        ax.scatter3D(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c='red', lw=lw)

    # legend and ticks
    ax.set_aspect('auto')
    # ax.elev = 20  # 45
    # ax.azim = -90
    ax.view_init(15, 45)  # 0 45 90 315
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-12)
    ax.set_ylabel('Y axis', labelpad=-12)
    ax.set_zlabel('Z axis', labelpad=-12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()

    if img.shape[0] != res:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((res, res)))

    return img


def visu_h36m_plus_pose3d(x, res=1024, lw_line=1, lw_dot=0.5, text=None):
    """
    x: tensor of shape [N,13,3]
    """

    bones = get_h36m_plus_skeleton()
    right = bones[0]
    left = bones[1]

    plt.style.use('dark_background')
    fig = Figure((res / 100., res / 100.))
    canvas = FigureCanvas(fig)

    # 3D
    ax = fig.gca(projection='3d')
    for i, pose3d in enumerate(x):
        if isinstance(pose3d, torch.Tensor):
            pose3d = pose3d.detach().cpu().numpy()
        if isinstance(pose3d, np.ndarray):
            pass
        else:
            raise NotImplementedError

        pose3d = np.stack([
            - pose3d[:, 0],
            - pose3d[:, 2],
            - pose3d[:, 1]
        ], 1)

        # draw green lines on the left side
        for i, j in left:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'g', scalex=None, scaley=None, lw=lw_line)

        # draw blue lines on the right side and center
        for i, j in right:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw_line)

        # red circle for all joints
        ax.scatter3D(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c='red', lw=lw_dot)

    # legend and ticks
    ax.set_aspect('auto')
    # ax.elev = 20  # 45
    # ax.azim = -90
    ax.view_init(15, 45)  # 0 45 90 315
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-12)
    ax.set_ylabel('Y axis', labelpad=-12)
    ax.set_zlabel('Z axis', labelpad=-12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()

    img = Image.fromarray(img).resize((res, res))
    if text is not None:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15)
        draw = ImageDraw.Draw(img)
        pad_width = int(0.025 * width)
        draw.text((1.5 * pad_width, 1.5 * pad_width), text, fill=(128, 128, 128), font=font)

    img = np.asarray(img)

    return img


def visu_dope_pose3d(x, res=1024, lw=2):
    """
    x: tensor of shape [N,13,3]
    """

    bones = get_dope_skeleton()
    right = bones[0]
    left = bones[1]

    plt.style.use('dark_background')
    fig = Figure((res / 100., res / 100.))
    canvas = FigureCanvas(fig)

    # 3D
    ax = fig.gca(projection='3d')
    for i, pose3d in enumerate(x):
        if isinstance(pose3d, torch.Tensor):
            pose3d = pose3d.detach().cpu().numpy()
        if isinstance(pose3d, np.ndarray):
            pass
        else:
            raise NotImplementedError

        pose3d = np.stack([
            - pose3d[:, 0],
            - pose3d[:, 2],
            - pose3d[:, 1]
        ], 1)

        # draw green lines on the left side
        for i, j in left:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'g', scalex=None, scaley=None, lw=lw)

        # draw blue lines on the right side and center
        for i, j in right:
            x = [pose3d[i, 0], pose3d[j, 0]]
            y = [pose3d[i, 1], pose3d[j, 1]]
            z = [pose3d[i, 2], pose3d[j, 2]]
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)

        # l/r-hip to hip
        hip = (pose3d[4] + pose3d[5]) / 2.
        for i, col in zip([4, 5], ['b', 'g']):
            ax.plot([pose3d[i, 0], hip[0]], [pose3d[i, 1], hip[1]], [pose3d[i, 2], hip[2]],
                    col, scalex=None, scaley=None, lw=lw)

        # l/r-should to neck
        neck = (pose3d[10] + pose3d[11]) / 2.
        for i, col in zip([10, 11], ['b', 'g']):
            ax.plot([pose3d[i, 0], neck[0]], [pose3d[i, 1], neck[1]], [pose3d[i, 2], neck[2]],
                    col, scalex=None, scaley=None, lw=lw)

        # middle and head
        ax.plot([hip[0], neck[0]], [hip[1], neck[1]], [hip[2], neck[2]], 'b', scalex=None, scaley=None, lw=lw)
        ax.plot([pose3d[12, 0], neck[0]], [pose3d[12, 1], neck[1]], [pose3d[12, 2], neck[2]], 'b', scalex=None,
                scaley=None, lw=lw)

        # red circle for all joints
        ax.scatter3D(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c='red', lw=2)

    # legend and ticks
    ax.set_aspect('auto')
    # ax.elev = 20  # 45
    # ax.azim = -90
    ax.view_init(15, 45)  # 0 45 90 315
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-12)
    ax.set_ylabel('Y axis', labelpad=-12)
    ax.set_zlabel('Z axis', labelpad=-12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()

    if img.shape[0] != res:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((res, res)))

    return img


def matching_poses(x, y, pa=True):
    """
    Find the matching between x and y (we must find a match for each y using x samples)
    Cost computed using the MPJPE
    Args:
         - x: np.array [N,K,3]
         - y: np.array [D,K,3]
    Return:
        - y_hat: np.array [D,K,3]
        - ids: list containing id of x for each y
    """
    from threed.geometry import compute_similarity_transform_batch

    # y = np.random.randn(3, 5, 3)
    # x = np.random.randn(2, 5, 3)

    # Compute the cost for each x vs all y
    cost_matrix = np.empty((y.shape[0], x.shape[0]))
    for i in range(y.shape[0]):
        if pa:
            y_ = compute_similarity_transform_batch(y[[i]].repeat(x.shape[0], 0), x)
            cost_matrix[i] = np.sqrt(np.power(y_ - x, 2).sum(-1)).mean(1)
        else:
            cost_matrix[i] = np.sqrt(np.power(y[[i]] - x, 2).sum(-1)).mean(1)

    # Hungarian algo
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Ids for matching and arrange y_hat
    ids = []
    y_hat = []
    for i in range(y.shape[0]):
        if i in row_ind:
            j = int(col_ind[np.where(row_ind == i)])
            ids.append(j)
            y_hat.append(x[j])
        else:
            ids.append(None)
            y_hat.append(None)

    return ids, y_hat


def preprocess_skeleton(pose, center_joint=[17], xaxis=[11, 14], yaxis=[7, 0], iter=5, sanity_check=True,
                        norm_x_axis=True, norm_y_axis=True):
    """
    Preprocess skeleton such that we disentangle the root orientation and the relative pose
    Following code from https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/preprocess.py
    Default values are for h36m_plus skeleton (center=hip, xaxis=left_shoulder/right_shoulder, yaxis=spine/hip
    Args:
        - pose: [t,k,3] np.array
        - center_joint: list
        - xaxis: list
        - yaxis: list
        - iter: int
    Return:
        - pose_rel: [t,k,3] np.array
        - pose_center: [t,3] np.array
        - matrix: [t,3,3] np.array
    """
    pose_rel = pose.copy()

    # Sub the center joint (pelvis 17)
    pose_center = pose_rel[:, center_joint].mean(1, keepdims=True)
    pose_rel = pose_rel - pose_center

    list_matrix = []
    list_diff = []
    for t in range(pose_rel.shape[0]):

        matrix = []
        inv_matrix = []
        for _ in range(iter):
            # parallel the bone between hip(jpt 0) and spine(jpt 7) to the Y axis
            if norm_y_axis:
                joint_bottom = pose_rel[t, yaxis[0]]
                joint_top = pose_rel[t, yaxis[1]]
                axis = np.cross(joint_top - joint_bottom, [0, 1, 0]).astype(np.float32)
                angle = angle_between(joint_top - joint_bottom, [0, 1, 0]).astype(np.float32)
                matrix_x = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_x.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_x)

            # parallel the bone between right_shoulder(jpt 0) and left_shoulder(jpt 7) to the X axis
            if norm_x_axis:
                joint_rshoulder = pose_rel[t, xaxis[0]]
                joint_lshoulder = pose_rel[t, xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0]).astype(np.float32)
                angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0]).astype(np.float32)
                matrix_y = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_y.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_y)

        # compute the center orient rotmat
        matrix.reverse()
        mat = matrix[0]
        for x in matrix[1:]:
            mat = mat @ x
        list_matrix.append(mat)

        if sanity_check:
            # sanity check for computing the inverse matrix step by step
            matrix.reverse()
            inv_mat = np.linalg.inv(matrix[0])
            for x in matrix[1:]:
                inv_mat = inv_mat @ np.linalg.inv(x)
            pose_centered_t_bis = (inv_mat.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            pose_centered_t = pose[t] - pose_center[t]
            err = np.abs(pose_centered_t_bis - pose_centered_t).sum()
            print(err)
            assert err < 1e-5
            inv_matrix.append(inv_mat)

            # sanity check for matrix multiplication
            pose_rel_bis = pose.copy() - pose_center
            pose_rel_t_bis = (mat.reshape(1, 3, 3) @ pose_rel_bis[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_rel_t_bis - pose_rel[t]).sum()
            print(err)
            assert err < 1e-5

            # inv bis
            inv_mat_bis = np.linalg.inv(mat)
            pose_centered_t_bis_bis = (inv_mat_bis.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_centered_t_bis_bis - pose_centered_t).sum()
            print(err)
            assert err < 1e-5

    orient_center = np.stack(list_matrix)
    return pose_rel, pose_center.reshape(-1, 3), orient_center


def unpreprocess_skeleton(pose_rel, pose_center, orient_center):
    """
    Move back to standard pose from root and relative information
    Args:
        - pose_rel: [t,k,3] np.array
        - pose_center: [t,3] np.array
        - orient_center: [t,3,3] np.array
    Return:
        - pose: [t,k,3] np.array
    """

    pose = []
    for t, mat in enumerate(orient_center):
        inv_mat = np.linalg.inv(mat)
        pose_ = (inv_mat.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
        pose.append(pose_)
    pose = np.stack(pose)

    # Center
    pose = pose + pose_center.reshape(-1, 1, 3)

    return pose


def get_smplx_pose(full_pose):
    """
    Args:
        - full_pose: [t,168]
    """
    assert full_pose.dim() == 2
    assert full_pose.size(1) == 168
    return {'global_orient': full_pose[:, :3],
            'body_pose': full_pose[:, 3:66],
            'jaw_pose': full_pose[:, 66:69],
            'leye_pose': full_pose[:, 69:72],
            'reye_pose': full_pose[:, 72:75],
            'left_hand_pose': full_pose[:, 75:120],
            'right_hand_pose': full_pose[:, 120:165],
            'transl': full_pose[:, 165:],
            }


def get_smpl_pose(full_pose):
    """
    Args:
        - full_pose: [t,75]
    """
    assert full_pose.dim() == 2
    assert full_pose.size(1) == 75
    return {'global_orient': full_pose[:, :3],
            'body_pose': full_pose[:, 3:72],
            'transl': full_pose[:, 72:],
            }

def get_smplh_pose(full_pose):
    """
    Args:
        - full_pose: [t,159]
    """
    assert full_pose.dim() == 2
    assert full_pose.size(1) == 159
    return {'global_orient': full_pose[:, :3],
            'body_pose': full_pose[:, 3:66],
            'left_hand_pose': full_pose[:, 66:111],
            'right_hand_pose': full_pose[:, 111:156],
            'transl': full_pose[:, 156:],
            }


def get_smplx_flip_params():
    root_body_flip_perm = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
    jaw_eye_flip_perm = [22, 24, 23]
    hand_flip_perm = list(range(25 + 15, 25 + 2 * 15)) + list(range(25, 25 + 15))
    trans_flip_perm = [55]
    ll = root_body_flip_perm + jaw_eye_flip_perm + hand_flip_perm + trans_flip_perm
    flip_perm = []
    flip_inv = []
    for i in ll:
        flip_perm.append(3 * i)
        flip_perm.append(3 * i + 1)
        flip_perm.append(3 * i + 2)
        if i == 55:
            # trans only
            flip_inv.extend([-1., 1., 1.])
        else:
            # smplx jts
            flip_inv.extend([1., -1., -1.])
    flip_inv = torch.Tensor(flip_inv).reshape(1, -1)
    return flip_perm, flip_inv

def get_smpl_flip_params():
    pose_flip_perm = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    trans_flip_perm = [24]
    ll = pose_flip_perm + trans_flip_perm
    flip_perm = []
    flip_inv = []
    for i in ll:
        flip_perm.append(3 * i)
        flip_perm.append(3 * i + 1)
        flip_perm.append(3 * i + 2)
        if i == 24:
            # trans only
            flip_inv.extend([-1., 1., 1.])
        else:
            # smplx jts
            flip_inv.extend([1., -1., -1.])
    flip_inv = torch.Tensor(flip_inv).reshape(1, -1)
    
    return flip_perm, flip_inv


if __name__ == "__main__":
    import sys

    exec(sys.argv[1])
