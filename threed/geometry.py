# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import numpy as np
from utils.constants import FOCAL_LENGTH, IMG_SIZE, IMG_RES_SPIN, FOCAL_LENGTH_SPIN
import roma
import math


def world2cam(x, translation, rotation):
    """
    Move from world coordintae system to camera coordinate system
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
    """
    x = torch.einsum('bij,bkj->bki', rotation, x)
    x = x + translation.unsqueeze(1)
    return x


def cam2world(x, translation, rotation):
    """
    Move from camera coordintae system to world coordinate system
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
    """
    x = x - translation.unsqueeze(1)
    x = torch.einsum('bij,bkj->bki', torch.inverse(rotation), x)
    return x


def perspective_projection(points, camera_center, f_x, f_y):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Input:
        points (bs, N, 3): 3D points
        camera_center (bs, 2): Camera center
        f_x, f_y (int): Focal length for z and y axis
    """
    batch_size = points.shape[0]

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Camera intrinsic params
    K = torch.zeros([batch_size, 3, 3], device=points.device)  # (bs, 3, 3)
    K[:, 0, 0] = f_x
    K[:, 1, 1] = f_y
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)  # (bs, N, 3)

    return projected_points[:, :, :2]


def inverse_perspective_projection(points, camera_center, f_x, f_y, distance):
    """
    This function computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
        distance (bs, N, 1): distance in the 3D world
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = f_x
    K[:, 1, 1] = f_y
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum('bij,bkj->bki', torch.inverse(K), points)

    # Apply perspective distortion
    points = points * distance

    return points


def estimate_translation_batch(pose3d, pose2d, n_jts_to_take=50):
    """
    Batch mode of estimate_translation_np and working with torch.tensor
    Args:
         - pose3d: : [b,k,3]
         - pose2d: : [b,k,2]
    Return:
        - camera: [b,3]
    """
    batch_size = pose2d.size(0)
    n_jts = pose3d.size(1)

    list_cam = []
    for i in range(batch_size):
        idx = torch.randperm(n_jts)[:n_jts_to_take]  # for fast estimation
        cam = estimate_translation_np(pose3d[i, idx].cpu().numpy(), pose2d[i, idx].cpu().numpy())
        list_cam.append(cam)
    return torch.from_numpy(np.stack(list_cam)).float()


def estimate_translation_np(pose3d, pose2d, f_x=FOCAL_LENGTH, f_y=FOCAL_LENGTH, c_x=IMG_SIZE / 2., c_y=IMG_SIZE / 2.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        pose_3d: (25, 3) 3D joint locations
        pose_2d: (25, 2) 2D joint locations - in range(0,1)
    Returns:
        (3,) camera translation vector
    """
    # pose2d = pose2d * IMG_SIZE
    pose2d = pose2d

    joints_conf = None
    if joints_conf is None:
        joints_conf = np.ones_like(pose2d[:, 0])

    num_joints = pose3d.shape[0]
    # focal length
    f = np.array([f_x, f_y])
    # optical center
    center = np.array([c_x, c_y])

    # transformations
    Z = np.reshape(np.tile(pose3d[:, 2], (2, 1)).T, -1)
    XY = np.reshape(pose3d[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
                  O - np.reshape(pose2d, -1)]).T
    c = (np.reshape(pose2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def get_squared_bbox(j2d, factor=1.0):
    """
    j2d is a np.array of shape [N,2]
    and return a tuple such as (left, top, right, bottom), (c_x, c_y, scale)
    """
    c_x = int((j2d[:, 0].min() + j2d[:, 0].max()) / 2.)
    c_y = int((j2d[:, 1].min() + j2d[:, 1].max()) / 2.)
    scale_x = int(factor * (j2d[:, 0].max() - j2d[:, 0].min()))
    scale_y = int(factor * (j2d[:, 1].max() - j2d[:, 1].min()))
    scale = max([scale_x, scale_y])
    left = int(c_x - scale / 2.)
    right = left + scale
    top = int(c_y - scale / 2.)
    bottom = top + scale
    return (left, top, right, bottom), (c_x, c_y, scale)


def get_bbox(j2d, factor=1.0):
    """
    j2d is a np.array of shape [N,2]
    and return a tuple such as (left, top, right, bottom), (c_x, c_y, scale)
    """
    c_x = int((j2d[:, 0].min() + j2d[:, 0].max()) / 2.)
    c_y = int((j2d[:, 1].min() + j2d[:, 1].max()) / 2.)
    scale_x = int(factor * (j2d[:, 0].max() - j2d[:, 0].min()))
    scale_y = int(factor * (j2d[:, 1].max() - j2d[:, 1].min()))
    left = int(c_x - scale_x / 2.)
    right = left + scale_x
    top = int(c_y - scale_y / 2.)
    bottom = top + scale_y
    return (left, top, right, bottom), (c_x, c_y, scale_x, scale_y)


def find_best_camera(joints, camera_translation_init=None, factor=1.1,
                     f_x=FOCAL_LENGTH, f_y=FOCAL_LENGTH, c_x=IMG_SIZE / 2., c_y=IMG_SIZE / 2., img_size=IMG_SIZE,
                     rotation=None):
    """
    joints: [T,K,3]
    camera_translation_init: [3]
    """
    batch_size = joints.size(0)

    if camera_translation_init is None:
        camera_translation_init = torch.as_tensor([[0.0, 0.0, 2 * f_x / img_size]]).repeat(batch_size, 1)

    if rotation is None:
        rotation = torch.eye(3).type_as(joints).unsqueeze(0).repeat(batch_size, 1, 1)

    camera_center = torch.as_tensor([[c_x, c_y]]).repeat(batch_size, 1)
    keypoints = perspective_projection(world2cam(joints, camera_translation_init, rotation), camera_center, f_x, f_y)

    list_camera_translation = []
    for i in range(batch_size):
        (left, top, right, bottom), (c_x, c_y, scale) = get_squared_bbox(keypoints[i].clone().numpy(), factor)
        keypoints_rescaled = torch.stack([((keypoints[i, :, 0] - left) / scale), ((keypoints[i, :, 1] - top) / scale)],
                                         1)
        keypoints_rescaled *= img_size
        camera_translation = estimate_translation_np(joints[i].numpy(), keypoints_rescaled.numpy(),
                                                     f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y
                                                     )
        list_camera_translation.append(camera_translation)
    camera_translation = torch.from_numpy(np.stack(list_camera_translation)).float()
    return camera_translation  # [T,3]


def find_best_camera_for_video(joints, camera_translation_init=None, factor=1.3, n_jts_to_take=50,
                               f_x=FOCAL_LENGTH, f_y=FOCAL_LENGTH,
                               c_x=IMG_SIZE / 2., c_y=IMG_SIZE / 2., img_size=IMG_SIZE, rotation=None):
    """
    joints: [T,K,3]
    camera_translation_init: [3]
    """

    batch_size = joints.size(0)

    if camera_translation_init is None:
        # camera_translation_init = torch.as_tensor([[0.0, 0.0, 2 * f_x / img_size]]).repeat(batch_size, 1)
        # move the cam way to make sure we can project the entire video in the 2d plane
        camera_translation_init = torch.as_tensor([[0.0, 0.0, 5 * 2 * f_x / img_size]]).repeat(batch_size, 1)

    if rotation is None:
        rotation = torch.eye(3).type_as(joints).unsqueeze(0).repeat(batch_size, 1, 1)

    camera_center = torch.as_tensor([[c_x, c_y]]).repeat(batch_size, 1)
    keypoints = perspective_projection(world2cam(joints, camera_translation_init, rotation), camera_center, f_x, f_y)

    (left, top, right, bottom), (c_x, c_y, scale) = get_squared_bbox(keypoints.flatten(0, 1).numpy(), factor=factor)
    keypoints_rescaled = torch.stack([((keypoints[..., 0] - left) / scale), ((keypoints[..., 1] - top) / scale)], -1)
    keypoints_rescaled *= img_size
    idx = torch.randperm(joints.size(0) * joints.size(1))[:n_jts_to_take]
    camera_translation = estimate_translation_np(joints.flatten(0, 1)[idx].numpy(),
                                                 keypoints_rescaled.flatten(0, 1)[idx].numpy(),
                                                 f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

    return torch.from_numpy(camera_translation).unsqueeze(0).float()


def compute_similarity_transform(S1, S2, return_transfom=False):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    if return_transfom:
        return S1_hat, (scale, R, t)

    return S1_hat


def compute_similarity_transform_batch(S1, S2, return_R=False):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    R = np.zeros((S1.shape[0], 3, 3))
    for i in range(S1.shape[0]):
        if return_R:
            S1_hat[i], R[i] = compute_similarity_transform(S1[i], S2[i])
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])

    if return_R:
        return S1_hat, R
    return S1_hat


def estimate_video_camera_params_wo_trans(jts, trans, factor=1.1, end_context=10000):
    """
    TODO this function needs to be updated
    Given joints w/ trans and the actual trans we estimate the overall action camera params
    Args:
        - jts: [B,T,K,3] - trans is not included into trans
        - trans: [B,T,3]
        - end_context: maximum timestep to tke into account for estimating the initial camera with trans
    Return:
        - cameras: [B,T,3]
    """
    batch_size = jts.size(0)

    jts_trans = jts + trans.unsqueeze(2)

    list_cameras = []

    for i in range(batch_size):
        # Find best cameras for jts+trans
        camera_trans = find_best_camera_for_video(jts_trans[i, :end_context], factor=factor)

        # Project jts+trans into 2d
        kps = perspective_projection(jts_trans[i], camera_trans.repeat(jts_trans.size(1), 1))  # range 0-1 - [t,k,3]

        # Find cameras for jts using 2d from jts+trans
        cameras = estimate_translation_batch(jts[i], kps)
        list_cameras.append(cameras)

    cameras = torch.stack(list_cameras).float()
    return cameras


def project_SPIN_jts_into_frame(jts, cam, bbox):
    """
    Args:
        - jts: [b,k,3]
        - cam: [b,3]
        - bbox: [b,4] in x1y1x2y2 format
    Return:
        - kps: [b,k,2]
    """

    batch_size = jts.size(0)

    # project into SPIN bbox
    kps = perspective_projection(world2cam(jts, cam, torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)),
                                 torch.Tensor([[IMG_RES_SPIN / 2., IMG_RES_SPIN / 2.]]).repeat(batch_size, 1),
                                 FOCAL_LENGTH_SPIN, FOCAL_LENGTH_SPIN)

    # move to real crop size
    crop_size_x, crop_size_y = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    scale_x = crop_size_x / IMG_RES_SPIN
    scale_y = crop_size_y / IMG_RES_SPIN
    kps[..., 0] *= scale_x
    kps[..., 1] *= scale_y

    # move to full image space
    kps[..., 0] += bbox[:, 0]
    kps[..., 1] += bbox[:, 1]

    return kps


def estimate_distance_from_smplh(jts, kps, f_x, f_y, central_joint='pelvis',
                                 joints_to_keep=['head',
                                                 'left_shoulder', 'right_shoulder',
                                                 'left_ankle', 'right_ankle',
                                                 ]):
    """
    Distance estimation following the fornulae: d = (X . f) / x , where X=3D and x=2D
    Compute only estimated distance between spine and certain joints
    Input:
         - jts: [52,3] np.array
         - kps: [52,2] np.array
         - f_x, f_y: int
    Return:
        - dist: float
    """
    from threed.skeleton import get_dict_bodymodel_joint_names
    my_map = get_dict_bodymodel_joint_names()
    inv_map = {v: k for k, v in my_map.items()}

    i = inv_map[central_joint]
    list_j = [inv_map[x] for x in joints_to_keep]
    #     list_j = [x for x in range(1,52)]

    list_distance = []

    for j in list_j:
        for k, f in enumerate([f_x, f_y]):
            x = np.abs(kps[i, k] - kps[j, k])
            X = np.abs(jts[i, k] - jts[j, k])
            distance = (X * f) / x
            list_distance.append(distance)
    return np.median(np.asarray(list_distance))


def estimate_distance_from_h36m(jts, kps, f_x, f_y):
    """
    Distance estimation following the fornulae: d = (X . f) / x , where X=3D and x=2D
    Only compute distance of the hip (with left and righ hip)
    Input:
         - jts: [17,3] np.array
         - kps: [17,2] np.array
         - f_x, f_y: int
    Return:
        - dist: float
    """
    #     i = 1 # left hip
    #     j = 4 # right hip
    list_i = list_j = list(range(17))
    #     list_i = [1]
    #     list_j = [4]

    list_distance = []
    for i in list_i:
        for j in list_j:
            if i != j:
                for k, f in enumerate([f_x, f_y]):
                    x = np.abs(kps[i, k] - kps[j, k])
                    X = np.abs(jts[i, k] - jts[j, k])
                    distance = (X * f) / x
                    list_distance.append(distance)
    return np.median(np.asarray(list_distance))


def project_SPIN_into_3d(pose, shape, cam, bbox, camera_center, f_x, f_y, bodymodel, J_regressor=None):
    """
    Step by step:
        - 1) Run bodymodel for extracting jts and vertices
        - 2) Project jts into the frame
        - 3) Estimate distance using the projected jts and the predicted shape
        - 4) Inverse projection of the human mesh into the 3d camera system (pelvis only)
        - 5) Adjust the entire human mesh according to the pelvis

    Args:
        - pose: tensor [24,3]
        - shape: tensor [10]
        - cam: tensor [3]
        - bbox: tensor [4] in x1y1x2y2 format
        - camera_center: tensor [2]
        - f_x, f_y: float
        - bodymodel: Bodymodel
    Return:
        - loc: tensor [3] - 3d location of the human in the scene in camera coordinate system - pelvis coord
    """

    # 1
    body = bodymodel(root_orient=pose[:1],
                     pose_body=pose[1:].flatten().unsqueeze(0)[:, :63],
                     betas=shape.unsqueeze(0))

    # 2-3 from h36m joints
    # jts = torch.matmul(J_regressor.unsqueeze(0), body.v)
    # kps = project_SPIN_jts_into_frame(jts, cam.unsqueeze(0), bbox.unsqueeze(0))
    # distance = estimate_distance_from_h36m(jts[0].numpy(), kps[0].numpy(), f_x, f_y)

    # 2-3 from smpl joints
    jts = body.Jtr
    kps = project_SPIN_jts_into_frame(jts, cam.unsqueeze(0), bbox.unsqueeze(0))
    distance = estimate_distance_from_smplh(jts[0].numpy(), kps[0].numpy(), f_x, f_y)

    # 4
    distance = torch.Tensor([[distance]])
    pelvis = inverse_perspective_projection(kps[:, :1], camera_center.unsqueeze(0), f_x, f_y, distance)

    return pelvis[0, 0]


def data_augment_root_orient(root_orient, trans,tilt, roll, ry):
    # Apply it on root_orient and trans
    rotation = roma.rotvec_composition(
        [torch.Tensor([tilt, 0., 0.]), torch.Tensor([0., ry, 0.]), torch.Tensor([0., 0., roll])]).type_as(root_orient)
    lll = list(root_orient.shape[:-1])
    lll.reverse()
    for x in lll:
        rotation = rotation.unsqueeze(0).repeat_interleave(x, 0)

    root_orient = roma.rotvec_composition([rotation, root_orient])
    rotation = roma.rotvec_to_rotmat(rotation)
    trans = None if trans is None else torch.matmul(rotation, trans.unsqueeze(-1))[..., 0]

    # Apply on jts
    # rotmat = roma.rotvec_to_rotmat(rotation.unsqueeze(-2))
    # jts_ = jts.unsqueeze(-1)
    # jts = torch.matmul(rotmat, jts_).squeeze(-1)

    return root_orient, trans, rotation


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
