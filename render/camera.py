from typing import List, Optional

import torch  # noqa
from torch import Tensor
import pytorch3d.transforms as t3d

import pyrender


def opencv_camera_to_opengl_proj(K: Tensor, size: List[int], znear: float = 0.1, zfar: float = 1000.0):
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=znear, zfar=zfar)
    proj = camera.get_projection_matrix(size[1], size[0])
    return proj


def opencv_camera_to_opengl_proj_extrinsic(K: Tensor, size: List[int], R: Tensor, C: Tensor, bounds: Tensor):
    """

    Args:
        K (Tensor):
        size (List[int]): [H W]
        R:
        C:
        bounds: [[minx, miny, minz], [maxx, maxy, maxz]]

    Returns:

    """
    # project the corners of the bounding box to the camera space, and get znear and zfar
    # get the 8 corners of the bounding box
    corners = torch.zeros(8, 3)
    corners[0, :] = bounds[0, :]
    corners[1, :] = bounds[1, :]
    corners[2, :] = bounds[0, :]
    corners[2, 0] = bounds[1, 0]
    corners[3, :] = bounds[0, :]
    corners[3, 1] = bounds[1, 1]
    corners[4, :] = bounds[1, :]
    corners[4, 2] = bounds[0, 2]
    corners[5, :] = bounds[1, :]
    corners[5, 0] = bounds[0, 0]
    corners[6, :] = bounds[1, :]
    corners[6, 1] = bounds[0, 1]
    corners[7, :] = bounds[1, :]
    corners[7, 2] = bounds[0, 2]

    # Xc = R * (X - C)
    corners = corners - C
    corners = torch.matmul(R, corners.T).T

    # get the znear and zfar
    znear = torch.min(corners[:, 2]).item()
    zfar = torch.max(corners[:, 2]).item()

    return opencv_camera_to_opengl_proj(K, size, znear, zfar)


def opencv_extrinsic_to_opengl_modelview(R: Tensor, C: Tensor, model_matrix: Optional[Tensor] = None):
    """
    For world (w), model (m), camera (c) coordinates:
    X_m = model_matrix @ X_w
    X_c = view @ model @ X_w
    X_c = Rx @ R @ (X_w - C) = modelview_matrix @ X_w
    modelview_matrix = [RxR -RxRC]
    If the model_matrix is inputted, the model input for subsequent rendering will be in the model coordinate
    system (X_m), but the R, C matrices are still in the camera's absolute coordinate system.
    X_c = modelview_matrix @ model_matrix^-1 @ X_m
    Args:
        R (Tensor): Rotation matrix 3x3
        C (Tensor): Camera center 3x1 in world coordinate system
        model_matrix (Tensor): 4X4 matrix to transform the mesh to the unit cube, unit = xform_bounds * mesh

    Returns:
        mv (Tensor): X_c = mv @ X_w or X_c = mv @ X_m (if input model_matrix)
    """
    if model_matrix is None:
        model_matrix = torch.eye(4, dtype=torch.float32)

    Rx = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    # X_c = view @ X_w
    # X_c = Rx @ R @ (X - C), Opencv 相机坐标系到 OpenGL 相机坐标系，差一个 X 轴的旋转
    view_matrix = torch.eye(4, dtype=torch.float32)
    view_matrix[:3, :3] = Rx @ R
    view_matrix[:3, 3] = -(Rx @ R) @ C

    # X_c = mv @ X_unit
    mv = view_matrix @ torch.inverse(model_matrix)
    return mv


def angle_axis_to_rotation(r: Tensor):
    """
    Convert angle axis to rotation matrix
    Args:
        r (Tensor): angle axis in shape (B, 3), the norm is angle in radians

    Returns:
        R (Tensor): rotation matrix in shape (B, 3, 3)
    """
    return t3d.axis_angle_to_matrix(r)


def dof6_to_matrix(extrinsic: Tensor):
    """
    Convert the six dof transformation to 4x4 matrix
    Args:
        extrinsic (Tensor): extrinsic matrix in shape (B, 6), the first 3 columns are angle axis, the last 3 columns
        are translation

    Returns:
        view (Tensor): view matrix in shape (B, 4, 4)

    """
    batch = extrinsic.shape[0]
    r = extrinsic[:, :3]
    t = extrinsic[:, 3:6]
    R = angle_axis_to_rotation(r)
    view = torch.zeros((batch, 4, 4), dtype=torch.float32, device=extrinsic.device)
    view[:, :3, :3] = R
    view[:, :3, 3] = t
    view[:, 3, 3] = 1.0
    return view
