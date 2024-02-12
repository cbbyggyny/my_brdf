import os
import glob
import json #处理JSON（JavaScript Object Notation）格式的数据。JSON是一种轻量级的数据交换格式，常用于在不同系统之间传输和存储数据
import yaml
import torch
from torch import Tensor
import pyrender
import numpy as np
import imageio.v2 as iio

from render import util

from tqdm import tqdm
from .dataset import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from render.transforms import CustomCropAndResizeTransformTensor, compute_camera_matrix
from render.camera import opencv_extrinsic_to_opengl_modelview
###############################################################################
# NERF image based dataset (synthetic)
###############################################################################


class DatasetBRDF(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
        self.n_images = len(self.cfg['frames'])

        self._names = list(self.cfg['frames'].keys())
        self._names = sorted(self._names)
        self._xform_bounds = torch.tensor(self.cfg['model_matrix'])
        self._K = torch.tensor(self.cfg['camera']['camera_matrix'])
        self._width = self.cfg['camera']['width']
        self._height = self.cfg['camera']['height']

        self.resolution = (self._height, self._width)
        self.aspect = self._width / self._height

        if self.FLAGS.local_rank == 0:
            print("Dataset: %d origin images with shape [%d, %d]" % (self.n_images, self._width, self._height))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in tqdm(range(self.n_images)):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        name = self._names[idx]
        frame = self.cfg['frames'][name]

        R = torch.tensor(frame['R'])
        C = torch.tensor(frame['C'])
        bbox = torch.tensor(frame['bbox'])

        img = DatasetBRDF._load_img(self, frame)

        #根据包围盒裁剪影像至训练分辨率
        res = self.FLAGS.train_res[0]
        bbox_image = torch.permute(img, (2, 0, 1))
        xmin, ymin, xmax, ymax = bbox
        scale_range: tuple = (0.08, 1.0)
        scale = (scale_range[0] * scale_range[0], scale_range[1] * scale_range[1])
        i, j, h, w = transforms.RandomResizedCrop.get_params(bbox_image, scale=scale, ratio=(1.0, 1.0))
        bbox_image = TF.resized_crop(bbox_image, i, j, h, w, [res, res],
                                     transforms.InterpolationMode.BILINEAR, antialias=True)
        mask = bbox_image[3, :, :] < 0.99
        bbox_image[:, mask] = 0.0

        scale_x = res / w
        scale_y = res / h

        # 新影像在原始影像的偏移
        xmin = xmin + j
        ymin = ymin + i

        # 新影像，在缩放后影像的偏移
        camera_matrix = compute_camera_matrix(xmin, ymin, self._width, self._height, self._K[0, 0],
                                              self._K[1, 1], scale_x=scale_x, scale_y=scale_y)

        # To [H W C] to make it compatible with the nvdiffrast renderer
        img = bbox_image.permute(1, 2, 0)

        K = camera_matrix

        mv = opencv_extrinsic_to_opengl_modelview(R, C, self._xform_bounds)
        # 变换后的相机位置，相机坐标系下的 [0,0,0,1] = mv * pos
        campos = torch.inverse(mv)
        # 最后一列
        campos = campos[:3, 3]

        # znear, zfar = self.compute_range(mv)
        znear, zfar = 0.1, 1000.0

        proj = DatasetBRDF.compute_proj(K, img.shape[1], img.shape[0], znear, zfar)
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def _load_img(self, frame):
        bbox = torch.tensor(frame['bbox'])
        path = os.path.join(self.base_dir,'images', frame['path'])
        image = iio.imread(path)
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        if image.dtype != np.float32: # LDR image
            image = torch.tensor(image / 255, dtype=torch.float32)
            image[..., 0:3] = util.srgb_to_rgb(image[..., 0:3])
        else:
            image = torch.tensor(image, dtype=torch.float32)
        return image
        
    def compute_proj(K: Tensor, width: int, height: int, znear: float = 0.1, zfar: float = 1000.0):
        """
            计算 MVP 矩阵
            Args:
                K: 相机的 K 矩阵
                width: 相机的影像宽度
                height: 相机的影像高度
                znear:
                zfar:

            Returns:

        """
            # compute the mvp matrix after update the K matrix
        icamera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=znear, zfar=zfar)
        proj = icamera.get_projection_matrix(width=width, height=height)
        proj = torch.tensor(proj, dtype=torch.float32)

            # nvdiffrast 的 proj 矩阵和 OpenGL 的第二行相差 -1
            # s.a. https://github.com/NVlabs/nvdiffrec/blob/e7f2181b8a60eb8fedcdb4ad4d05bff3c0cf9bc1/render/util.py#L187-L193
            # http://www.songho.ca/opengl/gl_projectionmatrix.html
        proj[1] = -proj[1]
        return proj

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []   #定义了一个空列表 img，用于存储图像数据
        fovy     = 2 * np.arctan(0.5 * 36 / self._K[0, 0])

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
