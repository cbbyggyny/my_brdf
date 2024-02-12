import torch  # noqa
import torchvision.transforms.functional as TF
from torchvision import transforms


def compute_camera_matrix(xmin: int, ymin: int, width: int, height: int, fx: float, fy: float, scale_x=1.0,
                          scale_y=1.0):
    """
    根据影像的平移、缩放计算裁剪影像的相机矩阵
    Args:
        xmin: 裁剪影像在原始影像起点的 x 坐标
        ymin: 裁剪影像在原始影像起点的 y 坐标
        width: 原始影像宽度
        height: 原始影像高度
        fx: 原始影像相机主距
        fy: 原始影像相机主距
        scale_x: 新影像与原始影像的缩放 new_width = width * scale_x
        scale_y: 新影像与原始影像的缩放 new_height = height * scale_y

    Returns:
        camera_matrix: 新影像的相机矩阵

    """
    # 新影像，在缩放后影像的偏移
    xmin = (xmin + 0.5) * scale_x - 0.5
    ymin = (ymin + 0.5) * scale_y - 0.5

    # 缩放后影像大小
    image_width = width * scale_x
    image_height = height * scale_y

    # Update the camera matrix
    camera_matrix = torch.eye(3, dtype=torch.float32)
    camera_matrix[0, 2] = (image_width - 1) / 2.0 - xmin
    camera_matrix[1, 2] = (image_height - 1) / 2.0 - ymin
    camera_matrix[0, 0] = fx * scale_x
    camera_matrix[1, 1] = fy * scale_y
    return camera_matrix


class CustomCropAndResizeTransformTensor:
    def __init__(self, image_size: tuple, patch_size: int = 512, scale_range: tuple = (0.08, 1.0)):
        """
        从给定的影像包围盒中，随机裁剪并缩放到 patch_size，并且更新相机矩阵
        Args:
            patch_size (int): patch size
        """
        self.patch_size = patch_size
        self.scale = (scale_range[0] * scale_range[0], scale_range[1] * scale_range[1])
        self.image_size = image_size

    def __call__(self, sample):
        # image is cropped by bbox and saved as RGBA (H, W, C) 8 bit LDR
        bbox_image, camera_matrix, bbox = sample['image'], sample['K'], sample['bbox']

        # torchvision needs [C H W]
        bbox_image = torch.permute(bbox_image, (2, 0, 1))
        image_height, image_width = self.image_size[1], self.image_size[0]

        # Extract bounding box and patch size
        xmin, ymin, xmax, ymax = bbox

        # Crop the image using the bounding box
        i, j, h, w = transforms.RandomResizedCrop.get_params(bbox_image, scale=self.scale, ratio=(1.0, 1.0))
        bbox_image = TF.resized_crop(bbox_image, i, j, h, w, [self.patch_size, self.patch_size],
                                     transforms.InterpolationMode.BILINEAR, antialias=True)

        # mask may be blurred
        # maskout the background
        mask = bbox_image[3, :, :] < 0.99
        bbox_image[:, mask] = 0.0

        scale_x = self.patch_size / w
        scale_y = self.patch_size / h

        # 新影像在原始影像的偏移
        xmin = xmin + j
        ymin = ymin + i

        # 新影像，在缩放后影像的偏移
        camera_matrix = compute_camera_matrix(xmin, ymin, image_width, image_height, camera_matrix[0, 0],
                                              camera_matrix[1, 1], scale_x=scale_x, scale_y=scale_y)

        # To [H W C] to make it compatible with the nvdiffrast renderer
        sample['image'] = bbox_image.permute(1, 2, 0)
        sample['K'] = camera_matrix
        return sample
