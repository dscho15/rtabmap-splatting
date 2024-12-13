import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from database import RTABSQliteDatabase
from o3d_utils import (
    registration,
    scalable_tdsf_integration,
    load_rgbd_as_point_cloud,
    draw_geometry_list,
)

import open3d as o3d


def resize_image(image: np.ndarray | Image.Image, size_hw: tuple) -> np.ndarray:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "/home/daniel/Desktop/rtabmap-splatting/data/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

indices = db.extract_opt_ids()
poses = db.extract_opt_poses()
rgb_images = db.extract_images()
depth_images = db.extract_depth_images()

_, depth_h, depth_w = depth_images.shape
_, rgb_h, rgb_w, _ = rgb_images.shape

rgb_images = np.asarray(
    [resize_image(image, (depth_h, depth_w)) for image in rgb_images]
)

s = rgb_w / depth_w
K = db.camera.K.copy()
K = K / s
K[2, 2] = 1

poses = db.extract_opt_poses()

volume = scalable_tdsf_integration(
    poses,
    rgb_images,
    depth_images,
    K,
    indices,
    depth_scale=1,
    voxel_length=4 / 512.0,
    sdf_trunc=8 / 512.0 * 3,
)

mesh = volume.extract_point_cloud()

draw_geometry_list([mesh])
