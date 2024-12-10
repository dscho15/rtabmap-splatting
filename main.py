import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from database import RTABSQliteDatabase
from o3d_utils import registration, scalable_tdsf_integration

import open3d as o3d


def resize_image(image: np.ndarray, size_hw: tuple) -> Image:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "data/241102-23114 PM.db"
 
db = RTABSQliteDatabase(PATH_TO_DB)

poses = db.extract_data_poses()
rgb_images = db.extract_images()
depth_images = db.extract_depth_images()
 
depth_h, depth_w = depth_images.shape[-2:]
rgb_h, rgb_w, _ = rgb_images.shape[-3:]
rgb_images = np.asarray([resize_image(image, (depth_h, depth_w)) for image in rgb_images])

s = rgb_w / depth_w
K = db.camera.K / s
K[2, 2] = 1

initial_pose = np.linalg.inv(poses[1]) @ poses[0]

result = registration(
    rgb_images[0],
    rgb_images[1],
    depth_images[0],
    depth_images[1],
    K,
    depth_scale=1,
    initial_pose=initial_pose,
    radius_normal_est=0.1,
    correspondence_distance=0.025,
    n_max_iters=2000,
    debugging_save_pcd=True,
)

data_poses = db.extract_data_poses()

volume = scalable_tdsf_integration(poses, rgb_images, depth_images, K)
mesh = volume.extract_point_cloud()
o3d.io.write_point_cloud("pointcloud.ply", mesh)
