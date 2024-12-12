import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from database import RTABSQliteDatabase
from o3d_utils import registration, scalable_tdsf_integration, load_rgbd_as_point_cloud, draw_geometry_list

import open3d as o3d

def resize_image(image: np.ndarray | Image.Image, size_hw: tuple) -> np.ndarray:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "/home/dts/Desktop/rtabmap-splatting/databases/241211-32334 PM.db"
 
db = RTABSQliteDatabase(PATH_TO_DB)

opt_ids = db.extract_opt_ids()
poses = db.extract_opt_poses()
rgb_images = db.extract_images()
depth_images = db.extract_depth_images()
 
_, depth_h, depth_w = depth_images.shape
_, rgb_h, rgb_w, _ = rgb_images.shape

rgb_images = np.asarray([resize_image(image, (depth_h, depth_w)) for image in rgb_images])

s = rgb_w / depth_w
K = db.camera.K.copy()
K = K / s
K[2, 2] = 1

L = np.vstack((db.camera.L, np.array([0, 0, 0, 1])))

poses = db.extract_opt_poses()

poses = [pose @ L for pose in poses]

pc1 = load_rgbd_as_point_cloud(rgb_images[0], depth_images[0], K, 1)
pc2 = load_rgbd_as_point_cloud(rgb_images[1], depth_images[1], K, 1)
pc3 = load_rgbd_as_point_cloud(rgb_images[2], depth_images[2], K, 1)
pc4 = load_rgbd_as_point_cloud(rgb_images[3], depth_images[3], K, 1)
pc5 = load_rgbd_as_point_cloud(rgb_images[4], depth_images[4], K, 1)
pc6 = load_rgbd_as_point_cloud(rgb_images[5], depth_images[5], K, 1)
pc7 = load_rgbd_as_point_cloud(rgb_images[6], depth_images[6], K, 1)
pc8 = load_rgbd_as_point_cloud(rgb_images[7], depth_images[7], K, 1)
pc9 = load_rgbd_as_point_cloud(rgb_images[8], depth_images[8], K, 1)
pc10 = load_rgbd_as_point_cloud(rgb_images[9], depth_images[9], K, 1)

pc1.transform(poses[0])
pc2.transform(poses[1])
pc3.transform(poses[2])
pc4.transform(poses[3])
pc5.transform(poses[4])
pc6.transform(poses[5])
pc7.transform(poses[6])
pc8.transform(poses[7])
pc9.transform(poses[8])
pc10.transform(poses[9])

draw_geometry_list([pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9])

poses = db.extract_opt_poses()

poses = [pose @ L for pose in poses]

volume = scalable_tdsf_integration(poses, rgb_images, depth_images, K)

mesh = volume.extract_point_cloud()

draw_geometry_list([mesh])

# o3d.io.write_point_cloud("pointcloud.ply", mesh)
