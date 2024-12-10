import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from camera_model import CameraModel
from database_utils import RTABSQliteDatabase
from o3d_interface import registration, tdsf_integration

import open3d as o3d


def resize_image(image: np.ndarray, size_hw: tuple) -> Image:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "/home/dts/rtabmap_interface/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

poses = db.extract_poses()
rgb_images = db.extract_images()
depth_images = db.extract_depth_images()

cam_model = CameraModel()
cam_model.deserialize_calibration(db.data["data"][0].calibration)

depth_h, depth_w = depth_images.shape[-2:]
rgb_h, rgb_w, _ = rgb_images.shape[-3:]

rgb_images = np.asarray(
    [resize_image(image, (depth_h, depth_w)) for image in rgb_images]
)

s = rgb_w / depth_w
K = cam_model.K / s
K[2, 2] = 1

initial_pose = np.linalg.inv(poses[1]) @ poses[0]

print("Initial Pose:")
print(initial_pose)

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

print("Registration Result:")
print(result)

# compute the discrepancy between the two poses
pose_diff = np.linalg.inv(initial_pose) @ result

poses = db.extract_poses()

# only show the first three decimals
pose_diff = np.round(pose_diff, 3)

volume = tdsf_integration(poses[:2], rgb_images[:2], depth_images[:2], K)
mesh = volume.extract_point_cloud()
o3d.io.write_point_cloud("pointcloud.ply", mesh)