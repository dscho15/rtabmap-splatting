import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from database import RTABSQliteDatabase
from o3d_utils import registration, scalable_tdsf_integration, load_rgbd_as_point_cloud, draw_geometry_list

import open3d as o3d

def resize_image(image: np.ndarray, size_hw: tuple) -> Image:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "data/241211-32334 PM.db"
 
db = RTABSQliteDatabase(PATH_TO_DB)

opt_ids = db.extract_opt_ids()
poses = db.extract_opt_poses()
rgb_images = db.extract_images()
depth_images = db.extract_depth_images()
 
depth_h, depth_w = depth_images.shape[-2:]
rgb_h, rgb_w, _ = rgb_images.shape[-3:]
rgb_images = np.asarray([resize_image(image, (depth_h, depth_w)) for image in rgb_images])

s = rgb_w / depth_w
K = db.camera.K / s
K[2, 2] = 1

opt_ids
np.linalg.inv(poses[1]) @ poses[0]
np.linalg.inv(poses[2]) @ poses[1]

pc1 = load_rgbd_as_point_cloud(rgb_images[0], depth_images[0], K, 1)
pc2 = load_rgbd_as_point_cloud(rgb_images[1], depth_images[1], K, 1)

pc1.transform(np.linalg.inv(poses[1]))
pc2.transform(np.linalg.inv(poses[2]))

draw_geometry_list([pc1, pc2])

# >>> opt_ids
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
#       dtype=int32)
# >>> poses = db.extract_opt_poses()
# >>> poses[0]
# array([[ 0.18787795,  0.96878719, -0.16171999, -0.17180113],
#        [-0.98113084,  0.1927669 ,  0.014947  , -0.00979134],
#        [ 0.04565473,  0.15586025,  0.98672348,  0.11460429],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
# >>> poses[1]
# array([[ 0.21416372,  0.96080518, -0.17603199, -0.16378519],
#        [-0.95994937,  0.24035096,  0.14397448,  0.02354243],
#        [ 0.18064089,  0.13814768,  0.97379881,  0.16355465],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
# >>> 

# result = registration(
#     rgb_images[0],
#     rgb_images[1],
#     depth_images[0],
#     depth_images[1],
#     K,
#     depth_scale=1000,
#     initial_pose=initial_pose,
#     radius_normal_est=0.1,
#     correspondence_distance=0.025,
#     n_max_iters=2000,
#     debugging_save_pcd=True,
# )

# poses = db.extract_opt_poses()

# poses[1] = poses[0] @ np.linalg.inv(result)

# volume = scalable_tdsf_integration(poses[:2], rgb_images[:2], depth_images[:2], K)

# mesh = volume.extract_point_cloud()

# o3d.io.write_point_cloud("pointcloud.ply", mesh)
