import PIL.Image
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

from segmentation import Segmentation2D

import open3d as o3d
from tqdm import tqdm


PATH_TO_DB = "/home/dts/rtabmap_interface/data/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

rgb_images, depth_images, poses, K = db.extract_data()

m = Segmentation2D()

rgb_images_l = []

for i, rgb_image in tqdm(enumerate(rgb_images), total=len(rgb_images)):
    pred = m.segment_image(rgb_image, desired_res=depth_images.shape[-2:])
    rgb_images_l.append(pred)
    
rgb_images_l = np.stack(rgb_images_l)
    
# extract color
color = np.array(m.colors[0] * 255.0)

PIL.Image.fromarray(rgb_image.astype(np.uint8)).save("test.png")

for i, (rgb_image, depth_image) in enumerate(zip(rgb_images_l, depth_images)):
    
    rgb_image = np.array(rgb_image) * 1.0
    
    mask = np.sum(np.abs(rgb_image - color), axis=-1) > 1
        
    depth_image[mask] = 10.
    
    depth_images[i] = depth_image
    
rgb_images_l = rgb_images_l.astype(np.uint8)
rgb_images_l = np.ascontiguousarray(rgb_images_l)
    

volume = scalable_tdsf_integration(
    poses,
    rgb_images_l,
    depth_images,
    K,
    depth_scale=1,
    voxel_length=16 / 512.0,
    sdf_trunc=16 / 512.0 * 3,
)

mesh = volume.extract_point_cloud()
o3d.io.write_point_cloud("pointcloud.ply", mesh)

# draw_geometry_list([mesh])

# 2D segmentation, run thrugh every photo

# masks = []
# for photo in photos:
# mask = run_segmentation
# append to masks
# create a volume-integration module with the rgb-values

#
# 3D segmentation, things become slightly easier
#
# mask_rgbd = []
#
