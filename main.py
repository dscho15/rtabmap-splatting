import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from PIL import Image

from database import RTABSQliteDatabase
from o3d_utils import (
    registration,
    scalable_tdsf_integration,
    load_rgbd_as_point_cloud,
    draw_geometry_list,
    extract_plane_from_pointcloud,
)

from segmentation import Segmentation2D
from tqdm import tqdm


def filter_depth_images(
    rgb_images: np.ndarray,
    depth_images: np.ndarray,
    colors: np.ndarray,
    threshold: float = 1.0,
):    
    assert rgb_images.shape[0] == depth_images.shape[0]
    assert len(rgb_images.shape) == 4
    assert len(depth_images.shape) == 3
    assert len(colors.shape) == 2
    
    for i, (rgb_image, depth_image) in enumerate(zip(rgb_images, depth_images)):
        
        mask = np.zeros_like(depth_image) # collect all the pixels that are close to the colors
        
        rgb_image = np.array(rgb_image) * 1.0
        
        for color in colors:
            
            diff = np.abs(rgb_image - color).sum(axis=-1)
                                    
            mask += (diff <= threshold) * 1.0
            
        mask = (mask <= 0)
                    
        depth_image[mask] = 0.0
        
        depth_images[i] = depth_image
        
    return depth_images


PATH_TO_DB = "/home/dts/rtabmap_interface/data/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

# data

rgb_images, depth_images, poses, K = db.extract_data()

# segmentation

m = Segmentation2D()

# rgb-sequence

rgb_images_l = []

for i, rgb_image in tqdm(enumerate(rgb_images), total=len(rgb_images)):

    pred = m.segment_image(rgb_image, desired_res=depth_images.shape[-2:])

    rgb_images_l.append(pred)

rgb_images_l = np.stack(rgb_images_l)

colors = np.array([m.colors[0], 
                   m.colors[3], 
                   m.colors[8], 
                   m.colors[14]]) * 255.0

depth_images = filter_depth_images(rgb_images_l, depth_images, colors)

rgb_images_l = rgb_images_l.astype(np.uint8)
rgb_images_l = np.ascontiguousarray(rgb_images_l)

# volume

volume = scalable_tdsf_integration(
    poses,
    rgb_images_l,
    depth_images,
    K,
    depth_scale=1,
    voxel_length=8 / 512.0,
    sdf_trunc=8 / 512.0 * 3,
)

mesh = volume.extract_point_cloud()
o3d.io.write_point_cloud("pointcloud.ply", mesh)

# extract colors and points from mesh
mesh_colors = np.array(mesh.colors)
mesh_points = np.array(mesh.points)

plane_equation = extract_plane_from_pointcloud(mesh, 0.1)
