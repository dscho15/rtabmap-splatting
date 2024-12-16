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


PATH_TO_DB = "/home/daniel/Desktop/rtabmap-splatting/data/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

rgb_images, depth_images, poses, K = db.extract_data()

volume = scalable_tdsf_integration(
    poses,
    rgb_images,
    depth_images,
    K,
    depth_scale=1,
    voxel_length=8 / 512.0,
    sdf_trunc=8 / 512.0 * 3,
)

mesh = volume.extract_point_cloud()

draw_geometry_list([mesh])

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
