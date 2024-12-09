import io

import matplotlib.pyplot as plt
import numpy as np

import rerun as rr
import rerun.blueprint as rrb

from PIL import Image

from camera_model import CameraModel
from data_utils import extract_depth_images, extract_images
from database_utils import RTABSQliteDatabase

from PIL import Image

def resize_image(image: np.ndarray, size_hw: tuple) -> Image:
    
    image = Image.fromarray(image)
    
    image = image.resize(size_hw[::-1])
    
    return np.asarray(image)


def decode_array(x: str, dtype=np.float32, shape=None) -> np.ndarray:

    pose = np.frombuffer(bytearray(x), dtype=dtype)

    if shape is not None:
        pose = pose.reshape(shape)

    return pose


def extract_poses(nodes: list, shape_rows_cols: tuple = (3, 4)) -> np.ndarray:

    poses = [decode_array(node.pose, shape=shape_rows_cols) for node in nodes]

    poses = np.array(poses).reshape(-1, *shape_rows_cols)

    return poses


PATH_TO_DB = "/home/dts/rtabmap_interface/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

nodes = db.extract_rows_from_table(db.models["Node"])

poses = extract_poses(nodes)

data = db.extract_rows_from_table(db.models["Data"])

rgb_images = extract_images(data)

depth_images = extract_depth_images(data)

dh, dw = depth_images.shape[-2:]

rgbh, rgbw, _ = rgb_images.shape[-3:]

rgb_images = np.asarray([resize_image(image, depth_images.shape[-2:]) for image in rgb_images])

cam_model = CameraModel()

cam_model.deserialize_calibration(data[0].calibration)

K = cam_model.K / (rgbw / dw)

print("rgb_images.shape:", rgb_images.shape)
print("depth_images.shape:", depth_images.shape)
print("K:", K)

height, width = rgb_images.shape[-3:-1]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

import open3d as o3d

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=8 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

volume.reset()

# calculate the relative pose of the camera 


for i in range(len(poses)):
    
    print("Integrate {:d}-th image into the volume.".format(i))
    
    color = o3d.geometry.Image(rgb_images[i])    
    depth = o3d.geometry.Image(depth_images[i])
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, 
                                                              depth,
                                                              1.0,
                                                              depth_trunc=4, 
                                                              convert_rgb_to_intensity=False)
    
    pose = np.eye(4)
    pose[:3, :] = poses[i]
    
    volume.integrate(
        rgbd,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy),
        extrinsic=np.linalg.inv(pose))

    if i >= 2:
        break

mesh = volume.extract_point_cloud()
o3d.io.write_point_cloud("pointcloud.ply", mesh)


print(mesh)
# # save pointcloud
o3d.io.write_point_cloud("pointcloud.ply", mesh)

# mesh = volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()

# o3d.io.write_triangle_mesh("mesh.ply", mesh)