import io

import matplotlib.pyplot as plt
import numpy as np

import rerun as rr
import rerun.blueprint as rrb

from PIL import Image

from camera_model import CameraModel
from data_utils import extract_depth_images, extract_images
from dep.database import RTABSQliteDatabase

from PIL import Image


def resize_image(image: np.ndarray, size_hw: tuple) -> Image:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


PATH_TO_DB = "/home/dts/rtabmap_interface/241102-23114 PM.db"

db = RTABSQliteDatabase(PATH_TO_DB)

nodes = db.extract_rows_from_table(db.supported_datatypes["Node"])

data = db.extract_rows_from_table(db.supported_datatypes["Data"])

poses = extract_poses(nodes)

rgb_images = extract_images(data)

depth_images = extract_depth_images(data)

dh, dw = depth_images.shape[-2:]

rgbh, rgbw, _ = rgb_images.shape[-3:]

rgb_images = np.asarray(
    [resize_image(image, depth_images.shape[-2:]) for image in rgb_images]
)

cam_model = CameraModel()

cam_model.deserialize_calibration(data[0].calibration)

K = cam_model.K / (rgbw / dw)

rr.init("test", spawn=True)

for i in range(poses.shape[0]):

    rr.set_time_seconds("sim_time", i)

    pose = poses[i]

    rr.log("world/camera/0", rr.Transform3D(translation=pose[:, 3], mat3x3=pose[:, :3]))

    rr.log(
        "world/camera/0/image",
        rr.Pinhole(
            focal_length=(K[0, 0], K[1, 1]), width=K[0, 2] * 2, height=K[1, 2] * 2
        ),
    )

    rr.log("world/camera/0/image", rr.DepthImage(depth_images[i], meter=1.0))

# save the rr log
rr.save("test")

db.close()
