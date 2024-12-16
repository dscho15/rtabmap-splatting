from peewee import (
    SqliteDatabase,
    Model,
    TextField,
    BlobField,
    DateTimeField,
    IntegerField,
)

import datetime
import numpy as np
import io
import struct
import zlib

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union

# Helper Functions
def decode_array(
    x: bytes,
    dtype: Union[np.dtype, str] = np.dtype(np.float32),
    shape: Optional[tuple] = None,
) -> np.ndarray:
    array = np.frombuffer(x, dtype=dtype)
    return array.reshape(shape) if shape else array


def resize_image(image: np.ndarray | Image.Image, size_hw: tuple) -> np.ndarray:
    image = Image.fromarray(image)
    image = image.resize(size_hw[::-1])
    return np.asarray(image)


# LinkData Class
class LinkData:
    def __init__(self):
        self.transforms = []
        self.from_ids = []
        self.to_ids = []
        self.information_matrices = []

    def add_link(self, transform, from_id, to_id, information_matrix):
        self.transforms.append(transform)
        self.from_ids.append(from_id)
        self.to_ids.append(to_id)
        self.information_matrices.append(information_matrix)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, idx):
        return (
            self.transforms[idx],
            self.from_ids[idx],
            self.to_ids[idx],
            self.information_matrices[idx],
        )

    def search(self, from_id, to_id):
        for i, (f, t) in enumerate(zip(self.from_ids, self.to_ids)):
            if f == from_id and t == to_id:
                pose, _, _, information = self[i]
                pose = np.vstack([pose, np.array([[0, 0, 0, 1]])])
                return pose, information
        return None


# CameraDataReader Class
class CameraDataReader:
    def __init__(self):
        self.K = None  # Intrinsic matrix
        self.D = None  # Distortion coefficients
        self.R = None  # Rotation matrix
        self.P = None  # Projection matrix
        self.L = np.eye(3, 4)  # Additional matrix L
        self.image_size = (0, 0)  # Width, Height

    def deserialize_calibration(self, data):
        header_size = 11
        if len(data) < header_size * 4:
            raise ValueError(
                f"Insufficient data size. Expected at least {header_size * 4} bytes, got {len(data)} bytes."
            )

        header = struct.unpack("11i", data[: header_size * 4])
        version_major, version_minor, version_patch, data_type = header[:4]

        if data_type != 0:
            raise ValueError(f"Unsupported calibration type: {data_type}")

        self.image_size = (header[4], header[5])
        index = header_size * 4

        matrices = {
            "K": (header[6], 9, 8),
            "D": (header[7], header[7], 8),
            "R": (header[8], 9, 8),
            "P": (header[9], 12, 8),
            "L": (header[10], 12, 4),
        }

        for attr, (size, elements, byte_size) in matrices.items():

            if size:
                matrix_bytes = data[index : index + byte_size * elements]
                if len(matrix_bytes) != byte_size * elements:
                    raise ValueError(
                        f"Matrix {attr} data parsing failed. Expected {byte_size * elements} bytes, got {len(matrix_bytes)} bytes."
                    )
                matrix = np.frombuffer(
                    matrix_bytes, dtype=np.float64 if byte_size == 8 else np.float32
                )
                setattr(
                    self,
                    attr.upper(),
                    matrix.reshape((3, 3) if elements == 9 else (3, 4)),
                )
                index += byte_size * elements
            else:
                setattr(self, attr.upper(), None)

        return index


# RTABSQliteDatabase Class
class RTABSQliteDatabase:
    def __init__(self, db_path: str):
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self.db = SqliteDatabase(db_path)
        self.db.connect()
        self.models = self.define_models()
        self.data = {
            name: self.extract_rows_from_table(model)
            for name, model in self.models.items()
        }

        self.camera = CameraDataReader()
        self.camera.deserialize_calibration(self.data["data"][0].calibration)
        self.db.close()

    def define_models(self):
        class BaseModel(Model):
            class Meta:
                database = self.db

        class Admin(BaseModel):
            version = TextField(primary_key=True)
            opt_poses = BlobField()
            opt_ids = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)

        class Node(BaseModel):
            id = IntegerField(primary_key=True)
            pose = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)

        class Link(BaseModel):
            from_id = IntegerField(primary_key=True)
            to_id = IntegerField()
            transform = BlobField()
            information_matrix = BlobField()

        class Data(BaseModel):
            id = IntegerField(primary_key=True)
            calibration = BlobField()
            image = BlobField()
            depth = BlobField()

        return {"admin": Admin, "nodes": Node, "links": Link, "data": Data}

    @staticmethod
    def extract_rows_from_table(model):
        return list(model.select())

    def get_opt_ids(self):
        opt_ids = zlib.decompress(self.data["admin"][0].opt_ids)
        opt_ids = np.frombuffer(opt_ids, dtype=np.int32)
        return np.asarray(opt_ids) - 1

    def get_opt_poses(self):
        poses = zlib.decompress(self.data["admin"][0].opt_poses)
        poses = decode_array(poses, np.dtype(np.float32), shape=(-1, 3, 4))
        poses = np.array(
            [np.vstack([pose, np.array([[0.0, 0.0, 0.0, 1.0]])]) for pose in poses]
        )
        local_transform = np.vstack((self.camera.L, np.array([0, 0, 0, 1])))
        poses = np.array([pose @ local_transform for pose in poses])
        return poses

    def extract_links(self):
        data = LinkData()
        for link in tqdm(self.data["links"]):
            data.add_link(
                decode_array(link.transform, np.dtype(np.float32), (3, 4)),
                link.from_id,
                link.to_id,
                decode_array(link.information_matrix, np.dtype(np.float64), (6, 6)),
            )
        return data

    def get_images(self) -> np.ndarray:
        images = [io.BytesIO(d.image) for d in self.data["data"]]
        images = [Image.open(image) for image in images]
        images = np.asarray(images)
        return images

    def __decode_depth_image(self, byte_sequence: bytes):
        image = Image.open(io.BytesIO(byte_sequence))
        depth_image = np.array(image)

        assert len(depth_image.shape) == 3, "Invalid image shape"
        assert depth_image.shape[-1] == 4, "Invalid image channels"

        h, w, c = depth_image.shape
        depth_image = depth_image.reshape((h * w, c))

        # decoding pattern (BGRA -> RGBA)
        depth_image = depth_image[:, [2, 1, 0, 3]]
        depth_image = depth_image.reshape(h, w, c)
        depth_bytes = depth_image.tobytes()

        image = np.frombuffer(depth_bytes, dtype=np.float32)

        return image.reshape(h, w)

    def get_depth_images(self) -> np.ndarray:
        depth_images = [self.__decode_depth_image(d.depth) for d in self.data["data"]]
        depth_images = np.array(depth_images)

        return depth_images

    def extract_data(self):

        indices = self.get_opt_ids()
        poses = self.get_opt_poses()
        rgb_images = self.get_images()[indices]
        depth_images = self.get_depth_images()[indices]

        _, depth_h, depth_w = depth_images.shape
        _, rgb_h, rgb_w, _ = rgb_images.shape

        # camera intrinsics
        s = rgb_w / depth_w
        K = self.camera.K.copy() # type: ignore # 
        K = K / s
        K[2, 2] = 1

        return (rgb_images, depth_images, poses, K)


# # Example Usage
# db = RTABSQliteDatabase("databases/241211-32334 PM.db")
# poses = db.get_opt_poses()
# links = db.extract_links()

# depth_images = db.get_depth_images()
# d_img_1 = depth_images[0]
# d_img_1[0, 0]
