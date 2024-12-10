from peewee import *

import datetime

from pathlib import Path

import numpy as np
import io
from PIL import Image


def decode_array(
    x: str, 
    dtype: np.dtype = np.float32, 
    shape: tuple = None) -> np.ndarray:
    pose = np.frombuffer(bytearray(x), dtype=dtype)
    
    if shape is not None:
        pose = pose.reshape(shape)
    
    return pose


class RTABSQliteDatabase:

    def __init__(self, p_db: str):

        path = Path(p_db)

        # chekc if the database file exists
        if not path.exists():
            raise FileNotFoundError(f"Database file {p_db} not found")

        # Initialize the database
        self.db = SqliteDatabase(p_db)
        self.db.connect()

        # Define the BaseModel
        class BaseModel(Model):
            class Meta:
                database = self.db

        class Feature(BaseModel):
            node_id = IntegerField()
            word_id = IntegerField()
            pos_x = IntegerField()
            pos_y = IntegerField()
            size = IntegerField()
            dir = IntegerField()
            response = FloatField()
            octave = IntegerField()
            depth_x = FloatField()
            depth_y = FloatField()
            depth_z = FloatField()
            descriptor_size = IntegerField()
            descriptor = BlobField()

        # Define the Data model
        class Data(BaseModel):
            id = IntegerField(primary_key=True)
            image = BlobField()
            depth = BlobField()
            calibration = BlobField()
            scan = BlobField()
            scan_info = BlobField()
            ground_cells = BlobField()
            obstacle_cells = BlobField()
            empty_cells = BlobField()
            cell_size = FloatField()
            view_point_x = FloatField()
            view_point_y = FloatField()
            view_point_z = FloatField()
            user_data = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)

        class Node(BaseModel):
            id = IntegerField(primary_key=True)
            map_id = IntegerField()
            weight = IntegerField()
            stamp = FloatField()
            pose = BlobField()
            ground_truth_pose = BlobField()
            velocity = BlobField()
            label = TextField()
            gps = BlobField()
            env_sensors = BlobField()
            time_enter = DateTimeField(default=datetime.datetime.now)

        self.data = {
            "nodes": self.extract_rows_from_table(Node),
            "data": self.extract_rows_from_table(Data),
        }

        self.db.close()

    @staticmethod
    def extract_rows_from_table(model):
        return list(model.select())

    # close model
    def close(self):
        self.db.close()

    def extract_poses(self) -> np.ndarray:
        poses = []
        
        for node in self.data["nodes"]:            
            pose = decode_array(node.pose, np.uint8).reshape(-1, 4)
            pose = pose.reshape(-1)
            pose = pose.tobytes()
            pose = decode_array(pose, np.float32, (3, 4))
            poses.append(pose)
        
        poses = np.array(poses)
        ones = np.zeros((poses.shape[0], 1, 4))
        ones[:, 0, 3] = 1
        poses = np.concatenate([poses, ones], axis=1)
        
        return poses

    def extract_images(self) -> np.ndarray:
        images = [io.BytesIO(d.image) for d in self.data["data"]]
        images = [Image.open(image) for image in images]
        images = np.asarray(images)
        return images

    def __decode_depth_image(self, byte_sequence: str):
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

    def extract_depth_images(self) -> list:
        depth_images = [self.__decode_depth_image(d.depth) for d in self.data["data"]]
        depth_images = np.array(depth_images)
        return depth_images
