from peewee import *
import datetime
import numpy as np
import io
import struct
import zlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Helper Functions
def decode_array(x: str, dtype: np.dtype = np.float32, shape: tuple = None) -> np.ndarray:
    array = np.frombuffer(bytearray(x), dtype=dtype)
    return array.reshape(shape) if shape else array

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
        return self.transforms[idx], self.from_ids[idx], self.to_ids[idx], self.information_matrices[idx]

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
        self.local_transform = None  # Local transform
        self.image_size = (0, 0)  # Width, Height

    def deserialize_calibration(self, data):
        header_size = 11
        if len(data) < header_size * 4:
            raise ValueError(f"Insufficient data size. Expected at least {header_size * 4} bytes, got {len(data)} bytes.")

        header = struct.unpack("11i", data[: header_size * 4])
        version_major, version_minor, version_patch, data_type = header[:4]

        if data_type != 0:
            raise ValueError(f"Unsupported calibration type: {data_type}")

        self.image_size = (header[4], header[5])
        index = header_size * 4

        matrices = {'K': (header[6], 9, 8),
                    'D': (header[7], header[7], 8),
                    'R': (header[8], 9, 8),
                    'P': (header[9], 12, 8),
                    'L': (header[10], 12, 4)}

        for attr, (size, elements, byte_size) in matrices.items():
            
            if size:
                matrix_bytes = data[index: index + byte_size * elements]
                if len(matrix_bytes) != byte_size * elements:
                    raise ValueError(
                        f"Matrix {attr} data parsing failed. Expected {byte_size * elements} bytes, got {len(matrix_bytes)} bytes."
                    )
                matrix = np.frombuffer(matrix_bytes, dtype=np.float64 if byte_size == 8 else np.float32)
                setattr(self, attr.upper(), matrix.reshape((3, 3) if elements == 9 else (3, 4)))
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
        self.data = {name: self.extract_rows_from_table(model) for name, model in self.models.items()}

        self.camera = CameraDataReader()
        self.camera.deserialize_calibration(self.data['data'][0].calibration)
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

        return {'admin': Admin, 'nodes': Node, 'links': Link, 'data': Data}

    @staticmethod
    def extract_rows_from_table(model):
        return list(model.select())

    def extract_opt_poses(self):
        opt_poses = zlib.decompress(self.data['admin'][0].opt_poses)
        return decode_array(opt_poses, np.float32, shape=(-1, 3, 4))

    def extract_data_poses(self):
        poses = [decode_array(node.pose, np.float32, (3, 4)) for node in self.data['nodes']]
        zeros = np.zeros((len(poses), 1, 4))
        zeros[:, 0, 3] = 1
        return np.hstack([poses, zeros])

    def extract_links(self):
        data = LinkData()
        for link in tqdm(self.data['links']):
            data.add_link(
                decode_array(link.transform, np.float32, (3, 4)),
                link.from_id,
                link.to_id,
                decode_array(link.information_matrix, np.float64, (6, 6))
            )
        return data

# Example Usage
db = RTABSQliteDatabase("/home/dts/rtabmap_interface/data/241102-23114 PM.db")
poses = db.extract_opt_poses()
data_poses = db.extract_data_poses()
links = db.extract_links()