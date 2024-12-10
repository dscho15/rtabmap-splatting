from peewee import *

import datetime
import numpy as np
import io
import struct

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def decode_array(
    x: str, dtype: np.dtype = np.float32, shape: tuple = None
) -> np.ndarray:
    pose = np.frombuffer(bytearray(x), dtype=dtype)

    if shape is not None:
        pose = pose.reshape(shape)

    return pose

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
    
    # search for from_ids to ids
    def search(self, from_id, to_id):
        for i, (f, t) in enumerate(zip(self.from_ids, self.to_ids)):
            if f == from_id and t == to_id:
                pose, _, _, information = self[i]
                pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)                
                return pose, information
        return None

class CameraDataReader:
    def __init__(self):
        self.K = None  # Intrinsic matrix
        self.D = None  # Distortion coefficients
        self.R = None  # Rotation matrix
        self.P = None  # Projection matrix
        self.local_transform = None  # Local transform
        self.image_size = (0, 0)  # Width, Height

    def deserialize_calibration(self, data):
        """
        Deserialize camera calibration data from a byte buffer.

        :param data: Byte buffer containing serialized calibration data
        :return: Number of bytes processed
        """
        # Ensure minimum data size for header
        header_size = 11
        if len(data) < header_size * 4:  # 4 bytes per int
            raise ValueError(
                f"Insufficient data size. Expected at least {header_size * 4} bytes."
            )

        # Unpack header
        header = struct.unpack("11i", data[: header_size * 4])
        print("Header:", header)

        # Extract header components
        version_major, version_minor, version_patch = header[0:3]
        data_type = header[3]
        print("Version:", version_major, version_minor, version_patch)

        # Check data type (0 for mono camera)
        if data_type != 0:
            raise ValueError(f"Unsupported calibration type: {data_type}")

        # Set image size
        self.image_size = (header[4], header[5])
        print("Image Size:", self.image_size)

        # Indices for different matrix components
        iK, iD, iR, iP, iL = 6, 7, 8, 9, 10

        # Calculate required data size
        required_data_size = (
            header_size * 4  # Header size in bytes
            + 8
            * (
                header[iK] + header[iD] + header[iR] + header[iP]
            )  # Double precision matrices
            + 4 * header[iL]  # Local transform (float)
        )

        # Verify total data size
        if len(data) < required_data_size:
            raise ValueError(
                f"Insufficient data size. "
                f"Actual: {len(data)} bytes, "
                f"Required: {required_data_size} bytes"
            )

        # Current index in the data buffer
        index = header_size * 4

        # Parse Intrinsic Matrix (K)
        if header[iK] != 0:
            assert header[iK] == 9, "K matrix must be 3x3"
            K_bytes = data[index : index + 8 * 9]
            self.K = np.frombuffer(K_bytes, dtype=np.float64).reshape((3, 3))
            index += 8 * 9
            print("Intrinsic Matrix (K):", self.K)

        # Parse Distortion Coefficients (D)
        if header[iD] != 0:
            D_bytes = data[index : index + 8 * header[iD]]
            self.D = np.frombuffer(D_bytes, dtype=np.float64).reshape((1, header[iD]))
            index += 8 * header[iD]
            print("Distortion Coefficients (D):", self.D)

        # Parse Rotation Matrix (R)
        if header[iR] != 0:
            assert header[iR] == 9, "R matrix must be 3x3"
            R_bytes = data[index : index + 8 * 9]
            self.R = np.frombuffer(R_bytes, dtype=np.float64).reshape((3, 3))
            index += 8 * 9
            print("Rotation Matrix (R):", self.R)

        # Parse Projection Matrix (P)
        if header[iP] != 0:
            assert header[iP] == 12, "P matrix must be 3x4"
            P_bytes = data[index : index + 8 * 12]
            self.P = np.frombuffer(P_bytes, dtype=np.float64).reshape((3, 4))
            index += 8 * 12
            print("Projection Matrix (P):", self.P)

        # Parse Local Transform
        if header[iL] != 0:
            assert header[iL] == 12, "Local transform must be 12 elements"
            local_transform_bytes = data[index : index + 4 * 12]
            self.local_transform = np.frombuffer(
                local_transform_bytes, dtype=np.float32
            )
            index += 4 * 12
            print("Local Transform:", self.local_transform)

        return index
    

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
                
        class Admin(BaseModel):
            version = TextField(primary_key=True)
            preview_image = BlobField()
            opt_cloud = BlobField()
            opt_ids = BlobField()
            opt_poses = BlobField()
            opt_last_localization = BlobField()
            opt_polygons_size = IntegerField()
            opt_polygons = BlobField()
            opt_tex_coords = BlobField()
            opt_tex_materials = BlobField()
            opt_map = BlobField()
            opt_map_x_min = FloatField()
            opt_map_y_min = FloatField()
            opt_map_resolution = FloatField()
            time_enter = DateTimeField(default=datetime.datetime.now)

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

        class Link(BaseModel):
            from_id = IntegerField(primary_key=True)
            to_id = IntegerField()
            type = IntegerField()
            information_matrix = BlobField()
            transform = BlobField()
            user_data = BlobField()

        self.data = {
            "nodes": self.extract_rows_from_table(Node),
            "data": self.extract_rows_from_table(Data),
            "links": self.extract_rows_from_table(Link),
            "admin": self.extract_rows_from_table(Admin),
        }
        
        self.camera = CameraDataReader()
        self.camera.deserialize_calibration(self.data["data"][0].calibration)

        self.db.close()

    @staticmethod
    def extract_rows_from_table(model):
        return list(model.select())

    def close(self):
        self.db.close()
        
    def extract_opt_poses(self) -> np.ndarray:
        opt_poses = self.data["admin"][0].opt_poses
        opt_ids = self.data["admin"][0].opt_ids
        
        poses = decode_array(opt_poses, np.float32, shape=(-1, 3, 4))
        
        return poses

    def extract_data_poses(self) -> np.ndarray:
        poses = [
            decode_array(node.pose, np.float32, (3, 4)) for node in self.data["nodes"]
        ]
        
        zeros = np.zeros((len(poses), 1, 4))
        zeros[:, 0, 3] = 1
        
        poses = np.concatenate([poses, zeros], axis=1)
        
        return poses
    
    def extract_links(self) -> LinkData:
        data = LinkData()

        for l in tqdm(self.data["links"]):
            data.add_link(
                decode_array(l.transform, np.float32, (3, 4)),         # 3x4 matrix
                l.from_id,
                l.to_id,
                decode_array(l.information_matrix, np.float64, (6, 6)) # 6x6 matrix
            )

        return data

    
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

# RTABSQliteDatabase("data/241102-23114 PM.db")