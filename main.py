from database_utils import RTABSQliteDatabase

import io
import numpy as np

from matplotlib import pyplot as plt

def decode_array(x: str, dtype=np.float32, shape=None) -> np.ndarray:
    
    pose = np.frombuffer(bytearray(x), dtype=dtype)
    
    if shape is not None:
        pose = pose.reshape(shape)
    
    return pose

def extract_poses(nodes: list, shape_rows_cols: tuple = (3, 4)) -> np.ndarray:
    
    poses = [decode_array(node.pose, shape=shape_rows_cols) for node in nodes]
        
    poses = np.array(poses).reshape(-1, *shape_rows_cols)
    
    return poses

db = RTABSQliteDatabase('241102-23114 PM.db')

nodes = db.extract_rows_from_table(db.models["Node"])

poses = extract_poses(nodes)

plt.plot(poses[:, 0, 3], poses[:, 1, 3])

plt.savefig('test.png')

db.close()