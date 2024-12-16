import numpy as np
import open3d as o3d
import o3d_utils as o3d_utils
from pathlib import Path

import os
from os.typing import PathLike

classes = [
    "wall",
    "door",
    "windows"
    "floor",
    "kitchen-cabinet",
    "toilet-seat",
    "toilet",
]


# what are we assuming here?
# basic idea: 
#   a) given a pointcloud, find the biggest consensus for a floor surface
#   b) segment the floor surface
#   c) plane fit the floor surface and find the up-vector
#   d) the up-vector should obey some constraints though, since we probably expect the ceiling to be above the floor
#   e) transform the map to the respecting frame

def ransac_planar_surface(mesh):
    pass    
    

def project_pcd_to_2D():
    pass

class Segmentation2D:
    
    def __init__(self,
                 
                 model_ckpt: PathLike,                                  
                 minimum_num_of_votes: int = 4,
                 depth_truncation: float = 2.5,
                 classes_of_interest: str[list] = classes
                 
                 ):
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        pass

    def project_segmentation_to_rgbd(
        self, image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        pass
    
    
class Segmentation3D:
    
    def __init__(self,
                 model_ckpt: PathLike                 
                 depth_truncation: float = 2.5,
                 classes_of_interest: str[list] = classes

                 
                 ):
        pass
    
    def segment_point_cloud_view(self, point_cloud_view: np.ndarray) -> np.ndarray:
        pass
    
    
    
    
    
    