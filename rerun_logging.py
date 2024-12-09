from pathlib import Path
from rerun_sdk import rerun as rr
import numpy as np

def log_camera(
    intri_path: Path,
    frame_id: str,
    poses_from_traj: dict[str, rr.Transform3D],
    entity_id: str,
) -> None:
    """Logs camera transform and 3D bounding boxes in the image frame."""
    w, h, fx, fy, cx, cy = np.loadtxt(intri_path)
    
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    camera_from_world = poses_from_traj[frame_id]

    # clear previous centroid labels
    rr.log(f"{entity_id}/bbox-2D-segments", rr.Clear(recursive=True))

    # pathlib makes it easy to get the parent, but log methods requires a string
    rr.log(entity_id, camera_from_world)
    
    rr.log(entity_id, rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))