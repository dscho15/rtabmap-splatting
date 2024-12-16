import open3d as o3d
import numpy as np
from tqdm import tqdm


def load_rgbd_as_point_cloud(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    K: np.ndarray,
    depth_scale: float = 1000,
):

    h, w = depth_image.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, h: {h}, w: {w}")

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image),
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, load_camera_intrinsics_obj(w, h, fx, fy, cx, cy)
    )

    return pcd


def draw_geometry_list(geometry_list):
    
    o3d.visualization.draw_geometries(geometry_list)


def registration(
    src_img: np.ndarray,
    target_img: np.ndarray,
    src_depth: np.ndarray,
    target_depth: np.ndarray,
    K: np.ndarray,
    depth_scale: float = 1000,
    initial_pose: np.ndarray = np.eye(4),
    radius_normal_est: float = 0.1,
    correspondence_distance: float = 0.025,
    n_max_iters: int = 2000,
    debugging_save_pcd: bool = False,
    path_save_pcd: str = "debug_pointcloud.ply",
):

    h, w = src_depth.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, h: {h}, w: {w}")

    # Load RGB-D images
    src_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(src_img),
        o3d.geometry.Image(src_depth),
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False,
    )

    target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(target_img),
        o3d.geometry.Image(target_depth),
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False,
    )

    # Load point clouds
    src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        src_rgbd, load_camera_intrinsics_obj(w, h, fx, fy, cx, cy)
    )

    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd, load_camera_intrinsics_obj(w, h, fx, fy, cx, cy)
    )

    src_pcd.paint_uniform_color([1, 0, 0])  # Red for source point cloud
    target_pcd.paint_uniform_color([0, 0, 1])  # Blue for target point cloud

    # Visualize point clouds
    o3d.visualization.draw_geometries([src_pcd, target_pcd])

    # Estimate normals
    src_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal_est, max_nn=30
        )
    )

    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal_est, max_nn=30
        )
    )

    registration_result = o3d.pipelines.registration.registration_icp(
        src_pcd,
        target_pcd,
        correspondence_distance,
        initial_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=n_max_iters),
    )

    print("Registration Result:")
    print(registration_result)

    if debugging_save_pcd:

        # Get the relative transformation
        relative_pose = registration_result.transformation

        # apply the transform to the second point cloud
        src_pcd.transform(relative_pose)

        # save the point clouds in a single file
        o3d.io.write_point_cloud(path_save_pcd, src_pcd + target_pcd)

    return registration_result.transformation


def load_camera_intrinsics_obj(width, height, fx, fy, cx, cy):

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
    )

    return camera_intrinsics


def extract_intrisics(K: np.ndarray) -> tuple:

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    return fx, fy, cx, cy


def extract_plane_from_pointcloud(
    pcd_np: np.ndarray | o3d.geometry.PointCloud, 
    threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000):
    
    assert pcd_np.ndim == 2, "The point cloud must be a 2D array"
    assert pcd_np.shape[1] == 3, "The point cloud must have 3 columns"
    
    pcd_np = np.ascontiguousarray(pcd_np)
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    
    # ensure inliers is 75% percent of the entire point cloud
    if len(inliers) < 0.75 * len(pcd_np):
        return None, None
    else:
        return plane_model, inliers

def scalable_tdsf_integration(
    poses: np.ndarray,
    rgb_images: np.ndarray,
    depth_images: np.ndarray,
    K: np.ndarray,
    depth_trunc: float = 2.5,
    depth_scale: float = 1,
    voxel_length: float = 8 / 512.0,
    sdf_trunc: float = 8 / 512.0 * 3,
):
    h, w = depth_images.shape[-2:]
    
    fx, fy, cx, cy = extract_intrisics(K)

    volume_obj = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    volume_obj.reset()

    for pose, img, depth in tqdm(zip(poses, rgb_images, depth_images)):

        volume_obj.integrate(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(img),
                depth=o3d.geometry.Image(depth),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            ),
            intrinsic=load_camera_intrinsics_obj(w, h, fx, fy, cx, cy),
            extrinsic=np.linalg.inv(pose),
        )

    return volume_obj