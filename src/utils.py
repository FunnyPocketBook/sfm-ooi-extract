from plyfile import PlyData
import numpy as np
import open3d as o3d

from colmap_loader import qvec2rotmat



def read_ply(file_path):
    print(f"Reading {file_path}")
    plydata = PlyData.read(file_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    if 'red' in plydata['vertex']:
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']
        colors = np.array([r, g, b]).T
    else:
        colors = None
    if 'nx' in plydata['vertex']:
        nx = plydata['vertex']['nx']
        ny = plydata['vertex']['ny']
        nz = plydata['vertex']['nz']
        normals = np.array([nx, ny, nz]).T
    else:
        normals = None
    points = np.array([x, y, z]).T
    return points, colors, normals


def get_extrinsic_matrix(qvec, tvec):
    """
    Create a 4x4 extrinsic matrix from quaternion and translation vector.
    
    Parameters:
    - qvec: np.ndarray - Quaternion (w, x, y, z) representing rotation.
    - tvec: np.ndarray - Translation vector (x, y, z).
    
    Returns:
    - extrinsic_matrix: np.ndarray - 4x4 extrinsic matrix.
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = qvec2rotmat(qvec)
    
    # Create 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec
    
    return extrinsic_matrix


def write_ply(original_points, points, colors, normals, out_path):
    dtype = [('x', float), ('y', float), ('z', float)]
    original_structured = np.array([tuple(p) for p in original_points], dtype=dtype)
    outliers_structured = np.array([tuple(p) for p in points], dtype=dtype)

    indices = np.nonzero(np.isin(original_structured, outliers_structured))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        end_colors = colors[indices]
        pcd.colors = o3d.utility.Vector3dVector(end_colors / 255.0)
    if normals is not None:
        end_normals = normals[indices]
        pcd.normals = o3d.utility.Vector3dVector(end_normals)

    o3d.io.write_point_cloud(out_path, pcd)