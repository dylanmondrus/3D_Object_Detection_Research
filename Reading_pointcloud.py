import numpy as np
import glob
import os

def read_point_cloud(file_path: str) -> np.ndarray:
    """
    Read a single .bin point cloud file.

    Args:
        file_path (str): Path to the .bin file.

    Returns:
        np.ndarray: Array of shape (N, 4) with [x, y, z, reflectance].
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32)
    return point_cloud.reshape(-1, 4)

def read_point_clouds(folder_path: str, limit: int = None) -> list[np.ndarray]:
    """
    Read multiple .bin point cloud files from a folder.

    Args:
        folder_path (str): Directory containing .bin files.
        limit (int, optional): Maximum number of files to read. Reads all if None.

    Returns:
        list[np.ndarray]: List of point cloud arrays.
    """
    pattern = os.path.join(folder_path, '*.bin')
    file_list = sorted(glob.glob(pattern))
    if limit is not None:
        file_list = file_list[:limit]

    clouds = []
    for file_path in file_list:
        cloud = read_point_cloud(file_path)
        clouds.append(cloud)
    return clouds
