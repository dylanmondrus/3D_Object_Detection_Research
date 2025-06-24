import numpy as np
import glob
import os
#hello

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


def read_label_file(label_path: str) -> list[dict]:
    """
    Parse a single KITTI label file (.txt) into a list of object dicts.

    Args:
        label_path (str): Path to the label .txt file.

    Returns:
        List[dict]: List of annotation objects.
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue  # Skip malformed lines
            obj = {
                'type': fields[0],
                'truncated': float(fields[1]),
                'occluded': int(fields[2]),
                'alpha': float(fields[3]),
                'bbox': [float(v) for v in fields[4:8]],
                'dimensions': [float(v) for v in fields[8:11]],  # h, w, l
                'location': [float(v) for v in fields[11:14]],   # x, y, z
                'rotation_y': float(fields[14])
            }
            objects.append(obj)
    return objects


def read_annotations(folder_path: str, limit: int = None) -> dict:
    """
    Read multiple KITTI label files from a folder.

    Args:
        folder_path (str): Directory containing .txt label files.
        limit (int, optional): Maximum number of files to read. Reads all if None.

    Returns:
        dict: Mapping from file ID to list of annotations.
    """
    all_annotations = {}
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    if limit is not None:
        files = files[:limit]

    for fname in files:
        file_id = os.path.splitext(fname)[0]
        full_path = os.path.join(folder_path, fname)
        annotations = read_label_file(full_path)
        all_annotations[file_id] = annotations

    return all_annotations




import numpy as np

def voxelize(point_cloud, voxel_size=(0.2, 0.2, 0.4), 
             point_cloud_range=(0, -40, -3, 70.4, 40, 1),
             max_points_per_voxel=5,
             max_voxels=20000):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    voxel_size_x, voxel_size_y, voxel_size_z = voxel_size

    mask = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] < x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] < y_max) &
        (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] < z_max)
    )
    pc = point_cloud[mask]

    # Convert to voxel indices (z, y, x)
    voxel_indices = np.floor((pc[:, :3] - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)
    coord_to_points = {}

    for i in range(pc.shape[0]):
        voxel_coord = tuple(voxel_indices[i])  # (z, y, x)
        if voxel_coord not in coord_to_points:
            coord_to_points[voxel_coord] = []
        if len(coord_to_points[voxel_coord]) < max_points_per_voxel:
            coord_to_points[voxel_coord].append(pc[i])

    voxel_coords = list(coord_to_points.keys())
    if len(voxel_coords) > max_voxels:
        voxel_coords = voxel_coords[:max_voxels]

    voxel_features = []
    final_coords = []
    for coord in voxel_coords:
        pts = coord_to_points[coord]
        while len(pts) < max_points_per_voxel:
            pts.append(np.zeros(4))
        voxel_features.append(pts)
        final_coords.append(coord)  # (z, y, x)

    voxel_features = np.array(voxel_features)           # (V, T, 4)
    voxel_features = np.expand_dims(voxel_features, 0)  # (1, V, T, 4)
    voxel_coords = np.array(final_coords)               # (V, 3)

    return voxel_features, voxel_coords




# Example usage
bin_path = "C:/Users/dylan/OneDrive/Desktop/Research_Summer_2025/velodyne"
label_path = "C:/Users/dylan/OneDrive/Desktop/Research_Summer_2025/label_2"

# Read a single file
sample_file = os.path.join(bin_path, '000000.bin')
pc = read_point_cloud(sample_file)
print("Single point cloud shape:", pc.shape)



from voxelnet import VoxelNet
import torch

# Assume you've already called:
voxel_features, voxel_coords = voxelize(pc)

# Convert to tensors
voxel_tensor = torch.tensor(voxel_features, dtype=torch.float32)
coord_tensor = torch.tensor(voxel_coords, dtype=torch.long)  # shape: (V, 3)

# Run model
model = VoxelNet()
model.eval()
with torch.no_grad():
    output = model(voxel_tensor, coord_tensor)

print("Output shape:", output.shape)

def decode_predictions(output, voxel_size=(0.2, 0.2), point_cloud_range=(0, -40, -3, 70.4, 40, 1), score_thresh=0.1):
    """
    Decode raw model output into 3D bounding box list.

    Args:
        output: (B, 7, H, W) tensor
        voxel_size: (vx, vy) — size of each voxel in XY
        point_cloud_range: min/max x/y/z
        score_thresh: Only keep predictions with nonzero box sizes

    Returns:
        boxes: List of dicts per batch item with predicted boxes
    """
    B, _, H, W = output.shape
    vx, vy = voxel_size
    x_offset = point_cloud_range[0]
    y_offset = point_cloud_range[1]

    boxes_batch = []

    for b in range(B):
        preds = output[b]  # (7, H, W)
        boxes = []

        for i in range(H):
            for j in range(W):
                x, y, z, l, w, h, theta, score = preds[:, i, j]

                if score.item() < score_thresh or l <= 0 or w <= 0 or h <= 0:
                    continue

                # Map grid cell (i, j) to real-world coordinates
                center_x = x_offset + j * vx + x.item()
                center_y = y_offset + i * vy + y.item()
                center_z = z.item()

                boxes.append({
                    'center': [center_x, center_y, center_z],
                    'dims': [l.item(), w.item(), h.item()],
                    'angle': theta.item()
                })

        boxes_batch.append(boxes)

    return boxes_batch

boxes = decode_predictions(output)
print(f"Predicted {len(boxes[0])} boxes in frame 0")
'''
import open3d as o3d
import numpy as np
import torch

def create_3d_box(center, size, yaw, color=[1, 0, 0]):
    l, w, h = size
    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = [l, w, h]
    R = box.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box.R = R
    box.color = color
    return box

def visualize_lidar_with_boxes(pc, output_tensor, voxel_size=(0.2, 0.2), pc_range=(0, -40, -3, 70.4, 40, 1), score_thresh=0.5):
    output = output_tensor[0].detach().cpu().numpy()  # (8, H, W)
    x_map, y_map, z_map, l_map, w_map, h_map, yaw_map, score_map = output

    x_min, y_min, *_ = pc_range
    vx, vy = voxel_size
    H, W = score_map.shape

    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray

    boxes = []

    for i in range(H):
        for j in range(W):
            score = score_map[i, j]
            if score < score_thresh:
                continue

            x, y, z = x_map[i, j], y_map[i, j], z_map[i, j]
            l, w, h = l_map[i, j], w_map[i, j], h_map[i, j]
            yaw = yaw_map[i, j]

            if l <= 0 or w <= 0 or h <= 0:
                continue

            # Compute world coordinates from voxel grid
            cx = x_min + (j + x) * vx
            cy = y_min + (i + y) * vy
            cz = z

            box = create_3d_box([cx, cy, cz], [l, w, h], yaw)
            boxes.append(box)

    o3d.visualization.draw_geometries([pcd, *boxes])

output = model(voxel_tensor, coord_tensor)
visualize_lidar_with_boxes(pc, output)
'''