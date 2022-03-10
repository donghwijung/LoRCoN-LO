# import os
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from common import load_files

def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
        current_vertex: raw point clouds
        Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx

def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
    """ Generate a normal image given the range projection of a point cloud.
        Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
        Returns: 
        normal_data: each pixel contains the corresponding normal
    """
    normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)

    # iterate over all pixels in the range image
    for x in range(proj_W):
        for y in range(proj_H - 1):
            p = current_vertex[y, x][:3]
            depth = current_range[y, x]
            
            if depth > 0:
                wrap_x = wrap(x + 1, proj_W)
                u = current_vertex[y, wrap_x][:3]
                u_depth = current_range[y, wrap_x]
                if u_depth <= 0:
                    continue
                
                v = current_vertex[y + 1, x][:3]
                v_depth = current_range[y + 1, x]
                if v_depth <= 0:
                    continue
                
                u_norm = (u - p) / np.linalg.norm(u - p)
                v_norm = (v - p) / np.linalg.norm(v - p)
                
                w = np.cross(v_norm, u_norm)
                norm = np.linalg.norm(w)
                if norm > 0:
                    normal = w / norm
                    normal_data[y, x] = normal

    return normal_data

def wrap(x, dim):
    """ Wrap the boarder of the range image.
    """
    value = x
    if value >= dim:
        value = (value - dim)
    if value < 0:
        value = (value + dim)
    return value

def gen_depth_data(scan_folder, dst_folder, normalize=False, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Generate projected range data in the shape of (64, 900, 1).
        The input raw data are in the shape of (Num_points, 3).
    """
    # specify the goal folder
    dst_folder = os.path.join(dst_folder, 'depth')
    try:
        os.stat(dst_folder)
        print('generating depth data in: ', dst_folder)
    except:
        print('creating new depth folder: ', dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)

    depths = []

    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        remains = current_vertex.shape[0] % 4
        if remains != 0:
            current_vertex = current_vertex[:current_vertex.shape[0]-remains].reshape((-1, 4))
        else:
            current_vertex = current_vertex.reshape((-1, 4))

        proj_range, _, _, _ = range_projection(current_vertex, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)

            # generate the destination path
        dst_path = os.path.join(dst_folder, str(idx).zfill(6) + ".npy")

        # save the semantic image as format of .npy
        np.save(dst_path, proj_range)
        depths.append(proj_range)
        print('finished generating depth data at: ', dst_path)

    return depths

def gen_intensity_data(scan_folder, dst_folder, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Generate projected intensity data in the shape of (64, 900, 1).
        The input raw data are in the shape of (Num_points, 1).
    """
    # specify the goal paths
    dst_folder = os.path.join(dst_folder, 'intensity')
    try:
        os.stat(dst_folder)
        print('creating intensity data in: ', dst_folder)
    except:
        print('creating new intensity folder: ', dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)
    intensities = []
    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        remains = current_vertex.shape[0] % 4
        if remains != 0:
            current_vertex = current_vertex[:current_vertex.shape[0]-remains].reshape((-1, 4))
        else:
            current_vertex = current_vertex.reshape((-1, 4))
        
        current_vertex[:,-1] *= 255 / 5.0

        # generate intensity image from point cloud
        _, _, proj_intensity, _ = range_projection(current_vertex, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)

        # generate the destination path
        dst_path = os.path.join(dst_folder, str(idx).zfill(6) + ".npy")

        # save the semantic image as format of .npy
        np.save(dst_path, proj_intensity)
        intensities.append(proj_intensity)
        print('finished generating intensity data at: ', dst_path)

    return intensities

def gen_normal_data(scan_folder, dst_folder, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Generate projected normal data in the shape of (64, 900, 3).
        The input raw data are in the shape of (Num_points, 3).
    """
    # specify the goal folder
    dst_folder = os.path.join(dst_folder, 'normal')
    try:
        os.stat(dst_folder)
        print('generating normal data in: ', dst_folder)
    except:
        print('creating new normal folder: ', dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)
    normals = []
    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        remains = current_vertex.shape[0] % 4
        if remains != 0:
            current_vertex = current_vertex[:current_vertex.shape[0]-remains].reshape((-1, 4))
        else:
            current_vertex = current_vertex.reshape((-1, 4))

        # generate range image from point cloud
        proj_range, proj_vertex, _, _ = range_projection(current_vertex, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)

        # generate normal image
        normal_data = gen_normal_map(proj_range, proj_vertex, proj_H=proj_H, proj_W=proj_W)

        # generate the destination path
        dst_path = os.path.join(dst_folder, str(idx).zfill(6) + ".npy")

        # save the semantic image as format of .npy
        np.save(dst_path, normal_data)
        normals.append(normal_data)
        print('finished generating intensity data at: ', dst_path)

    return normals