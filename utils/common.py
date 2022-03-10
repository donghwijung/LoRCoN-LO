import datetime
from time import time
import numpy as np
import math
import os

def euler_to_rot_mat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

def rot_mat_to_euler(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def get_original_poses(pose_folder, preprocessed_folder, data_seqs, depth_name="depth"):
    Y_origin_data = np.array([],dtype=np.object)
    for idx, seq in enumerate(data_seqs):
        pose_file = os.path.join(pose_folder, seq + "_vel.txt")
        with open(pose_file, "r") as f:       
            parent_directory = os.path.join(preprocessed_folder, seq)
            npy_directory = os.path.join(parent_directory, depth_name)
            npy_names = os.listdir(npy_directory)
            npy_names.sort()
            npy_names_len = len(npy_names)

            lines = f.readlines()
            Y_row = np.zeros((npy_names_len), dtype=np.object)
            for i, line in enumerate(lines):
                Y_row[i] = line
            
            Y_origin_data = np.append(Y_origin_data, Y_row)
    return Y_origin_data

def save_poses(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset="KITTI"):
    start_idx = 0
    end_idx = 0
    additional_row = np.array([0, 0, 0, 1], dtype=np.float32)
    for i, seq in enumerate(data_seqs):
        end_idx += seq_sizes[seq]
        poses = np.zeros((Y_origin_data[start_idx:end_idx].shape[0], 4,4),dtype=np.float32)
        
        for idx in range(rnn_size):
            current_pose = np.array(list(map(float, Y_origin_data[start_idx+idx].strip().split(" "))), dtype=np.float32)
            current_pose = np.concatenate((current_pose, additional_row))
            current_pose = current_pose.reshape(4,4)
            poses[idx] = current_pose
        
        for idx, relative_pose in enumerate(Y_estimated_data[start_idx-i*rnn_size:end_idx-(i+1)*rnn_size]):
            rot_mat = euler_to_rot_mat(relative_pose[5],relative_pose[4],relative_pose[3])
            trans_mat = np.identity(4)
            trans_mat[:3,:3]=rot_mat
            trans_mat[0,3]=relative_pose[0]
            trans_mat[1,3]=relative_pose[1]
            trans_mat[2,3]=relative_pose[2]

            current_pose = np.dot(current_pose, trans_mat)
            poses[idx + rnn_size] = current_pose
        
        est_pose_folder = os.path.join("result", dataset, "pose")
        os.makedirs(est_pose_folder, exist_ok=True)
        with open(os.path.join(est_pose_folder, f"{seq}.txt"), "w") as f_w:
            for p_i, p in enumerate(poses):
                added_str = ""
                for r_i in range(3):
                    for c_i in range(4):
                        added_str += str(p[r_i,c_i])
                        if r_i != 2 or c_i != 3:
                            added_str += " "
                if p_i < poses.shape[0] - 1:
                    added_str += "\n"
                f_w.write(added_str)
        start_idx += seq_sizes[seq]

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args: 
        pose_path: (Complete) filename for the pose file
        Returns: 
        A numpy array of size nx4x4 with n poses as 4x4 transformation 
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)

def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)

def load_files(folder):
    """ Load all files in a folder and sort.
    """
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths