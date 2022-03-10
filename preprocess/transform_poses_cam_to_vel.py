import numpy as np
import yaml

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from common import load_poses, load_calib

if __name__ == "__main__":

    config_filename = 'config/config.yml'

    config = yaml.load(open(config_filename), yaml.Loader)

    dataset = config["dataset"]
    if dataset == "KITTI":
        data_seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    else:
        data_seqs = ["00", "01", "02", "03", "04"]

    for seq in data_seqs:
        # set the related parameters
        poses_file = os.path.join(config["pose_folder"], seq + ".txt")
        scan_folder = os.path.join(config["scan_folder"], seq)

        # load calibrations
        if dataset == "KITTI":
            calib_file = os.path.join(config["calib_folder"], seq + ".txt")
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        else: # Rellis-3D
            T_cam_velo = np.array([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]) ## yaw 180 degree
            T_cam_velo = T_cam_velo.reshape(4,4)
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # load poses
        poses = load_poses(poses_file)
        pose0_inv = np.linalg.inv(poses[0])

        # for KITTI dataset, we need to convert the provided poses 
        # from the camera coordinate system into the LiDAR coordinate system  
        poses_vel = []
        for pose in poses:
            poses_vel.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
        
        poses = np.array(poses_vel)
        vel_poses_file = os.path.join(config["pose_folder"], seq + "_vel.txt")
        with open(vel_poses_file, "w") as f_w:
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