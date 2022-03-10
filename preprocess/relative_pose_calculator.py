import numpy as np
import yaml

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from common import rot_mat_to_euler

if __name__ == "__main__":
    config_filename = 'config/config.yml'
    config = yaml.load(open(config_filename), yaml.Loader)
    
    poses_path = config["pose_folder"]
    relative_poses_path = config["relative_pose_folder"]
    file_list = os.listdir(poses_path)
    file_list.sort()

    additional_row = np.array([0, 0, 0, 1], dtype=np.float64)
    for f_name in file_list:
        if "_vel" in f_name:
            file_path = os.path.join(poses_path, f_name)
            with open(file_path, "r") as f:
                lines = f.readlines()
                lines_len = len(lines)
                relative_pose_file_path = os.path.join(relative_poses_path, f_name.split("_")[0] + ".txt")
                with open(relative_pose_file_path, "w") as f_w:
                    rot_mats = np.zeros((lines_len, 4, 4), dtype=np.float64)
                    for idx, line in enumerate(lines):
                        rot_mat = np.array(list(map(float, line.strip().split(" "))), dtype=np.float64)
                        rot_mat = np.concatenate((rot_mat, additional_row))
                        rot_mat = rot_mat.reshape(4,4)

                        rot_mats[idx] = rot_mat

                    for i in range(rot_mats.shape[0]):
                        if i < rot_mats.shape[0] - 1:
                            rot_mat_pre = rot_mats[i]
                            rot_mat_nxt = rot_mats[i+1]

                            rot_mat_pre_inv = np.linalg.inv(rot_mat_pre)
                            rot_mat_pre_inv.dtype = np.float64
                            relative_transform = np.dot(rot_mat_pre_inv, rot_mat_nxt)
                            (x, y, z) = (relative_transform[0,3], relative_transform[1,3], relative_transform[2,3])
                            (roll, pitch, yaw) = rot_mat_to_euler(relative_transform[:3,:3])

                            added_str = str(x) + " " + str(y) + " " + str(z) + " " + str(roll) + " " + str(pitch) + " " + str(yaw)
                            if i < rot_mats.shape[0] - 1 - 1:
                                added_str += "\n"
                            f_w.write(added_str)