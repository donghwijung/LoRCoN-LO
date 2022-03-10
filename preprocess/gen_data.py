import os
import yaml

import gen_data_utils

from tqdm import tqdm

def gen_data(scan_folder, dst_folder, fov_up, fov_down, proj_H, proj_W, max_range):
  gen_data_utils.gen_depth_data(scan_folder, dst_folder, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)[0]
  gen_data_utils.gen_normal_data(scan_folder, dst_folder, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)[0]
  gen_data_utils.gen_intensity_data(scan_folder, dst_folder, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)[0]

if __name__ == "__main__":
    # load config file
    config_filename = 'config/config.yml'
    
    config = yaml.load(open(config_filename), yaml.Loader)

    data_seqs = config["data_seqs"].split(",")

    for seq in tqdm(data_seqs):
        
        # set the related parameters
        scan_folder = os.path.join(config["scan_folder"], seq)
        # scan_folder = os.path.join(config["scan_folder"], seq, "velodyne")
        # scan_folder = os.path.join(config["scan_folder"], seq, "os1_cloud_node_kitti_bin")
        dst_folder = os.path.join(config["preprocessed_folder"], seq)
        
        os.makedirs(scan_folder, exist_ok=True)
        os.makedirs(dst_folder, exist_ok=True)

        fov_up = config["fov_up"]
        fov_down = config["fov_down"]
        proj_H = config["proj_H"]
        proj_W = config["proj_W"]
        max_range = config["max_range"]
        
        # start the demo1 to generate different types of data from LiDAR scan
        gen_data(scan_folder, dst_folder, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range)
