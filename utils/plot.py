import os

import numpy as np
import matplotlib.pyplot as plt

# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import process_data
import common

def plot_gt(Y_origin_data, pose_folder, preprocessed_folder, data_seqs, seq_sizes, dim="2d", save_graph=True, dataset="KITTI"):
    start_idx = 0
    end_idx = 0
    additional_row = np.array([0, 0, 0, 1], dtype=np.float64)
    for seq in data_seqs:
        end_idx += seq_sizes[seq]
        origin_poses = np.zeros((Y_origin_data[start_idx:end_idx].shape[0], 4,4),dtype=np.float64)
        for idx, row in enumerate(Y_origin_data[start_idx:end_idx]):
            new_pose = np.array(list(map(float, row.strip().split(" "))), dtype=np.float64)
            new_pose = np.concatenate((new_pose, additional_row))
            new_pose = new_pose.reshape(4,4)
            origin_poses[idx] = new_pose
        fig = plt.figure(figsize=(10,10))

        if dim == "2d":
            plt.scatter(origin_poses[:,0,3],origin_poses[:,1,3], c=origin_poses[:,2,3], s=20, alpha=0.5)
        else: # 3d
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(origin_poses[:,0,3],origin_poses[:,1,3],origin_poses[:,2,3],c=origin_poses[:,1,3], s=20, alpha=0.5)

        if save_graph:
            graph_folder = os.path.join('result', dataset, 'graph')
            os.makedirs(graph_folder, exist_ok=True)
            plt.savefig(os.path.join(graph_folder, f"gt_{seq}_{dim}.png"))
        # plt.close(fig)
        start_idx += seq_sizes[seq]
        
def plot_results(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dim="2d", save_graph=True, dataset="KITTI"):        
    start_idx = 0
    end_idx = 0
    additional_row = np.array([0, 0, 0, 1], dtype=np.float64)
    for i, seq in enumerate(data_seqs):
        end_idx += seq_sizes[seq]
        poses = np.zeros((Y_origin_data[start_idx:end_idx].shape[0], 4,4),dtype=np.float64)

        for idx in range(rnn_size):
            current_pose = np.array(list(map(float, Y_origin_data[start_idx+idx].strip().split(" "))), dtype=np.float64)
            current_pose = np.concatenate((current_pose, additional_row))
            current_pose = current_pose.reshape(4,4)
            poses[idx] = current_pose

        for idx, relative_pose in enumerate(Y_estimated_data[start_idx-i*rnn_size:end_idx-(i+1)*rnn_size]):
            rot_mat = common.euler_to_rot_mat(relative_pose[5],relative_pose[4],relative_pose[3])
            trans_mat = np.identity(4)
            trans_mat[:3,:3]=rot_mat
            trans_mat[0,3]=relative_pose[0]
            trans_mat[1,3]=relative_pose[1]
            trans_mat[2,3]=relative_pose[2]

            current_pose = np.dot(current_pose, trans_mat)
            poses[idx + rnn_size] = current_pose

        fig = plt.figure(figsize=(10,10))
        if dim == "2d":
            plt.scatter(poses[:,0,3],poses[:,1,3], c=poses[:,2,3], s=20, alpha=0.5)
        else: # 3d
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(poses[:,0,3],poses[:,1,3],poses[:,2,3],c=poses[:,1,3], s=20, alpha=0.5)

        if save_graph:
            graph_folder = os.path.join('result', dataset, 'graph')
            os.makedirs(graph_folder, exist_ok=True)
            plt.savefig(os.path.join(graph_folder, f"est_{seq}_{dim}.png"))
        # plt.close(fig)
        start_idx += seq_sizes[seq]