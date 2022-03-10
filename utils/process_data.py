import os
import numpy as np

import torch
from torch.utils.data import Dataset

def count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes, depth_name="depth"):
    for seq in data_seqs:
        parent_directory = os.path.join(preprocessed_folder, seq)
        npy_directory = os.path.join(parent_directory, depth_name)
        npy_names = os.listdir(npy_directory)
        npy_names_len = len(npy_names)
        seq_sizes[seq] = npy_names_len
    
    return seq_sizes

def process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes, depth_name="depth"):
    
    Y_data = np.empty((0, 2), dtype=object)

    for idx, seq in enumerate(data_seqs):
        with open(os.path.join(relative_pose_folder, seq + ".txt"), "r") as f:       
            parent_directory = os.path.join(preprocessed_folder, seq)
            npy_directory = os.path.join(parent_directory, depth_name)
            npy_names = os.listdir(npy_directory)
            npy_names.sort()
            npy_names_len = len(npy_names)

            lines = f.readlines()
            Y_row = np.zeros((len(lines), 2), dtype=object)
            for i, line in enumerate(lines):
                Y_row[i, 0] = seq + " " + npy_names[i]
                Y_row[i, 1] = line
                
            Y_data = np.vstack((Y_data, Y_row))
    return Y_data

class LoRCoNLODataset(Dataset):
    def __init__(self, img_dir, Y_data, data_idx, seq_sizes, rnn_size, width, height, depth_name, intensity_name, normal_name, dni_size, normal_size):
        self.img_dir = img_dir
        self.Y_data = Y_data
        self.seq_sizes = seq_sizes
        self.rnn_size = rnn_size
        self.data_idx = data_idx
        self.width = width
        self.height = height
        self.depth_name = depth_name
        self.intensity_name = intensity_name
        self.normal_name = normal_name
        self.dni_size = dni_size
        self.normal_size = normal_size

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        start_id = self.data_idx[idx]
        image_pre_path_args = self.Y_data[start_id, 0].split(" ")
        current_seq = image_pre_path_args[0]
        current_npy_filename = image_pre_path_args[1]
        current_npy_id = int(image_pre_path_args[1].split(".")[0])
        current_seq_size = self.seq_sizes[current_seq]
        images = np.zeros((self.rnn_size, 10, self.height, self.width), dtype=np.float32)
        labels = np.zeros((self.rnn_size, 6), dtype=np.float64)
        
        for i in range(self.rnn_size):
            current_npy_filename = "{:06d}".format(current_npy_id + i) + ".npy"
            nxt_npy_filename = "{:06d}".format(current_npy_id + i + 1) + ".npy"
            images_wrapper = np.zeros((10, self.height, self.width), dtype=np.float32)
            
            names_list = [self.depth_name, self.intensity_name, self.normal_name]
            for idx, n in enumerate(names_list):
                image_pre_path = os.path.join(self.img_dir, current_seq)
                image_pre_path = os.path.join(image_pre_path, n)
                image_pre_path = os.path.join(image_pre_path, current_npy_filename)

                image_pre = np.load(image_pre_path)
                image_pre = image_pre.astype("float32")
                if idx == len(names_list) - 1:
                    image_pre = image_pre.transpose((2, 0, 1))

                image_nxt_path = os.path.join(self.img_dir, current_seq)
                image_nxt_path = os.path.join(image_nxt_path, n)
                image_nxt_path = os.path.join(image_nxt_path, nxt_npy_filename)

                try:
                    image_nxt = np.load(image_nxt_path)
                    image_nxt = image_nxt.astype("float32")
                    if idx == len(names_list) - 1:
                        image_nxt = image_nxt.transpose((2, 0, 1))

                    if idx < 2:
                        image_pre /= 255.0
                        image_nxt /= 255.0
                    else:
                        image_pre = (image_pre + 1.0) / 2.0
                        image_nxt = (image_nxt + 1.0) / 2.0

                except FileNotFoundError:
                    print(idx, start_id, image_nxt_path, current_npy_id, i)

                image_pre.dtype = np.float32
                image_nxt.dtype = np.float32
                if idx < 2: # depth, intensity
                    images_wrapper[idx] = image_pre
                    images_wrapper[idx+self.dni_size] = image_nxt
                else: # normal
                    images_wrapper[idx:idx+self.normal_size] = image_pre
                    images_wrapper[idx+self.dni_size:] = image_nxt

            label = np.array(list(map(float, self.Y_data[start_id + i, 1].strip().split(" "))), dtype=np.float64)
            labels[i] = label
            images[i] = images_wrapper
        labels = torch.tensor(labels)
        images = torch.tensor(images)
        return images, labels