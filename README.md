# <img src="https://user-images.githubusercontent.com/73815549/157867543-bc994b16-5dda-4e30-bcff-55c522c91f50.png" width=75/> LoRCoN-LO: Long-term Recurrent Convolutional Network-based LiDAR Odometry
Video [[youtube]](https://youtu.be/_Ld58Rn_y-s)
## Download datasets
KITTI dataset (http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- velodyne laser data, calibration files, ground truth poses

Rellis-3D dataset (https://github.com/unmannedlab/RELLIS-3D).
- SemanticKITTI Format, Scan Poses files

## Setup directories
```bash
bash setup_directories.sh
```
Enter dataset name
```
KITTI # KITTI dataset
Rellis-3D # Rellis-3D dataset
```

## Move datasets
### KITTI
Move calib files (from 00 to 10)
```bash
mv data_odometry_calib/dataset/sequences/00/calib.txt data/KITTI/calib/00.txt
```
Move pose files (from 00 to 10)
```bash
mv data_odometry_poses/dataset/poses/00.txt data/KITTI/pose/00.txt
```
Move scan files (from 00 to 10)
```bash
mv data_odometry_velodyne/dataset/sequences/00/velodyne data/KITTI/scan/00/
```

### Rellis-3D
Move pose files (from 00 to 04)
```bash
mv Rellis_3D_lidar_poses_20210614/Rellis_3D/000000/poses.txt data/Rellis-3D/pose/00.txt
```
Move scan files (from 00 to 04)
```bash
mv Rellis_3D_os1_cloud_node_kitti_bin/Rellis_3D/000000/os1_cloud_node_kitti_bin data/Rellis-3D/scan/00/
```

## Setup the environment
```bash
conda create -n LoRCoN-LO python=3.8
conda activate LoRCoN-LO
pip install -r requirements.txt
```

## Change the config file
Change the config file (`config/config.yaml`) depending on your directory configuration.

## Pre-process
- transform ground truth poses from cam to vel
```bash
python preprocess/transform_poses_cam_to_vel.py
```

## Compute relative poses
```bash
python preprocess/relative_pose_calculator.py
```

## Generate input data
```bash
python preprocess/gen_data.py
```

## Train and Test
```bash
python train.py
python test.py
```

## Pre-trained models
KITTI model wass trained with 00 to 08 sequences.

- [Download](https://mysnu-my.sharepoint.com/:u:/g/personal/donghwijung_seoul_ac_kr/EbqmbaFUjVVPvYDUpvbhkTYBg4g9JmgIImRW-sq1B3oRRQ?e=uDcSjw)

Rellis-3D model was trained with 00 to 03 sequences.

- [Download](https://mysnu-my.sharepoint.com/:u:/g/personal/donghwijung_seoul_ac_kr/ETm4-EVG4jhCndW3U38fapgB__2eNsQwi8umnpyenXFN8w?e=bnn9rB)

When downloading and running the model, please modify the checkpoint related code in `confg/config.yaml`.

## Paper
```
@inproceedings{jung2023lorcon,
  title={LoRCoN-LO: Long-term Recurrent Convolutional Network-based LiDAR Odometry},
  author={Jung, Donghwi and Cho, Jae-Kyung and Jung, Younghwa and Shin, Soohyun and Kim, Seong-Woo},
  booktitle={2023 International Conference on Electronics, Information, and Communication (ICEIC)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement
The input data generation module is adapted from [Overlapnet](https://github.com/PRBonn/OverlapNet).

## License

Copyright 2022, Donghwi Jung, Jae-Kyung Cho, Younghwa Jung, Soohyun Shin, Seong-Woo Kim, Autonomous Robot Intelligence Lab, Seoul National University.

This project is free software made available under the MIT License. For details see the LICENSE file.
