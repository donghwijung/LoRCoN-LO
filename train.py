import os
import yaml
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import model as model_util
import process_data
import common

if __name__ == "__main__":
    config_filename = 'config/config.yml'
    config = yaml.load(open(config_filename), yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda_visible_devices"]
    
    cuda = torch.device('cuda')
    seq_sizes= {}
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    torch.set_num_threads(num_workers)

    preprocessed_folder = config["preprocessed_folder"]
    # preprocessed_folder = config["preprocessed_folder_prev"]
    
    relative_pose_folder = config["relative_pose_folder"]

    dataset = config["dataset"]

    data_seqs = config["data_seqs"].split(",")
    test_seqs = config["test_seqs"].split(",")

    rnn_size = config["rnn_size"]
    image_width = config["image_width"]
    image_height = config["image_height"]

    depth_name = config["depth_name"]
    intensity_name = config["intensity_name"]
    normal_name = config["normal_name"]
    dni_size = config["dni_size"]
    normal_size = config["normal_size"]

    seq_sizes = process_data.count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
    Y_data = process_data.process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes)
    
    start_idx = 0
    end_idx = 0
    train_idx = np.array([], dtype=int)
    test_idx = np.array([], dtype=int)
    for seq in data_seqs:
        end_idx += seq_sizes[seq] - 1
        if seq in test_seqs:
            test_idx = np.append(test_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        else:
            train_idx = np.append(train_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        start_idx += seq_sizes[seq] - 1

    training_data = process_data.LoRCoNLODataset(preprocessed_folder, Y_data, train_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
    test_data = process_data.LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)

    train_dataloader = DataLoader(training_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    
    model = model_util.LoRCoNLO(batch_size=batch_size, batchNorm=False)
    
    criterion = model_util.WeightedLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

    from_checkpoint = config["from_checkpoint"]
    is_same_dataset = config["is_same_dataset"]
    start_epoch = 1
    if from_checkpoint:
        cp_folder = config["cp_folder"]
        cp_folder = os.path.join(cp_folder, dataset)
        checkpoint_path = os.path.join(cp_folder, config["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_same_dataset:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Train from checkpoint {checkpoint_path}, start_epoch: {start_epoch}")
    else:
        print("Train from scratch")
    
    log_folder = config["log_folder"]
    log_folder = os.path.join(log_folder, dataset)
    os.makedirs(log_folder, exist_ok=True)
    _, prev_log_dirs, _ = next(os.walk(log_folder))
    new_log_dir = os.path.join(log_folder, str(len(prev_log_dirs)).zfill(4))
    os.makedirs(new_log_dir, exist_ok=True)
    writer = SummaryWriter(new_log_dir)

    cp_folder = config["cp_folder"]
    cp_folder = os.path.join(cp_folder, dataset)
    os.makedirs(cp_folder, exist_ok=True)
    _, prev_cp_dirs, _ = next(os.walk(cp_folder))
    new_cp_dir = os.path.join(cp_folder, str(len(prev_cp_dirs)).zfill(4))
    os.makedirs(new_cp_dir, exist_ok=True)
    model_path = os.path.join(new_cp_dir, "cp-{epoch:04d}.pt")

    data_loader_len = len(train_dataloader)
    test_data_loader_len = len(test_dataloader)

    epochs = config["epochs"]
    log_epoch = config["log_epoch"]
    cp_epoch = config["checkpoint_epoch"]
    for epoch in tqdm(range(start_epoch, epochs+1)):  # loop over the dataset multiple times
        model.train()
        criterion.train()
        running_loss = 0.0
        rmse_error_train = 0.0
        rmse_t_error_train = 0.0
        rmse_r_error_train = 0.0
        for i, data in tqdm(enumerate(train_dataloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = Variable(inputs.float().to('cuda:1')), Variable(labels.float().to('cuda:1'))

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            rmse_error_train += model_util.RMSEError(outputs, labels).item()
            rmse_t_error_train += model_util.RMSEError(outputs[:,:,:3], labels[:,:,:3]).item()
            rmse_r_error_train += model_util.RMSEError(outputs[:,:,3:], labels[:,:,3:]).item()
        if epoch % log_epoch == 0:
            print('[%d, %5d] training loss: %.10f' %
                  (epoch + 1, i + 1, running_loss / data_loader_len))
            writer.add_scalar('Loss/train', running_loss / data_loader_len, epoch)
            writer.add_scalar('RMSE/train', rmse_error_train / data_loader_len, epoch)
            writer.add_scalar('RMSE_t/train', rmse_t_error_train / data_loader_len, epoch)
            writer.add_scalar('RMSE_r/train', rmse_r_error_train / data_loader_len, epoch)

            model.eval()
            criterion.eval()
            with torch.no_grad():
                test_loss = 0
                rmse_error_test = 0
                rmse_t_error_test = 0
                rmse_r_error_test = 0
                for t_i, t_data in enumerate(test_dataloader):
                    t_inputs, t_labels = t_data
                    t_inputs, t_labels = Variable(t_inputs.float().to('cuda:1')), Variable(t_labels.float().to('cuda:1'))
                    t_outputs = model(t_inputs)

                    t_loss = criterion(t_outputs, t_labels)

                    test_loss += t_loss.item()
                    rmse_error_test += model_util.RMSEError(t_outputs, t_labels).item()
                    rmse_t_error_test += model_util.RMSEError(t_outputs[:,:,:3], t_labels[:,:,:3]).item()
                    rmse_r_error_test += model_util.RMSEError(t_outputs[:,:,3:], t_labels[:,:,3:]).item()
                print('[%d, %5d] validation loss: %.10f' %
                  (epoch + 1, i + 1, test_loss / test_data_loader_len))
                writer.add_scalar('Loss/val', test_loss / test_data_loader_len, epoch)
                writer.add_scalar('RMSE/val', rmse_error_test / test_data_loader_len, epoch)
                writer.add_scalar('RMSE_t/val', rmse_t_error_test / test_data_loader_len, epoch)
                writer.add_scalar('RMSE_r/val', rmse_r_error_test / test_data_loader_len, epoch)

        if epoch % cp_epoch == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
            }, model_path.format(epoch=epoch))
            print("Model saved in ", model_path.format(epoch=epoch))

    print('Finished Training')