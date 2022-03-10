import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRCoNLO(nn.Module):
    def __init__(self, batch_size, batchNorm=True):
        super(LoRCoNLO,self).__init__()
        
        self.batch_size = batch_size
        
        self.simple_conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, stride=(1,2), padding=(1,0)).to("cuda:1")
        self.simple_conv2 = nn.Conv2d(32, 64, 3, (1,2), (1,0)).to("cuda:1")
        self.simple_conv3 = nn.Conv2d(64, 128, 3, (1,2), (1,0)).to("cuda:1")
        self.simple_conv4 = nn.Conv2d(128, 256, 3, (2,2), (1,0)).to("cuda:1")
        self.simple_conv5 = nn.Conv2d(256, 512, 3, (2,2), (1,0)).to("cuda:1")
        self.simple_conv6 = nn.Conv2d(512, 128, 1, 1, (1,0)).to("cuda:1")
        
        # RNN
        self.rnn = nn.LSTM(
                    input_size=128*306,
                    hidden_size=1024, 
                    num_layers=4, 
                    dropout=0, 
                    batch_first=True,
                    bidirectional=True).to("cuda:0")
        self.rnn_drop_out = nn.Dropout(0.4)
        
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(2048, 512).to("cuda:1")
        # self.fc2 = nn.Linear(512, 128).to("cuda:1")
        # self.fc3 = nn.Linear(128, 64).to("cuda:1")
        # self.fc4 = nn.Linear(64, 16).to("cuda:1")
        # self.fc5 = nn.Linear(16, 6).to("cuda:1")
        self.fc1 = nn.Linear(2048, 6).to("cuda:1")
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))
        
        self.conv_bn1 = nn.BatchNorm2d(32).to("cuda:1")
        self.conv_bn2 = nn.BatchNorm2d(64).to("cuda:1")
        self.conv_bn3 = nn.BatchNorm2d(128).to("cuda:1")
        self.conv_bn4 = nn.BatchNorm2d(256).to("cuda:1")
        self.conv_bn5 = nn.BatchNorm2d(512).to("cuda:1")
        self.conv_bn6 = nn.BatchNorm2d(128).to("cuda:1")

    def forward(self, x):
        batch_size = x.size(0)
        rnn_size = x.size(1)
        
        x = x.view(batch_size * rnn_size, x.size(2), x.size(3), x.size(4))
        
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.maxpool(x)
        
        # CNN
        x = self.encode_image(x)
        
        x = x.view(batch_size, rnn_size, -1)
        
        x, hc = self.rnn(x.to("cuda:0"))

        x = self.rnn_drop_out(x)
        
        x = x.reshape(batch_size * rnn_size, -1)
        
        output = self.fc_part(x.to("cuda:1"))
        
        output = output.reshape(batch_size, rnn_size, -1)

        return output
    
    def encode_image(self, x):
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv1(x)
        x = self.conv_bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv2(x)
        x = self.conv_bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv3(x)
        x = self.conv_bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv4(x)
        x = self.conv_bn4(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv5(x)
        x = self.conv_bn5(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv6(x)
        x = self.conv_bn6(x)
        x = F.leaky_relu(x, 0.1)
        return x
    
    def fc_part(self, x):
        x = F.leaky_relu(x, 0.2)
        x = self.fc1(x)
        # x = F.leaky_relu(x, 0.2)
        # x = self.fc1(x)
        # x = F.leaky_relu(x, 0.2)
        # x = self.fc2(x)
        # x = F.leaky_relu(x, 0.2)
        # x = self.dropout1(x)
        # x = self.fc3(x)
        # x = F.leaky_relu(x, 0.2)
        # x = self.dropout2(x)
        # x = self.fc4(x)
        # x = F.leaky_relu(x, 0.2)
        # x = self.fc5(x)
        return x
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class WeightedLoss(nn.Module):
    def __init__(self, learn_hyper_params=True, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = 100

    def forward(self, pred, target):
        L_t = F.mse_loss(pred[:,:,:3], target[:,:,:3])
        L_r = F.mse_loss(pred[:,:,3:], target[:,:,3:])
        loss = L_t + L_r * self.w_rot
        return loss
    
def RMSEError(pred, label):
    return torch.sqrt(torch.mean((pred-label)**2))