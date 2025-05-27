import torch
import torch.nn as nn
import torch.nn.functional as F
class DNN(nn.Module):
        def __init__(self, input_dim, n_outputs=10, dropout_rate=0.25, top_bn=False):
            super(DNN, self).__init__()
            self.dropout_rate = dropout_rate
            self.top_bn = top_bn
            
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 256)
            self.fc5 = nn.Linear(256, 128)
            self.fc6 = nn.Linear(128, 128)
            self.fc7 = nn.Linear(128, 64)
            self.fc8 = nn.Linear(64, 64)
            self.fc9 = nn.Linear(64, 32)
            self.fc10 = nn.Linear(32, n_outputs)
            
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(512)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(256)
            self.bn5 = nn.BatchNorm1d(128)
            self.bn6 = nn.BatchNorm1d(128)
            self.bn7 = nn.BatchNorm1d(64)
            self.bn8 = nn.BatchNorm1d(64)
            self.bn9 = nn.BatchNorm1d(32)
            self.bn10 = nn.BatchNorm1d(n_outputs)

        def forward(self, x):
            h = x
            h = F.leaky_relu(self.bn1(self.fc1(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn2(self.fc2(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn3(self.fc3(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn4(self.fc4(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn5(self.fc5(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn6(self.fc6(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn7(self.fc7(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn8(self.fc8(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            h = F.leaky_relu(self.bn9(self.fc9(h)), negative_slope=0.01)
            h = F.dropout(h, p=self.dropout_rate)
            
            logit = self.fc10(h)
            if self.top_bn:
                logit = self.bn10(logit)
            return logit