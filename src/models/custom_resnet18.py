import torch
import torch.nn as nn
from torchvision.models import resnet18
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        
        # Expose specific layers as attributes
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        
        # Define new layers
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, 100)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Forward through custom layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x