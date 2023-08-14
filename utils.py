
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

#### Dataset ####

class MyDataset(Dataset):
    def __init__(self, images, labels, norm=False):
        self.images = images
        self.labels = labels
        self.norm = norm

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        img = self.images[index,:,:,:] 
        transf = transforms.Resize((227, 227))
        img = transf(torch.from_numpy(img)).detach().numpy()
        if self.norm:  # normalize 
            img = (img - np.mean(img))/np.std(img) # normalize to std normal dist
        image = np.array(img, dtype=np.float32)
        # Get labels
        label = self.labels[index]
        mask = np.zeros((3), dtype=np.float32) # One-hot encode label 
        if label == '': mask[0] = 1
        elif label == '': mask[1] = 1
        elif label == '': mask[2] = 1
        return image, mask
    
def check_inputs(train_ds, train_loader, savefig=False, name=None):
    ''' 
    Check data is loaded correctly
    '''
    print('Train data:')
    print(f'     {len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_features, train_labels = next(iter(train_loader))
    print(train_features.dtype)
    shape = train_features.size()
    print(f'     Each batch has data of shape {train_features.size()}, e.g. {shape[0]} images, {[shape[2], shape[3]]} pixels each, {shape[1]} layers (features)')
    shape = train_labels.size()
    print(f'     Each batch has labels of shape {train_labels.size()}, e.g. {shape[0]} images, {shape[1]} layers (classes)') # {[shape[2], shape[3]]} pixels each, {shape[1]} layers (classes)')
    if savefig:
        fig, axs = plt.subplots(4, 1)
        axs[0].set_title('images')
        for i in range(0, 8, 2):
            X, y = next(iter(train_loader))
            im1 = axs[i].imshow(X[0,0,:,:]); plt.colorbar(im1, ax=axs[i]) # first img in batch, first channel
        plt.savefig(f'traindata_{name}')

#### Models ####

class DoubleConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_chans)
        self.relu  = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.relu(self.batchnorm(self.conv1(X)))
        X = self.relu(self.batchnorm(self.conv2(X)))
        X = self.maxpool(X)
        return X
    
class TripleConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_chans)
        self.relu  = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.relu(self.batchnorm(self.conv1(X)))
        X = self.relu(self.batchnorm(self.conv2(X)))
        X = self.relu(self.batchnorm(self.conv3(X)))
        X = self.maxpool(X)
        return X
    
class MyVGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(MyVGG16, self).__init__() 
        self.convs = nn.Sequential(
            DoubleConv(3, 64), 
            DoubleConv(64, 128), 
            TripleConv(128, 256), 
            TripleConv(256, 512), 
            TripleConv(512, 512))
        self.linear1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096), # 4096
            nn.ReLU())
        self.linear2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), # 4096
            nn.ReLU())
        self.linear3 = nn.Sequential(
            nn.Linear(4096, num_classes)) # 4096
        
    def forward(self, X):
        out = self.convs(X) # [16, 3, 227, 227] -> [16, 512, 7, 7]
        out = out.reshape(out.size(0), -1) # [16, 512, 7, 7] -> [16, 25088]
        out = self.linear1(out) # [16, 25088] -> [16, 2048]
        out = self.linear2(out) # [16, 2048] -> [16, 2048]
        out = self.linear3(out) # [16, 2048] -> [16, 3]
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class MyResNet(nn.Module):
    def __init__(self, layers, num_classes = 10):
        super(MyResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(ResBlock, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(ResBlock, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(ResBlock, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(ResBlock, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#### Training ####
 
def train(train_loader, model, loss_fn, optimizer, device):
    '''
    Train one epoch
    '''
    for i, (images, labels) in enumerate(train_loader):  
        print(f'\t   Batch {i}', end='\r')
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(loader, model, device):
    '''
    Compute results on validation set
    '''
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        accuracy = correct/total
        return accuracy


#### Misc ####

