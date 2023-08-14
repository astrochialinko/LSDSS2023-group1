import utils
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

Net_name = 'VGG1' # 'ResNet1'
in_channels = 1
n_classes = 3
num_epochs = 20 
channels = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_data = True

# Get the data
batch_size = 16
if load_data:
    X_train = np.transpose(np.load('../Data/X_train.npy'), axes=(0,3,1,2)) 
    y_train = np.load('../Data/y_train.npy')
    X_test = np.transpose(np.load('../Data/X_test.npy'), axes=(0,3,1,2)) 
    y_test = np.load('../Data/y_test.npy')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    X_val = np.transpose(X_val, axes=(0,3,1,2)) 
    train_ds = utils.MyDataset(X_train, y_train, norm=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    val_ds = utils.MyDataset(X_val, y_val, norm=False) 
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
    test_ds = utils.MyDataset(X_test, y_test, norm=False) 
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
    utils.check_inputs(train_ds, train_loader, savefig=False, name=Net_name)

# Define model, optimizer, and transforms
model = utils.MyVGG16(num_classes=n_classes).to(device) # utils.ResNet(ResidualBlock, layers=[3, 4, 6, 3]).to(device)
learning_rate = 0.005 # 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)  
loss_fn = nn.CrossEntropyLoss() 

# Train
print('Training:')
for epoch in range(num_epochs):
    print(f'\tEpoch {epoch}')

    # Train (not saving snapshot)
    utils.train(train_loader, model, loss_fn, optimizer, device)
    
    # check accuracy 
    accuracy, = utils.validate(val_loader, model, device)
    print(f"\tGot validation accuracy {accuracy:.2f}")
    model.train() # set model back into train mode

# Save model 
torch.save(model.state_dict(), f'../NN_storage/{Net_name}.pth')
print(f'Saving trained model as {Net_name}.pth')

# Load it back in and compute results on test set
model = utils.MyVGG16(in_channels, out_channels=n_classes)
model.load_state_dict(torch.load(f'../NN_storage/{Net_name}.pth'))
accuracy = utils.validate(test_loader, model=model)