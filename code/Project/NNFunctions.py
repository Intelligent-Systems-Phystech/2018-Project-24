
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F 
  
class IndConv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mode='nearest'):
        super(IndConv2, self).__init__()
        
        self.mode = mode
        self.in_channels = in_channels
        self.A00 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A01 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A02 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A10 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A11 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A12 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A20 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A21 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.A22 = nn.Conv2d(in_channels, out_channels, (3,3))
    
    def forward(self, x):
        ins = F.interpolate(x.reshape(x.shape[0], self.in_channels, 3, 3), scale_factor=3, mode=self.mode)
        out = torch.ones((x.shape[0], self.in_channels, 3, 3)).type(torch.cuda.FloatTensor)
    
        out[:, :, 0, 0] = self.A00(ins[:, :, 2:5, 2:5])[:, :, 0, 0]
        out[:, :, 0, 2] = self.A02(ins[:, :, 2:5, 4:7])[:, :, 0, 0]
        out[:, :, 2, 2] = self.A22(ins[:, :, 4:7, 4:7])[:, :, 0, 0]
        out[:, :, 2, 0] = self.A20(ins[:, :, 4:7, 2:5])[:, :, 0, 0]
        out[:, :, 1, 0] = self.A10(ins[:, :, 3:6, 2:5])[:, :, 0, 0]
        out[:, :, 0, 1] = self.A01(ins[:, :, 2:5, 3:6])[:, :, 0, 0]
        out[:, :, 1, 2] = self.A12(ins[:, :, 3:6, 4:7])[:, :, 0, 0]
        out[:, :, 2, 1] = self.A21(ins[:, :, 4:7, 3:6])[:, :, 0, 0]
        out[:, :, 1, 1] = self.A11(ins[:, :, 3:6, 3:6])[:, :, 0, 0]

        return out
    
def net_training(net, train_loader, test_loader, criterion, learning_rate, N, train_MSE=None, test_MSE=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if train_MSE is None:
        train_MSE = []
    if test_MSE is None:
        test_MSE = []
    for iter in tqdm.tqdm_notebook(range(N)):
        epoch_MSE = 0.0
        epoch_iter = 0
        error = []
        for item in train_loader:

            inputs = torch.tensor(item[0], dtype=torch.float32)
            labels = torch.tensor(item[1], dtype=torch.float32)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            epoch_MSE += F.mse_loss(outputs.squeeze(1), labels)
            epoch_iter += 1
        errors = 0.
        for test_item in test_loader:
            inputs = torch.tensor(test_item[0], dtype=torch.float32)
            labels = torch.tensor(test_item[1], dtype=torch.float32)
            outputs = net(inputs)
            errors += F.mse_loss(outputs.squeeze(1), labels)
        test_MSE.append(errors.data)
        epoch_MSE /= epoch_iter
        train_MSE.append(epoch_MSE.data)
    return train_MSE, test_MSE

def net_cuda_training(net, train_loader, test_loader, criterion, learning_rate, N, train_MSE=None, test_MSE=None):
    """ Train model on GPU. Be careful maybe you have to change some tensors type to cuda.FloatTensor.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if train_MSE is None:
        train_MSE = []
    if test_MSE is None:
        test_MSE = []
    for iter in tqdm.tqdm_notebook(range(N)):
        epoch_MSE = 0.0
        epoch_iter = 0
        error = []
        for item in train_loader:

            inputs = item[0].type(torch.cuda.FloatTensor)
            labels = item[1].type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            epoch_MSE += F.mse_loss(outputs.squeeze(1), labels)
            epoch_iter += 1
        errors = 0.
        for test_item in test_loader:
            inputs = item[0].type(torch.cuda.FloatTensor)
            labels = item[1].type(torch.cuda.FloatTensor)
            outputs = net(inputs)
            errors += F.mse_loss(outputs.squeeze(1), labels)
        test_MSE.append(errors.data)
        epoch_MSE /= epoch_iter
        train_MSE.append(epoch_MSE.data)
    return train_MSE, test_MSE

def rec_pic(net, X):
    """ Return recovered picture in numpy array format.
    X is array of pictures wich sizes satisfy to net.
    """
    return net(torch.cuda.FloatTensor(X)).squeeze(1).type(torch.FloatTensor).detach().numpy()