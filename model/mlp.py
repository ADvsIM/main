import numpy as np
import pandas as pd
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('/home/ppleeqq/IMvsAD/')
from utils.loss import *

from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, feature):
        super(MLP,self).__init__()

        if feature < 100:
            self.fc1 = nn.Linear(feature, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 8)
            self.fc4 = nn.Linear(8, 1)
            self.dropout = nn.Dropout(0.2)
        
        elif feature >= 100 :
            self.fc1 = nn.Linear(feature, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.dropout(self.fc4(x)))



class MLPClassifier(nn.Module):
    

    r"""
    FocalLoss : alpha, gamma ; FL(pt) = -alpha_t*((1-pt)^gamma)*log(pt)
    MFE, MSFE : X
    ClassBalancedLoss : beta, loss_type, (if focal: gamma) ; CB(p,y) = ((1-beta)/(1-beta^n_y)) * L(p,y) 
    """


    def __init__(self, loss_name = 'mfe', lr = 0.001,
                gamma = 10, alpha = [0.01, 1], beta = 0.5, loss_type='cross-entropy', device = 'cpu'):
        super(MLPClassifier, self).__init__()

        self.lr = lr
        self.loss_name = loss_name
        self.num_of_label_list = None
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.device = device
        self.batch_size = 0
        self.dataset = None

        self.model = None
        self.model_path = ''
        
    def _loss(self, gamma, alpha, beta, loss_type): 

        if self.loss_name == 'mfe': 
            criterion = MFELoss(device = self.device)
        elif self.loss_name == 'msfe':
            criterion = MSFELoss(device = self.device)
        elif self.loss_name == 'focal':
            criterion = FocalLoss(device = self.device ,gamma = gamma, alpha = alpha)
        elif self.loss_name == 'class-balanced':
            criterion = ClassBalancedLoss(device = self.device ,num_of_label_list = self.num_of_label_list, 
                                            beta = beta, loss_type = loss_type, gamma = gamma)
        return criterion

    def fit(self, dataset, epoch = 200, batch_size = 256, lr = 0.001):

        self.dataset = dataset
        self.num_of_label_list =np.unique(self.dataset.train_y, return_counts=True)[1].tolist()
        self.batch_size = batch_size

        # import pdb
        # pdb.set_trace()
        self.model = MLP(feature = self.dataset.train_data.shape[1])
        criterion = self._loss(self.gamma, self.alpha, self.beta, self.loss_type)
        # criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        loss_ = []
        self.model.to(self.device)
        self.model.train()

        train_dataloader = DataLoader(self.dataset.train_dataset, batch_size = self.batch_size, shuffle=True, drop_last = True)
        n = len(train_dataloader)

        for epoch in tqdm(range(epoch)):

            running_loss = 0

            for X, y in train_dataloader:
                optimizer.zero_grad()
                
                X=X.to(self.device)
                y=y.to(self.device)
                
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                # print(loss)
                optimizer.step()
                
                running_loss += loss.item()

            loss_.append(running_loss / n)
            #print([x for x in self.model.parameters()][5])

        self.model_path = '/home/ppleeqq/IMvsAD/log/model_parameter/model.pt'
        torch.save(self.model.state_dict(),self.model_path )

        plt.plot(loss_)
        plt.title('Loss')
        plt.xlabel('epoch')
        now = datetime.now()
        formatted = now.strftime('%m-%d-%H-%M-%S')
        plt.savefig(f'/home/ppleeqq/IMvsAD/log/plot/{self.loss_name}_loss_plot{formatted}.png')
        plt.show()
        
                
    
    def predict_pro(self, test_dataset, load_parameter = False):

        # test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last = False)

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        y_hat_total = self.model(torch.tensor(test_dataset).float().to(self.device))

        return y_hat_total
    
    def predict(self, test_dataset, load_parameter = False):

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        y_hat_total = self.model(torch.tensor(test_dataset).float().to(self.device))
        
        threshold = 0.5
        out_binary = torch.where(y_hat_total > threshold, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))

        return out_binary
            
