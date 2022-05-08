# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:08:48 2022

@author: thoma
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, sys

pathModel = "pre-trained_model"
sys.path.append(os.path.abspath(pathModel))
from initModel import Linear_QNet, xdef, initModel


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.lr = lr
        self.gamma = gamma
        
        self.model = Linear_QNet
        self.model = initModel(pathModel+"/model.pth").to(self.device)
        # self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.device)


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        # print(pred)
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][int(torch.sigmoid(action[idx]).item())-1] = Q_new    

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

