# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:42:26 2022

@author: thoma
"""

#%% import
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
import sys
import os

pathModel = "../sources/model/model-V1-TP20"
sys.path.append(os.path.abspath(pathModel))
from initModel import Net, xdef

pathFunctions = "../sources"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f


#%% fonctions ------------------------------
def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def calculate_accuracy_percent(x_test, y_test, per):
    y_test_pred = net(x_test)
    y_test_pred = torch.squeeze(y_test_pred)
    i = 0
    nb_test = 0
    test_good = 0

    for ft in y_test_pred :
        ft= ft.item()
        if(ft >= (0.5+per)) :
            nb_test = nb_test + 1
            if(y_test[i] == 1.0):
                test_good = test_good + 1
        elif(ft <= (0.5-per)) :
            nb_test = nb_test + 1
            if(y_test[i] == 0.0):
                test_good = test_good + 1
        i = i + 1

    if(nb_test):
        print("nombre de test :", nb_test, " / ", len(x_test))
        print("test à +/-",per,"% : ", round((test_good/nb_test*100),1),"% de réussite")
        print("soit ",nb_test/4, " trade par ans." )
        return test_good/nb_test, nb_test
    else :
        print("Pas de résultat.")
        return -1,-1

#%% variables
nbEpoch = 500
lr=0.005
pathTrainData = '../data/trainData/allData.csv'

#%% model IA ------------------------------
df = pd.read_csv(pathTrainData)
x = df[xdef] #sans var c'est mieux
y = df[['y']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

x_test = torch.from_numpy(x_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

#definition du model
net = Net(x_train.shape[1]) # l'importer du .py du model
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr)

# to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)


#%% train

for epoch in range(nbEpoch):
    y_pred = net(x_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)

    if epoch % 100 == 0:
      train_acc = calculate_accuracy(y_train, y_pred)

      y_test_pred = net(x_test)
      y_test_pred = torch.squeeze(y_test_pred)

      test_loss = criterion(y_test_pred, y_test)

      test_acc = calculate_accuracy(y_test, y_test_pred)
      print(
f'''epoch {epoch}
Train set - loss: {f.round_tensor(train_loss)}, accuracy: {f.round_tensor(train_acc)}
Test  set - loss: {f.round_tensor(test_loss)}, accuracy: {f.round_tensor(test_acc)}
''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

torch.save(net, pathModel+"model.pth")
net = torch.load(pathModel+"model.pth")

calculate_accuracy_percent(x_test, y_test,0)
calculate_accuracy_percent(x_test, y_test,0.15)





