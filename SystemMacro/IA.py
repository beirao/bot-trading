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
import torch.nn.functional as F


#%% fonctions ------------------------------
def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)



def calculate_accuracy_percent(x_test, y_test, per):
    y_test_pred = net(x_test)
    y_test_pred = torch.squeeze(y_test_pred)
    i = 0
    nb_test = 0
    test_good = 0

    for f in y_test_pred :
        f= f.item()
        if(f >= (0.5+per)) :
            nb_test = nb_test + 1
            if(y_test[i] == 1.0):
                test_good = test_good + 1
        elif(f <= (0.5-per)) :
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



#%% model IA ------------------------------

#importation data
df = pd.read_csv('../data/allData.csv')

x = df[["rsi14","var","ma25","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]] #sans var c'est mieux
y = df[['y']]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

x_test = torch.from_numpy(x_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

#definition du model

class Net(nn.Module):
    def __init__(self,enter):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(enter,32)
        self.fc2 = nn.Linear(32, 54)
        self.fc3 = nn.Linear(54, 40)
        self.fc4 = nn.Linear(40, 10)
        self.fcf = nn.Linear(10, 1) #si 1 hausse si 0 baisse

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fcf(x))
        return x

net = Net(x_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#device = "cuda:0"

x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)

net = net.to(device)
criterion = criterion.to(device)





#%% train

for epoch in range(1000):
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
Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

MODEL_PATH = '../model/modelTP20.pth'
torch.save(net, MODEL_PATH)
net = torch.load(MODEL_PATH)

calculate_accuracy_percent(x_test, y_test,0)
calculate_accuracy_percent(x_test, y_test,0.15)





