# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:34:53 2022

@author: thoma
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

#%% fonctions ------------------


#%% Chargement du model deja entrainé
MODEL_PATH = '../model/modelTP20.pth'

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

netl = torch.load(MODEL_PATH)

#importation data
df = pd.read_csv('../data/btcData.csv')

x = df[["rsi14","var","ma25","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]] #sans var c'est mieux
y = df[['y']]
p = df[['open']]

x = torch.tensor(x.values).float()
y = torch.tensor(y.values).float()
p = torch.tensor(p.values).float()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

x = x.to(device)
y = y.to(device)
netl = netl.to(device)


#%% simu ------------------------------
y_pred = netl(x)
startWallet = 1000
walletUSD = startWallet #USDT

fee = 0.001
buy = False
nbTrade = 0
soldeCOIN = 0
soldeCOIN_HOLD = (startWallet/p[0])-(startWallet/p[0]*fee)

py = []
py = np.array(py)

threshold = 0.3

for i in range(len(x)) :
    if y_pred[i] > 0.5 + threshold:
        if not buy :
            soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
            buy = True

        py = np.append(py,(soldeCOIN*p[i]))
        walletUSD = (soldeCOIN*p[i])

        nbTrade = nbTrade + 1

    elif y_pred[i] <= 0.5 - threshold:
        if buy :
            walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
            buy = False
        py = np.append(py,walletUSD)

plt.title('Evolution du wallet')
#plt.yscale("log")
plt.plot(py)
plt.show()

print("\nwallet HOLD : ",soldeCOIN_HOLD*p[-1])
print("nwallet final : ",walletUSD)
print("nb trade : ",nbTrade)
print("\n")

#%% affichage ---------------------------------
plt.figure(figsize=(10,7))

plt.subplot(3,1,1)
plt.title("Prix pur")
plt.yscale("log")
plt.plot(dff['open'][tp*2:len(dff)-tp*2])
displayBuySell(dff['dAjustedMA'],tp)

plt.subplot(3,1,2)
plt.title("MA")
plt.yscale("log")
plt.plot(dff['ajustedMA'][tp*2:len(dff)-tp*2])
displayBuySell(dff['dAjustedMA'],tp)

plt.subplot(3,1,3)
plt.title("Derivé de la MA")
plt.plot(np.full(shape=len(dff['dAjustedMA'][tp*2:len(dff)-tp*2]), fill_value = 0), 'r')
plt.plot(dff['dAjustedMA'][tp*2:len(dff)-tp*2])
displayBuySell(dff['dAjustedMA'],tp)

plt.show()























































