# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:34:53 2022

@author: thoma
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from initModel import initModel


#%% fonctions ------------------
def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

def simulation(netl,x,walletUSD,threshold):
    y_pred = netl(x)

    fee = 0.001
    buy = False
    nbTrade = 0
    soldeCOIN = 0
    soldeCOIN_HOLD = (walletUSD/p[0])-(walletUSD/p[0]*fee)
    sumFee = 0

    py=[]
    py = np.array(py)

    for i in range(len(x)) :
        if y_pred[i] > 0.5 + threshold:
            if not buy :
                soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
                sumFee += walletUSD/p[i]*fee
                buy = True

            py = np.append(py,(soldeCOIN*p[i]))
            nbTrade = nbTrade + 1

        elif y_pred[i] < 0.5 - threshold:
            if buy :
                walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
                sumFee += (p[i]*soldeCOIN)*fee
                buy = False
            py = np.append(py,walletUSD)

    plt.figure(figsize=(13,10))
    plt.title('Evolution du wallet')
#    plt.yscale("log")
    plt.plot(py)
    plt.show()

    print("\nwallet HOLD : ",(soldeCOIN_HOLD*p[-1]))
    print("wallet final : ", (walletUSD))
    print("nb trade : ",nbTrade)
    print("frais payé : ",sumFee)
    print("\n")

#%% Chargement du model deja entrainé
path = "../model/model-V1-TP20"
sys.path.append(os.path.abspath(path))
netl  = initModel(path+"/model.pth")

#importation data
df = pd.read_csv('../data/btcData.csv')
x = df[["rsi14","var","ma25","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]] #sans var c'est mieux
p = df[['open']]

x = torch.tensor(x.values).float()
p = torch.tensor(p.values).float()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

x = x.to(device)
netl = netl.to(device)


#%% simu ------------------------------
simulation(netl,x,1000,0.4)






















































