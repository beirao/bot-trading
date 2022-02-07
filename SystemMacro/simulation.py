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
path = "../model/model-V1-TP20"
sys.path.append(os.path.abspath(path))
from initModel import Net,initModel


#%% fonctions ------------------
def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

def simulationThreshold(netl,x,walletUSD,threshold,p):
    y_pred = netl(x)

    fee = 0.001
    buy = False
    nbTrade = 0
    soldeCOIN = 0
    soldeCOIN_HOLD = (walletUSD/p[0])-(walletUSD/p[0]*fee)
    sumFee = 0

    py=[]
    py = np.array(py)
    fig = plt.figure(figsize=(13,10))

    for i in range(len(x)) :
        if y_pred[i] > 0.5 + threshold:
            if not buy :
                soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
                sumFee += walletUSD/p[i]*fee
                buy = True
                plt.axvline(i,color='green')
            nbTrade = nbTrade + 1

        elif y_pred[i] < 0.5 - threshold:
            if buy :
                walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
                sumFee += (p[i]*soldeCOIN)*fee
                buy = False
                plt.axvline(i,color='red')

        if buy :
             py = np.append(py,(soldeCOIN*p[i]))
        else :
            py = np.append(py,walletUSD)


    plt.title('Evolution du wallet')
    plt.yscale("log")
    plt.plot(py,'r',label="wallet evolution from 1000 USD")
    plt.plot(p, label="prix asset")
    fig.legend()
    plt.show()

    print("\nwallet HOLD : ",(soldeCOIN_HOLD*p[-1]))
    print("wallet final : ", (walletUSD))
    print("nb trade : ",nbTrade)
    print("frais payé : ",sumFee)
    print("\n")

def simulationConsecutive(netl,x,walletUSD, threshold, p):
    y_pred = netl(x)

    fee = 0.001
    buy = False
    nbTrade = 0
    soldeCOIN = 0
    soldeCOIN_HOLD = (walletUSD/p[0])-(walletUSD/p[0]*fee)
    sumFee = 0

    py=[]
    py = np.array(py)
    fig = plt.figure(figsize=(16,8))

    for i in range(len(x)) :
        if y_pred[i] > 0.5 + threshold and y_pred[i-1] > 0.5  + threshold and y_pred[i-2] > 0.5 + threshold :
            if not buy :
                soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
                sumFee += walletUSD/p[i]*fee
                buy = True
                plt.axvline(i,color='green')
            nbTrade = nbTrade + 1

        elif y_pred[i] < 0.5 - threshold and y_pred[i-1] < 0.5 - threshold and y_pred[i-2] < 0.5 - threshold :
            if buy :
                walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
                sumFee += (p[i]*soldeCOIN)*fee
                buy = False
                plt.axvline(i,color='red')

        if buy :
            py = np.append(py,(soldeCOIN*p[i]))
        else :
            py = np.append(py,walletUSD)


    plt.title('Evolution du wallet')
    plt.yscale("log")
    plt.plot(py,'r',label="wallet evolution from 1000 USD")
    plt.plot(p, label="prix asset")
    fig.legend()
    plt.show()

    print("\nwallet HOLD : ",(soldeCOIN_HOLD*p[-1]))
    print("wallet final : ", (walletUSD))
    print("nb trade : ",nbTrade)
    print("frais payé : ",sumFee)
    print("\n")

#%% Chargement du model deja entrainé

model = Net
model, xdef = initModel(path+"/model.pth")

#importation data
df = pd.read_csv('../data/btcData.csv')
x = df[xdef] #sans var c'est mieux
p = df[['close']]

x = torch.tensor(x.values).float()
p = torch.tensor(p.values).float()
model = model.to("cpu")


#%% simu ------------------------------
#♠simulationThreshold(model,x,1000,0.4,p)
simulationConsecutive(model,x,1000, 0.3,p)






















































