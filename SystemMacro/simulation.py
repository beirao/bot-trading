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

pathModel = "../sources/model/model-V1-TP20"
sys.path.append(os.path.abspath(pathModel))
from initModel import Net, xdef, initModel

pathFunctions = "../sources"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f

#%% fonctions ------------------

def simulationConsecutive(netl,x, p, walletUSD, threshold, fee, logScale):
    y_pred = netl(x)

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
    if logScale:
        plt.yscale("log")
    plt.plot(py,'r',label="wallet evolution from 1000 USD")
    plt.plot(p, label="prix asset")
    fig.legend()
    plt.show()

    print("\nwallet HOLD : ",(soldeCOIN_HOLD*p[-1]).item())
    print("wallet final : ", walletUSD.item())
    print("nb trade : ",nbTrade)
    print("frais payé : ",sumFee.item())
    print("\n")

    # return : wallet hold | wallet trade | nb total trades | frais total payé
    return (soldeCOIN_HOLD*p[-1]).item(), walletUSD.item(), nbTrade, sumFee.item()

#%% Chargement du model deja entrainé

model = Net
model = initModel(pathModel+"/model.pth")

#importation data
df = pd.read_csv('../data/trainData/btcData.csv')
x = df[xdef] #sans var c'est mieux
p = df[['close']]

x = torch.tensor(x.values).float()
p = torch.tensor(p.values).float()
model = model.to("cpu")


#%% simu ------------------------------
simulationConsecutive(model,x,p, walletUSD = 1000, threshold = 0.3, fee = 0.001, logScale = True)
f.simulationThreshold(model,x,p, walletUSD = 1000, threshold = 0.3, fee = 0.001, logScale = True)






















































