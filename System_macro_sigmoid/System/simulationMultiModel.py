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
import talib as ta


pathModel = "../sources/model/model-V1-TP5"
sys.path.append(os.path.abspath(pathModel))
from initModel import Net, xdef, simulation

pathModel1 = "../sources/model/model-V1-TP13"
sys.path.append(os.path.abspath(pathModel1))
import initModel as im13

pathModel2 = "../sources/model/model-V1-TP20"
sys.path.append(os.path.abspath(pathModel2))
import initModel as im20

pathModel3 = "../sources/model/model-V1-TP35"
sys.path.append(os.path.abspath(pathModel3))
import initModel as im35

pathFunctions = "../sources"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f

#%% fonctions ------------------
def yProcess(net, x, tp) :
    y_pred = net(x)

    yp = y_pred.detach().numpy()
    yp = [float(x) for x in yp]
    yp = np.array(yp)
    return ta.MA(yp,tp)

def modelProcess(net, x, tp) :
    y_pred = net(x)

    yp = y_pred.detach().numpy()
    yp = [float(x) for x in yp]
    yp = np.array(yp)
    return ta.MA(yp,tp)

def simulationMultiModel(net1,net2,net3,x, p, walletUSD, threshold, fee, tp, graph, logScale, plotBuySell):
    yp1 = yProcess(net1, x, tp)
    yp2 = yProcess(net1, x, tp)
    yp3 = yProcess(net1, x, tp)

    buy = False
    nbTrade = 0
    soldeCOIN = 0
    soldeCOIN_HOLD = (walletUSD/p[0])-(walletUSD/p[0]*fee)
    sumFee = 0

    py=[]
    py = np.array(py)

    if graph :
        fig = plt.figure(figsize=(14,10))

    for i in range(len(x)) :
        if yp1[i] > 0.5 + threshold and yp2[i] > 0.5 + threshold and yp3[i] > 0.5 + threshold :
            if not buy :
                soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
                sumFee += walletUSD/p[i]*fee
                buy = True
                if plotBuySell and graph :
                    plt.axvline(i,color='green')
            nbTrade = nbTrade + 1

        elif yp1[i] < 0.5 + threshold and yp2[i] < 0.5 + threshold and yp3[i] < 0.5 + threshold   :
            if buy :
                walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
                sumFee += (p[i]*soldeCOIN)*fee
                buy = False
                if plotBuySell and graph:
                    plt.axvline(i,color='red')

        if buy :
            py = np.append(py,(soldeCOIN*p[i]))
        else :
            py = np.append(py,walletUSD)

    if graph :
        plt.title('Evolution du wallet')
        if logScale:
            plt.yscale("log")
        plt.plot(py,'r',label="wallet evolution with the trading bot")
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

def variablesFinding(model13,model20,model35,x,p) :
    Lth=[]
    Lth = np.array(Lth)
    Ltp=[]
    Ltp = np.array(Ltp)

    rangeTH = range(0,0.49,0.01)
    rangeTP = range(1,1,60)

    for th in rangeTH :
        for tp in rangeTP :
            ph, pbot, _, _, _ = simulationMultiModel(model13,model20,model35,x,p, walletUSD = p[0], threshold = th, fee = 0.001, tp = tp, graph = False, logScale = True, plotBuySell = False)
            lth = np.append(Lth,)


    return bTreshold, bTp


#%% Chargement du model deja entrainé

model13, model20, model35 = Net, Net, Net
model13 = im13.initModel(pathModel1+"/model.pth").to("cpu")
model20 = im20.initModel(pathModel2+"/model.pth").to("cpu")
model35 = im35.initModel(pathModel3+"/model.pth").to("cpu")

#importation data
df = pd.read_csv('../data/trainData/dotData.csv') #.iloc[220000:235000]
x = df[xdef] #sans var c'est mieux
p = df[['close']]

x = torch.tensor(x.values).float()
p = torch.tensor(p.values).float()
p = p.detach().numpy()
p = [float(x) for x in p]
p = np.array(p)


#%% simu ------------------------------
simulationMultiModel(model13,model20,model35,x,p, walletUSD = p[0], threshold = 0, fee = 0.001, tp = 8, graph = False, logScale = True, plotBuySell = False)























































