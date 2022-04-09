# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:59:32 2022

@author: thoma
"""
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import talib as ta


#xdef = ["rsi14","var","ma25","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]
#xdef = ["quoteVolMoy25", "deltaOpenClose", "deltaHighClose", "deltaLowClose", "rsi14","var","ma25","stochRsiD","stochRsiK","stochRsiInf03","stochRsiSup07","deltaSMA25close"]
xdef = ["quoteVolMoy25", "deltaHighClose", "deltaLowClose", "rsi14","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]


class Net(nn.Module):
    def __init__(self,enter):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(enter,128)
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 3000)
        self.fc4 = nn.Linear(3000, 10)
        self.fcf = nn.Linear(10, 1) #si 1 hausse si 0 baisse

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fcf(x))
        return x

def initModel(pathModel):
    return torch.load(pathModel)


def simulation(netl,x, p, walletUSD, threshold, fee, tp, logScale, plotBuySell):
    y_pred = netl(x)
    yp = y_pred.detach().numpy()
    yp = [float(x) for x in yp]
    yp = np.array(yp)
    yp = ta.MA(yp,tp)

    buy = False
    nbTrade = 0
    soldeCOIN = 0
    soldeCOIN_HOLD = (walletUSD/p[0])-(walletUSD/p[0]*fee)
    sumFee = 0

    py=[]
    py = np.array(py)

    fig = plt.figure(figsize=(14,10))
    #plt.subplot(2,1,1)

    for i in range(len(x)) :
        if yp[i] > 0.5 + threshold and yp[i-1] > 0.5  + threshold  :
            if not buy :
                soldeCOIN = (walletUSD/p[i])-(walletUSD/p[i]*fee)
                sumFee += walletUSD/p[i]*fee
                buy = True
                if plotBuySell :
                    plt.axvline(i,color='green')
            nbTrade = nbTrade + 1

        elif yp[i] < 0.5 - threshold and yp[i-1] < 0.5 - threshold  :
            if buy :
                walletUSD = (p[i]*soldeCOIN) - (p[i]*soldeCOIN)*fee
                sumFee += (p[i]*soldeCOIN)*fee
                buy = False
                if plotBuySell :
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

#    plt.subplot(2,1,2)
#    plt.plot(ta.MA(yp,tp))

    plt.show()

    print("\nwallet HOLD : ",(soldeCOIN_HOLD*p[-1]).item())
    print("wallet final : ", walletUSD.item())
    print("nb trade : ",nbTrade)
    print("frais payé : ",sumFee.item())
    print("\n")

    # return : wallet hold | wallet trade | nb total trades | frais total payé
    return (soldeCOIN_HOLD*p[-1]).item(), walletUSD.item(), nbTrade, sumFee.item()

#simulationConsecutive(model,x,p, walletUSD = p[0], threshold = 0.3, fee = 0.001, tp = 2, logScale = True, plotBuySell = False)
