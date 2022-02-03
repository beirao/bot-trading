# -*- coding: utf-8 -*-
"""
Created on Sun feb 01 20:22:19 2022

@author: thoma
"""
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt

#%% fonctions---------------------------------
def ajustedMA(p, tp) :
    i = 0
    rl  = [0.0 for i in range(len(p))]
    rl = np.array(rl)
    for ip in p :
        if(i >= tp and i <= len(p)-tp):
            rl[i] = np.sum([p[t] for t in range(i-tp,i+tp)])
        else:
            rl[i] = np.nan
        i = i + 1
    return rl

def derive(x) :
    xn  = [0.0 for i in range(len(x))]
    for i in range(len(x)-1):
        xn[i] = x[i+1]-x[i]
    #xn[len(x)] = x[len(x-1)]
    return np.array(xn)

def displayBuySell(dMA,tp):
    for i in range(tp*2,len(dMA[tp:len(dMA)-tp])):
        if(dMA[i-1] <= 0 and dMA[i] >= 0) :
            plt.axvline(i,color='green')
        elif(dMA[i-1] > 0 and dMA[i] < 0) :
            plt.axvline(i,color='red')

def signalBuySell(dMA) :
    rl  = [0.0 for i in range(0,len(dMA))]
    rl = np.array(rl)
    for i in range(tp*2,len(dMA)-tp*2):
        if(dMA[i] > 0) :
            rl[i] = 1
        elif(dMA[i] <= 0) :
            rl[i] = 0
    return rl


#%% dataframe ---------------------------------
#variables
tp = 20

##Importation de la data
klines = np.loadtxt('../rawData/rawTestMatic.csv')

#df raw data
dfi = pd.DataFrame(data=klines, columns=["timestampOpen", "open", "high", "low", "close", "volume", "timestampClose", "quoteVol", "nbTrade", "TakerBaseVol", "TakerQuoteVol", "ignore"])

# calcule des datas x
dff = pd.DataFrame()
dff["open"] = dfi["open"]
dff["high"] = dfi["high"]
dff["low"] = dfi["low"]
dff["close"] = dfi["close"]
dff["quoteVol"] = dfi["quoteVol"]
dff = dff.assign(rsi14 = lambda x: ta.RSI(x['open'], timeperiod=14)/100)
dff = dff.assign(var = lambda x: ((x["close"]-x["open"])*100)/x["open"])
dff = dff.assign(ma25 = lambda x: ta.MA(x["open"], timeperiod=25))
dff = dff.assign(stochRsiD = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1])
dff = dff.assign(stochRsiK = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0])
dff['stochRsiInf03'] = dff['stochRsiK'].apply(lambda x: 1 if x <= 30 else 0)
dff['stochRsiSup07'] = dff['stochRsiK'].apply(lambda x: 1 if x >= 70 else 0)
dff = dff.assign(deltaSMA25close = lambda x: x['close'] - x['ma25'])
#dff = dff.assign(stochRsiCross = lambda x: 1 if(x["stochRsiK"] >= x["stochRsiD"]  and x["stochRsiK"] <= x["stochRsiD"]) else 0

# calcule des datas y
dff = dff.assign(ajustedMA = lambda x: ajustedMA(x["open"], tp))
dff = dff.assign(dAjustedMA = lambda x: ajustedMA(derive(x["ajustedMA"]),tp))
dff = dff.assign(y = lambda x: signalBuySell(x["dAjustedMA"]))

#["open","high","low","close","quoteVol","rsi14","var","ma25","stochRsiD","stochRsiK","stochRsiInf03","stochRsiSup07","deltaSMA25close","y"]

#replace missing value
dff = dff.fillna(0)

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

#%%analyse ----------------------------
startWallet = 1000
walletUSD = startWallet #USDT

fee = 0.00075
buy = False
nbTrade = 0
soldeCOIN = 0
soldeCOIN_HOLD = (startWallet/dff['close'][tp*2])-(soldeCOIN*fee)
py = []
py = np.array(py)


for i in range(tp*2,len(dff['close'])-tp*2):
    if(dff['y'][i] == 1 and buy == False) :
        soldeCOIN = (walletUSD/dff['close'][i])-(soldeCOIN*fee)
        walletUSD = 0
        buy = True
        nbTrade = nbTrade + 1
    elif(dff['y'][i] == 0 and buy == True) :
        walletUSD = (dff['close'][i]*soldeCOIN)
        walletUSD = walletUSD - walletUSD*fee
        buy = False

    if(buy == True) :
        py = np.append(py,(soldeCOIN*dff['close'][i]))
        walletUSD = (soldeCOIN*dff['close'][i])
    else :
        py = np.append(py,walletUSD)

plt.title('Evolution du wallet')
plt.yscale("log")
plt.plot(py)
plt.show()
print("\nwallet HOLD : ",soldeCOIN_HOLD*dff['close'][len(dff['close'])-tp*2])
print("nwallet final : ",walletUSD)
print("nb trade : ",nbTrade)
print("\n")

#%% exporter en ---------------------------------
adff = dff[tp*2:len(dff)-tp*2] #*2 parceque il y a le lissage de la ma et de la dma
adff.to_csv('../data/maticData.csv', index=False)
print(adff)





























































































