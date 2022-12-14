# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:40:47 2022

@author: thoma
"""
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import numpy as np

#%% Fonctions d'initialisation
def initBinanceClient(path_to_config):
    config = eval(open(path_to_config).read())
    api_key = config['api_key']
    secret_key = config['secret_key']
    return Client(api_key, secret_key)

def getCandlesRealTime(client, trading_pair, time_frame) :
    return client.get_klines(symbol=trading_pair, interval=time_frame)

#%% Fonction de traitement
def processDataX(candles) :
    #df raw data
    dfi = pd.DataFrame(data=candles, columns=["timestampOpen", "open", "high", "low", "close", "volume", "timestampClose", "quoteVol", "nbTrade", "TakerBaseVol", "TakerQuoteVol", "ignore"])

    # calcule des datas x
    df = pd.DataFrame()
    df["open"] = dfi["open"].astype(float, errors = 'raise')
    df["high"] = dfi["high"].astype(float, errors = 'raise')
    df["low"] = dfi["low"].astype(float, errors = 'raise')
    df["close"] = dfi["close"].astype(float, errors = 'raise')
    df["quoteVol"] = dfi["quoteVol"].astype(float, errors = 'raise')
    df = df.assign(quoteVolMoy25 = lambda x: x["quoteVol"]/np.mean(x["quoteVol"].iloc[-25:]))
    df = df.assign(deltaOpenClose = lambda x: (np.abs(x["open"]-x["close"]))/x["close"])
    df = df.assign(deltaHighClose = lambda x: (np.abs(x["high"]-x["close"]))/x["close"])
    df = df.assign(deltaLowClose  = lambda x: (np.abs(x["low"]-x["close"]))/x["close"])
    df = df.assign(rsi14 = lambda x: ta.RSI(x['open'], timeperiod=14)/100)
    df = df.assign(var = lambda x: ((x["close"] - x["open"])*100)/x["open"])
    #df = df.assign(varMoy25 = lambda x: )
    df = df.assign(ma25 = lambda x: ta.MA(x["open"], timeperiod=25))
    df = df.assign(stochRsiD = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1])
    df = df.assign(stochRsiK = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0])
    df['stochRsiInf03'] = df['stochRsiK'].apply(lambda x: 1 if x <= 30 else 0)
    df['stochRsiSup07'] = df['stochRsiK'].apply(lambda x: 1 if x >= 70 else 0)
    df = df.assign(deltaSMA25close = lambda x: x['close'] - x['ma25'])

    #["open","high","low","close","quoteVol", "quoteVolMoy25", "deltaOpenClose", "deltaHighClose", "deltaLowClose", "rsi14","var","ma25","stochRsiD","stochRsiK","stochRsiInf03","stochRsiSup07","deltaSMA25close"]

    return df


#%% Fonctions maths
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
    return np.array(xn)

#%% Fonctions diverses
def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)