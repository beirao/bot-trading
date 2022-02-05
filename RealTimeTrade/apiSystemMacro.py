# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:34:53 2022

@author: thoma
"""

from binance.client import Client
import numpy as np
import pandas as pd
import talib as ta
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

#%%variables -------
time_frame = Client.KLINE_INTERVAL_1MINUTE
trading_period = 2
trading_pair = 'BNBUSDT'
threshold = 0

fee = 0.001
start_wallet = 1000
wallet_stable = start_wallet
wallet_coin = 0

fin = "1 Feb 2022"
api_path = "../api.config"
model_path = "../model/modelTP20.pth"


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

#%% fonctions -----
def initBinanceClient(path_to_config):
    config = eval(open(path_to_config).read())
    api_key = config['api_key']
    secret_key = config['secret_key']
    return Client(api_key, secret_key)

def initModel(path_to_model) :
    return torch.load(path_to_model)

def getCandles(trading_pair) :
    return client.get_klines(symbol=trading_pair, interval=time_frame)

def processData(candles) :
    #df raw data
    dfi = pd.DataFrame(data=candles, columns=["timestampOpen", "open", "high", "low", "close", "volume", "timestampClose", "quoteVol", "nbTrade", "TakerBaseVol", "TakerQuoteVol", "ignore"])

    # calcule des datas x
    df = pd.DataFrame()
    df["open"] = dfi["open"].astype(float, errors = 'raise')
    df["high"] = dfi["high"].astype(float, errors = 'raise')
    df["low"] = dfi["low"].astype(float, errors = 'raise')
    df["close"] = dfi["close"].astype(float, errors = 'raise')
    df["quoteVol"] = dfi["quoteVol"].astype(float, errors = 'raise')
    df = df.assign(rsi14 = lambda x: ta.RSI(x['open'], timeperiod=14)/100)
    df = df.assign(var = lambda x: ((x["close"] - x["open"])*100)/x["open"])
    df = df.assign(ma25 = lambda x: ta.MA(x["open"], timeperiod=25))
    df = df.assign(stochRsiD = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1])
    df = df.assign(stochRsiK = lambda x: ta.STOCH(x["high"], x["low"], x["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0])
    df['stochRsiInf03'] = df['stochRsiK'].apply(lambda x: 1 if x <= 30 else 0)
    df['stochRsiSup07'] = df['stochRsiK'].apply(lambda x: 1 if x >= 70 else 0)
    df = df.assign(deltaSMA25close = lambda x: x['close'] - x['ma25'])

    return df

def predict(df, model) :
    x = df[["rsi14","var","ma25","stochRsiD","stochRsiInf03","stochRsiSup07","deltaSMA25close"]] #sans var c'est mieux
    p = df[['close']]

    x = torch.tensor(x.values).float()
    p = torch.tensor(p.values).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    x = x.to(device)
    model = model.to(device)

    return model(x)[-1]

def simuTrade(prediction, price, threshold, fee, wallet_stable, wallet_coin):
    order_fee = 0

    if(prediction > (0.5 + threshold) and wallet_stable > 0) :
        order_amount = wallet_stable / price
        order_fee = wallet_stable * fee
        wallet_coin = order_amount - order_amount*fee
        wallet_stable = 0


    elif( prediction < 0.5 - threshold and wallet_coin > 0) :
        order_amount = wallet_coin * price
        order_fee =  order_amount * fee
        wallet_stable =  order_amount - order_fee
        wallet_coin = 0

    print("prediction : ", prediction)
    return wallet_coin, wallet_stable, order_fee

def visualizations(results) :
    total_fee = results['fee'].sum()
    nb_trade = results['fee'].astype(bool).sum(axis=0)
    wallet = results["wallet_stable"] + results["wallet_coin"]*results['price']

    plt.figure(figsize=(10,8))
    plt.title('Evolution du wallet')
    plt.plot(wallet)
    plt.show()

    print("prix : ", results['price'].iloc[-1])
    print("\nwallet HOLD : ", (start_wallet/results['price'].iloc[0])*results['price'].iloc[-1])
    print("wallet final : ", wallet.iloc[-1])
    print("nb trade : ", nb_trade)
    print("frais payés : ", total_fee)

    print("\n")



#%% main
client = initBinanceClient(api_path)
model = initModel(model_path)
results = pd.DataFrame()

try :
    while True :
        candles = getCandles(trading_pair)
        df = processData(candles)
        y_pred = predict(df, model)
        wallet_coin, wallet_stable, order_fee = simuTrade(y_pred, df['close'].iloc[-1], threshold, fee, wallet_stable, wallet_coin)
        results = results.append(pd.DataFrame([[wallet_coin, wallet_stable, order_fee, df['close'].iloc[-1]]], columns=['wallet_coin', 'wallet_stable', 'fee', 'price']), ignore_index=True)
        visualizations(results)
        time.sleep(trading_period)

except :
    pass

























































