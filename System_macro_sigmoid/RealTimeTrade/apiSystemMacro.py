# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:34:53 2022

@author: thoma
"""
from binance.client import Client
import pandas as pd
import torch
import matplotlib.pyplot as plt
import talib as ta
import time
import sys
import os

pathModel = "../model/model-V1-TP20"
sys.path.append(os.path.abspath(pathModel))
from initModel import Net, xdef, initModel

pathFunctions = "../../src"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f

#%%variables -------
time_frame = Client.KLINE_INTERVAL_1MINUTE
trading_period = 0.5
trading_pair = 'BTCUSDT'
threshold = 0

fee = 0.001
start_wallet = 1000
wallet_stable = start_wallet
wallet_coin = 0

fin = "1 Feb 2022"
api_path = "../../api.config"

#%% Chargement du model deja entrainé
model = Net
model = initModel(pathModel+"/model.pth")

#%% fonctions --------------

def predict(df, model, xdef) :
    x = df[xdef] #sans var c'est mieux
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

    print("prediction : ", prediction.item())
    return wallet_coin, wallet_stable, order_fee

def visualizations(results) : # [wallet_coin, wallet_stable, fee, price]
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
client = f.initBinanceClient(api_path)
results = pd.DataFrame()

try :
    while True :
        candles = f.getCandlesRealTime(client, trading_pair,time_frame)
        df = f.processDataX(candles)
        y_pred = predict(df, model,xdef)
        wallet_coin, wallet_stable, order_fee = simuTrade(y_pred, df['close'].iloc[-1], threshold, fee, wallet_stable, wallet_coin)
        results = pd.concat([results, pd.DataFrame([[wallet_coin, wallet_stable, order_fee, df['close'].iloc[-1]]], columns=['wallet_coin', 'wallet_stable', 'fee', 'price'])], ignore_index=True)
        visualizations(results)
        time.sleep(trading_period)
except :
    pass




















































