# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:07:39 2022

@author: thoma
"""
import random
import pandas as pd 
import numpy as np
import sys
import os
from enum import Enum 
import keyboard
from collections import namedtuple
import matplotlib.pyplot as plt


pathFunctions = "../src/"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f

class Trade(Enum):
    BUY = 1
    NEUTRAL = 2
    SELL = 3

class TradeGameAI() :
    
    def __init__(self, rawTradingDataPath) :
        self.rawTradingDataPath = rawTradingDataPath
        self.reset()

    def reset(self) :
        # Importation de la data
        self.get_data()
        
        # Paramz
        self.walletValue = 1000
        self.fee = 0.001
        
        # Temp
        self.currentState = Trade.NEUTRAL
        self.lastState = Trade.NEUTRAL
        self.currentTradingData = [self.tradingData.iloc[0], self.tradingData.iloc[1]]
        self.currentIndex = 1
        self.walletHistory = [self.walletValue]
        self.lastTradeWallet = 0
    
    def get_data(self) :
        self.tradingData = f.processDataX(np.loadtxt(self.rawTradingDataPath))

    def play_step(self, action): 
        # init
        self.currentTradingData.append(self.tradingData.iloc[self.currentIndex])
        self.currentIndex += 1   
        
        # action traduction
        if action > 0.5 :
            self.currentState = Trade.BUY
        else:
            self.currentState = Trade.SELL
        
        # update
        if(self.currentState == Trade.BUY) :
            self.walletValue = self.walletValue*(self.currentTradingData[-1]['open']/self.currentTradingData[-2]['open'])
        
        # fee update
        if(self.currentState != self.lastState) :
            self.walletValue -= self.walletValue*self.fee       
        self.walletHistory.append(self.walletValue)
        
        # lastTradewallet update
        if(self.currentState == Trade.BUY and self.lastState == Trade.SELL) :
            self.lastTradeWallet = self.walletValue
        
        # reward trigger
        reward = 0
        if(self.currentState == Trade.SELL and self.lastState == Trade.BUY) :
            if(self.lastTradeWallet < self.walletValue) :
                reward = 10
            else :
                reward = -10
        
        # plot
        plt.figure(figsize=(14,10))
        plt.figure(1)             
        plt.subplot(211)           
        plt.plot([self.currentTradingData[i]['open'] for i in range(len(self.currentTradingData))])
        plt.title('Price')
        plt.subplot(212)          
        plt.plot(self.walletHistory)
        plt.title('Wallet')
        plt.show()
        
        # check if it is the end 
        end = False
        if self.walletValue == 0 or self.currentIndex-1 >= len(self.tradingData):
            end = True
            return reward, end, self.walletValue
        
        # end  
        self.lastState = self.currentState
        return reward, end, self.walletValue


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    