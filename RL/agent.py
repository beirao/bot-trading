import torch 
import random
import numpy as np
from collections import deque
from tradGame import TradeGame


MAX_MEMORY = 100000
BATCH_SIZE = 1000
lr = 0.001

class Agent : 
    def __init__(self) :
        self.nb_trade = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen = MAX_MEMORY)
        # TODO : model, trainer
        
    
    def get_state(self, tradeGame) :
        pass
    
    def remember(self, state, action, reward, next_state, done) :
        pass
    
    def train_long_memory(self) :
        pass
    
    def train_short_memory(self) :
        pass
    
    def get_action(self,state):
        pass
        
def train() :
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    tradeGame = TradeGame()
    
    while True :
        # get old state
        state_old = agent.get_state(tradeGame)
        
        # get move 
        final_trade = agent.get_action(state_old)
        
        # perform trade and get new state
        reward, done, score = tradeGame.play_step(final_trade)
        state_new = agent.get_state(tradeGame)
        
        # train short memory
        agent.train_short_memory(state_old, final_trade, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_trade, reward , state_new, done)
        
        if done :
            # train long memory, plot result
            tradeGame.reset()
            agent.nb_trade += 1
            agent.train_long_memory()
            
            if score > record :
                record = score
                # agent.model.save()
            
            print('nb_trade : ', agent.nb_trade,'Score : ', score, 'Record : ', record)
            # TODO : plot
            
        
if __name__ == '__main__' :
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    