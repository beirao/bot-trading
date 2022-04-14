import torch 
import random
import numpy as np
from collections import deque
from tradGameAI import TradeGameAI
import sys
import os
from model import Linear_QNet, QTrainer
from helper import plot

pathFunctions = "../src/"
sys.path.append(os.path.abspath(pathFunctions))
import functions as f

MAX_MEMORY = 1000000
BATCH_SIZE = 10000
LR = 0.001

class Agent : 
    def __init__(self) :
        self.n_games = 0 
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen = MAX_MEMORY)
        
        self.model = Linear_QNet(17)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, tradeGame) :
        return tradeGame.currentTradingData[-1].to_numpy()
    
    def remember(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    def train_long_memory(self) :
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done) :
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self,state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 40 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            prediction = random.random()
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.trainer.device)
            prediction = self.model(state0)
    
        return prediction
        
def train() :
    path = '../src/data/rawData/rawEth.csv'
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    tradeGame = TradeGameAI(path)
    
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
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record :
                record = score
                # agent.model.save()
            
            print('n_games : ', agent.n_games,'Score : ', score, 'Record : ', record)
            
            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
if __name__ == '__main__' :
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    