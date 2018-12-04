#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:03:49 2018

@author: hans


implementation of sarsa
"""
import numpy as np
import scipy.sparse as sp

import definition_MDP

class agent:
    def __init__(self, mdp: definition_MDP.MDP, learning_rate, epsilon=0.5):
        """SARSA learning algorithm
        MDP - problem to control
        learning_rate (alpha) - balances new information
        epsilon - epsilon-greedy action-selection parameter
        """
        # Q-function
        self.Q = sp.lil_matrix( (mdp.A, mdp.S) )
        # actions to select from, assume all action to be valid in each state...
        self.A, self.actions = mdp.get_A()
        self.R = mdp.reward
        
        self.alpha = learning_rate
        self.gamma = mdp.discount
        self.eps = epsilon
    # ----
    
    def sarsa_control(self, obs):
        """update Q-function an choose action according to the last observation (s,a,s')"""
        s, a, ss = obs # s, a, s'
        r = self.R(s,a,ss)
        aa = self.greedy(ss) # a'
        # update Q
        self.Q[a, s] += self.alpha*( r + 
              self.gamma*self.Q[aa, ss] - self.Q[a, s] )
        print('Q updated:', s,a,r,ss,aa, self.Q[a,s])
        return aa
        
    def greedy(self, state):
        """epsilon greedy policy, treating multiple best actions uniform"""
        # all Q maximizing actions, indices
        aidx, _, q = sp.find(self.Q[:, state].toarray())
        abest = aidx[np.argwhere(q == np.max(q))].squeeze() if np.size(q) > 0 else self.actions 
        puniform = self.eps/ self.A
        p = np.full((self.A,), puniform)
        pbest = ( 1 - self.eps ) / np.size(abest)
        p[abest] += pbest
        a = np.random.choice(self.actions, p=p)
        return a
# ------------------

def get_policy(mdp, learning_rate, epsilon=0.5):
    """ returns a policy for 'mdp': pol: S x H -> A"""
    sarsa_agent = agent(mdp, learning_rate, epsilon)
    return ( lambda state, hist: sarsa_agent.sarsa_control(hist[-1]) if len(hist) > 0
            else sarsa_agent.greedy(state) )
    # initial action chosen according to greedy policy -> uniformly random
        
