#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:47:18 2018

defines the chain problem MDP and some policies for testing

@author: Hans Stenglein
"""

import numpy
from definition_MDP import MDP, State, Action

# -------------------------

# define the chain problem:

# transition probabilitias for chain problem
# 5 states, 2 actions
# a0 gets to next state with 0.8, with 0.2 to state s0
# a1 is the reverse, with 0.8 to s0
_transition = numpy.zeros( (2,5,5) )
_transition[0,:,0] = 0.2

_transition[0,0,1] = 0.8
_transition[0,1,2] = 0.8
_transition[0,2,3] = 0.8
_transition[0,3,4] = 0.8
_transition[0,4,4] = 0.8


_transition[1,:,0] = 0.8

_transition[1,0,1] = 0.2
_transition[1,1,2] = 0.2
_transition[1,2,3] = 0.2
_transition[1,3,4] = 0.2
_transition[1,4,4] = 0.2
# -------

# reward function only depends on next state s'
# 2 by returning to s0, 10 by staying (!) in s4, (A,S,S)!
_reward = numpy.zeros( (2,5,5) )
_reward[0,4,4] = 10
_reward[1,4,4] = 10
_reward[:,:,0] = 2


# define the MDP subclass
#       transition, reward, discount
class _chain(MDP):
    
    def __init__(self, discount=0.95):
        super().__init__(discount)
        self.maxReward = 10
        self.counts = 10 * _transition # prior for dirichlet distr., rates as int
        
    def get_S(self):
        return 5, numpy.arange(5)
    def get_A(self):
        return 2, numpy.arange(2)
    
    def true_transition(self, state: State, action: Action) -> State:
        return numpy.random.choice(self.states, 
                                   p=_transition[action, state, :])

    def reward(self, state, action, next_state):
        """Reward function for the given s =(a)=> s'. Must be overwritten
        by subclasses.
        returns a real number"""
        return _reward[action, state, next_state]

# -------------
        
chainProblem = _chain()    

# -------------
   
# some examples of different policies
# policies in the form of (s x hist -> a) mappings

# this should be the optimal policy
# allways choose action_0 -> P(0 | sx) = 1, P(1 | sx) = 0
optimalChainPolicy = lambda s, h: 0
# this allways tries to get to s0 for reward 2
# iths the reverse of the optimal policy
greedyAllBackPolicy = lambda s, h: 1

def __bad(s, hist):
    if s < 4:   # foreward to max. s3
        return 1
    else:       # go back
        return 0

someBadPolicy  = __bad

#def _randPol():
#    """return random (row stochastic) policy"""
#    m = numpy.random.rand(5,2)
#    m /= m.sum(axis=1)[:,None]
#    return m
#
# used to sample two random policies:
    
__randPolicy1 = numpy.array([[0.57159428, 0.42840572],
       [0.56290676, 0.43709324],
       [0.52330081, 0.47669919],
       [0.50071266, 0.49928734],
       [0.46796381, 0.53203619]])
randPolicy1 = lambda s, h: numpy.random.choice(chainProblem.actions, p=__randPolicy1[s, :])
    
__randPolicy2 = numpy.array([[0.07830137, 0.92169863],
       [0.79437975, 0.20562025],
       [0.58037476, 0.41962524],
       [0.88494259, 0.11505741],
       [0.43441979, 0.56558021]])
randPolicy2 = lambda s, h: numpy.random.choice(chainProblem.actions, p=__randPolicy2[s, :])

def __sampleRate():
    p = numpy.random.rand()
    a_p = numpy.array([p, 1-p])
    return a_p

nonstationaryBernoulli = lambda s, h: numpy.random.choice(chainProblem.actions,                                                  p=__sampleRate())

chainPolicies = [optimalChainPolicy, greedyAllBackPolicy, someBadPolicy, randPolicy1, randPolicy2, nonstationaryBernoulli]

# --------------------