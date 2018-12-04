#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:38:16 2018

@author: Hans Stenglein

defines a function to sample a trajectory from an MDP

uses the true transition model of the given MDP
"""

import numpy as np
from scipy.sparse import lil_matrix

from functools import reduce
import typing
# ----
from definition_MDP import History, MDP, Policy, TransitionMatrix


def sample(mdp: MDP, policy: Policy, iterations: int
           ) -> typing.Tuple[History, float, TransitionMatrix]:
    """samples a trajectory from a MDP with a given (stochastic, markovian) policy and returns a trajectory and a reward
        input:
        - mdp is a MDP defined by definition_MDP.py
        - policy as a function representing a (not nessessary markovian or stationary) policy: 
            s x [(s,a,s')] -> a
        - iterations is the number of samples, which are taken from this MDP following the given policy, it is also the final horizon T
        output (tuple):
        - traj: a list of (s, a, s') tuples representing s =(a)=> s' transitions
        - reward: the obtained accumulated reward
        - counts: counting of s =(a)=> s' transitions, can be used as prior
        """
    traj = []
    reward = 0
    # counts as sparse matrices
    tmp = []
    for aa in mdp.actions:
        m = lil_matrix((mdp.S, mdp.S), dtype=np.uint) 
        # row-based linked list implementation for inserting elements
        tmp.append(m)
    # tuple[a] -> sparse[s,s]
    counts = tuple(tmp)
    
    # simulate
    s = mdp.initial_state()
    for t in range(iterations):
        a = policy(s, traj) # traj(ectory) = history
        s_next = mdp.true_transition(s, a)
        traj.append( ( s, a, s_next ) )
        counts[a][s,s_next] += 1 # count transition
        # cleanup for next iteration
        s = s_next
    
    # list of immediate rewards for each point in trajectory
    # reward: s x a x s -> R
    immediate_rewards = [mdp.reward(o[0], o[1], o[2]) for o in traj]
    if mdp.discount < 1:
        # create list of discount factors (gamma**t)..1
        discounts = [ mdp.discount**x for x in reversed(range(iterations)) ]        
        reward = reduce( lambda x,y: x+y, list( map( lambda x, y: x * y, discounts, immediate_rewards)) )
    else:
        reward = reduce( lambda x,y: x+y, immediate_rewards)
        
    return ( traj, reward, counts )

# -------------------------