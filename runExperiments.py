#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:40:14 2018

@author: Hans Stenglein

Run script for testing and simulating
"""
import numpy as np
import scipy.stats as stats

import time
import sys
#TODO get numba to work
#from numba import njit, prange
from chain import chainProblem
from chain import chainPolicies
from sampleMDP import sample
import queue4D
from bamcp import BAMCP

def chainTest():
    horizons = [100, 300, 500, 1000, 10000]
    rewards = np.zeros( (len(horizons), len(chainPolicies)) )
    trials = 20
    for h in range(len(horizons)):
        for p in range(len(chainPolicies)):
            for i in range(trials):
                    traj, rew, c = sample(chainProblem, chainPolicies[p], horizons[h])
                    rewards[h, p] += rew
            rewards[h,p] /= trials # averaging the rewards obtained
        idx = np.argmax(rewards[h])
        print("best policy for chain problem with horizon",horizons[h], "is policy #",idx, "with ", rewards[h, idx], "obtained as average of", trials, "trials!")
        

import fsc

def testFSCChain():
    horizons = [100]#[300, 800, 1000, 5000]
    values, policies = [], []
    #for q in [1,2,3,4]:
    #queue4D.sample4Dqueue(totalsizeQueue, trials=10)
    v, p = fsc.finiteStateController(chainProblem, 5, 3000, 
            stepsize=[-0.8,-1.2e-2, -25*1.7e1], learning_rate=0.7, verbose=1)
    values.append(v)
    policies.append(p)
    
    for i in range(250):
        _, r, _ = sample(chainProblem, policies[0], 100)
        print(r)

def queueFSC():
    buffer = [2, 2, 2, 2]
    arrive = [0.08, 0.08]
    delay = [0.12, 0.12, 0.28, 0.28]
    
    totalsizeQueue = queue4D.queue4D(buffer, delay, arrive, discount=0.8)
    #for q in [1,2,3,4]:
    #queue4D.sample4Dqueue(totalsizeQueue, trials=10)
    v, p = fsc.finiteStateController(totalsizeQueue, 6, 5000, learning_rate=0.8, verbose=1)
    trials = 20
    r = np.zeros(trials)
    for i in range(trials):
        _, rew, _ = sample(totalsizeQueue, p, 100)
        r[i] = rew
    print(r)
    

def testBAMCP():
    buffer = [20, 20, 20, 20]
    arrive = [0.08, 0.08]
    delay = [0.12, 0.12, 0.28, 0.28]
    totalsizeQueue = queue4D.queue4D(buffer, delay, arrive, discount=0.8)
    
    #rollout = lambda s, h, mdp: np.random.choice(mdp.actions)
    #planner = BAMCP(totalsizeQueue, 0.5, max_time=0.5, exploration_scale=3)
    
    trials = 10
    r = np.zeros(trials)
    for i in range(trials): #TODO do not learn over trials!
        planner = BAMCP(totalsizeQueue, 0.2, max_time=0.5, exploration_scale=3)
        _, r[i], _ = sample(totalsizeQueue, planner.search, 100)
    print(r)

def BAMCP_Chain():
    planner = BAMCP(chainProblem, 0.7, max_time=0.2, exploration_scale=3)
    
    trials = 250
    r = np.empty(trials)
    for i in range(trials):
        print(i)
        _, rew, _ = sample(chainProblem, planner.search, 100)
        r[i] = rew
    print(r)

def determineC_Chain():
    rollout = lambda s, h: np.random.choice(chainProblem.actions)
    # c > Rmax / (1 - y)
    cs = [2.01, 2.5, 3, 3.5, 4, 5, 7, 8, 10, 20]
    trials = 20
    rewards = np.empty( (len(cs), trials) )
    #@njit(parallel=True)
    for k in range(len(cs)):
        planner = BAMCP(chainProblem, rollout, max_time=0.5, exploration_scale=cs[k])
        for i in range(trials):
            _, rew, _ = sample(chainProblem, planner.search, 1000)
            rewards[k, i] = rew
    # stats
    stat_c = stats.gmean(rewards, axis=1) # gmean over rewards
    best = np.argmax(stat_c)
    print('best c is', cs[best])
    header = ("c -        gmean(reward) trials: " 
              + str(trials) + "__" + time.strftime("%c"))
    np.savetxt('bamcp/determineC_Chainproblem.txt', np.array([cs, stat_c]),             header=header)
    print("run saved")

def queueTest():
    buffer = [2, 2, 2, 2]
    arrive = [0.08, 0.08]
    delay = [0.12, 0.12, 0.28, 0.28]
    
    totalsizeQueue = queue4D.queue4D(buffer, delay, arrive)
    maxsizeQueue = queue4D.queue4D(buffer, delay, arrive, loss='maxsize', discount=0.7)
    throughputsizeQueue = queue4D.queue4D(buffer, delay, arrive, loss='throughput')
    
    uniformPol = lambda s, traj, mdp: np.random.choice(mdp.actions)
    
    traj1, r1, count1 = sample(throughputsizeQueue, uniformPol, 1000)
    print("---------------------------")
    print()
    #traj2, r2, count2 = sample(throughputsizeQueue, queue4D.pol_LBFS, 1000)
    print("throughputsizeQueue with LONGER policy: reward", r1)
    #print("throughputsizeQueue with LBFS policy: reward", r2)


def test_SARSA():
    buffer = [20, 20, 20, 20]
    arrive = [0.08, 0.08]
    delay = [0.12, 0.12, 0.28, 0.28]
    totalsizeQueue = queue4D.queue4D(buffer, delay, arrive, discount=0.8)
    
    import sarsa
    
    trials = 20
    r = np.zeros(trials)
    for i in range(trials): #TODO do not learn over trials!
        pol = sarsa.get_policy(totalsizeQueue, 0.68, 0.3)
        _, r[i], _ = sample(totalsizeQueue, pol, 100)
    print(r)
    
def sarsa_chain():
    import sarsa
    trials = 250
    r = np.empty(trials)
    for i in range(trials):
        pol = sarsa.get_policy(chainProblem, 0.68, 0.3)
        _, rew, _ = sample(chainProblem, pol, 100)
        r[i] = rew
    print(r)

# --------------
# running code here

#BAMCP_Chain()
#testFSCChain()
#_, r, _ = sample(chainProblem, chainPolicies[0], 100)
#print("reward with optimal pol", r)
#BAMCP_Chain()
#if len(sys.argv) == 1:
#    test_different_Q_FSC()
#else:
#sarsa_chain()
#test_SARSA()
testBAMCP()
#testBAMCP()
#BAMCP_Chain()
#queueFSC()
#testFSCChain()
#    determineC_Chain()
#print("############ now on 4Dqueue ############")
#testFSC()
#testBAMCP()
    
