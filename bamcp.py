#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:04:52 2018

@author: Hans Stenglein

Implementation of BAMCP

Reference:
A. Guez, D. Silver, and P. Dayan, 
“Scalable and Efficient Bayes-Adaptive Reinforcement Learning Based 
on Monte-Carlo Tree Search,” 
Journal of Artificial Intelligence Research, vol. 48, pp. 841–883, Nov. 2013.
"""

import numpy as np
import scipy.sparse as sp

#import typing
from definition_MDP import State, Action, History, TransitionMatrix
from definition_MDP import MDP

from time import time as time_now

class BAMCP():
    """Bayes Adaptive Monte Carlo Planning
    
    Class provides method: search(s, h) -> a
    
    following Guez et. al. 2013"""
     # definition of epsilon greedy policy
    def greedy(self, eps, state):
                """epsilon greedy policy, treating multiple best actions uniform"""
                # all Q maximizing actions, indices
                aidx, _, q = sp.find(self.realQ[:, state].toarray())
                # all actions are best if Q is empty (= 0 everywhere)
                abest = aidx[np.argwhere(q == np.max(q))].squeeze() if np.size(q) > 0 else self.problem.actions
                # epsilon / A basis value
                puniform = eps/self.problem.A
                p = np.full((self.problem.A,), puniform)
                pbest = ( 1 - eps ) / np.size(abest)
                p[abest] += pbest
                print("greedy probabilities ", p)
                a = np.random.choice(self.problem.actions, p=p)
                return a
            
    def greedy_update(self, alpha, hist):
        if not hist:
            pass
        else:
            state = hist[-1][0]
            action = hist[-1][1]
            next_state = hist[-1][2]
            # standard Q-learning (Watkins 1989)
            self.realQ[action, state] += ( alpha 
                      * (self.problem.reward(state, action, next_state) # deterministic reward!
                      + self.problem.discount * np.max(self.realQ.getcol(state).toarray())
                      - self.realQ[action, state] ) )
    
    def __init__(self, problem: MDP, 
                 rollout_policy=None, max_time=0.5,
                 exploration_scale=3, precision=0.5, learning_rate=0.7):
        self.problem = problem
        self.update = lambda hist: self.greedy_update(0, None) # default: do nothing
        if type(rollout_policy) is float:
            # epsilon-greedy policy
            self.realQ = sp.lil_matrix( (self.problem.A, self.problem.S) )
            # A x S because rows are in lists
            epsilon = rollout_policy # given as arg
            if epsilon < 0 or epsilon > 1:
                raise ValueError('epsilon for epsilon-greedy policy must be in 0..1! input was', str(epsilon))
            if learning_rate < 0 or learning_rate > 1:
                raise ValueError('alpha for Q-learing for epsilon-greedy policy must be in 0..1! input was', str(learning_rate))
            # instanciating greedy policy
            self.rollout_policy = lambda state, hist: self.greedy(epsilon, state)
            self.update = lambda hist: self.greedy_update(learning_rate, hist)
            
        elif rollout_policy:
            # some other policy
            #TODO type check p: s x h -> a
            self.rollout_policy = rollout_policy
        else:
            # uniform policy
            self.rollout_policy = lambda s, h: np.random.choice(self.problem.actions)
            
        self.exploration_scale = exploration_scale # default 3
        # determine max depth
        # reforulation of y**d Rmax < epsilon
        # rounded to next int
        self.max_depth = np.rint( np.log(precision / self.problem.maxReward) 
                        / np.log( problem.discount) )
        
        self.max_time = max_time # default 0.5s
        
        self.__Qfunction = {}
        self.__N = {}
    # -- end init --
        
    # define structure and functions maintaining visits and Qfunction
    #
    # defined as Q: hist -> action -> state
    def __updateX(self, X, state, hist, action, increment):
        """updates X (dict of hist - matrix) by +="""
        k = tuple(hist)
        if k in X.keys():
            # k -> sparse -> el
            X[k][action, state] += increment
        else:
            # insert new value
            values = sp.lil_matrix( (self.problem.A, self.problem.S) )
            values[action, state] = increment 
            X[k] = values
    # --------------
    def __getX(self, X, state, hist):
        """returns an action indexed vector of entitiy X"""
        k = tuple(hist)
        if k in X.keys():
            # return as dense vector
            return X[k][:, state].toarray().squeeze()
        else:
            return np.zeros_like( self.problem.actions )
    # --------------   
    def __updateQ(self, state, hist, action, R): 
        """updates Q function by TD(0) with alpha = 1/N_a
        updates also N_a
        
        'Monte-Carlo' tree search backup"""
        self.__updateX(self.__N, state, hist, action, 1) # update visit
        q = self.__getQ_a(state, hist)
        r = (R - q[action])
        n = self.__N_sa(state, hist)
        v = r / n[action] # Q-function increment
        self.__updateX( self.__Qfunction, state, hist, action, v)
    
    # getters from data structures
    __getQ_a = lambda self, state, hist: self.__getX(self.__Qfunction, state, hist)
    __N_sa = lambda self, state, hist: self.__getX(self.__N, state, hist)
    # ----
    # =============================
    def todeep(self, d: int) -> bool:
        # random modification
        #return (np.random.sample()**d < self.problem.discount)
        return d > self.max_depth
    #
    # define procedures
    # =============================
    def _search(self, root_state: State, root_hist: History) -> Action:
        self.update(root_hist) # update Q for real MDP, 
        #or if rollout other than epsilon-greedy: do nothing
        t = time_now()
        while (time_now() - t) < self.max_time:
            transModel = self.problem.sample_transition_model(root_hist)
            print("==== new simulation ====")
            self._simulate(root_state, root_hist, transModel, 0)
        # --- planning time depleted ---
        # return action that maximizes Q; break ties
        print('--- searched ---')
        print(self.__getQ_a(root_state, root_hist))
        best_a = np.argmax( self.__getQ_a(root_state, root_hist) )
        print('best a',best_a)
        action = best_a
        return action

        
    def _rollout(self, state: State, hist: History, transModel: TransitionMatrix, 
                 depth: int) -> float:
        s = state
        h = hist.copy() # no update of given history
        d = depth
        print('---')
        print('rollout at state', str(s))
        
        r = 0 # accumulator for reward
        y_d = 1 # accumulator for recursive multiplication with discount
        while not self.todeep(d): # iterative formulation for efficiency
            a = self.rollout_policy(s, h)
            next_state = self.problem.bayes_transition(s, a, transModel)
            r += y_d * self.problem.reward(s, a, next_state)
            # "recursive call" = prepare next iteration
            h.append( (s,a,next_state) )
            s = next_state
            d += 1
            y_d *= self.problem.discount
        print('rollout reward ', r)
        print('---')
        return r # breaks at depth ...
            
    #TODO iterative deepening???
    # started above
    #
    # suggested by 
    # L. Kocsis and C. Szepesvári, 
    # “Bandit Based Monte-Carlo Planning,”
    # in Machine Learning: ECML 2006, 2006, pp. 282–293.
    def _simulate(self, state: State, hist: History, model: TransitionMatrix, 
                  depth: int) -> float:
        s = state
        N_a = self.__N_sa(state, hist) # N(s,h,a) vector indexed by a
        N = np.sum(N_a)   # N(s,h)
        print("simulate at d=", depth)
        if self.todeep(depth):
            return 0
        elif N == 0:
            print("============")
            print("init ", s, len(hist))
            # N, Q implicit initialized to 0 
            # init one action
            a = self.rollout_policy(s, hist)
            print('init action', a)
            s_next = self.problem.bayes_transition(s, a, model)
            h_next = hist + [(s, a, s_next)] #no history update, needed for indexing
            # rollout
            r = self.problem.reward(s, a, s_next) + self.problem.discount*self._rollout(s_next, h_next, model, depth)
            self.__updateQ(s, hist, a, r)
            hist = h_next # change history at last
            return r
        # --- end init ---
        #
        # Q function: vector indexed by a
        else:
            if np.any(N_a == 0):
                # play an undiscovered action, get indices of undiscovered
                unknown_a = np.array(np.where(N_a == 0)).flatten()
                a = np.random.choice(unknown_a)
            else:
                # all N_a > 0 => N > 0, no problem in computing exploration bonus
                Q_b = self.__getQ_a(state, hist)
                # UCB exploration estimate: vector -a (N_a is vector => vector)
                #                       scalar          *              scalar / vector
                explore_b = np.multiply( self.exploration_scale, np.sqrt(np.divide(np.log(N), N_a)) )
                tradeoff = Q_b + explore_b
                print('--- UCB ---')
                print(tradeoff)
                a = np.random.choice(np.argmax( tradeoff ).flatten())
            print('a -> ', a)
            
            s_next = self.problem.bayes_transition(state, a, model)
            h_next = hist + [(state, a, s_next)] # no update, needed for indexing
            R = self.problem.reward(state, a, s_next) + self.problem.discount * self._simulate(
                    s_next, h_next, model, depth+1)
            # updates Q function as Q + (R - Q) / N_a
            # updates N_a, N implicitly
            self.__updateQ(state, hist, a, R)
            return R
        # ------------------

# =================================

#    #TODO
#    def _rolloutIterativeDeepening(state, hist, transModel, depth, policy):
#        """iterative deepening gives more weight to nearer states"""
#        r = np.zeros(depth)
#        for d in range(depth): # d off by one?
#            r[d] += _rollout(state, hist, transModel, d+1, policy)
#        # normalize by visits
#        for d in range(depth):
#            # values are once normalized by discount**d ??
#            # a second time needed??
#            #r[d] *= problem.discount**d
#            # normalize by visits, which are something like ?**d or d**?; branching factor b==1
#            r[d] /= (depth - d) # d off by one (depth - d + 1)
#        return sum(r)

# ==============================
# wrap as policy
    def search(self, s: State, h: History) -> Action:
        return self._search(s, h)
    