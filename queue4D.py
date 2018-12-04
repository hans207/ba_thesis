#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:50:20 2018

MDP model for 4D queueing network

@author: Hans Stenglein

Reference on the 4D-queueing network:
Y. Abbasi-Yadkori, P. L. Bartlett, and A. Malek, 
“Linear Programming for Large-Scale Markov Decision Problems,” Feb. 2014.
https://arxiv.org/abs/1402.6763
"""
import numpy as np
import scipy.sparse as sp
#from numba import jit, njit # tried parallelization

import warnings
from time import strftime
import glob # file handling

import typing # tried to use typing
from definition_MDP import State, History, Action, TransitionMatrix, TransitionModel

from definition_MDP import MDP
from sampleMDP import sample

# types
X1D = int
X4D = np.ndarray

class queue4D(MDP):
    
    def __init__(self, buffer, delay, arrive, loss='totalsize', discount=0.95):
        """
        - buffer should be 4 values, 1 is broadcasted to all, everything else raises an value error
        - delay should be 4 values, 1 is broadcasted to all, everything else raises an value error
        - arrive should be 2 values, 1 is broadcasted to all, (4 values is allowed) everything else raises an value error
        - loss is a key for a loss function, allowed:
                - 'totalsize': reward as neg loss which is the sum of all queues
                - 'maxsize': reward as neg loss of the biggest queue
                - 'throughput': reward as througput of last (outgoing) queues (q2, q4)
                - 'throughputall': reward as througput of all queues
        """
        # sanity check for arguments
        # -----
        # buffer should be 4 values, 1 is broadcasted to all, everything else raises an value error
        if np.size(buffer) == 1:
            self.buffer = np.full(4, buffer)
        elif np.size(buffer) == 4:
            self.buffer = np.array(buffer).reshape(4)
        else:
            raise ValueError("buffer must be 1 value for all queues or 4 values, each for one queue")
        if np.size(delay) == 1:
            self.delay = np.full(4, delay)
        elif np.size(delay) == 4:
            self.delay = np.array(delay).reshape(4)
        else:
            raise ValueError("delay must be 1 value for all queues or 4 values, each for one queue")
        if np.size(arrive) == 1:
            self.arrive = np.full(4, arrive)
        elif np.size(arrive) == 2:
            self.arrive = np.array([arrive[0], None, arrive[1], None])            
        elif np.size(arrive) == 4:
            self.arrive = np.array(arrive).reshape(4)
        else:
            raise ValueError("arrive must be 1 value for all incoming jobs or 2 values, each for one incoming jobs (4 values are also exepted but some are ignored!)")
        
        # calculate possible state-space, raise a warning if too large
        __s_max = 100 #TODO maximum size of statespace
        __nStates = (buffer[0]+1) * (buffer[1]+1) * (buffer[2]+1) * (buffer[3]+1)
        if(__nStates > __s_max):
            hint = ("Statespace is soo big with " + str(__nStates) 
            + " sample_transition_model() will not return a matrix!")
            warnings.warn(hint, RuntimeWarning)
        # overwrite get_S(), now we know size of statespace
        self.get_S = lambda: (__nStates, np.arange(__nStates))
        # now we can use this, 
        super().__init__(discount)
        # being an MDP from here on
        # -----
        #TODO sanity check for loss function
        self.loss = loss
        # set the maximum reward
        self.maxReward = self.__maxReward()
        
        # TODO get prior transition counts from simulation
        # sample only if needed!
        print('sampling with uniform policy to estimate prior probabilities for transitions')
        sample4Dqueue(self, 1000, 5) # 1000 iterations, 5 times = 5000 transitions
        print('simulations done, now count transitions for sampling a prior')
        self.counts = produceData(self) # counts acts as prior
        print('counts as parameters for dirichlet distribution are as follows: ')
        print(self.counts)
    # ---- end init() ----
    
    def get_S(self):
        """number of states, returning N_states and a iterable of states (1D).
        must be overwritten by subclasses"""
        # will be overwritten by init()
        # bad style
        pass    
    def get_A(self):
        """number of actions, returning N_actions and a iterable of actions (1D).
        must be overwritten by subclasses"""
        return 4, np.arange(4)
    
    def initial_state(self):
        """samples an initial state, the default is state 0, but could be overwritten"""
        return 0 #TODO initial

    def sample_transition_model(self, hist: History) -> TransitionModel:
        """samples a transition model, which is passed to bayes_Transition()
        
        with no history given samples only from prior
        """
        __s_max = 100 #TODO maximum size of statespace
        if( self.S > __s_max): # statespace is to big
            return {'history': hist, 'P':None} 
            # lazy sampling for BAMCP, save history as list
        else:
            # invoke the standart sampling implemented by MDP
            # uses counts as prior information, which should be sampled by init()
            # returns a matrix
            return {'P': super(queue4D, self).sample_transition_model([])}
    
    
    def lazy_sample(self, state: State, action: Action, 
                    model: TransitionModel,
                    counts):
        # no return, instead update given model
        key = (state, action)
        if not key in model.keys():
            # sample new row distribution from counts
            print('lazy sample transition distr. P(', state, action, ': | hist)')
            if np.size(counts) > 1: #0 is valid
                posterior_counts = self.counts[action][state, :].tocsr(copy=True) + counts
            else:
                posterior_counts = self.counts[action][state, :].tocsr(copy=True)
            # ----
            # get dirichlet from non-zero (!) counts;
            # idx0 is all 0 in size of idxc
            idx0, idxc, par_c = sp.find(posterior_counts)
            probs = np.random.dirichlet(par_c)
            # fill row with probabilities from dirichlet distribution
            # at the places where the counts were taken from
            dirichlet_row = sp.csr_matrix( (probs, (idx0,idxc) ), shape=(1, self.S) )
            # insert in transition model
            model[key] = dirichlet_row
            # ---
        # transition distribution should now be in       
        _, next_idx, prob = sp.find(model[key])
        if np.size(next_idx) > 0:
            return np.random.choice(next_idx, p=prob)
        else:
            # if empty there is a state for which no prior/posterior transitions exist
            # use a uniform prior instead
            return np.random.choice(self.actions)
            
    def bayes_transition(self, state: State, action: Action,
                         model: TransitionModel) -> State:
        """sample a next state from given state, action and (estimated) transition model"""
        if model['P']:
            # Transition model is given as matrix
            return super(queue4D, self).bayes_transition(state, 
                        action, model['P'])
        else:
            # build posterior from observations
            hist = model['history']
            counts = sp.lil_matrix( (1, self.S) ) # only one row
            if len(hist) > 0:
                for o in hist:
                    # update only relevant counts
                    # from (s^,a^) to s'
                    if o[0] == state and o[1] == action:
                        counts[0,o[2]] += 1
            # else counts are 0 -> only prior
            #                                   csr for efficient addition
            next_state = self.lazy_sample(state, action, model, counts.tocsr())
            return next_state
        
        
    def true_transition(self, state: State, action: Action) -> State:
        """Transition function for the given state/action."""
        X = self._state1to4(state)
        S = _actionVec(action)
        A1 = np.random.binomial(1, self.arrive[0])
        A3 = np.random.binomial(1, self.arrive[2])
        #broadcasting
        D = np.random.binomial(1, np.multiply(self.delay, S)) # D_it ~ Bernoulli(d_i*s_it)
        # A1 - D1
        # D1 - D2
        # A3 - D3
        # D3 - D4           D1          D3
        Z = np.array( [A1, D[0], A3, D[2]] ) - D
        # bound at 0, B_i
        X_next = np.clip(X + Z, 0, self.buffer)
        return self._state4to1(X_next)
    # --- end true_transition ---

    def reward(self, state: State, action: Action, next_state: State) -> float:
        """Reward function for the given s =(a)=> s'.
        returns a real number"""
        X = self._state1to4(state)
        
        if self.loss is 'totalsize':
        # reward as neg loss which is the sum of all queues
            return -np.sum(X)  
        elif self.loss is 'maxsize':
        # reward as neg loss of the biggest queue
            return -np.max(X)
        
        elif self.loss is 'throughput':
        # reward as througput of last queues (q2, q4)
            X_next = self._state1to4(next_state)
            # clipping at 0
            reward = max(0, X[1]-X_next[1]) + max(0, X[3]-X_next[3])
            return reward
        elif self.loss is 'throughputall':
        # reward as througput of all queues    
            X_next = self._state1to4(next_state)
            reward = max(0, X[0]-X_next[0]) + max(0, X[1]-X_next[1]) + max(0, X[2]-X_next[2]) + max(0, X[3]-X_next[3])
            return reward
            
        else:
            raise ValueError("unknown loss function", self.loss)
    # --- end reward ---

    def __maxReward(self) -> float:
        """returns the maximum one-step reward, that can be obtained
        the maximum one-step reward is an upper bound 
        concerning the given reward/loss function
        """
        if self.loss is 'totalsize':
            return 1 # neutral element for multiplication
            # the totalsize is in [0, B]**4, as it is a loss, the reward is 
            # negative, we want an upper bound: max -> - min
            # actually maxR = - min(totalsize) = -0**4 = 0 < 1
        elif self.loss is 'maxsize':
            return 1 # neutral element for multiplication
            # the maximum size is in [0, B], as it is a loss, the reward is 
            # negative, we want an upper bound: max -> - min
            # actually maxR = - min([0,B]) = -0 = 0 < 1
        elif self.loss is 'throughput':
        # reward as througput of last queues (q2, q4)
            reward = 2 # two downstream queues
            return reward
        elif self.loss is 'throughputall':
            return 4 # four queues

    def valid_actions(self, state):
        """Returns a list of valid actions that can be taken in the given
        state. by default all actions are valid in all states"""
        return self.actions # [0,1,2,3]
    
    def toTransitionParameterStr(self) -> str:
        b = "b" + str(self.buffer)
        d = "d" + str(self.delay)
        a = "a" + str(self.arrive)
        return b + d + a
    
    def repr(self) -> str:
        p = self.toTransitionParameterStr()
        l = self.loss 
        g = "gamma" + str(self.discount*100)
        return "Queue_with_" + p + l + g   

    #@njit
    def _state4to1(self, x4D):
        """convert 4D state representation to 1D"""
        # digit a_i, base {b}
        # b**i is prod(b_k, k<=i)
                                # b0 = 1
        akk = x4D[0] # *1       # a0 = q1
        
        base = self.buffer[0]+1    # b1 = B1
        akk += x4D[1]*base      # a1 = q2
        
        base *= self.buffer[1]+1   # b2 = B1*B2
        akk += x4D[2]*base      # a2 = q3
        
        base *= self.buffer[2]+1   # b3 = B1*B2*B3
        akk += x4D[3]*base      # a3 =  q4
        
        x1D = akk    
        return x1D
    def _state1to4(self, x1D):
        """inversion of _state4to1()"""
    
        res, q1 = divmod(x1D, self.buffer[0]+1)    # /% B1
        res, q2 = divmod(res, self.buffer[1]+1)    # /% B2
        q4,  q3 = divmod(res, self.buffer[2]+1)    # /% B3
        # res /% b = 0, %q4
        # == q4 = res
        return np.array([q1,q2,q3,q4])
    
def _actionVec(action_num: X1D) -> X4D:
    """given a number for an combined action returns the 4D action"""
    if action_num == 0:
        return  np.array([1,1,0,0]) # action 0 = s1,s2
    elif action_num == 1:
        return np.array([1,0,1,0]) # action 1 = s1,s3
    elif action_num == 2:
        return np.array([0,1,0,1]) # action 2 = s4, s2
    elif action_num == 3:
        return np.array([0,0,1,1]) # action 3 = s4, s3
    else:
        raise ValueError("invalid action number!", action_num)

def _actionIdx(action_vec: X4D) -> X1D:
    """gives number of 4D action vector"""
    if np.array_equiv(action_vec, np.array([1,1,0,0]).T): # action 0 = s1,s2
        return 0
    elif np.array_equiv(action_vec, np.array([1,0,1,0])): # action 1 = s1,s3
        return 1
    elif np.array_equiv(action_vec, np.array([0,1,0,1])): # action 2 = s4, s2
        return 2
    elif np.array_equiv(action_vec, np.array([0,0,1,1])): # action 3 = s4, s3
        return 3
    else:     
        raise ValueError("invalid action!", action_vec)

# =========================================================
# functions for sampling a prior distribution
# =========================================================
        
def __fileName(q):
    f = "sim_4Dqueue/sim_4Dqueue_transitions-" + q.toTransitionParameterStr() + "-action"
    return f

def sample4Dqueue(q, iterations=1000, trials=5):
    pol = (lambda s, traj: np.random.choice(q.actions)) # uniform policy
    for i in range(trials):
        _,_, counts = sample(q, pol, iterations)
        #sumnon0 = sum([counts[aa].count_nonzero() for aa in self.actions])
        #n = self.A*(self.S**2)
        #frac = sumnon0/n * 100
        #print(n-sumnon0," transitions inactive, total", sumnon0,"/",n, "= ", frac, "% used")
        f = __fileName(q)
        t = strftime("%c")
        for aa in q.actions:
            name = f + str(aa) + "_" + t
            sp.save_npz(name, counts[aa].tocsr())
# --- end save ---

def produceData(q):
    """constructing a prior from sampled trajectories (uniform policy) of the given queue
    
    WARNING does not work properly"""
    hint = "produceData() can not decide if enougth data is availabe, only loads simulations for the exact Queue parameters (B, d, a) of this queue ! - this may be none, resultung in an empty = uniform prior!"
    warnings.warn( hint, RuntimeWarning)
    
    counts = __readSampledCounts(q)
    # ----
    # check if enough samples are drawn
    toless = np.ones_like(q.actions)
    def tolesssamples():
        for aa in q.actions: # check for each action
            if counts[aa].count_nonzero() < 10: #TODO at least 10 transitions?
                toless[aa] = True
            else:
                toless[aa] = False
        return toless.any() 
    # any == or; if there is at least one action with to less samples ...
    # TODO problems with loop??
    #while tolesssamples():
    #    print("sampling ", q.toStr(), "!")
    #    __sample4Dqueue(q)
    #    counts = __readSampledCounts(q)
    # -----
    # now we have enough samples
    return counts

def __readSampledCounts(q):
    f = __fileName(q)
    c = []
    for aa in q.actions:
        c.append(sp.csr_matrix((q.S, q.S)))
        pat = glob.escape(f + str(aa)) + "*.npz"
        listfs = glob.glob(pat)
        for x in listfs:
            delta = sp.load_npz(x)
            c[aa] += delta
            print(x, " added")
    return c 

# =========================================================
# =========================================================
# --------------
# Policies given by Abbasi-Yadkori et al. 2014 as functions
# or de Farias and Van Roy 2003

def __pol_LONGER(state, hist, mdp):
    """servers first serve longer queue"""
    q = mdp._state1to4(state)
    action = np.zeros(4)

    # server 1: Q1, Q4
    if q[0] > q[3]:
        action[0] = 1
    elif q[0] < q[3]:
        action[3] = 1
    else: # ties broken uniform at random
         p = np.random.rand() 
         if p > 0.5:
             action[0] = 1
         else:
            action[3] = 1
        
    # server 2: Q2, Q3
    if q[1] > q[2]:
        action[1] = 1
    elif q[1] < q[2]:
        action[2] = 1
    else: # ties broken uniform at random
         p = np.random.rand() 
         if p > 0.5:
             action[1] = 1
         else:
            action[2] = 1
    return _actionIdx(action)

def pol_LONGER(mdp):
    return lambda s, h: __pol_LONGER(s, h, mdp)

def __pol_LBFS(state, hist, mdp):
    """servers first serve downstream queue"""
    q = mdp._state1to4(state)
    action = np.zeros(4)
    
    idx1 = 3 if q[3] > 0 else 0
    idx2 = 1 if q[1] > 0 else 2
    
    action[idx1] = 1
    action[idx2] = 1
    return _actionIdx(action)

def pol_LBFS(mdp):
    return lambda s, h: __pol_LBFS(s, h, mdp)
