#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:47:36 2018

@author: Hans Stenglein

defines an MDP with transition, reward and policy as functions.
those could for example be sampled from underlying distributions

inspired by: 
# Demo: Bayes-Adaptive Monte-Carlo Tree Search
# originally written by Jessica B. Hamrick; 2015 at UC Berkeley
# < http://www.jesshamrick.com/ >
# original code from
# < https://github.com/jhamrick/quals/blob/master/notebooks/Guez2013.ipynb >
"""

from abc import ABCMeta, abstractmethod
import typing

import numpy as np
import scipy.sparse as sp

# type definitions
State = int
Action = int
History = typing.Iterable[typing.Tuple[State, Action, State]] #TODO

#MDPType = typing.Any #TODO typing
# typing.Type[MDP]
# see below
#TransitionModel = typing.Dict[str, typing.Any]

Matrix = typing.Any#typing.Union[typing.Type[ndarray], typing.Type[spmatrix]] #TODO
Vector = typing.Any#typing.Union[typing.Type[ndarray], typing.Type[spmatrix]] #TODO

TransitionMatrix = typing.Tuple[Action, Matrix]
TransitionModel = typing.Union[TransitionMatrix, typing.Any]

class MDP(object, metaclass=ABCMeta):
    """class representing an Markov Decision Process. this is an abstract class 
    providing only basic methods. They must be overwritten by subclasses"""

    """stationary distributions as initial state distr., transition matrix or reward distr.
        must be defined by subclasses (for use in transition_dist, reward etc.)"""

    def __init__(self, discount=0.95):
        self.discount = discount

        self.V = None # Value function unused
        self.Q = None # Q function
        
        self.maxReward = None # maximum reward
        self.counts = None # prior for Transition Distribution (dirichlet distribution)
        
        # set of states, set of actions as 1D representation
        self.S, self.states = self.get_S()
        self.A, self.actions = self.get_A()
        
    @abstractmethod
    def get_S(self) -> typing.Tuple[State, typing.Iterable[State]]:
        """number of states, returning N_states and a iterable of states (1D).
        must be overwritten by subclasses"""
        pass
    @abstractmethod
    def get_A(self) -> typing.Tuple[Action, typing.Iterable[Action]]:
        """number of actions, returning N_actions and a iterable of actions (1D).
        must be overwritten by subclasses"""
        pass
    
    def initial_state(self) -> State:
        """samples an initial state, the default is state 0, but could be overwritten"""
        return 0
    
    def sample_transition_model(self, hist: History) -> TransitionModel:
        """samples a transition model, which is passed to bayes_Transition()
        
        with no history given samples only from prior
        """
        posterior_counts = self.counts.copy()
        if len(hist) > 0:
            # posterior update
            for o in hist: # count transitions from history
                posterior_counts[o[1]][o[0], o[2]] += 1
            print('posterior updated')
        print('sampling transition model, wait...')
        P = self._sampleTransitionMatrix_fromCounts(posterior_counts)
        print("transition matrix generated")
        return P
    
    def bayes_transition(self, state: State, action: Action, 
                         model: TransitionModel) -> State:
        """sample a next state from given state, action and (estimated) transition model"""
        _, next_idx, prob = sp.find(model[action][state])
        #if empty there is a transition for which no prior/posterior exists
        # use uniform instead
        next_idx = next_idx if np.size(next_idx) > 0 else self.actions
        prob = prob if np.size(prob) > 0 else np.full_like(next_idx, 1/np.size(next_idx))
        
        next_state = np.random.choice(next_idx, p=prob)
        return next_state

    @abstractmethod
    def true_transition(self, state: State, action: Action) -> State:
        """Transition function for the given state/action. 
            
        problem specific, must be overwritten by subclasses"""
        pass

    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> float:
        """Reward function for the given s =(a)=> s'. Must be overwritten
        by subclasses.
        returns a real number"""
        pass

    def valid_actions(self, state: State) -> typing.Iterable[State]:
        """Returns a list of valid actions that can be taken in the given
        state. by default all actions are valid in all states"""
        return self.actions
    
    def _sampleTransitionMatrix_fromCounts(self, counts: TransitionMatrix
                                       ) -> TransitionMatrix:
        """TODO docs"""
        #TODO sample lazy, use dict
        #
        # build transition matrix of action-wise SxS sparse matrices
        pr_tmp = []
        for aa in self.actions:
            row_list = []
            for xx in self.states:
                # build row: xx =aa=> xx'
                # (dirichlet does not broadcast)
                # ----
                # get dirichlet from non-zero (!) counts;
                # idx0 is all 0 in size of idxc
                idx0, idxc, par_c = sp.find(self.counts[aa][xx])
                if np.size(par_c) > 0:
                    # use dirichlet counts
                    probs = np.random.dirichlet(par_c) 
                    # fill row with probabilities 
                    # at the places where the counts were taken from
                    row = sp.csr_matrix( (probs, (idx0,idxc) ), shape=(1, self.S) )
                else:
                    # counts are empty -> initialize uniform
                    row = sp.csr_matrix( np.full( (1, self.S), 1/self.S ) )
                row_list.append(row)
            # construct a sparse matrix from sparse rows
            pr_tmp.append(sp.vstack(row_list,'csr'))
            # --- transitions for aa done ---
        # transition matrix build as tuple of spare row orientated matrices
        return tuple(pr_tmp)
# ------
# Type definitions using MDP class

Policy = typing.Callable[[State, History], Action]
    
    