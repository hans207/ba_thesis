#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:47:47 2018

implementation of duffs FSC

@author: Hans Stenglein

Implementation of "Monte-Carlo Improvement for Stochastic Finite-State Controller"

Reference:
M. O. Duff, “Monte-Carlo Algorithms for the Improvement of Finite-State Controllers: 
Application to Bayes-Adaptive Markov Decision Processes,” 
presented at the International Workshop on Artificial Intelligence and Statistics, 2001.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv as inverse

from itertools import product as iterprod # nicer loops
import typing
import time

from definition_MDP import State, Action, History
import definition_MDP

# ----------

def finiteStateController(problem: definition_MDP.MDP, Q: int, 
                          max_iter=5000, 
                          stepsize=[1.0e-1, 2.1e-2, 2.4e-3],
                          learning_rate=0.7, verbose=0
                ) -> typing.Tuple[definition_MDP.Matrix, definition_MDP.Policy]:
    """implements Monte-Carlo Improvement for Stochastic Finite-State Controller
    Input:
        - problem: Instance of an MDP
        - Q: number of memory-states, for which a controller is generated
        - [max_iter]: maximum iterations
        - [learning_rate]: learning rate for TD(0) learning of problems value function
        - [verbose]:
            0 no output until finish
            1 outputs parameters, distributions, gradient updates and trajectory
            2 outputs also matricies and asks for progressing
    returns:
        - V: Value function for given problem
        - polily: policy represented by the FSC
        
    
    Reference:
    M. O. Duff, “Monte-Carlo Algorithms for the Improvement of Finite-State Controllers: 
    Application to Bayes-Adaptive Markov Decision Processes,” 
    presented at the International Workshop on Artificial Intelligence and Statistics, 2001.
    """
    # ---------------------------------------------
    # initialize controller 
    # 
    # value function for original "physical" MDP
    V = np.zeros((problem.S, 1))    
    
    memory = range(Q) # space of memory-states
    # states and actions given by problem
    
    # parameters for distributions
    # phi, chi, psi 
    # distributions
    # alpha, xsi, eta
    fsc_parameters, fsc_distributions = __init_FSC(problem.S, problem.A, Q)
    
    # TODO usefull definition of convergence
    par_old = fsc_parameters.copy()             # track change
    dist_old = fsc_distributions.copy()
    par_convergence = np.zeros( (max_iter, 3) ) # save sum of parameter changes
    dist_convergence = np.zeros( (max_iter, 3) ) # abs sum of differences in distributions per step
        
    # ---------------------------------------------
    # "Do forever", break at convergence
    for iteration in range(max_iter):
        if verbose >= 3:
            print("phi:\n", fsc_parameters['phi'])
            print("alpha:\n",fsc_distributions['alpha'])
            print()
            print("chi:\n",fsc_parameters['chi'])
            print("xsi:\n",fsc_distributions['xsi'])
            print()
            print("psi:\n", fsc_parameters['psi'])
            print("eta:\n", fsc_distributions['eta'])
        if verbose >= 4:
            s = "go to iteration: " + str(iteration+1)
            input(s)
            
        # sample a generalized transition matrix
        # without any knowledge of history -> pure prior sampling
        Pmat = problem.sample_transition_model([])
        # sample the initial hybrid-state
        i_0 = problem.initial_state()
        i = i_0
        q = np.random.choice(Q, p=fsc_distributions['alpha'])
        # hybrid-state is now (i,q)
            
        # inverse policy averaged transition matrix
        Amat = __calc_matrix(problem, Q, Pmat, fsc_distributions)
        if verbose >= 3:
            print("sampled P:\n", Pmat)
            print("A:\n", Amat)
        # batch updates
        updates = {'phi' : np.zeros_like(fsc_parameters['phi']), 
                   'chi' : np.zeros_like(fsc_parameters['chi']), 
                   'psi' : np.zeros_like(fsc_parameters['psi'])}
            
        # follow trajectory
        traj_length = 0
        while True: # break with probability
            traj_length += 1
            if verbose >= 2:
                print("current traj length: ", traj_length)
            
            # current hyperstate is (i,q)
            #
            # sample action, next state, next memory-state and reward
            a = np.random.choice(problem.actions, p=fsc_distributions['xsi'][q, :])
            _, idx_j, prob_ij = sp.find(Pmat[a][i,:])
            i_next = np.random.choice(idx_j, p=prob_ij)
            q_next = np.random.choice(memory, p=fsc_distributions['eta'][q, :, i, i_next, a])
            r = problem.reward(i, a, i_next)
            if verbose >= 2:
                print("--------------------")
                print("i,  q,  i', q', a")
                print(i," ", q," ", i_next," ", q_next," ",a)
                if verbose >= 4:
                    input("go?")
            
            # TD(0) update for V[i] == V[ physical MDP ] !
            # TODO
            #
            # Sutton, Barto 2015 p.134 Eq. 6.2
            # as argument: learning_rate
            # update    learing rate ?      reward              temporal difference
            V[i] += learning_rate * ( r + problem.discount * V[i_next] - V[i])
            if verbose >= 4:
                print("TD(0) updated V\n", V)
            
            # gradient ascent
            # 
            # solve equation systems to obtain gradient
            grad = __gradient_estimate(i_0, i, q, i_next, q_next, a, 
                                       problem, Q, Amat, Pmat, fsc_distributions)
            if verbose >= 3:
                print("grad['phi']\n", grad['phi'].T)
                print("grad['chi']\n", grad['chi'].T)
                print("grad['psi']\n", grad['psi'][:,:,i, i_next, a].T)
            # move controller parameters in gradient direction
            # stepsizes are done by update_parameters_distributions() !
            updates['phi'] += grad['phi']
            updates['chi'] += grad['chi']
            updates['psi'] += grad['psi']
            if verbose >= 2:
                print("updates['phi']\n", updates['phi'].T)
                print("updates['chi']\n", updates['chi'].T)
                print("updates['psi']\n", updates['psi'][:,:,i, i_next, a].T)
                print("----------")
            # ---- cleanup iteration ----
            # set next hybridstate
            i, q = i_next, q_next
            
            # terminate trajectory with probability 1-gamma
            if (np.random.sample() > problem.discount): 
                fsc_parameters, fsc_distributions = __update_FSC(fsc_parameters,
                                        updates, stepsize)
                # convergence in parameters
                #TODO usefull convergence formulation
                par_convergence[iteration, 0] = np.sum( np.fabs( 
                        fsc_parameters['phi'] - par_old['phi'] ))
                par_convergence[iteration, 1] = np.sum( np.fabs( 
                        fsc_parameters['chi'] - par_old['chi'] ))
                par_convergence[iteration, 2] = np.sum( np.fabs( 
                        fsc_parameters['psi'] - par_old['psi'] ))
                par_old = fsc_parameters
                # convergence in distribution                
                #TODO usefull convergence formulation
                dist_convergence[iteration, 0] = np.sum( np.fabs( 
                        fsc_distributions['alpha'] - dist_old['alpha'] ))
                dist_convergence[iteration, 1] = np.sum( np.fabs( 
                        fsc_distributions['xsi'] - dist_old['xsi'] ))
                dist_convergence[iteration, 2] = np.sum( np.fabs( 
                        fsc_distributions['eta'] - dist_old['eta'] ))
                dist_old = fsc_distributions # update for next iteration
                
                if verbose >= 1:
                    print("------ iteration ", iteration+1, " finished with trajectory length of ", traj_length, " ------")
                    print("convergence in parameters", par_convergence[iteration])
                    print("convergence in distributions", dist_convergence[iteration])
                break
            # -- break trajectory --
        # -- end for loop iterations --
    # --- run finished ---
    saveFSC(fsc_distributions, Q, max_iter, problem, 
            par_convergence, dist_convergence, stepsize)
    # ======================================================================
    print("\n\n\n")
    print("#===== run finished =====#")
    print("iterations:          ", iteration+1)
    print("discount:            ", problem.discount)
    print("step sizes:          ", stepsize)
    print("TD(0) learning rate: ", learning_rate)
    print()
    print("alpha(q):\n",fsc_distributions['alpha'])
    print("xsi(q,a):\n",fsc_distributions['xsi'])
    for ii, aa, jj in iterprod(problem.states, problem.actions, problem.states):
        print("o = s,a,s': ", ii, aa, jj)
        print("eta(q,q'|o):\n", fsc_distributions['eta'][:, :, ii, jj, aa])
    print()
    print("Value function for original MDP\n", V)
    print("#========================#")
    
    return (V, FSC(problem.actions, memory, fsc_distributions).search)
              
# =============================================================================
# =============================================================================
### --- functions for maintaining controller parameters --- ###
def __init_FSC(S, A, Q): #TODO random init etc.
    """
    this function initializes the parameters for the controller distribution
    
    values are initialized equally (uniform) distributed
    Input:
        - S number of states
        - A number of actions
        - Q number of memory states
    returns:
        parameters: dict with parameters 'phi', 'chi' and 'psi'
        distributions: dict with distributions 'alpha', 'xsi', 'eta'
    """
    # distribution over initial memory state
    # uniform values
    phi = np.ones( (Q, 1) )
    # action distributions
    # uniform values
    chi = np.ones( (Q, A) )
    # memory-state transitions
    # uniform values
    psi = np.ones( (Q, Q, S, S, A) )
    z = {'phi':np.zeros_like(phi),
         'chi':np.zeros_like(chi),
         'psi':np.zeros_like(psi) }
    return __update_FSC({'phi':phi, 'chi':chi, 'psi':psi}, z, [0,0,0])
# --- end __init_parameters() ---
        
def __update_FSC(parameters, delta_parameters, stepsize):
    """
    this function calculates the controller distributions (alpha, xsi, eta)
    initial distribution: alpha(phi)
    action distribution: xsi(chi)
    memory-state transition distribution: eta(psi)
    Input:
        parameters: parameters dict with elements 'phi', 'chi' and 'psi'
        delta_parameters: dict of updates with elements 'phi', 'chi' and 'psi'
        stepsize:       3 small floats as iterable
    returns:
        parameters: dict with updated parameters 'phi', 'chi' and 'psi'
        distributions: dict with updated distributions 'alpha', 'xsi', 'eta'
    updates according to the normalized delta_parameters * stepsize
    """
    # all controller transition probabilities 
    # are given as parameterized, exponentialized distributions (nd Boltzman)
    # ----
    min_preci = 1e-12 # this is small, dont care under this precision
    max_preci = 500   # as it gets exponentiated must be clipped at some point to not overflow
    a_phi = stepsize[0]
    a_chi = stepsize[1]
    a_psi = stepsize[2]
    # step rule: steps should be small, smaller than one
    # d > 1 => s/d # d > min => s*d # else 0
    step = (lambda s, d:
        s/d if d > 1 else (s*d if d > min_preci else 0) )
    # ----------------------------------------
    # distribution over *initial* memory state
    d_phi = np.fabs(delta_parameters['phi']).sum() # 1-norm
    step_phi = step(a_phi, d_phi)
    t_phi = parameters['phi'] + step_phi * delta_parameters['phi']
    phi = np.clip( np.nan_to_num(t_phi), min_preci, max_preci ) # lock values
    # --
    alpha = np.exp(phi)
    alpha = (alpha / alpha.sum( axis=0 ))[:,0] #normalize
    # ----
    # action distributions
    d_chi = np.fabs(delta_parameters['chi']).sum() # 1-norm
    step_chi = step(a_chi, d_chi)
    t_chi = parameters['chi'] + step_chi * delta_parameters['chi']
    chi = np.clip( np.nan_to_num(t_chi), min_preci, max_preci ) # lock values
    # --
    xsi = np.exp( chi )
    xsi /= xsi.sum( axis=1 )[:, np.newaxis] #normalize over actions
    # ----
    # memory-state transitions
    d_psi = np.fabs(delta_parameters['psi']).sum() # 1-norm
    step_psi = step(a_psi, d_psi)
    t_psi = parameters['psi'] + step_psi * delta_parameters['psi']
    psi = np.clip( np.nan_to_num(t_psi), min_preci, max_preci ) # lock values
    # --
    eta = np.exp( psi )
    eta /= eta.sum( axis=1 )[:, np.newaxis] #normalize over next memory-states (q')    
    return ({'phi':phi,     'chi':chi, 'psi':psi}, # parameters
            {'alpha':alpha, 'xsi':xsi, 'eta':eta}) # distributions
# --- end __update_FSC() ---
    
### --- end functions for maintaining controller parameters --- ###
# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================
### --- functions for calculating gradient --- ###
def __calc_matrix(problem, Q, Pmat, fsc_distributions):
    """calculates the inverse "policy averaged transition matrix"
    Input:
        problem MDP
        S number of states
        Q number of memory states
        
        Pmat transition matrix
        fsc_distributions"""
    
    S = problem.S
    #states = range(S)
    actions = range(problem.A)
    memory = range(Q)
    
    Mentries = sp.lil_matrix( (S*Q, S*Q) ) # NQxNQ
    # policy averaged transition
    for aa in actions:
        akk = 0 # sum over a
        for ii, jj, prob in np.nditer(sp.find(Pmat[aa])): 
            # consider only indices with nonzero P
            for qq, pp in iterprod(memory, memory):
                    xsi = fsc_distributions['xsi'][qq, aa]
                    eta = fsc_distributions['eta'][qq, pp, ii, jj, aa]
                    akk +=  xsi * prob * eta
            Mentries[ii*Q +qq, jj*Q +pp] = akk
    
    Mmat = Mentries.tocsc()
    # csc efficient for inversion
    SystemMat = sp.identity(S*Q, format='csc') - problem.discount*Mmat
    # A = (I-gamma M)inv; sparse inversion
    # get compressed row format as rows are needed for multiplication
    # csr efficient for row slicing
    Inv = sp.csr_matrix( inverse(SystemMat) )
    return Inv
# --- end __calc_Amatrix() ---
    
#TODO sparse
def __calc_bv(Q, problem, Pmatrix, fsc_distributions):
    b = np.zeros( (problem.S*Q, 1) ) # NQx1
    # loop i, q
    for i, q in iterprod(problem.states, range(Q)):
            akk = 0 # sum over j, a
            for aa in problem.actions:
                ak = 0 # expected reward, sum over j
                for jj in problem.states:
                    ak += Pmatrix[aa][i,jj] * problem.reward(i, aa, jj)
                akk += fsc_distributions['xsi'][q, aa] * ak
            b[i*Q+q] = akk
    return b

#TODO sparse
def __calc_bchi(q_current, a_current, Q, problem, Pmatrix, fsc_distributions, hyperV):
    
    b = np.zeros( (problem.S*Q, 1) ) # NQx1
    memory = range(Q)
    # element of bx -- (i,q)
    for ii in problem.states:
        part1 = 0 # sum over j
        for q in memory:
            for jj in problem.states:
                    # first sum over q'
                    sumq1 = problem.discount * np.dot( fsc_distributions['eta'][q, :, ii, jj, a_current], hyperV[jj, :] )
                    part1 += Pmatrix[a_current][ii, jj] * (problem.reward(ii, a_current, jj) + sumq1)
            # --------
        part2 = 0 # sum over a
        for aa in problem.actions: 
            akk_jj = 0 # sum over j'
            for jj in problem.states: 
                sumq2 = 0 # sum over q'
                for qq in memory: 
                    sumq2 += problem.discount * fsc_distributions['eta'][q_current, qq, ii, jj, aa] * hyperV[jj, qq]
                akk_jj += Pmatrix[aa][ii, jj] * ( problem.reward(ii, aa, jj) + sumq2)
                # self.P[aa][i, jj] * ( self.R[aa][i, jj] + sumq2 )
            # -------
            part2 += fsc_distributions['xsi'][q_current, aa] * akk_jj
        # -----------
        b[ii*Q+q_current] = fsc_distributions['xsi'][q_current, a_current] * ( part1 - part2 )
    return b

#TODO sparse  
def __calc_bpsi(i_current, i_next, q_current, q_next, a_current, Q, problem, Pmatrix, fsc_distributions, hyperV):
    
    b = np.zeros( (problem.S*Q, 1) ) # NQx1
    b[ i_current*Q + q_current ] = ( problem.discount * fsc_distributions['xsi'][q_current, a_current] * Pmatrix[a_current][i_current, i_next] * 
        fsc_distributions['eta'][q_current, q_next, i_current, i_next, a_current] *
        ( hyperV[i_next, q_next] - 
         np.dot( fsc_distributions['eta'][q_current, :, i_current, i_next, a_current], hyperV[i_next, :] ) ))
    return b


def __gradient_estimate(i_0, i_current, q_current, i_next, q_next, a_current, problem, Q, Amatrix, Pmatrix, fsc_distributions):
    """ estimates the gradient with respect to FSC parameters
    calculates vectors b_v, b_chi, b_psi
    
    Input:
        todo
    
    --------
    returns: gradient: {'phi', 'chi', 'psi'}
    """    
    memory = range(Q)
    #=======================================
    # solve linear equations for value function of hyperstates
    # V(i,q) == hyperV
    hyperV = np.zeros( (problem.S, Q) )
    # gradPhi only needs V(i0, qq) but bv (-> gradChi) needs V(jj,qq), also by (-> gradPsi)
    bv = __calc_bv(Q, problem, Pmatrix, fsc_distributions)
    for ii, qq in iterprod(problem.states, memory):
        hyperV[ii, qq] = Amatrix[ii*Q + qq, :].dot(bv) #TODO vectorize loop with slice
    #=======================================
    # should give Qx1 non-zero elements = all
    gradPhi = np.multiply( fsc_distributions['alpha'], (hyperV[i_0, : ] - np.dot( fsc_distributions['alpha'], hyperV[i_0, :]) ) )[:,np.newaxis]
    #=======================================
    
    #=======================================
    gradChi = np.zeros((Q, problem.A)) # QxA
    chi_comp = np.zeros( (Q, 1) )
    bx = __calc_bchi(q_current, a_current, Q, problem, Pmatrix, fsc_distributions, hyperV)
    # over Q
    for qq in memory: #TODO only N elements contribute, cut out those matrix parts
        chi_comp[qq] = Amatrix[i_0*Q + qq, :].dot(bx)
    gradChi[q_current, a_current] = np.dot(fsc_distributions['alpha'], chi_comp)
    #=======================================
    
    #=======================================
    gradPsi = np.zeros( (Q, Q, problem.S, problem.S, problem.A) )
    psi_comp = np.zeros( (Q, 1) )
    by = __calc_bpsi(i_current, i_next, q_current, q_next, a_current, Q, problem, Pmatrix, fsc_distributions, hyperV)
    #idx = np.nonzero(by) # only one nonzero element
    #TODO only column (i-1)Q+q
    for qq in memory:
        psi_comp[qq] = Amatrix[i_0*Q + qq, :].dot(by)
    gradPsi[q_current, q_next, i_current, i_next, a_current] = np.dot(
            fsc_distributions['alpha'], psi_comp)
    
    #=======================================
    
    return {'phi' : gradPhi, 'chi' : gradChi, 'psi' : gradPsi}

# ----------------------------------------
### --- functions for calculating gradient --- ###
# =============================================================================
# =============================================================================
    
    
# ------------------
# non-stationary, in S; Markov in SxH
def _fsc_pol(s, traj, fsc_distributions):
    # deprecated, calculates the memory state at each call from q0 on!
    """returns an action based on the given trained controller distributions
    - s: state for which a action should be chosen
    - traj: trajectory so far
    - 
    - fsc_distributions
        - alpha: P(q0=q), initial memory-state distribution
        - xsi: P(a|q), action for each memory state distribution, "policy in memory space"
        - eta: P(q'|q, s, s', a)"""
    # notation: state i (==s), next state j
    memory = np.size(fsc_distributions['alpha'])
    actions = np.size(fsc_distributions['xsi'][0]) # -> xsi[0] == P(a|q=0); all equally sized
    
    # simulating posterior
    q = np.random.choice(memory, p=fsc_distributions['alpha'])
    for obs in traj: #TODO maybe off by one error
        # at each call guess hybrid-state from (i0,q0) on
        #                                  q ->q'    i       j       a
        q = np.random.choice(memory, p=fsc_distributions['eta'][q, :, obs[0], obs[2], obs[1]])
    a = np.random.choice(actions, p=fsc_distributions['xsi'][q, :])
        
    return a

# -------------------------
class FSC():
    # using a class to save q
    def __init__(self, actions, memory, fsc_distributions):
        self.actions = actions # set of actions
        self.memory = memory # set of memory-states
        self.q = np.random.choice(memory, p=fsc_distributions['alpha']) # init q
        self.mem_pol = fsc_distributions['xsi']
        self.mem_trans = fsc_distributions['eta']
    
    def __react(self, obs):
        self.q = np.random.choice(self.memory, 
                    p=self.mem_trans[self.q, :, obs[0], obs[2], obs[1]])
        a = np.random.choice(self.actions, p=self.mem_pol[self.q, :])
        return a
    
    def search(self, s: State, traj: History) -> Action:
        """
        one step policy search
        
        uses only the last observation
        non-stationary in S; Markov in SxH
        """
        if traj:
            return self.__react(traj[-1]) # last observation
        else: # avoid empty history
            return np.random.choice(self.actions, p=self.mem_pol[self.q, :])    

def saveFSC(fsc_distributions, Q, i, mdp, par_convergence, dist_convergence, stepsize):
    name = 'sim_fsc/fsc_Q_' + str(Q) + 'on_' + repr(mdp) + 'with_iterations_' + str(i) + '_' + time.strftime('%j-%X')
    head = "FSC run " + str(i) + "iterations, stepsizes: " + str(stepsize)
    np.savetxt(name + '_alpha', fsc_distributions['alpha'], header=head)
    np.savetxt(name + '_xsi', fsc_distributions['xsi'], header=head)
    for ii, aa, jj in iterprod(mdp.states, mdp.actions, mdp.states):
        obs = '::'.join(["o", str(ii), str(aa), str(jj)])
        np.savetxt(name + '_eta_' + obs, 
                   fsc_distributions['eta'][:, :, ii, jj, aa], header=head)
    #np.savetxt(name + '_parameters_conv', par_convergence, 
    #           header=head + "\n abs sum of updates, phi, chi, psi")
    #np.savetxt(name + '_distribution_conv', dist_convergence, 
    #           header=head + "\n abs sum of differences t - (t-1), alpha, xsi, eta")