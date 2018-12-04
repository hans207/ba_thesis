# ba_thesis
Code implemented for my BSc thesis "Optimal planning under Model Uncertainty" at TU Darmstadt 2018


Abstract

Bayesian model-based reinforcement learning is an elegant formulation to learning and planning optimal
behavior under model uncertainty. In this work an extension to the Markov decision process model
(MDP), used throughout in the field of reinforcement learning, is studied. The formalism of Bayes-
Adaptive Markov decision processes (BAMDP) allows an intrinsic representation of model uncertainty
and gathered information for action-selection. Thus solving a BAMDP is equivalent to finding an optimal
exploration / exploitation tradeoff in the underlying MDP.
I reviewed to approaches to solving BAMDPs: one offline approach based on policy improvement for
stochastic finite-state controllers and one online approach using sample-based tree-search with given
heuristics. I applied both approaches to two problems famous in literature: the chain problem and the
four-dimensional queueing network.
