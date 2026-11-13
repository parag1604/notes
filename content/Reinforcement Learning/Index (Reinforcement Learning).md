---
title: Index (Reinforcement Learning)
next: "[[Multi-Armed Bandits]]"
---

A blog-style lecture series on reinforcement learning, meant to be read in this order. Each lecture lists its prerequisites at the top.

The series has two halves. The first is practical: after bandits and the MDP framework, it builds the methods that learn from experience without a model of the environment, from imitating an expert to policy gradients, actor-critic, temporal-difference learning and Q-learning, with and without function approximation. The second half asks **why these methods work**. With a known model, the Bellman equations can be solved exactly (dynamic programming), which shows what TD and Q-learning are chasing. Stochastic approximation then explains why their noisy, sampled versions still converge. The series closes by joining the two halves: learn a model from experience, then plan with it.

## Lectures

1. [[Multi-Armed Bandits]]: RL with a single state. Exploration vs exploitation, incremental estimates (a first stochastic-approximation algorithm), ε-greedy, UCB with a regret proof, gradient bandits and Thompson sampling.
2. [[Introduction to Reinforcement Learning|Introduction to RL]]: sequential decisions, MDPs (routing, recycling robot, pole balancing), policies, returns and horizons, value functions, and the Bellman equations, with hand-checked gridworlds.
3. [[Imitation Learning]]: behavior cloning as supervised learning, why errors compound ($O(\epsilon T^2)$), and DAgger.
4. [[Policy Gradients]]: REINFORCE, reward-to-go and baselines, off-policy policy gradients, and straight-through Gumbel-softmax.
5. [[Actor-Critic Algorithms]]: a learned critic for the policy gradient: advantage estimation, batch and online actor-critic, A2C/A3C, n-step returns and GAE, off-policy actor-critic (DDPG, TD3, SAC), and two-timescale convergence.
6. [[Monte Carlo and Temporal-Difference Learning]]: learning value functions from samples: Monte Carlo prediction, TD(0), MC vs TD, n-step returns and TD(λ) with eligibility traces.
7. [[Q-Learning and SARSA]]: learning optimal action values: Q-learning and its convergence conditions, SARSA and exploration (GLIE), cliff walking, maximization bias and Double Q-learning.
8. [[Linear Function Approximation]]: TD with linear features, the projected Bellman equation, why TD converges (and where), error bounds, TD(λ) and LSTD.
9. [[Off-Policy Learning with Function Approximation]]: importance sampling, the deadly triad and Baird's counterexample, emphatic TD, and linear Q-learning.
10. [[Finite-Horizon Dynamic Programming]]: backward induction and the principle of optimality, with a chess-match and an inventory example.
11. [[Dynamic Programming]]: discounted infinite-horizon planning: Bellman operators and contractions, policy evaluation, policy iteration, value iteration, error bounds, modified and asynchronous policy iteration, and discounting as termination.
12. [[Stochastic Shortest Path Problems]]: episodic problems without discounting: proper and improper policies, existence and uniqueness of the optimal value, value and policy iteration, and weighted contractions.
13. [[Stochastic Approximation]]: why the sampled methods of lectures 1–9 converge: Robbins–Monro iterations, the ODE method, martingale noise, Lyapunov stability, asynchronous and two-timescale stochastic approximation.
14. [[Model-Based Reinforcement Learning]]: learning the model: certainty equivalence, Dyna, planning with learned models (MPC, CEM), and model-based policy learning.

## Coming soon

- Deep Q-networks, and trust-region and proximal policy optimization (TRPO/PPO).
- Exploration beyond bandits, offline RL and multi-agent RL.

---

*Acknowledgements:* the structure of these notes follows CS 285 (Deep Reinforcement Learning) at UC Berkeley by Prof. Sergey Levine.
