---
title: Index (Reinforcement Learning)
next: "[[Multi-Armed Bandits]]"
---

A blog-style lecture series on reinforcement learning. It starts with the methods that need the fewest assumptions: learning from demonstrations and from trial and error, without a model of the environment. Then it turns to planning when the model is known (or learned), and ends with the mathematics that proves the learning algorithms converge. Each lecture lists its prerequisites at the top.

## Getting started

1. [[Multi-Armed Bandits]]: RL with a single state. Exploration vs exploitation, incremental estimates (a first stochastic-approximation algorithm), ε-greedy, UCB with a regret proof, gradient bandits and Thompson sampling.
2. [[Introduction to Reinforcement Learning|Introduction to RL]]: sequential decisions, MDPs (routing, recycling robot, pole balancing), policies, returns and horizons, value functions, and the Bellman equations, with hand-checked gridworlds.

## Part I: model-free reinforcement learning

We cannot assume the transition probabilities are known; the agent learns from experience.

3. [[Imitation Learning]]: behavior cloning as supervised learning, why errors compound ($O(\epsilon T^2)$), and DAgger.
4. [[Policy Gradients]]: REINFORCE, reward-to-go and baselines, off-policy policy gradients, and straight-through Gumbel-softmax.
5. [[Actor-Critic Algorithms]]: a learned critic for the policy gradient: advantage estimation, batch and online actor-critic, A2C/A3C, n-step returns and GAE, off-policy actor-critic (DDPG, TD3, SAC), and two-timescale convergence.
6. [[Monte Carlo and Temporal-Difference Learning]]: learning value functions from samples: Monte Carlo prediction, TD(0), MC vs TD, n-step returns and TD(λ) with eligibility traces.
7. [[Q-Learning and SARSA]]: learning optimal action values: Q-learning and its convergence conditions, SARSA and exploration (GLIE), cliff walking, maximization bias and Double Q-learning.
8. [[Linear Function Approximation]]: TD with linear features, the projected Bellman equation, why TD converges (and where), error bounds, TD(λ) and LSTD.
9. [[Off-Policy Learning with Function Approximation]]: importance sampling, the deadly triad and Baird's counterexample, emphatic TD, and linear Q-learning.

## Part II: model-based reinforcement learning

When the transition model is known we can plan exactly; when it is not, we can learn it.

10. [[Finite-Horizon Dynamic Programming]]: backward induction and the principle of optimality, with a chess-match and an inventory example.
11. [[Dynamic Programming]]: discounted infinite-horizon planning: Bellman operators and contractions, policy evaluation, policy iteration, value iteration, error bounds, modified and asynchronous policy iteration, and discounting as termination.
12. [[Stochastic Shortest Path Problems]]: episodic problems without discounting: proper and improper policies, existence and uniqueness of the optimal value, value and policy iteration, and weighted contractions.
13. [[Model-Based Reinforcement Learning]]: learning the model: certainty equivalence, Dyna, planning with learned models (MPC, CEM), and model-based policy learning.

## Part III: analysis

14. [[Stochastic Approximation]]: the convergence toolkit behind Part I: Robbins–Monro iterations, the ODE method, martingale noise, Lyapunov stability, asynchronous and two-timescale stochastic approximation.

## Coming soon

- Deep Q-networks, and trust-region and proximal policy optimization (TRPO/PPO).
- Exploration beyond bandits, offline RL and multi-agent RL.

---

*Acknowledgements:* the structure of these notes follows CS 285 (Deep Reinforcement Learning) at UC Berkeley by Prof. Sergey Levine.
