---
title: Index (Reinforcement Learning)
---

A blog-style lecture series on reinforcement learning, from the formal foundations (MDPs and Bellman equations) to modern deep RL algorithms. Read in order; each lecture lists its prerequisites at the top.

## Foundations

1. [[Introduction to Reinforcement Learning|Introduction to RL]]: why sequential decisions are hard, MDPs, policies, value functions, and the Bellman equations.
2. [[Dynamic Programming]]: solving the Bellman equations with a known model. Contractions, policy evaluation, policy iteration, value iteration and GPI.

## Learning policies from data

3. [[Imitation Learning]]: behavior cloning, why errors compound ($O(\epsilon T^2)$), and DAgger.
4. [[Policy Gradients]]: REINFORCE, variance reduction with baselines and reward-to-go, off-policy policy gradients, and straight-through Gumbel-softmax.

## Coming soon

- Multi-armed bandits: exploration vs exploitation, ε-greedy, UCB, gradient bandits.
- Temporal-difference learning and Q-learning: dynamic programming from samples.
- Actor-critic methods.

---

*Acknowledgements:* the structure of these notes is loosely based on CS285 (Deep Reinforcement Learning) at UC Berkeley by Prof. Sergey Levine, and E0277 at the Indian Institute of Science (IISc) by Prof. Shalabh Bhatnagar.
