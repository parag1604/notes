---
title: Index (Reinforcement Learning)
---

A blog-style lecture series on reinforcement learning, from the formal foundations (MDPs and Bellman equations) to modern deep RL algorithms. Read in order; each lecture lists its prerequisites at the top.

## Foundations

1. [[Multi-Armed Bandits]]: RL with a single state. Exploration vs exploitation, incremental estimates, ε-greedy, UCB (with a regret proof), gradient bandits and Thompson sampling.
2. [[Introduction to Reinforcement Learning|Introduction to RL]]: why sequential decisions are hard, MDPs, policies, value functions, and the Bellman equations.
3. [[Dynamic Programming]]: solving the Bellman equations with a known model. Contractions, policy evaluation, policy iteration, value iteration and GPI.

## Learning policies from data

4. [[Imitation Learning]]: behavior cloning, why errors compound ($O(\epsilon T^2)$), and DAgger.
5. [[Policy Gradients]]: REINFORCE, variance reduction with baselines and reward-to-go, off-policy policy gradients, and straight-through Gumbel-softmax.

## Coming soon

- Temporal-difference learning and Q-learning: dynamic programming from samples.
- Actor-critic methods.

---

*Acknowledgements:* the structure of these notes is loosely based on CS285 (Deep Reinforcement Learning) at UC Berkeley by Prof. Sergey Levine, and E0277 at the Indian Institute of Science (IISc) by Prof. Shalabh Bhatnagar.
