---
title: Introduction to Reinforcement Learning
---

Reinforcement Learning (RL) is a discipline within machine learning that formalizes how to make sequential decisions under uncertainty. While machine learning techniques include supervised learning (<span style="color:red">cite</span>) at one end of its spectrum (where given data $\mathbf{X}$ and corresponding labels $\mathbf{y}$ one needs to find the mapping from $\mathbf{X}$ to $\mathbf{y}$ by minimizing empirical risk) and unsupervised learning (<span style="color:red">cite</span>) at the other end of its spectrum (where only data $\mathbf{X}$ is provided and one needs to find pattern within $\mathbf{X}$ by some clustering algorithm), none of these techniques deal with sequential decision making, i.e. although in the real world, the current decision has an effect on the future, such feedback is not considered in the predictions made by the machine learning models (which are used for single step of decision making).

### Why is RL Hard?

Typically supervised learning problems have assumptions that make them “easy”:​

- Independent datapoints​
- Outputs don’t influence next inputs​
- Ground truth labels are provided at training time​

Decision-making problems often don’t satisfy these assumptions​:

- Current actions influence future actions​
- Goal is to maximize some utility (reward)​
- Optimal actions are not provided

In many cases, real-world deployment of ML has these same feedback issues.​ For instance, decisions made by a traffic prediction system may affect the route that people take, which in turn changes the traffic.​

----------
## Markov Decision Process
Pre-requisite: Markov Random Process (<span style="color:red">cite</span>) and Markov Chains (<span style="color:red">cite</span>)
We model a sequential decision-making process under uncertainty as a Markov Decision Process (MDP). An MDP is specified by the tuple: $$\mathcal{M}=\{\mathcal{S}, \mathcal{A}, \mathcal{T}, r, \rho\}$$where $\mathcal{S}\in\mathbb{R}^{d_s}$ is the ($d_s$ dimensional) state space, $\mathcal{A}\in\mathbb{R}^{d_a}$ is the ($d_a$ dimensional) action space, $\mathcal{T}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the transition tensor (defines the dynamics governing the transition probabilities over next states given current state and action), $r:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\in\mathbb{R}$ is the reward function, and $\rho:\mathcal{S}\to[0,1]$ is a probability distribution over the states according to which the initial states are sampled.

A **policy** $\pi:\mathcal{S}\to\mathcal{A}$ (the software inside the agent) takes as input a state $s\in\mathcal{S}$ and outputs an action $a\in\mathcal{A}$. The policy can be learned by interacting with the MDP $\mathcal{M}$ (environment). Initially, state $s_0$ is sampled from $\rho$ and provided to the agent. The agent takes some action according to its internal policy $\pi_\theta$ (the policy is learnable and parameterized by $\theta$), i.e. $a_0\sim\pi_\theta(\cdot|s_0)$. This action $a_0$ is provided to $\mathcal{M}$ which then executes the action and provides the resultant next state $s_1$, reward $r_0=r(s_0,a_0,s_1)$, and $d_0$ (denoting whether the next state is a terminal state or not) to the agent. Thus at any given time instance $t$, an experience tuple (of agent-environment interaction) is of the form $(s_t,a_t,r_t,s_{t+1},d_t)$. When $d_t=True$, the environment is reset and a new initial state is sampled and returned to the agent. Thus a trajectory (assume it ran for $T$ steps) is of the form $\tau=\{s_0,a_0,r_0,s_1,a_1,r_1,s_2,...,s_T,a_T,r_T,s_{T+1}\}$.

Given $$p_\theta(\tau)=\rho(s_0)\prod\limits_{t=0}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$$the **goal** of reinforcement learning is to learn the optimal parameters $\theta^*$ such that $\pi_{\theta^*}\equiv\pi^*$ as follows:
$$
\theta^*=\arg\max_\theta\mathbb{E}_{\tau\sim p_\theta(\tau)}\sum\limits_{t=0}^{T} r_t
$$

***Note***:

- An alternative way to define MDPs is as follows: $\mathcal{M}'=\{\mathcal{S}, \mathcal{A}, \mathcal{T}, c, \rho\}$ where $c$ is the cost function. Therefore, by convention, the objective of the policy for such MDPs would be to minimize the cost rather than maximize the reward. Therefore, by setting $r(\cdot)=-c(\cdot)$, one can show that both the MDP definitions are equivalent.
- It can be shown that the reward function can also be defined as $r:\mathcal{S}\times\mathcal{A}\in\mathbb{R}$ without changing the expressive power of $\mathcal{M}$.
- A **Partially Observable/Observed Markov Decision Process** (POMDP) is defined by the tuple $\mathcal{M}=\{\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{E}, r, \rho\}$ where $\mathcal{O}$ is the ($d_o$ dimensional) observation space and $\mathcal{E}:\mathbb{R}^{d_s}\to\mathbb{R}^{d_o}$ defines the emission probabilities $p(o_t|s_t)$. All other symbols have their usual meanings. POMDPs are a generalization of MDPs where the agent is not given access to the internal state $s$, rather a (partial) observation $o\in\mathcal{O}$. So, the policy has to take actions based on $o_t$ instead of $s_t$. For instance, $s\in\mathcal{S}$ can the positions, velocities and torques of the joints of a quadruple legged robot whereas the $o\in\mathcal{O}$ can be an image of the robot taken using an external camera.

----------

## Some additional topics

### State and action space

<span style="color:red">Populate</span>

Until otherwise mentioned, we shall work with the case of continuous state and action space since they are very general, and almost all discrete setting analog are easy to derive/code. 

### Model of the environment

In machine learning, we often call the function approximator that maps data to labels as the "model". Similarly in RL, model often refers to the approximation of the environment that can be learned by interacting with the MDP. Recall that during a step of agent-environment interaction: (i) the environment gives a state to the agent, (ii) the agent gives an action to the environment corresponding to the state according to some internal policy, and (iii) the environment performs an internal step and returns the next state, reward and whether it was a terminal state or not.
Hence, in order to learn a successful "model" of the environment, one can learn three approximators as follows:

1. $\hat{s}_{t+1}=s_{t}+\hat{\Delta}_{t+1}$; where $\hat{\Delta}_{t+1}=f^s_\phi(s_t,a_t)$
2. $\hat{r}_{t+1}=f^r_{\phi}(s_t,a_t)$
3. $\hat{d}_{t+1}=\sigma(f^d_{\phi}(s_t,a_t))$

where, $f_\phi$ is a (non-)linear function approximator parameterized by $\phi$ and $\sigma(x)=\frac{1}{1+e^{-x}}$ is the sigmoid function.

### Horizon Length of Trajectories

The length of trajectories depends on the MDP, and can be classified into the following three categories: (i) Fixed Horizon, (ii) Finite Horizon, and (iii) Infinite Horizon.

<span style="color:red">Populate</span>

**Note** that in the notes, we will be using the finite horizon case unless otherwise specified.

### Value of a state/state-action pair

$$
\begin{align}
\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\sum\limits_{t=1}^{T}r(s_{t},a_{t})\right]
&=\mathbb{E}_{s_{0}\sim\rho(s_{0})}[\mathbb{E}_{a_{0}\sim\pi(a_{0}|s_{0})}[ \overbrace{r(s_{0},a_{0}) + \mathbb{E}_{s_{1}\sim p(s_{1})}[\mathbb{E}_{a_{1}\sim \pi(a_{1}|s_{1})}[ r(s_{1},a_{1}) + ... |s_{1}]]}^{\text{This term is known as the Q-value of the state-action pair $(s_{0},a_{0})$}} |s_{0}]]\\
&=\mathbb{E}_{s_{0}\sim\rho(s_{0})}[\mathbb{E}_{a_{0}\sim\pi(a_{0}|s_{0})}[Q(s_{0},a_{0})|s_{0}]]
\end{align}
$$
One can think of the **Q-value function** as the function that gives the expected total reward obtained from taking action $a_{t}$ when starting from state $s_{t}$ by following policy $\pi$:
$$
Q^{\pi}(s_{t},a_{t})=\sum\limits_{t'=t}^T\mathbb{E}_{\pi}\left[r(s_{t'},a_{t'})|s_{t},a_{t}\right]
$$
Similarly the **Value function** can be defined as the expected total reward obtained when starting from state $s_{t}$ under policy $\pi$:
$$
\begin{align}
V^{\pi}(s_{t})
&=\sum\limits_{t'=t}^T\mathbb{E}_{\pi}\left[r(s_{t'},a_{t'})|s_{t}\right]\\
&=\mathbb{E}_{a_{t}\sim\pi(a_{t}|s_{t})}[Q^{\pi}(s_{t},a_{t})]
\end{align}
$$

**Note**: $\mathbb{E}_{s_{0}\sim \rho(s_0)}\left[V^\pi(s_0)\right]$ is nothing but the aforementioned RL objective and the goal of RL is to maximize the same.

### Anatomy of a RL algorithm

```
                  -------------> fit a model
                /             (estimate returns)
               |                      |
               |                      |
        generate samples              |
        (execute policy)              |
               ^                      |
               |                      v
                \                 improve the
                  ----------------- policy
```

### On-policy vs Off-policy Algorithm

On policy:

- Experience is collected using current policy $\pi_\theta^t$ and after policy update, $\pi_\theta^{t+1}$ is the new policy.
- Since the experiences were collected using $\pi_\theta^t$ they are discarded and new trajectories are collected using $\pi_\theta^{t+1}$ for updating the policy.
- This is quite "sample" inefficient since old trajectories are not re-used.
- In practice however, the "wall-clock" time might actually be less if parallel simulations/real world experiments can be run.

Off policy:

- Update algorithm for $\pi_\theta^{t}$ is not restricted to experiences collected using $\pi_\theta^{t}$.
- Sample efficient learning algorithms.
- Catastrophic forgetting can be remedied by using previously collected samples.
- Also since off-policy samples can be used to learn, some of the samples can be collected using random policy (say w.p. $\epsilon$). Thus, "exploration-exploitation" tradeoff can be incorporated into the algorithm.
