---
title: Policy Gradients
---

Recall that the goal of Reinforcement Learning is:
$$
\begin{align}
\theta^{*}
&=\arg\max_\theta\sum\limits_{t=0}^{T}\mathbb{E}_{(s_{t},a_{t})\sim p_{\theta}(s_{t},a_{t})}[r(s_{t},a_{t})]\\
&=\arg\max_\theta\underbrace{\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum\limits_{t=0}^{T}r(s_{t},a_{t})\right]}_{\text{This term can be denoted by }J(\theta)}
\end{align}
$$
$J(\theta)$ can be evaluated from samples as follows:
$$
J(\theta)
=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum\limits_{t=0}^{T}r(s_{t},a_{t})\right]
\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=0}^{T}r(s_{i,t},a_{i,t})
$$
i.e. we collect $N$ trajectories by running policy $\pi_\theta$ and estimate the value of the policy $J(\theta)$. This is essentially a Monte-Carlo estimate.
Letting $r(\tau) = \sum\limits_{t=0}^{T}r(s_{t},a_{t})$, we can find the $\nabla_\theta J(\theta)$:
$$
\begin{align}
\nabla_{\theta}J(\theta)
&=\nabla_{\theta}\mathbb{E}_{\tau\sim p_{\theta}(\tau)}[r(\tau)]\\
&=\nabla_{\theta}\int p_\theta(\tau)\cdot r(\tau)\cdot d\tau\\
&=\int(\nabla_{\theta}p_\theta(\tau))\cdot r(\tau)\cdot d\tau\\
\end{align}
$$
We can use the identity $p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)=p_\theta(\tau)\frac{\nabla_{\theta}p_\theta}{p_\theta}=\nabla_{\theta}p_\theta(\tau)$ to simplify the above equation as follows:
$$
\begin{align}
\nabla_{\theta}J(\theta)
&=\int(p_\theta(\tau)\cdot\nabla_{\theta}\log p_\theta(\tau))\cdot r(\tau)\cdot d\tau\\
&=\mathbb{E}_{p_{\theta}(\tau)}\left[\nabla_{\theta}\log p_\theta(\tau)\cdot r(\tau)\right]...[1]
\end{align}
$$
We already know that:
$$
\begin{align}
p_\theta(\tau)
&=p_\theta(s_{0},a_{0},s_{1},a_{1},...,s_{T},a_{T},s_{T+1})\\
&=\rho(s_{0})\prod\limits_{t=0}^{T}\pi_{\theta}(a_{t}|s_{t})\cdot p(s_{t+1}|s_{t},a_{t})\\
\implies\log p_\theta(\tau)
&=\log\left(\rho(s_{0})\prod\limits_{t=0}^{T}\pi_{\theta}(a_{t}|s_{t})\cdot p(s_{t+1}|s_{t},a_{t})\right)\\
&=\underbrace{\log\rho(s_{0})}_{\text{independent of }\theta}+\sum\limits_{t=0}^{T}\log\pi_{\theta}(a_{t}|s_{t})+\underbrace{\log p(s_{t+1}|s_{t},a_{t})}_{\text{independent of }\theta}\\
\implies\nabla_\theta\log p_\theta(\tau)&=\nabla_\theta\sum\limits_{t=0}^{T}\log\pi_{\theta}(a_{t}|s_{t})=\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})
\end{align}
$$
Thus equation $[1]$ becomes
$$
\begin{align}
\nabla_{\theta}J(\theta)
&=\mathbb{E}_{p_{\theta}(\tau)}\left[\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}|s_{t})\cdot \sum\limits_{t=0}^{T}r(s_{t},a_{t})\right]\\
&\approx\underbrace{\frac{1}{N}\sum\limits_{i=1}^{N}}_{\text{Generate Samples}}\left(\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\right)\underbrace{\left(\sum\limits_{t=0}^{T}r(s_{t}^{i},a_{t}^{i})\right)}_{\text{Estimate Rewards}}\\
\end{align}
$$
Looking back at the anatomy of RL, The first part of the equation corresponds to the trajectory collection phase and the last part corresponds to the value estimation part. The policy update part is essentially:
$$
\theta\gets\theta+\alpha\nabla_{\theta}J(\theta)
$$
where $\alpha$ is the learning rate.

----

### REINFORCE Algorithm

Goal: Obtain the optimal policy $\pi^*$<br>
While $|V^{\pi_{\theta}}-V^{\pi^*}|<\epsilon$:<br>
    1. Collect $N$ trajectories $\{\tau^{i}\}_{i=1}^{N}$ by executing policy $\pi_\theta(a_{t}|s_{t})$ in the environment<br>
    2. Estimate $\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum\limits_{i=1}^{N}\left(\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\right)\left(\sum\limits_{t=0}^{T}r(s_{t}^{i},a_{t}^{i})\right)$<br>
    3. Update $\theta\gets\theta+\alpha\nabla_{\theta}J(\theta)$, i.e. take step towards the positive gradient

**Comments**:

1. Recall that $\nabla_{\theta}J_{MLE}(\theta)\approx\frac{1}{N}\sum\limits_{i=1}^{N}\left(\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\right)$, i.e. finding the maximum likelihood estimate (MLE) using ERM in behavior cloning (BC). Comparing with the PG algorithm, we can see that there is an extra term $r(\tau)$ in PG which is "weighing" the action log likelihoods given states. Thus PG can be thought of as a "trial-and-error" algorithm where the actions which lead to more rewarding trajectories are given more importance during gradient update by reinforcing the behavior to take more rewarding actions conditional on states.
2. PG algorithms have very high variance. For instance, we can take a running example: consider the reward system +1 for each time-step (fixed horizon length of 100), and only if the agent reaches the goal, then there is an extra reward of +1. Now consider an alternate situation where at each step there is 0 reward instead for survival. PG algorithms work better in the latter situation, since we can interpret it as giving 100 weightage to all bad trajectories and 101 to good trajectories.
3. To reduce variance, we can simply introduce the concept of baselines:$$\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum\limits_{i=1}^{N}\left(\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\right)\left(\sum\limits_{t=0}^{T}r(s_{t}^{i},a_{t}^{i})-b\right)$$which is essentially just subtracting a baseline value to give more importance to good trajectories, i.e. say if we subtract $b=100$, then good trajectories get 1 weightage and everything else gets 0 weightage. We can simply set $b=\frac{1}{N}\sum\limits_{i=1}^{N}r(\tau)$ since:$$\mathbb{E}[\nabla_\theta\log p_{\theta}(\tau)b]=\int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)b\cdot d\tau=b\int\nabla_{\theta}p_\theta(\tau)d\tau=b\nabla_\theta\int p_\theta(\tau)d\tau=b\nabla_\theta1=0$$which essentially means that subtracting a baseline is *unbiased* in expectation. The baseline can also be learned during training (using back-propagation).
4. One other way to reduce variance is by noticing the causality during calculating the rewards, i.e. a policy at time $t'$ cannot affect the rewards obtained at time $t$ where $t<t'$. Thus gradient calculation equation can be modified as:<br><center>$$\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\underbrace{\left(\sum\limits_{t'=t}^{T}r(s_{t'}^{i},a_{t'}^{i})\right)}_{\hat{Q}(s_{t'}^{i},a_{t'}^{i})}$$</center><br>
where $\hat{Q}(s_{t'}^{i},a_{t'}^{i})$ is essentially the MC estimate of the state-action value function. Now notice that in this situation, we cannot introduce a constant baseline $b$, since for each state $s_{t}$ the baseline will be different. For our running example, say at $s_0$, the baseline value should be 100, but at $s_{50}$ the baseline value should be 50, and finally at $s_{100}$ the baseline value should be 0. Therefore there is a need to learn state dependent baselines:
$$
\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}( a_{t}^{i}|s_{t}^{i})\left(\hat{Q}(s_{t'}^{i},a_{t'}^{i})-b(s_{t})\right)
$$

#### ANOVA (for point 3)

From $[1]$ and including baselines we get $\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\nabla_{\theta}\log p_\theta(\tau)(r(\tau)-b)\right]$. Now the variance is:
$$
\begin{align}
Var
&=\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[(\nabla_{\theta}\log p_\theta(\tau)(r(\tau)-b))^{2}\right]-\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\nabla_{\theta}\log p_\theta(\tau)(r(\tau)-b)\right]^{2}\\
&=\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[(\nabla_{\theta}\log p_\theta(\tau)(r(\tau)-b))^{2}\right]-\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\nabla_{\theta}\log p_\theta(\tau)(r(\tau))\right]^{2}
\end{align}
$$
since baselines are unbiased in expectation
$$
\begin{align}
\frac{dVar}{db}&=\frac{d}{db}\mathbb{E}[g(\tau)^{2}(r(\tau)-b)^{2}]+0\\
&=\frac{d}{db}(\mathbb{E}[g(\tau)^{2}r(\tau)^{2}]-2\mathbb{E}[g(\tau)^{2}r(\tau)b+b^{2}\mathbb{E}[g(\tau)^{2}]])\\
&=0-2\mathbb{E}[g(\tau)^{2}r(\tau)]+2b\mathbb{E}[g(\tau)^{2}]=0\\
\implies b&=\frac{\mathbb{E}[g(\tau)^{2}r(\tau)]}{\mathbb{E}[g(\tau)^{2}]}
\end{align}
$$
which is essentially just the expected rewards, weighted by gradient magnitudes

----

### Off-policy Policy Gradients

The problem with the above algorithm is that it is completely on-policy, i.e. once the update happens, all the collected trajectories are discarded and new trajectories must to collected to perform the next policy gradient update. It would be more efficient if we did not have to discard all the previously collected trajectories since it took effort to collect the same. Hence, we look at ways to adapt the PG algorithm for the off-policy case.

We start with Importance Sampling, say we have an expectation of some function of $x$ w.r.t. distribution $p(x)$:
$$
\begin{align}
\mathbb{E}_{x\sim p(x)}[f(x)]
&=\int\limits_{x}p(x)\cdot f(x)\cdot dx\\
&=\int\limits_{x}\frac{q(x)}{q(x)}\cdot p(x)\cdot f(x)\cdot dx\\
&=\int\limits_{x}q(x)\cdot \frac{p(x)}{q(x)}\cdot f(x)\cdot dx\\
&=\mathbb{E}_{x\sim q(x)}\left[\frac{p(x)}{q(x)}\cdot f(x)\right]
\end{align}
$$
We can choose any distribution $q(x)$ (it's under our control) as long as the support of the distribution is same as that of the $p(x)$ and $q(x)>0$ for all such $x$ where $p(x)>0$.

Assuming we have trajectories from some other distribution $\bar{p}$ instead of $p_{\theta}$, we can modify the PG algorithm as follows:
$$
\begin{align}
J(\theta)&=\mathbb{E}_{\tau\sim\bar{p}(\tau)}\left[\frac{p_\theta(\tau)}{\bar{p}(\tau)}\cdot r(\tau)\right]
\end{align}
$$
Now,
$$
\begin{align}
\frac{p_\theta(\tau)}{\bar{p}(\tau)}=\frac{\rho(s_0)\prod\limits_{t=0}^{T}\pi_{\theta}(a_{t}|s_{t})\cdot p(s_{t+1}|a_{t+1})}{\rho(s_0)\prod\limits_{t=0}^{T}\bar{\pi}_{\theta}(a_{t}|s_{t})\cdot p(s_{t+1}|a_{t+1})}=\frac{\prod\limits_{t=0}^{T}\pi_{\theta}(a_{t}|s_{t})}{\prod\limits_{t=0}^{T}\bar{\pi}_{\theta}(a_{t}|s_{t})}...[2]
\end{align}
$$
Let's assume that our current policy (after the $t^\text{th}$) policy update is parameterized by $\theta^t$ (that is used for collecting new transitions) whereas the policies that collected the previous samples (which are stored in a buffer) can be parameterized by $\theta^{t-k}$. In doing so we are overloading some notations, therefore we will assume that the previously collected dataset was collected by some policy $\pi^{\theta}$ and the current policy is $\pi^{\theta'}$ (note that by convention objects in the past are denoted by a letter and for corresponding future objects a prime is added, e.g. if current state is $s$, next state is often denoted as $s'$). Therefore we  would like to update $\pi_{\theta'}$ using samples from $\pi_{\theta}:$
$$
\begin{align}
J(\theta') &= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}\cdot r(\tau)\right]\\
\nabla_{theta'}J(\theta') &= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\frac{\nabla_{\theta'}p_{\theta'}(\tau)}{p_{\theta}(\tau)}\cdot r(\tau)\right]\\
&= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}\cdot \nabla_{\theta'}\log p_{\theta'}(\tau) \cdot r(\tau)\right]\\
\end{align}
$$
Replacing with $[2]$ and expanding the remaining terms we get:
$$
\begin{align}
\nabla_{\theta'}J(\theta')
&= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\left(\prod\limits_{t=0}^{T}\frac{\pi_{\theta'}(a_{t}|s_{t})}{\pi_{\theta}(a_{t}|s_{t})}\right)\left(\sum\limits_{t=0}^{T}\nabla_{\theta'}\log\pi_{\theta'}( a_{t}|s_{t})\right)\left(\sum\limits_{t=0}^{T}r(s_{t},a_{t})\right)\right]\\
\end{align}
$$
Now in order to decrease the variance in estimates, we can modify the above by introducing the notion of causality. We have already seen how the current reward is independent of the trajectory history. In this case notice further that the importance weights also have a notion of causality involved, i.e. current importance weights are not affected by future actions:
$$
\begin{align}
\nabla_{\theta'}J(\theta')
&= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\sum\limits_{t=0}^{T}\nabla_{\theta'}\log\pi_{\theta'}( a_{t}|s_{t})\left(\prod\limits_{t'=0}^{t}\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}\right)\left(\sum\limits_{t'=t}^{T}r(s_{t'},a_{t'})\left(\prod\limits_{t''=t}^{t'}\frac{\pi_{\theta}(a_{t''}|s_{t''})}{\pi_{\theta}(a_{t''}|s_{t''})}\right)\right)\right]\\
\end{align}
$$
If we ignore the final term, we can recover an algorithm called policy iteration (<span style="color:red">cite</span>) which has some nice convergence guarantees. However even in the reduced expression
$$
\begin{align}
\nabla_{\theta'}J(\theta')
&= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\sum\limits_{t=0}^{T}\nabla_{\theta'}\log\pi_{\theta'}( a_{t}|s_{t})\left(\prod\limits_{t'=0}^{t}\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}\right)\left(\sum\limits_{t'=t}^{T}r(s_{t'},a_{t'})\right)\right]
\end{align}
$$
notice that the second term is actually exponentially hard to compute and doesn't work very well for long horizon tasks. Hence we can look into something like a "first-order" approximation of the same, which is essentially just modifying the on-policy algorithm by adding importance weights:
$$
\begin{align}
\nabla_{\theta}J_{\text{On Policy}}(\theta)&\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=1}^{T}\nabla_{\theta}\log\pi_\theta(a_{t}^{i}|s_{t}^{i})\hat{Q}_{t}^{i}\\
\nabla_{\theta'}J_{\text{Off Policy}}(\theta')&\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=1}^{T}\frac{\pi_{\theta'}(s_{t}^{i},a_{t}^{i})}{\pi_{\theta}(s_{t}^{i},a_{t}^{i})}\nabla_{\theta'}\log\pi_{\theta'}(a_{t}^{i}|s_{t}^{i})\hat{Q}_{t}^{i}\\
&\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=1}^{T}\frac{\pi_{\theta'}(s_{t}^{i})\pi_{\theta'}(a_{t}^{i}|s_{t}^{i})}{\pi_{\theta}(s_{t}^{i})\pi_{\theta}(a_{t}^{i}|s_{t}^{i})}\nabla_{\theta'}\log\pi_{\theta'}(a_{t}^{i}|s_{t}^{i})\hat{Q}_{t}^{i}\\
&\approx\frac{1}{N}\sum\limits_{i=1}^{N}\sum\limits_{t=1}^{T}\frac{\pi_{\theta'}(a_{t}^{i}|s_{t}^{i})}{\pi_{\theta}(a_{t}^{i}|s_{t}^{i})}\nabla_{\theta'}\log\pi_{\theta'}(a_{t}^{i}|s_{t}^{i})\hat{Q}_{t}^{i}\\
\end{align}
$$
It is reasonable to assume that the marginals are similar (<span style="color:red">cite</span>) and can be ignored.

----

### Straight Through Gumble Softmax (STGS) for PG Algorithms
