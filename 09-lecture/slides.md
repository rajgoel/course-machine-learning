# Policy-based methods

---

## Policy functions

Policy-based methods learn a **policy function** $\pi(S,X)$ to determine probabilities of taking a decision $X$ when in state $S$.

> [!NOTE]
> - During training, policy-based methods make decisions by sampling the probabilities $\pi(S,X)$.
> - After training, policy-based methods make decisions by selecting the decision with the highest probability $\pi(S,X)$.

---

## Policy learning with neural networks

We can use a neural network with parameters $\theta$ to learn a **policy function** $\pi_\theta(S,X)$.

The goal is to find a policy that can be used to 

- create a trajectory `$(S_0, X_1, r_1, S_1, X_1, r_1, \ldots, S_T)$`  and
- maximise `$J(\theta) = \sum_{t=1}^T r_t$`

---

## Gradient ascent

In order to 

`$$\textrm{maximise } J(\theta)$$`

we can use gradient ascent to update our parameters by

`$$\theta \leftarrow \theta + \alpha \cdot \nabla_{\!\theta}\ J(\theta)$$`


---

## Policy gradient theorem

According to the [policy gradient theorem](http://incompleteideas.net/book/RLbook2020.pdf#page=346), we have 

`$$\nabla_{\!\theta}\ J(\theta) \propto \sum_S \Big( \mu_{\pi_\theta}(S) \cdot \sum_X Q_{\pi_\theta}(S,X) \cdot \nabla_{\!\theta} \ \pi_\theta(S,X) \Big)$$`

where

- $\mu_{\pi_\theta}(S)$ is the probability of entering state $S$ under policy $\pi_\theta$
- $Q_{\pi_\theta}(S,X)$ is the action-value function for policy $\pi_\theta$.

> [!IMPORTANT]
> This requires knowing $\mu_{\pi_\theta}(S)$ and $Q_{\pi_\theta}(S,X)$ for all states and decisions. In  practice, we estimate these using observed trajectories.

---

<!-- .slide: data-auto-animate="true" -->

## Trajectory-based gradient ascent

For a given trajectory $(S_0, X_1, r_1, S_1, X_1, r_1, \ldots, S_T)$ of an episode, we approximate

`$$\nabla_{\!\theta}\ J(\theta) \propto \sum_S \Big( \mu_{\pi_\theta}(S) \cdot \sum_X Q_{\pi_\theta}(S,X) \cdot \nabla_{\!\theta} \ \pi_\theta(S,X) \Big)$$`

by

`$$\sum_{t=1}^{T}  \Big( 1 \cdot \underbrace{\sum_{k=t}^T r_k}_{\approx Q_{\pi_\theta}(S_{t-1},X_t)} \cdot   \nabla_{\!\theta}\ \pi_\theta(S_{t-1},X_t) \Big)$$`

> [!NOTE]
> For the given trajectory, we use $\mu_{\pi_\theta}(S)=1$ for all states in the trajectory and $\mu_{\pi_\theta}(S)=0$ for all other states.


---

<!-- .slide: data-auto-animate="true" -->

<!--

## Policy gradient estimation (sample-based)

From a trajectory $(S_0, X_0, r_0, S_1, X_1, r_1, \ldots, S_T)$ of an episode, we can estimate policy gradients as:

$$\nabla_\theta J_t(\theta) = \nabla_\theta \log \pi_\theta(X_t,S_t) \cdot  \underbrace{\sum_{k=t}^T r_k}_{\textrm{value of} X_t} \textrm{ for all } 0 \leq t < T$$

We can use the averaged gradient to update the policy parameters:

$$\theta \leftarrow \theta + \alpha \cdot  \frac{1}{T} \sum_{t=0}^{T-1}\nabla_\theta J_t(\theta)$$

> [!NOTE]
> The log comes from the **log-derivative trick**: 
> $$\nabla_\theta \pi_\theta(S,X) = \pi_\theta(S,X) \cdot \frac{\nabla_\theta \pi_\theta(S,X)}{\pi_\theta(S,X)} = \pi_\theta(S,X) \cdot \nabla_\theta \log \pi_\theta(S,X)$$
> The sample-based average in the update, approximates multiplication with $\pi_\theta(S,X)$.

---

Objective:
  $$J(\theta) = \sum_{t=0}^T r_t$$

  This is the total reward from one episode following policy $\pi_\theta$.

  Gradient ascent:
  To maximize J(θ), we update:
  $$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$$

  The gradient:
  $$\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \nabla_\theta
  \pi_\theta(X_t, S_t) \cdot \left(\sum_{k=t}^T r_k\right)$$

  Log-derivative trick:
  Since $\nabla_\theta \pi_\theta(S,X) = \pi_\theta(S,X) \cdot
  \nabla_\theta \log \pi_\theta(S,X)$:

  $$\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \pi_\theta(X_t, S_t) \cdot
  \nabla_\theta \log \pi_\theta(X_t, S_t) \cdot \left(\sum_{k=t}^T
  r_k\right)$$

  When we sample actions according to $\pi_\theta$, the $\pi_\theta(X_t,
  S_t)$ weighting is automatically included.

  Final update:
  $$\theta \leftarrow \theta + \alpha \cdot \sum_{t=0}^{T-1} \nabla_\theta
  \log \pi_\theta(X_t, S_t) \cdot \left(\sum_{k=t}^T r_k\right)$$

---
-->

## REINFORCE algorithm

> [!TODO]

<!--
```
Initialize:
  - Policy network π_θ with random weights θ
  - Environment

For each episode:
  1. Collect trajectory: (S₀, X₀, r₀, S₁, X₁, r₁, ..., S_T)
  
  2. For each time step t = T-1 down to 0:
     - Compute cumulative reward: R_t = r_t + R_{t+1} (with R_T = 0)
     - Compute gradient: ∇_θ J_t(θ) = ∇_θ log π_θ(X_t|S_t) × R_t
  
  3. Update policy: θ ← θ + α × (1/T) × Σ(t=0 to T-1) ∇_θ J_t(θ)
```

> [!NOTE]
> - REINFORCE learns from complete episodes (Monte Carlo approach)
> - Higher rewards increase probability of actions that led to them
> - Can have high variance due to using full trajectory returns
-->

---

> [!WARNING]
> REINFORCE is essentially outdated for practical usage and training can be unstable and slow.

===

## Actor-critic methods

---

> [!TODO]
> PPO algorithm


