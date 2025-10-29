# Policy-based methods

---

## Policy functions

Policy-based methods learn policy functions to determine probabilities of taking a decision

- **Policy function** $\pi(X,S)$: probability of taking action $x$ when in state $S$

> [!NOTE]
> During training, policy-based methods make decisions by sampling the probabilities $\pi(X,S)$.
> After training, policy-based methods make decisions by selecting the decision with the highest probability $\pi(X,S)$.

---

## Policy learning with neural networks

We can use a neural network with parameters $\theta$ to learn a **policy function** $\pi_\theta(X,S)$ that estimates the probability of taking decision $X$ in state $S$.

The goal is to find a policy that maximise the cumulative rewards when always selecting the decision with the highest probability $\hat X =
  \arg\max_X \pi_\theta(X,S)$.

---

## Policy gradient estimation (sample-based)

From a trajectory $(S_0, X_0, r_0, S_1, X_1, r_1, \ldots, S_T)$ of an episode, we can estimate policy gradients as:

$$\nabla_\theta J_t(\theta) = \nabla_\theta \log \pi_\theta(X_t|S_t) \cdot  \underbrace{\sum_{k=t}^T r_k}_{\textrm{value of} X_t} \textrm{ for all } 0 \leq t < T$$

We can use the averaged gradient to update the policy parameters:

$$\theta \leftarrow \theta + \alpha \cdot  \frac{1}{T} \sum_{t=0}^{T-1}\nabla_\theta J_t(\theta)$$

> [!NOTE]
> Observed transitions allow us to omit computing expected values.

---

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

===

# Actor-critic methods

---

> [!TODO]
> PPO algorithm


