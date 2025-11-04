# Policy-based methods

---

## Policy functions

Policy-based methods learn a **parameterised policy function** $\pi_\theta(S,X)$ with parameters $\theta$ to determine probabilities of taking a decision $X$ when in state $S$.

> [!NOTE]
> - After training, the decision with the highest probability $\pi_\theta(S,X)$ is taken.
> - During training, the decision is chosen by sampling using probabilities $\pi_\theta(S,X)$.

---

## Policy learning with neural networks

We can use a neural network with parameters $\theta$ to learn $\pi_\theta(S,X)$ to be used to 

- create a trajectory `$(S_0, X_1, r_1, S_1, \ldots, S_T)$`  and
- maximise `$J(\theta) = \displaystyle\sum_{t=1}^T r_t$`

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

- `$\mu_{\pi_\theta}(S)$` is the probability of entering state $S$ under policy $\pi_\theta$
- `$Q_{\pi_\theta}(S,X)$` is the action-value function for policy $\pi_\theta$
- `$\nabla_{\!\theta} \ \pi_\theta(S,X) \Big)$` is the gradient of policy $\pi_\theta$ w.r.t. $\theta$

> [!IMPORTANT]
> This requires knowing $\mu_{\pi_\theta}(S)$ and $Q_{\pi_\theta}(S,X)$ for all states and decisions. In  practice, we estimate these using observed trajectories.

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\nabla_{\!\theta} \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> $\Big)$

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> $\cdot$ `$\displaystyle\frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`<!-- .element: data-id="del-J-b" --> $\Big)$

> [!NOTE]
> Multiplication with $\frac{\pi_\theta(S,X)}{\pi_\theta(S,X)}$.

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

<span data-id="del-J-final">
`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> $\cdot$ `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> $\Big)$
</span>

> [!NOTE]
> We have `$\nabla_{\!\theta} \ln \ \pi_\theta(S,X) = \frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`.

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)\Big)$$`<!-- .element: data-id="del-J-final" --> 

which can be estimated using a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$ and replacing probabilities by 1 or 0 based on the actual observation:

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> $1 \cdot$ `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot  1 \cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> $\cdot$ `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> $\Big)$

which can be estimated by using a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$ and replacing probabilities by 1 or 0 based on the actual observation:

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

## Trajectory-based gradient ascent

Averaged over multiple trajectories of different episodes, 

`$$\sum_{t=1}^{T}  \Big( \sum_{k=t}^T r_k \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

is assumed to proportionally approximate `$\nabla_{\!\theta}\ J(\theta)$`.

---

## REINFORCE algorithm

> [!TODO]

<!--
```
Initialize:
  - Policy network π_θ with random weights θ
  - Environment

For each episode:
  1. Collect trajectory: (S₀, X₁, r₁, S₁, ..., S_T)
  
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


