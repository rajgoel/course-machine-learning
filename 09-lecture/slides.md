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

> [!WARNING]
> `$\nabla_{\!\theta} \ \pi_\theta(S,X)$` can be extremely small and may cause numerical issues (underflow).

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> $\cdot$ `$\displaystyle\frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`<!-- .element: data-id="del-J-b" --> $\Big)$

> [!NOTE]
> Multiplication with $\frac{\pi_\theta(S,X)}{\pi_\theta(S,X)}$.

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$`<!-- .element: data-id="del-J-dot1" --> `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> `$\cdot$`<!-- .element: data-id="del-J-dot2" --> `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

> [!NOTE]
> We have `$\nabla_{\!\theta} \ln \ \pi_\theta(S,X) = \frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`. 

> [!NOTE]
> Computing gradients of log-probabilities avoids numerical underflow issues. Modern deep learning frameworks have auto-differentiation, allowing us to determine these values easily.
<!-- .element: class="fragment" -->


---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$`<!-- .element: data-id="del-J-dot1" --> `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> `$\cdot$`<!-- .element: data-id="del-J-dot2" --> `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

which can be estimated by using a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$ and replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> $1 \cdot$ `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot  1 \cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$`<!-- .element: data-id="del-J-dot1" --> `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> `$\cdot$`<!-- .element: data-id="del-J-dot2" --> `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

which can be estimated by using a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$ and replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

## Trajectory-based policy updates

The main idea of policy-based methods is to apply gradient ascent using 

`$$\sum_{t=1}^{T}  \Big( \sum_{k=t}^T r_k \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

as an estimate of the true gradient `$\nabla_{\!\theta}\ J(\theta)$`, assuming that with a sufficiently large number of trajectories, the gradient steps converge toward the optimum.

<!--
> [!NOTE]
> With **Flux.jl** we can compute `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t)$` using the final layer `activations` the network parameters `θ`, and the chosen `action` by determining `∇_θ = Flux.gradient( () -> Flux.logsoftmax(activations)[action], θ )`.
-->

---

## REINFORCE algorithm

```
Initialize:
  - Policy network π_θ with random weights θ
  - Environment

For each episode:
  1. Collect trajectory: (S₀, X₁, r₁, S₁, ..., S_T)
  
  2. For each time step t = T down to 1:
     - Compute cumulative reward: Rₜ = rₜ + R_{t+1} (with R_{T+1} = 0)
     - Compute gradient: ∇ₜ = ∇_θ log π_θ(Sₜ₋₁,Xₜ) × Rₜ
  
  3. Update policy: θ ← θ + α × (1/T) × Σ(t=1 to T) ∇ₜ
```

> [!NOTE]
> - REINFORCE learns from complete episodes (Monte Carlo approach)
> - Higher rewards increase probability of actions that led to them
> - Can have high variance due to using full trajectory returns

---

> [!WARNING]
> REINFORCE is essentially outdated for practical usage and training can be unstable and slow.

===

## Actor-critic methods

---

> [!TODO]
> PPO algorithm


