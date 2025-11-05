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
- `$\nabla_{\!\theta} \ \pi_\theta(S,X)$` is the gradient of policy $\pi_\theta$ w.r.t. $\theta$

---

<!-- .slide: data-auto-animate="true" -->

`$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> $\cdot$ `$\nabla_{\!\theta} \ \pi_\theta(S,X)$`<!-- .element: data-id="del-J-b" --> `$\Big)$`

> [!WARNING]
> In general, we cannot compute this sum as we can neither determine all possible states $S$, their probabilities $\mu_{\pi_\theta}(S)$, nor do we know $Q_{\pi_\theta}(S,X)$ for all possible actions $X$ that could be taken in a state $S$.

---

<!-- .slide: data-auto-animate="true" -->

`$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$` `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> $\cdot$ `$\displaystyle\frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

> [!NOTE]
> Multiplication with $\frac{\pi_\theta(S,X)}{\pi_\theta(S,X)}$.

---

<!-- .slide: data-auto-animate="true" -->

`$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$`<!-- .element: data-id="del-J-dot1" --> `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> `$\cdot$`<!-- .element: data-id="del-J-dot2" --> `$ \class{highlight}{\nabla_{\!\theta} \ln \ \pi_\theta(S,X)}$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

> [!NOTE]
> We have `$\nabla_{\!\theta} \ln \ \pi_\theta(S,X) = \frac{\nabla_{\!\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}$`. 

> [!TIP]
> Computing gradients of log-probabilities  `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$` can easily be done using **auto-differentiation** of modern deep learning frameworks.


---

<!-- .slide: data-auto-animate="true" -->

Given a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$, we can estimate

`$$\displaystyle\sum_S \Big( \class{highlight}{\mu_{\pi_\theta}(S)} \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \class{highlight}{\pi_\theta(S,X)} \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="del-J-final" -->

using the observed rewards and by replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> $1 \cdot$ `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot  1 \cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

Given a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$, we can estimate

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="del-J-final" -->

using the observed rewards and by replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot$ `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

## Trajectory-based policy updates

The main idea of policy-based methods is to apply gradient ascent using 

`$$\sum_{t=1}^{T}  \Big( \sum_{k=t}^T r_k \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

as a proportional estimate of the true gradient `$\nabla_{\!\theta}\ J(\theta)$`

> [!NOTE]
> The hope is that gradient estimates over many trajectories will lead to policy improvement similar to using true gradients.


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
  
  3. Update policy: θ ← θ + α × Σ(t=1 to T) ∇ₜ
```

> [!NOTE]
> - REINFORCE learns from complete episodes (Monte Carlo approach)
> - Higher rewards increase probability of actions that led to them
> - Can have high variance due to using full trajectory returns

---

> [!WARNING]
> REINFORCE is essentially outdated for practical usage and training can be unstable and slow.

===

# Actor-critic methods

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \nabla_{\!\theta}\ 1$`<!-- .element: data-id="zero-1" --> 

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \nabla_{\!\theta}\ 1$`<!-- .element: data-id="zero-1" --> 
`$= \nabla_{\!\theta}\ \displaystyle\sum_X \pi_\theta(S,X)$`<!-- .element: data-id="zero-2" class="fragment" -->

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \nabla_{\!\theta}\ \displaystyle\sum_X \pi_\theta(S,X)$`<!-- .element: data-id="zero-2" -->
`$= \displaystyle\sum_X \nabla_{\!\theta}\  \pi_\theta(S,X)$`<!-- .element: data-id="zero-3" class="fragment" -->

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \displaystyle\sum_X \nabla_{\!\theta}\  \pi_\theta(S,X)$`<!-- .element: data-id="zero-3" -->
`$= B(S)  \cdot \displaystyle\sum_X \nabla_{\!\theta}\ \pi_\theta(S,X)$`<!-- .element: data-id="zero-4" data-fragment-index="0" class="fragment" -->

for any given baseline $B(S)$. <!-- .element: data-fragment-index="0" class="fragment" -->

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= B(S)  \cdot \displaystyle\sum_X \nabla_{\!\theta}\ \pi_\theta(S,X)$`<!-- .element: data-id="zero-4" -->
`$= \displaystyle\sum_X B(S)  \cdot \nabla_{\!\theta}\ \pi_\theta(S,X)$`<!-- .element: data-id="zero-5" class="fragment" -->

for any given baseline $B(S)$.

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="pgt" -->

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \displaystyle\sum_X B(S)  \cdot \nabla_{\!\theta}\ \pi_\theta(S,X)$`<!-- .element: data-id="zero-5" -->
`$= \displaystyle\sum_X B(S)  \cdot \pi_\theta(S,X)  \cdot \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$`<!-- .element: data-id="zero-6" class="fragment" -->

for any given baseline $B(S)$.

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \class{highlight}{\displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X)} \Big)$$`<!-- .element: data-id="pgt" -->

Moreover, we have

`$0$`<!-- .element: data-id="zero" -->
`$= \class{highlight}{\displaystyle\sum_X B(S)  \cdot \pi_\theta(S,X)  \cdot \nabla_{\!\theta} \ln \ \pi_\theta(S,X)}$`<!-- .element: data-id="zero-6" -->

and `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 
`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \class{highlight}{\big( Q_{\pi_\theta}(S,X) - B(S) \big)} \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

for any given baseline $B(S)$.

---

## Value-function baseline

The main idea of actor-critic methods is to use a learnable **state-value function** `$V_{\theta_\text{critic}}(S)$` with parameters `$\theta_\text{critic}$` as a baseline.

Then, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 
`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \big( Q_{\pi_\theta}(S,X) \class{highlight}{- V_{\theta_\text{critic}}(S)} \big) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`

<span class="fragment">
which can be estimated by 

`$$\sum_{t=1}^{T}  \Big( \big( \sum_{k=t}^T r_k \class{highlight}{- V_{\theta_\text{critic}}(S_{t-1})} \big) \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

</span>

---

`$$\sum_{t=1}^{T}  \Big( \big( \class{highlight}{\sum_{k=t}^T r_k} - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

can be estimated by

`$$\sum_{t=1}^{T}  \Big( \big( \class{highlight}{r_t + V_{\theta_\text{critic}}(S_t)} - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

> [!NOTE]
> Every term can now be computed directly when the reward and new state become known  after taking an action. 

---


As we do not need to wait for termination of an episode we can do our gradient updates using 

`$$\big( \class{highlight}{r_t + V_{\theta_\text{critic}}(S_t)} - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t)$$`

---

## TD-error and policy gradient

Inspired by the Bellmann equation, we can use a **discount factor** $\gamma \in [0,1]$ to define the **temporal difference (TD) error**

`$$\delta_t = r_t + \class{highlight}{\gamma} \cdot V_{\theta_\text{critic}}(S_t) - V_{\theta_\text{critic}}(S_{t-1})$$`

and

`$$\delta_t \cdot  \nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t)$$`

as an estimation of the policy gradient. 

---

## TD-error and loss of the critic

To learn the **state-value function** `$V_{\theta_\text{critic}}(S)$` we minimise the loss of the critic

`$$\mathscr{L}(\theta_\text{critic}) = \Big( \underbrace{\big( r_t + \gamma \cdot V_{\theta_\text{critic}}(S_t) \big)}_{\textrm{Bellman target}} - \underbrace{V_{\theta_\text{critic}}(S_{t-1})}_{\textrm{Prediction}}  \Big)^2$$`

The **temporal difference (TD) error** appears in the gradient of this loss:

`$$\frac{\partial \mathscr{L}}{\partial \theta_\text{critic}} = 2 \delta_t \cdot \nabla_{\theta_\text{critic}} V_{\theta_\text{critic}}(S_{t-1})$$`

where `$\delta_t = r_t + \gamma \cdot V_{\theta_\text{critic}}(S_t) - V_{\theta_\text{critic}}(S_{t-1})$`

<!--
The gradient of this loss is the **temporal difference (TD) error** 

`$$\delta_t = r_t + \gamma \cdot V_{\theta_\text{critic}}(S_t) - V_{\theta_\text{critic}}(S_{t-1})$$`
-->


---

## Actor-critic algorithm

```
Initialize:
  - Policy network π_θ with random weights θ
  - Critic network V_θ_critic with random weights θ_critic
  - Environment

For each episode:
  1. Initialize state S₀
  
  2. For each time step t = 1 to T:
     a. Sample action Xₜ ~ π_θ(Sₜ₋₁, ·)
     b. Execute action Xₜ, observe reward rₜ and next state Sₜ
     c. Compute TD error: δₜ = rₜ + γ V_θ_critic(Sₜ) - V_θ_critic(Sₜ₋₁)
     d. Update policy: θ ← θ + α × δₜ × ∇_θ log π_θ(Sₜ₋₁,Xₜ)
     e. Update critic: θ_critic ← θ_critic + α_critic × δₜ × ∇_θ_critic V_θ_critic(Sₜ₋₁)
```

