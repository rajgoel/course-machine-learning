# Policy-based methods

---

## Policy functions

Policy-based methods learn a **parameterised policy function** $\pi_\theta(S,X)$ with parameters $\theta$ to determine probabilities of taking a decision $X$ when in state $S$.

> [!NOTE]
> - During training, the decision is chosen by sampling using probabilities $\pi_\theta(S,X)$.
> - After training, the decision with the highest probability $\pi_\theta(S,X)$ is taken.

---

## Policy learning with neural networks

We can use a neural network with parameters $\theta$ to learn $\pi_\theta(S,X)$ for 
creating trajectories 
`$$\tau = (S_0, X_1, r_1, S_1, \ldots, S_T)$$` 
 and maximising 
`$$J(\theta) = \mathbb{E}_\tau \left[R(\tau)\right]$$`
where $R(\tau)$ represents the total rewards of trajectory $\tau$.

---

## Gradient ascent

In order to 

`$$\textrm{maximise } J(\theta)$$`

we can use gradient ascent to update our parameters by

`$$\theta \leftarrow \theta + \eta \cdot \nabla_{\!\theta}\ J(\theta)$$`


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

`$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X)$`<!-- .element: data-id="del-J-a" --> `$\cdot$`<!-- .element: data-id="del-J-dot1" --> `$\pi_\theta(S,X)$`<!-- .element: data-id="del-J-pi" --> `$\cdot$`<!-- .element: data-id="del-J-dot2" --> `$ \underbrace{\class{highlight}{\nabla_{\!\theta} \ln \ \pi_\theta(S,X)}}_{= \frac{\nabla_{\tiny\theta} \pi_\theta(S,X)}{\pi_\theta(S,X)}}$`<!-- .element: data-id="del-J-b" --> `$\Big)$`<!-- .element: data-id="del-J-)" -->

> [!TIP]
> Computing gradients of log-probabilities  `$ \nabla_{\!\theta} \ln \ \pi_\theta(S,X)$` can easily be done using **auto-differentiation** of modern deep learning frameworks.


---

<!-- .slide: data-auto-animate="true" -->

Given a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$, we can estimate

`$$\displaystyle\sum_S \Big( \class{highlight}{\mu_{\pi_\theta}(S)} \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \class{highlight}{\pi_\theta(S,X)} \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="del-J-final" -->

using the observed rewards and by replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> $1 \cdot$ `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot  1 \cdot$ `$\nabla_{\!\theta}$`<!-- .element: data-id="nabla" --> `$\ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

Given a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$, we can estimate

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="del-J-final" -->

using the observed rewards and by replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot$ `$\nabla_{\!\theta}$`<!-- .element: data-id="nabla" --> `$\ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

<!-- .slide: data-auto-animate="true" -->

Given a trajectory of observations $(S_0, X_1, r_1, S_1, \ldots, S_T)$, we can estimate

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X Q_{\pi_\theta}(S,X) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="del-J-final" -->

using the observed rewards and by replacing probabilities by 1 or 0 based on the actual observation:
<!-- .element: data-id="trajectories" -->

`$\nabla_{\!\theta}$`<!-- .element: data-id="nabla" --> `$\displaystyle\sum_{t=1}^{T}  \Big($`<!-- .element: data-id="del-J-estimate-a" --> `$\displaystyle\sum_{k=t}^T r_k$`<!-- .element: data-id="del-J-estimate-b" --> $\cdot$ `$\ln \pi_\theta(S_{t-1},X_t) \Big)$`<!-- .element: data-id="del-J-estimate-c" -->

---

## Trajectory-based policy updates

We can apply gradient ascent using 

`$$\tilde J(\theta) = \nabla_{\!\theta} \sum_{t=1}^{T}  \Big( \sum_{k=t}^T r_k \cdot  \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

as a proportional estimate of the true gradient `$\nabla_{\!\theta}\ J(\theta)$`.

> [!NOTE]
> The hope is that gradient estimates over many trajectories will lead to policy improvement similar to using true gradients.


<!--
> [!NOTE]
> With **Flux.jl** we can compute `$\nabla_{\!\theta} \ln \pi_\theta(S_{t-1},X_t)$` using the final layer `activations` the network parameters `θ`, and the chosen `action` by determining `∇_θ = Flux.gradient( () -> Flux.logsoftmax(activations)[action], θ )`.
-->

---

## REINFORCE algorithm

<!-- .slide: data-fullscreen="yes"  -->


```julia [1-5|7-15|18-25|27-30|32-48|56-68]
function REINFORCE(env; 
    hidden_layers=[64, 32], η=1e-3, 
    γ=0.99, T=20_000, max_episodes=100_000, 
    callback=EpisodeLogger()
)
    
    # Initialize environment and get dimensions
    RL.reset!(env)
    
    # Create policy network for discrete actions
    layers = [length(RL.observe(env)), hidden_layers..., length(RL.actions(env))]
    policy = PolicyNetwork(layers)
    
    # Set up optimizer (Adam works well for policy gradients)
    optimizer = Flux.setup(Adam(η), policy)
    
    enable_interrupt() # allow to interrupt training by pressing ENTER key
    # Training loop
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        
        trajectory = Tuple{Vector{Float32}, Int32, Float32}[] # (state,action,reward)

        t = 0
        # Loop over at most T steps within episode
        while t < T
            t += 1
            
            # Sample action from policy
            π = Flux.softmax(policy(reshape(Sₜ₋₁, :, 1))[:, 1])            
            action_idx = sample_action(π)

            Xₜ = RL.actions(env)[action_idx]
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)          
            
            push!(trajectory, (Sₜ₋₁, Xₜ, rₜ))
            ∑rₜ += rₜ
            Sₜ₋₁ = Sₜ

            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Callback for logging
        if callback !== nothing
            callback(i, t, ∑rₜ)
        end

        ∇J̃ = Flux.gradient(policy) do model
            J̃ = 0.0f0 # Trajectory-based estimate of objective J(θ)
            R = 0.0f0
            while t > 0
                Sₜ₋₁, Xₜ, rₜ = trajectory[t] 
                R = rₜ + γ * R
                log_π = Flux.logsoftmax(model(reshape(Sₜ₋₁, :, 1))[:, 1])
                action_idx = findfirst(==(Xₜ), RL.actions(env))
                J̃ -= R * log_π[action_idx] # negative objective for gradient descent
                t -= 1
            end
            return J̃ / length(trajectory)  # Average objective per step
        end
              
        Flux.update!(optimizer, policy, ∇J̃[1])         

        if QUIT[]
            break
        end
    end
    
    return policy
end
```
<!-- .element: class="fullscreen stretch" -->


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
`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \class{highlight}{\big( Q_{\pi_\theta}(S,X) - B(S) \big)} \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="pgt-baseline" -->

for any given baseline $B(S)$.

---

<!-- .slide: data-auto-animate="true" -->

## Policy gradient theorem with baseline

According to the policy gradient theorem, `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 

`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \class{highlight}{\big( Q_{\pi_\theta}(S,X) - B(S) \big)} \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="pgt-baseline" -->

for any given baseline $B(S)$.

---

<!-- .slide: data-auto-animate="true" -->

## Value-function baseline

The main idea of actor-critic methods is to use a baseline obtained from a learnable **state-value function** `$V_{\theta_\text{critic}}(S)$` with parameters `$\theta_\text{critic}$`.

We know that `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 
`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \big( Q_{\pi_\theta}(S,X) \class{highlight}{- V_{\theta_\text{critic}}(S)} \big) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="critic" -->

---


<!-- .slide: data-auto-animate="true" -->

We know that `$\nabla_{\!\theta}\ J(\theta)$` is proportional to 
`$$\displaystyle\sum_S \Big( \mu_{\pi_\theta}(S) \cdot \displaystyle\sum_X \big( Q_{\pi_\theta}(S,X) \class{highlight}{- V_{\theta_\text{critic}}(S)} \big) \cdot \pi_\theta(S,X) \cdot  \nabla_{\!\theta} \ln \ \pi_\theta(S,X) \Big)$$`<!-- .element: data-id="critic" -->

which can be estimated by  

`$$\nabla_{\!\theta}  \sum_{t=1}^{T}  \Big( \big( \sum_{k=t}^T r_k \class{highlight}{- V_{\theta_\text{critic}}(S_{t-1})} \big) \cdot   \ln \pi_\theta(S_{t-1},X_t) \Big)$$`<!-- .element: data-id="critic-2" -->

---

<!-- .slide: data-auto-animate="true" -->

`$$\nabla_{\!\theta} \sum_{t=1}^{T}  \Big( \big( \class{highlight}{\sum_{k=t}^T r_k} - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \ln \pi_\theta(S_{t-1},X_t) \Big)$$`<!-- .element: data-id="critic-2" -->


can be estimated by

`$$\nabla_{\!\theta} \sum_{t=1}^{T}  \Big( \big( \class{highlight}{r_t + V_{\theta_\text{critic}}(S_t)} - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \ln \pi_\theta(S_{t-1},X_t) \Big)$$`

> [!IMPORTANT]
> For each step $t$ we can now compute all terms without waiting for the episode to be completed. 

---


As we do not need to wait for termination of an episode, we can estimate the policy gradient **for step** $t$ by 

`$$\nabla_{\!\theta}  \big( r_t + V_{\theta_\text{critic}}(S_t) - V_{\theta_\text{critic}}(S_{t-1}) \big) \cdot  \ln \pi_\theta(S_{t-1},X_t)$$`

---

## Training the actor 

As before, we can add a **discount factor** $\gamma\in [0,1]$ to obtain

`$$\nabla_{\!\theta}  \big( \underbrace{r_t + \class{highlight}{\gamma} \cdot V_{\theta_\text{critic}}(S_t)}_{\textrm{Bellman target}} - \underbrace{V_{\theta_\text{critic}}(S_{t-1})}_{\textrm{Prediction}} \big) \cdot  \ln \pi_\theta(S_{t-1},X_t)$$`

as a proportional approximate of $\nabla_\theta J(\theta)$ for training the actor.

---

## Training the critic 

For training the critic we use the squared error loss
`$$\mathscr{L}(\theta_{\text{critic}}) = \big( \underbrace{r_t + \class{highlight}{\gamma} \cdot V_{\theta_\text{critic}}(S_t)}_{\textrm{Bellman target}} - \underbrace{V_{\theta_\text{critic}}(S_{t-1})}_{\textrm{Prediction}} \big)^2$$`

---

<!-- .slide: data-fullscreen="yes"  -->

## Actor-critic algorithm

```julia [1-5|7-21|24-31|33-36|38-53|58-66|68-77|79-80|85-91]
function ActorCritic(env; 
    hidden_layers=[64, 32], η=1e-4, η_critic=1e-3,
    γ=0.99, T=20_000, max_episodes=100_000, 
    batch_size=32, callback=EpisodeLogger()
)
    
    # Initialize environment and get dimensions
    RL.reset!(env)
    feature_dim = length(RL.observe(env))
    n_actions = length(RL.actions(env))
    
    # Create actor (policy) and critic (value) networks
    actor_layers = [feature_dim, hidden_layers..., n_actions]
    critic_layers = [feature_dim, hidden_layers..., 1]
    
    actor = PolicyNetwork(actor_layers)    # π_θ(s,a)
    critic = ValueNetwork(critic_layers)   # V_θ_critic(s)
    
    # Set up optimizers
    actor_optimizer = Flux.setup(Adam(η), actor)
    critic_optimizer = Flux.setup(Adam(η_critic), critic)
    
    enable_interrupt() # allow to interrupt training by pressing ENTER key
    # Training loop
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        # Batch storage (state,action,reward,value of state, value of next state)
        batch = Tuple{Vector{Float32}, Int32, Float32, Float32, Float32}[]
        
        t = 0
        # Loop over at most T steps within episode
        while t < T
            t += 1
            
            # Sample action from policy (actor)
            π = Flux.softmax(actor(reshape(Sₜ₋₁, :, 1))[:, 1])            
            action_idx = sample_action(π)
            Xₜ = RL.actions(env)[action_idx]
            
            # Execute action and observe next state
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)
            
            # Compute value functions for batch storage
            Vₜ₋₁ = critic(reshape(Sₜ₋₁, :, 1))[1]
            Vₜ = terminal ? 0.0f0 : critic(reshape(Sₜ, :, 1))[1]
            
            # Store in batch
            push!(batch, (Sₜ₋₁, Xₜ, rₜ, Vₜ₋₁, Vₜ))
            
            # Update networks when batch is full or episode ends
            if length(batch) >= batch_size || terminal
                # Batch update actor
                ∇_actor = Flux.gradient(actor) do model
                    loss = 0.0f0
                    for (Sₜ₋₁, Xₜ, rₜ, Vₜ₋₁, Vₜ) in batch
                        log_π = Flux.logsoftmax(model(reshape(Sₜ₋₁, :, 1))[:, 1])
                        action_idx = findfirst(==(Xₜ), RL.actions(env))
                        loss -= (rₜ + γ * Vₜ - Vₜ₋₁) * log_π[action_idx]
                    end
                    loss / length(batch)
                end
                
                # Batch update critic
                ∇_critic = Flux.gradient(critic) do model
                    loss = 0.0f0
                    for (Sₜ₋₁, _, rₜ, _, Vₜ) in batch
                        target = rₜ + γ * Vₜ 
                        prediction = model(reshape(Sₜ₋₁, :, 1))[1]
                        loss += (target - prediction)^2
                    end
                    loss / length(batch)
                end
                
                Flux.update!(actor_optimizer, actor, ∇_actor[1])
                Flux.update!(critic_optimizer, critic, ∇_critic[1])
                
                empty!(batch)
            end
            
            ∑rₜ += rₜ
            Sₜ₋₁ = Sₜ

            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Callback for logging
        if callback !== nothing
            callback(i, t, ∑rₜ)
        end

        if QUIT[]
            break
        end
    end
    
    return actor
end
```
<!-- .element: class="fullscreen stretch" -->

---

You find a full implementation of REINFORCE and Actor-critic in the [course repository](https://rajgoel.github.io/course-machine-learning/julia).

> [!TIP]
> - **REINFORCE:**
> ```julia
> using MachineLearningCourse
> Lecture09.demo(:REINFORCE)
> ```
>
> - **Actor-critic:**:
> ```julia
> using MachineLearningCourse
> Lecture09.demo(:ActorCritic)
> ```

---

## REINFORCE for the Breakout game

> [!TIP]
> A visualisation of the trained agent playing the Breakout game can be launched by:
> 
> ```julia
> using MachineLearningCourse
> Lecture09.breakout()
> ```

