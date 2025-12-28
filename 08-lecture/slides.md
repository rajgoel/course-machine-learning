# Reinforcement learning

---

In reinforcement learning, an **agent learns** to make **sequential decisions** by interacting with an **environment** and adapting its behavior based on **feedback** obtained from the environment.

---

## Sequential decision problems

[Sequential decision problems](https://castle.princeton.edu/wp-content/uploads/2021/09/RLSO-Cover-Chapter1-Sept32021.pdf) provide a unified modelling framework for reinforcement learning that naturally handles both deterministic and stochastic environments. They consist of

- **State variables**
- **Decision variables**
- **Exogenous information**
- **Transition function**
- **Objective function**

> [!NOTE]
> An alternative framework for sequential decision problems are [Markov decision processes (MDPs)](https://en.wikipedia.org/wiki/Markov_decision_process). 

---

### State variables

The set of state variables $S_t$ comprises all variables required to describe the state of a system at (the end of) time $t$.

> [!NOTE]
> The term **time** does not necessarily represent clock time and can also be understood as a metaphor for step.

---

## Decision variables

The set of decision variables $X_t$ comprises all variables required to describe a decision (action) that can be taken at time $t$.

---

## Exogenous information

The set of information $W_t$ comprises all information that is revealed exogenously (from the world around) at time $t$.

> [!IMPORTANT]
> This includes random events, observations, and information arrivals that are **neither controlled nor known by the decision-maker**.

---

## Transition function

The transition function $g$ describes how the system evolves from one state to the next. 

Given the latest observed state $S_{t-1}$, decision $X_t$, and new exogenous information $W_t$, it updates the state by

$$S_{t} = g(S_{t-1},X_t,W_t)$$

---

## Objective function

The objective function $f$ evaluates the quality of a state $S_t$. 

The goal is to **maximize** $f(S_T)$, where $T$ is the final time step.

> [!IMPORTANT]
> We assume that the sequential decision process is finite, i.e., there is a final time step $T$. Moreover, we assume that $f(S_t)$ can be evaluated for all $0 \leq t \leq T$.

---

## Reinforcement learning for sequential decision processes

<!--
@startuml
!pragma teoz true
skinparam ParticipantPadding 240
skinparam defaultFontSize 16
participant Agent
participant Environment
||20||
activate Agent
activate Environment
Agent -> Environment: get_state()
Agent <-- Environment: state S₀
||20||
loop for t=1 to T
    Environment -> Environment: Wₜ ← get_exogenous_information()
&   Agent -> Agent: Xₜ ← decide(Sₜ₋₁)
||20||
    Agent -> Environment: make_decision(Xₜ)
    Environment -> Environment: Sₜ ← g(Sₜ₋₁, Xₜ, Wₜ)
    Environment -> Environment: rₜ ← f(Sₜ) - f(Sₜ₋₁)
    Agent <-- Environment: reward rₜ, state Sₜ
||20||

    Agent -> Agent: learn(Sₜ₋₁, Xₜ, rₜ, Sₜ)
||20||
end
||20||
deactivate Agent
deactivate Environment
@enduml
-->

![Image](08-lecture/Reinforcement_learning.svg)

===

# Value-based methods

---

## Value functions

Value-based methods learn to estimate:

- **State-value function** $V(S)$: expected cumulative reward from state $S$
- **Action-value function** $Q(S,X)$: expected cumulative reward from taking action (or decision) $X$ in state $S$

---

## Q-learning

In Q-learning, we learn a **parameterised action-value function** $Q_\theta(S,X)$ with parameters $\theta$ that estimates the expected cumulative reward from taking action $X$ in state $S$.

---

## $\epsilon$-greedy algorithm

The $\epsilon$-greedy algorithm is a simple approach to balance **exploration** and **exploitation** when selecting actions by

`$$X_t = \left\{  \begin{array}{l} \textsf{ any action with probability } \epsilon \\ \displaystyle \,\,\arg\!\max_X Q_\theta(𝑆,𝑋) \textsf{ with probability } 1-\epsilon \end{array}\\   \right.$$`

> [!NOTE]
> We can gradually reduce $\epsilon$ during training.

---

## Deep Q-Networks (DQN)

In DQN we represent $Q_\theta(S,X)$ by a neural network with parameters $\theta$ representing the weights and biases. 

> [!NOTE]
> - The state $S$ defines the activations of the **input layer**.
> - Each **output neuron** estimates the Q-value for one possible action.


---

## Q-learning principle

If we had a perfectly learned action-value function $Q_\theta(S,X)$, and always select the decision with highest Q-value, we would have

`$$Q_\theta(S_{t-1}, X_t) = r_t + \max_{X} Q_\theta(S_t, X)$$`


---

## Bellman equation (sample-based)

To learn the action-value function $Q_\theta(S,X)$, we use a sample-based form of the Bellman equation:

`$$Q_\theta(S_{t-1}, X_t) = r_t + \class{highlight}{\gamma} \cdot \max_{X} Q_\theta(S_t, X)$$`

where $\gamma \in [0,1]$ is a so-called **discount factor**.


> [!NOTE]
> $\gamma < 1$ is used to reduce the impact of imperfect $Q$-values for future actions.

---

## Loss function

We can train our neural network by minimizing the squared error between the target obtained by the Bellman equation and the prediction

`$$\mathscr{L}(\theta) = \Big( \underbrace{\big( r_t + \gamma \cdot \max_{X} Q_\theta(S_t, X) \big)}_{\textrm{Bellman target}} - \underbrace{Q_\theta(S_{t-1}, X_t)}_{\textrm{Prediction}} \Big)^2$$`

> [!WARNING]
> Consecutive observations are highly correlated. This can cause **overfitting** to recent training data.
<!-- .element: class="fragment" -->

---

## Experience replay

To reduce correlation of observations used for training, we can create a **replay buffer** in which we store observed **state transitions** $(S_{t-1},X_t,r_t,S_t)$.

We create mini-batches by **randomly** picking state transitions to run stochastic gradient descent.

> [!NOTE]
> With a sufficiently large replay buffer, observations selected for a mini-batch are likely to come from different episodes.

---

## Instability

Both the Bellman target and the prediction

`$$\mathscr{L}(\theta) = \Big( \underbrace{\big( r_t + \gamma \cdot \max_{X} Q_\theta(S_t, X) \big)}_{\textrm{Bellman target}} - \underbrace{Q_\theta(S_{t-1}, X_t)}_{\textrm{Prediction}} \Big)^2$$`

depend on $\theta$ which is updated during training.

> [!IMPORTANT]
> Even small changes in $\theta$ can cause large changes in the loss. This can create **instability** and can prevent convergence.

---

## Target networks

To address the **instability** problem, we can use a **target network** with parameters $\theta_{\text{target}}$ and train our neural network by minimising

`$$\mathscr{L}(\theta) = \Big( \underbrace{\big( r_t + \gamma \cdot \max_{X} \class{highlight}{Q_{\theta_{\text{target}}}}(S_t, X) \big)}_{\textrm{Bellman target}} - \underbrace{Q_\theta(S_{t-1}, X_t)}_{\textrm{Prediction}} \Big)^2$$`

> [!NOTE]
> The target network has the same architecture as the main network and is updated periodically by copying the parameters.

---

<!-- .slide: data-fullscreen="yes"  -->

## DQN-Algorithm

```julia [1-9|10-24|26-28|30|32-33|34-37|39-44|46-51|53-56|58-59|61-65|67-71|73-80|88-89]
function DQN(env; 
    hidden_layers=[128, 64], η=1e-4, 
    γ=0.99, T=20_000,  
        ϵ=(0.5, 0.01), Δϵ = 1e-4, 
    replay_memory_size=1_000_000, replay_start_size=100_000, batch_size=32, update_frequency=4,
    target_evaluation=ddqn_target_evaluation,  target_update_frequency=25_000,
    max_episodes=100_000,  
    callback=EpisodeLogger()
)
    RL.reset!(env)

    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(replay_memory_size)
    
    # Create the neural network architecture with flattened game state as input and action Q-values as output
    layers = [length(RL.observe(env)), hidden_layers..., length(RL.actions(env))]
    q_network = ValueNetwork(layers)        # Main Q-network
    target_network = ValueNetwork(layers)   # Target network for stable learning
    
    # Initialize target network with same weights as main network
    update_target_network!(target_network, q_network)
    
    # Set up Adam optimizer for gradient-based learning
    optimizer = Flux.setup(Adam(η), q_network)

    # Initialize update counters
    next_update = replay_start_size
    next_target_update = replay_start_size + target_update_frequency

         ϵᵢ = first(ϵ)  # Will be reduced by Δϵ after every episode

    # Loop over episodes
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        
        t = 0
        # Loop over at most T steps within episode
        while t < T
            t += 1
            next_update -= 1
            next_target_update -= 1
            
            # ϵᵢ-greedy action selection
            if rand() < ϵᵢ
                Xₜ = rand(RL.actions(env)) # Random action
            else               
                Xₜ = RL.actions(env)[argmax(q_network(reshape(Sₜ₋₁, :, 1)))]  # Greedy action 
            end
            
            # Execute the chosen action in the environment
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)
            
            # Store experience in replay buffer for later learning
            push!(replay_buffer, StateTransition(Sₜ₋₁, Xₜ, rₜ, Sₜ, terminal))
            
            # Perform training step when counter reaches zero
            if next_update <= 0
                loss = train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, γ, RL.actions(env), target_evaluation)
                next_update = update_frequency  # Reset counter
            end
            
            # Update target network when counter reaches zero
            if next_target_update <= 0
                update_target_network!(target_network, q_network)
                next_target_update = target_update_frequency  # Reset counter
            end
            
            # Update episode reward
            ∑rₜ += rₜ
            Sₜ₋₁ = Sₜ
            
            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Callback for monitoring progress
        if callback !== nothing
            callback(i, t, ∑rₜ)
        end

        # Reduce ϵᵢ for reduced exploration and increased exploitation
                ϵᵢ = max(last(ϵ), ϵᵢ - Δϵ)
    end
    
    return q_network
end
```
<!-- .element: class="fullscreen stretch" -->


---

## DQN vs. Double DQN (DDQN)

In DQN the target network is target network is used for both action selection and evaluation. As Q-values may be wrong, especially early in training, any overestimation may amplify if used for selection and evaluation.

To reduce such an **overestimation bias** DDQN uses the target network for evaluation, and the main network for action selection. 

> [!NOTE]
> Decoupling selection from evaluation prevents the same network's overestimation errors from compounding. If the main network overestimates an action, the target network provides an independent evaluation that is less likely to have the same overestimation bias.

