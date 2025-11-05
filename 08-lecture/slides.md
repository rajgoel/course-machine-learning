# Reinforcement learning

---

In reinforcement learning, an **agent learns** to make **sequential decisions** by interacting with an **environment** and adapting its behavior based on **feedback** obtained from the environment.

---

## Sequential decision processes

Sequential decision processes provide a [unified modelling framework](https://link.springer.com/chapter/10.1007/978-3-030-60990-0_3) for reinforcement learning that naturally handles both deterministic and stochastic environments.
They consist of

- **State variables**
- **Decision variables**
- **Exogenous information**
- **Transition function**
- **Objective function**

> [!NOTE]
> Reinforcement learning is often based on [Markov decision processes (MDPs)](https://en.wikipedia.org/wiki/Markov_decision_process), a framework for sequential decision processes in stochastic environments. 

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
> We assume that the sequential decision process is finite, i.e., there is a final time step $T$. Moreover, we assume that the objective function $f$ is evaluable for all states $S_t$, including intermediate and terminal states.

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

Value-based methods learn value functions to estimate expected cumulative rewards:

- **State-value function** $V(S)$: expected cumulative reward from state $S$
- **Action-value function** $Q(S,X)$: expected cumulative reward from taking action (or decision) $X$ in state $S$

---

## Q-learning

In Q-learning, we learn a **parameterised action-value function** $Q_\theta(S,X)$ with parameters $\theta$ that estimates the expected cumulative reward from taking action $X$ in state $S$.

> [!NOTE]
> - After training, the action $X$ with the highest value of $Q_\theta(𝑆,𝑋)$ is taken.
> - During training, exploration strategies are used that occasionally select other actions. 

---

## Deep Q-Networks (DQN)

In DQN, $Q_\theta(S,X)$ is represented by a neural network and $\theta$ represents the weights and biases of the neural network. 

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

`$$Q_\theta(S_{t-1}, X_t) = r_t + \gamma \cdot \max_{X} Q_\theta(S_t, X)$$`

where
- $\gamma \in [0,1]$ is a so-called **discount factor** for future rewards.


> [!NOTE]
> - Observed transitions `$\big(S_{t-1},X_t,r_t,S_t\big)$` allow us to omit computing expected values.
> - With $\gamma < 1$, future errors due to imperfect $Q$-values are discounted.

---

## Loss function and challenges in training

We can train our neural network by minimizing the squared error between the target obtained by the Bellman equation and the prediction

`$$\mathscr{L}(\theta) = \Big( \underbrace{\big( r_t + \gamma \cdot \max_{X} Q_\theta(S_t, X) \big)}_{\textrm{Bellman target}} - \underbrace{Q_\theta(S_{t-1}, X_t)}_{\textrm{Prediction}} \Big)^2$$`

> [!WARNING]
> Consecutive observations are highly correlated. This can cause **overfitting** to recent training data.
<!-- .element: class="fragment" -->

> [!WARNING]
> Both prediction and target depend on $\theta$ which is updated during training. This can create **instability** and can prevent convergence.
<!-- .element: class="fragment" -->

---

## Experience replay

To reduce correlation of observations used for training, we can create a **replay buffer** in which we store observations.

During training we randomly pick observations from the replay buffer, and run stochastic gradient descent on those mini-batches. 

> [!NOTE]
> With a sufficiently large replay buffer, observations selected for a mini-batch are likely to come from different episodes.


---

## Target networks

To address the **instability** problem, we can use a **target network** with parameters $\theta_{\text{target}}$ and train our neural network by minimising

`$$\mathscr{L}(\theta) = \Big( \underbrace{\big( r_t + \gamma \cdot \max_{X} \class{highlight}{Q_{\theta_{\text{target}}}}(S_t, X) \big)}_{\textrm{Bellman target}} - \underbrace{Q_\theta(S_{t-1}, X_t)}_{\textrm{Prediction}} \Big)^2$$`

As the Bellman target no longer depends on $\theta$, we have
$$\frac{\partial \mathscr{L}}{\partial \theta} = 2( \big( r_t + \gamma \cdot \max_{X} \class{highlight}{Q_{\theta_{\text{target}}}}(S_t, X) \big) - 
  Q_\theta(S_{j-1}, X_j)) \cdot (-1) \cdot \nabla_\theta Q_\theta(S_{j-1},
  X_j)$$

> [!NOTE]
> The target network has the same architecture as the main network and is updated periodically by copying the parameters.

---

## DQN Algorithm

```
Initialize:
  - Q-network with random weights θ
  - Target network Q_target with weights θ_target = θ 
  - Replay buffer D
  - Target network update frequency C
  - Environment

For each episode:
  1. Initialize state S₀
  
  For each time step t:
    1a. With probability ε: select random action Xₜ
        Otherwise: select Xₜ = argmax Q(Sₜ₋₁, X)
    
    1b. Execute action Xₜ, observe reward rₜ and state Sₜ
    
    1c. Store transition (Sₜ₋₁, Xₜ, rₜ, Sₜ) in replay buffer D
    
    1d. Sample random mini-batch from D
    
    1e. For each transition in the mini-batch:
        Compute target: yⱼ = rⱼ + γ · max Q_target(Sⱼ₋₁, X)
    
    1f. Update θ: θ ← θ - α × avg( 2(yⱼ - Q(Sⱼ₋₁, Xⱼ)) × (-1) × ∇_θ Q(Sⱼ₋₁, Xⱼ) ) 
    
    1h. if t mod C == 0: update target network θ_target ← θ

  2. Update ε
```

> [!NOTE]
> - ε-greedy exploration balances exploitation vs exploration
> - Mini-batch sampling breaks correlation between consecutive updates
> - Target network updates every C steps maintain training stability

