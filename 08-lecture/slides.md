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

The set of state variables $S_t$ comprises all variables required to describe the state of a system at time $t$.

> [!NOTE]
> The term **time** does not necessarily represent clock time and can also be understood as a metaphor for step.

---

## Decision variables

The set of decision variables $X_t$ comprises all variables required to describe a decision (action) that can be taken at time $t$.

---

## Exogenous information

The set of information $W_t$ comprises all information that is 
revealed endogenously (from the world around) between time $t-1$ and $t$.

This includes random events, observations, and information arrivals that are not controlled by the decision-maker.

---

## Transition function

The transition function $g$ describes how the system evolves from one state to the next. 

Given the current state $S_t$, decision $X_t$, and new exogenous information $W_{t+1}$, it determines the next state by

$$S_{t+1} = g(S_t,X_t,W_{t+1}).$$

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
loop for t=0 to T-1
||20||
    Agent -> Agent: Xₜ ← decide(Sₜ)
    Agent -> Environment: make_decision(Xₜ)
    Environment -> Environment: Wₜ₊₁ ← get_exogenous_information()
    Environment -> Environment: Sₜ₊₁ ← g(Sₜ, Xₜ, Wₜ₊₁)
    Environment -> Environment: rₜ ← f(Sₜ₊₁) - f(Sₜ)
    Agent <-- Environment: reward rₜ, state Sₜ₊₁
||20||

    Agent -> Agent: learn(Sₜ, Xₜ, rₜ, Sₜ₊₁)
||20||
end
||20||
deactivate Agent
deactivate Environment
@enduml
-->

![Image](08-lecture/Reinforcement_learning.svg)

===

## Deep Q-Networks (DQN)

In DQN, we use a neural network with parameters $\theta$ to learn an **action-value function** $Q_\theta(S,X)$ that estimates the expected cumulative reward from taking decision $X$ in state $S$.

- The state $S$ defines the activations of the **input layer**.
- Each **output neuron** provides the $Q$-value for one possible decision (i.e. action).

> [!NOTE]
> - After training, the agent selects the decision with highest $Q$-value.
> - During training, exploration strategies (e.g., ε-greedy) are used to improve the accuracy of $Q$-values of other actions.

---

## Q-learning principle

If we had a perfectly learned action-value function $Q_\theta(S,X)$, and always select the decision with highest $Q$-value, we would have

`$$Q_\theta(S_t, X_t) = \underbrace{R(S_t, X_t)}_{= f(S_{t+1}) - f(S_t)} + \max_{X} Q_\theta(S_{t+1}, X)$$`


---

## Bellman equation (sample-based)

To learn the action-value function $Q_\theta(S,X)$, we use a sample-based form of the Bellman equation:

`$$Q_\theta(S_t, X_t) = R(S_t, X_t) + \class{highlight}{\gamma} \cdot \max_{X} Q_\theta(S_{t+1}, X)$$`

where
- $\gamma \in [0,1]$ is a so-called **discount factor** for future rewards.


> [!NOTE]
> - Observed transitions `$\big(S_t,X_t,R(S_t, X_t),S_{t+1}\big)$` allow us to omit computing expected values.
> - With $\gamma < 1$, future errors due to imperfect $Q$-values are discounted.

---

## Loss function and challenges in training

We can train our neural network by minimizing the squared error between the prediction and the target obtained by the Bellman equation

`$$\mathscr{L}(\theta) = \Big( \underbrace{Q_\theta(S_t, X_t)}_{\textrm{Prediction}} - \underbrace{R(S_t, X_t) + \gamma \max_{X} Q_\theta(S_{t+1}, X)}_{\textrm{Bellman target}}\Big)^2$$`

> [!WARNING]
> Consecutive observations are highly correlated. This can cause **overfitting** to recent training data.
<!-- .element: class="fragment" -->

> [!WARNING]
> Both prediction and target depend on $\theta$ which is updated during training. This can create **instability** and can prevent convergence.
<!-- .element: class="fragment" -->

---


> [!TODO]
> Full algorithm with experience replay, target networks, etc.
