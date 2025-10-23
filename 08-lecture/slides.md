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

> [!NOTE]
> For terminal states $S_t$ with $t < T$, we can prematurely exit the loop.
