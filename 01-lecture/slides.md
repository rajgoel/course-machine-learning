# Introduction

> [!CAUTION]
> Not yet complete!

===

## Machine learning

Machine learning (ML) is a branch of artificial intelligence (AI) concerned with the development of algorithms that can learn from data and generalise to unseen data.

> [!NOTE]
> The goal of artificial intelligence is to develop algorithms that can perform tasks without being explicitly programmed.

---

### Types of machine learning

Machine learning is broadly classified into:

- **Supervised Learning**
- **Unsupervised Learning**
- **Reinforcement Learning**

---

### Supervised learning

Supervised Learning trains a model to predict an output from a given input. 

It uses **labelled data**, where each input is paired with the correct output, to adjust the model’s parameters and improve its predictions.

> [!NOTE]
> - The term *supervised* is used to indicate that guidance is given in form of correct outputs for each input.
> - In ML, the term **predict** does not necessarily mean forecasting future events. It refers to assuming the output corresponding to a given input based on the patterns learned from the training data.

---

#### Examples

Common examples for supervised learning are:

- **Regression:** Predict a continuous value, e.g., predict the arrival time of a shipment.

- **Classification:** Predict a category or class, e.g., predict whether a shipment will be delayed or not.


---

### Unsupervised learning

Unsupervised learning trains a model to find patterns, relationships, or structures in the data.

> [!NOTE]
> Unlike supervised learning, unsupervised learning does **not** rely on given labels or outputs.

---

#### Examples

Common examples for unsupervised learning are:

- **Clustering:** Group similar data points together based on their features.
- **Dimensionality reduction:** Simplify or compress data while keeping important information.

---

### Reinforcement learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment in order to maximize cumulative reward.

- The agent takes actions in the environment.
- The environment provides feedback in the form of rewards or penalties.
- The agent learns to choose actions that lead to higher long-term rewards.

> [!NOTE]
> RL differs from supervised learning because the agent learns from trial and error with feedback, rather than given labeled input-output pairs.

---

#### Examples

Common examples of reinforcement learning include sequential decision-making problems:

- **Exogenous uncertainty:** Decisions affected by external factors outside the agent’s control, e.g., unknown customer demand.
- **Endogenous uncertainty:** Decisions affected by internal factors within the agent’s control, e.g., the current positions of vehicles in a fleet.

===

## Deep learning

Deep Learning (DL) is a subfield of machine learning that uses **artificial neural networks** with **multiple layers** to automatically learn from data.

---

### Artificial neural networks

An artificial neural network (ANNs) is a computational model inspired by the structure and functions of biological neural networks.

Like biological neural networks, which contain neurons connected by synapses that carry signals, an ANN is made up of:

- **Artificial neurons:** Each neuron receives input signals, processes them, and produces an output signal.
- **Edges:** Connections between neurons that carry signals of varying intensity.


---

## Artificial neurons

Artificial neurons have an activation value that is calculated with an *activation function* that uses the input of connected neurons.

<table class="stretch">
<tr>
<td style="vertical-align: middle;">
$$
\left(
\begin{align}
a^\textrm{in}_{1} \\
a^\textrm{in}_{2} \\
a^\textrm{in}_{3} 
\end{align}
\right)
$$

</td>
<td style="vertical-align: middle;width:500px;">
<div class="neuralnetwork" style="height: 500px; width: 600px;">
<!--
{"type": "neuron" }
-->
</div>
</td>
<td style="vertical-align: middle;">
$$a^\textrm{out} = f( a^\textrm{in}_{1}, a^\textrm{in}_{2}, a^\textrm{in}_{3})$$
</td>
</tr>
</table>


===

## Classification with neural networks

---

### Example: Digit recognition

<table style="table-layout: fixed!important;width:900px;">
<tr>
<td style="vertical-align: middle;padding:0;width:400px!important;">
<canvas class="drawDigit" width="400" height="400" data-prevent-swipe>
</canvas>
</td>
<td style="vertical-align: middle;padding:0;width:300px!important;">
<i class="fas fa-arrow-right" style="font-size:100px;padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:400px!important;">
<svg class="togglePrediction" data-model="0" width="400" height="400">
  <rect width="400" height="400" style="fill:black;" />
  <text class="predictedDigit" x="200" y="350" font-size="400" text-anchor="middle"  style="fill:white;stroke:white;">?</text>
</svg>
</td>
</tr>
</table>

---

How can we recognise digits using neural networks?

===


## Simple world: Two symbols and 2x2 black and white pixel input


<table style="table-layout: fixed!important;width:700px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;width:300px!important;">
<i class="fas fa-arrow-right" style="font-size:100px;padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">/</text>
</svg>
</td>
</tr>
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;width:300px!important;">
<i class="fas fa-arrow-right" style="font-size:100px;padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">&bsol;</text>
</svg>
</td>
</tr>
</table>

---

### Vectorisation / Flattening

Vectorisation or flattening converts a multi-dimensional data structure, such as a matrix or tensor, into a one-dimensional vector by sequentially arranging all elements in a specified order.


![Image](01-lecture/flatten.svg)

---

### Input encoding

The input is represented by an encoding `$(a^\textrm{in}_{1}, a^\textrm{in}_{2}, a^\textrm{in}_{3}, a^\textrm{in}_{4})$`.

<table style="table-layout: fixed!important;width:700px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:100px;width:400px!important;">
(0,1,1,0)
</td>
</tr>
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:100px;width:400px!important;">
(1,0,0,1)
</td>
</tr>
</table>

---

### One-hot encoding

Given a value $v$ from list of alternative values $V$,  *one-hot encoding* creates a vector of size $|V|$ where only the value at the position corresponding to value $v$ is set to 1, and all other values are set to 0.

---

### Output encoding

The output for our list of symbols $V$ = ( /, \ ) is represented by a one-hot encoding  `$(a^\textrm{out}_{1}, a^\textrm{out}_{2})$`.

<table style="table-layout: fixed!important;width:700px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">/</text>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:100px;width:300px!important;">
(1,0) 
</td>
</tr>
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">&bsol;</text>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:100px;width:300px!important;">
(0,1) 
</td>
</tr>
</table>


---

### A simple neural network

<table class="stretch">
<tr>
<td style="vertical-align: middle;">
$$
\left(
\begin{align}
a^\textrm{in}_{1} \\
a^\textrm{in}_{2} \\
a^\textrm{in}_{3} \\
a^\textrm{in}_{4}
\end{align}
\right)
$$
</td>
<td>
<div class="neuralnetwork" style="height: 500px; width: 900px;">
<!--
{"type": "simple" }
-->
</div>
</td>
<td style="vertical-align: middle;">
$$ \left( \begin{align} a^\textrm{out}_{1} \\  a^\textrm{out}_{2} \end{align} \right) $$
</td>
</tr>
</table>


---

### Linear activation function

Given a 

- *weight* $w_{i,j}$  for each output $i$ and input $j$, and 
- a bias $b_i$ for each output $i$, 

we can determine the output activation values as a linear combination of the input activation values by

`$$a^\textrm{out}_1 = w_{1,1} a^\textrm{in}_{1} + w_{1,2} a^\textrm{in}_{2} + w_{1,3} a^\textrm{in}_{3} + w_{1,4} a^\textrm{in}_{4} + b_1$$`

and 

`$$ a^\textrm{out}_2 = w_{2,1} a^\textrm{in}_{1} + w_{2,2} a^\textrm{in}_{2} + w_{2,3} a^\textrm{in}_{3} + w_{2,4} a^\textrm{in}_{4} + b_2.$$`

---

### Matrix notation

`$$\left(
\begin{array}{c}
a^\textrm{out}_{1} \\
a^\textrm{out}_{2} 
\end{array}
\right) = \left(
\begin{array}{cccc}
w_{1,1} & w_{1,2} & w_{1,3} & w_{1,4} \\
w_{2,1} & w_{2,2} & w_{2,3} & w_{2,4} 
\end{array}
\right)
\left(
\begin{array}{c}
a^\textrm{in}_{1} \\
a^\textrm{in}_{2} \\
a^\textrm{in}_{3} \\
a^\textrm{in}_{4}
\end{array}
\right) + \left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right)$$`

or

`$$a^\textrm{out} = Wa^\textrm{in} + b$$`

---

### Determining weights and biases

We want to find weights and biases, so that our neural network can reliably recognise the given input.

<div class="twocolumn"  style="transform:scale(0.5);width:900px;margin-top:-100px">
<div style="margin-right:150px;">
<table style="table-layout: fixed!important;height: 500px;width:700px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;width:300px!important;">
<i class="fas fa-arrow-right" style="font-size:100px;padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">/</text>
</svg>
</td>
</tr>
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <rect x="0" y="0" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="0" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="0" y="100" width="100" height="100" style="fill:black;stroke:gray;stroke-width:3px;"/>
  <rect x="100" y="100" width="100" height="100" style="fill:white;stroke:gray;stroke-width:3px;"/>
</svg>
</td>
<td style="vertical-align: middle;padding:0;width:300px!important;">
<i class="fas fa-arrow-right" style="font-size:100px;padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:200px!important;">
<svg width="200" height="200">
  <rect width="200" height="200" style="fill:black;" />
  <text  x="100" y="175" font-size="200" text-anchor="middle"  style="fill:white;stroke:white;">&bsol;</text>
</svg>
</td>
</tr>
</table>
</div>
<div style="margin-top:50px;margin-left:150px;">
<div class="neuralnetwork" style="height: 500px; width: 900px;">
<!--
{"type": "simple" }
-->
</div>
</div>
</div>

---

If we find weights and biases such that

`$$\left(
\begin{array}{cccc}
w_{1,1} & w_{1,2} & w_{1,3} & w_{1,4} \\
w_{2,1} & w_{2,2} & w_{2,3} & w_{2,4} 
\end{array}
\right)
\left(
\begin{array}{c}
0 \\
1 \\
1 \\
0
\end{array}
\right) + \left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right) = \left(
\begin{array}{c}
1 \\
0 
\end{array}
\right)$$`

and

`$$\left(
\begin{array}{cccc}
w_{1,1} & w_{1,2} & w_{1,3} & w_{1,4} \\
w_{2,1} & w_{2,2} & w_{2,3} & w_{2,4}  
\end{array}
\right)
\left(
\begin{array}{c}
1 \\
0 \\
0 \\
1
\end{array}
\right) + \left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right) = \left(
\begin{array}{c}
0 \\
1 
\end{array}
\right) $$`

our neural network can perfectly recognise both symbols. 

---

### Optimal weights and biases

Below weights and biases are optimal.

`$$\left(
\begin{array}{cccc}
0.5 & 0.5 & 0.5 & -0.5 \\
0.5 & 0.5 & -0.5 & 0.5  
\end{array}
\right)
\left(
\begin{array}{c}
0 \\
1 \\
1 \\
0
\end{array}
\right) + \left(
\begin{array}{c}
0 \\
0 
\end{array}
\right) = \left(
\begin{array}{c}
1 \\
0 
\end{array}
\right)$$`

and

`$$\left(
\begin{array}{cccc}
0.5 & 0.5 & 0.5 & -0.5 \\
0.5 & 0.5 & -0.5 & 0.5  
\end{array}
\right)
\left(
\begin{array}{c}
1 \\
0 \\
0 \\
1
\end{array}
\right) + \left(
\begin{array}{c}
0 \\
0 
\end{array}
\right) = \left(
\begin{array}{c}
0 \\
1 
\end{array}
\right)$$`

We can also find other weights and biases that are optimal for this example.
<!-- .element: class="fragment" -->

---

### Conclusion

A simple neural network with

- 4 input neurons
- 2 output neurons with linear activation function

can be used to reliable recognize both symbols given a 2x2 black and white pixel image.


