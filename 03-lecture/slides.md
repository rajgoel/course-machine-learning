> [!CAUTION]
> Not yet complete!

---


# Feedforward neural networks

---

## Feedforward neural network with 4 layers

<div class="neuralnetwork predictions" style="height: 700px; width: 1300px!important;">
<!--
{"type": "feedforward" }
-->
</div>

In a feedforward neural network, the activations of one layer determine the activations of the next layer.

---

### Activations in feedforward neural networks

For each layer $l$ in the feedforward neural network let
- $a^l$ denote the vector of activation values,
- $W^l$ and $b^l$ denote the weights and biases, and
- $\sigma^l$ denote the vector of activation functions.

Then, the activation values of layer $l$ can be computed by

`$$z^l = W^l a^{l-1} + b^l,$$`
`$$a^{l} = \sigma^l(z^l).$$`

---

### Loss

For a given input/output pair  $(a,a^*)$, the sum of squared errors of a feedforward neural network with $L$ layers is

`$$\mathscr{L}_{(a,a^*)}(W^1,b^1,\ldots,W^{L},b^{L}) = \sum_{i=1}^{n^L}(\hat{a}_i - a^*_i)^2.$$`

where $n^L$ denotes the number of neurons in the output layer.

===

### Gradient descent for feed forward networks

> [!TIP]
> The following deep dive is helpful for an understanding of the mechanics of deep learning. 
> As the mechanics are already implemented in deep learning frameworks, you do not need to understand them when simply using the frameworks.

---


### Gradient descent

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!NOTE]
> Remember, that gradient descent works by iteratively changing weights and biases in opposite direction of the average gradient of the loss. To compute the gradient, we need the partial derivatives for **all** weights and biases.

---

### Partial derivatives for weights and biases in the last layer

- For each output neuron $i$ and each neuron $j$ in the last hidden layer, we have
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^L_{i,j} } = 2(a^L_i - a^*_i) \cdot a^{L-1}_j.$$`
- For each output neuron $i$, we have
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^L_{i} } = 2(a^L_i - a^*_i)$$`

> [!NOTE]
> Apart from notational differences, this is the same as for the single layer neural network studied in the last session.

---

### Total derivatives

Changing a weight or bias in layer $l < L$, does not only change the activations of layer $l$, but also the activations of subsequent layers. Instead of using the partial derivative, we use the total derivative for gradient descent.

> [!NOTE]
> **Total derivative:** The [total derivative](https://www.geeksforgeeks.org/engineering-mathematics/total-derivative/) of a function $f( g_1(x), \ldots, g_n(x) )$ measures the rate of change of a function with respect to one variable while considering the effect of all other variables changing as well. 
> We have $\genfrac{}{}{1pt}{1}{d f}{d x }  = \displaystyle\sum_{i=1}^n \genfrac{}{}{1pt}{1}{\partial f}{\partial g_i } \cdot \genfrac{}{}{1pt}{1}{\partial g_i}{\partial x }$. 


---

### Total derivatives for weights and biases in hidden layers

According to the chain rule, we have
- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d w^l_{i,j} } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i }
\cdot  \displaystyle\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i }
 \cdot \displaystyle\genfrac{}{}{1pt}{1}{d z^l_i}{d w^l_{i,j} } $` for each neuron $i$ of layer $l$ and each neuron $j$ of layer $l-1$, and  

- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d b^l_i } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i } 
\cdot \displaystyle\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i }
\cdot \displaystyle\genfrac{}{}{1pt}{1}{\partial z^l_i}{d b^l_i }$` for each neuron $i$ of layer $l$.  


where `$\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i } = \genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i }$`, `$\genfrac{}{}{1pt}{1}{d z^l_i}{d w^l_{i,j}} = a^{l-1}_j$`, and `$\genfrac{}{}{1pt}{1}{d z^l_i}{d b^l_j} = 1$`.

---





---



### Partial derivatives for weights

For any particular weight $\bar w$, we have

`$$
\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \bar w} = \sum_{i\in I^L} \underbrace{\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i}}_{\textrm{independent of} \atop \textrm{weights and biases}} \cdot \frac{\partial \hat{a}_i}{\partial \bar w} 
\class{fragment}{= \sum_{i\in I^L}2(\hat{a}_i - a^*_i) \frac{\partial \hat{a}_i}{\partial \bar w}}
$$`

> [!NOTE]
> Remember, that `$\mathscr{L}_{(a,a^*)} = \displaystyle\sum_{i\in I^L}(\hat{a}_i - a^*_i)^2$`.

---

The partial derivative $\displaystyle\frac{\partial \hat{a}_i}{\partial \bar w}$ indicates how much a change in $\bar w$ affects an output activation $\hat{a}_i$ and can be determined using the **chain rule**.

---

Assume that $\bar w$ is a weight belonging to $W^{\bar l}$ used to compute the activation in layer $\bar l$.

According to the chain rule, we have
`$$
\begin{array}{ccl}
\frac{\partial \hat{a}}{\partial \bar w} & = & \frac{\partial a^L}{\partial \bar w} \\
& = & 
\underbrace{
\frac{\partial a^L}{\partial a^{L-1}}
\ \cdot\  
\frac{\partial a^{L-1}}{\partial a^{L-2}}
\ \cdot\  \ldots \ \cdot\  
\frac{\partial a^{\bar l+1}}{\partial a^{\bar l}}}_{\textrm{each term indicates how much a change in the activation of a neuron} \atop \textrm{in one layer affects the activation of the a neuron in the next layer}}
\ \cdot\ 
\frac{\partial a^{\bar l}}{\partial \bar w}
\end{array}
$$`


---

### Weight and bias sensitivities

We have
`$$
\frac{\partial a^{l}}{\partial a^{l-1}} = 
\frac{\partial a^{l}}{\partial z^{l}} \cdot \frac{\partial z^{l}}{\partial a^{l-1}}$$`

Since $\sigma^l$ is applied element-wise, we have
`$$
\frac{\partial a^{l}}{\partial z^{l}} = \operatorname{diag}\left(
\frac{\partial \sigma^{l}(z_1^{l})}{\partial z_1^{l}}, \ldots, 
\frac{\partial \sigma^{l}(z_{n^{l}}^{l})}{\partial z_{n^{l}}^{l}}
\right)
$$`

Moreover, we have
`$$
\frac{\partial z^{l}}{\partial a^{l-1}} = W^l
$$`


---

### Jacobian matrix of partial derivatives

We have
`$$
\frac{\partial a^{l+1}}{\partial a^{l}} = \frac{\partial F^{l}_{W^{l},b^{l}}}{\partial a^{l}}
=\left(
\begin{array}{ccc}
\tfrac{\partial f_1}{\partial a^{l}_1} & \ldots & \tfrac{\partial f_1}{\partial a^{l}_{h}} \\
\vdots & \ddots & \vdots \\
\tfrac{\partial f_{k}}{\partial a^{l}_1} & \ldots & \tfrac{\partial f_{k}}{\partial a^{l}_{h}} \\
\end{array}
\right)
$$`
where 
- $h$ is the number of neurons in layer $l$,
- $k$ is the number of neurons in layer $l+1$, and 
- $f_i$ is the activation function of the $i$-th neuron in layer $l+1$. 

> [!NOTE]
> Remember, that $F^l_{W^l,b^l}$ is a **vector** of activation functions.

---


### Partial derivatives for biases

For any particular bias $\bar b$, we have

`$$
\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \bar b} = \sum_{i\in I^L}\underbrace{\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i}}_{\textrm{independent of} \atop \textrm{weights and biases}} \cdot \frac{\partial \hat{a}_i}{\partial \bar b} 
\class{fragment}{= \sum_{i\in I^L}2(\hat{a}_i - a^*_i) \frac{\partial \hat{a}_i}{\partial \bar b}}
$$`

> [!NOTE]
> Remember, that `$\mathscr{L}_{(a,a^*)} = \displaystyle\sum_{i\in I^L}(\hat{a}_i - a^*_i)^2$`.

---

The partial derivative $\displaystyle\frac{\partial \hat{a}_i}{\partial \bar b}$ indicates how much a change in $\bar b$ affects an output activation $\hat{a}_i$ and can also be determined using the **chain rule**.

> [!NOTE]
> All the steps are analogously to those to determine the partial derivatives for weights.


---

### Back propagation

We can iteratively determine the product of the Jacobian matrices 
<span class="fragment">
`$
\frac{\partial a^L}{\partial a^{L-1}},
$`
</span>
<span class="fragment">
`$
\frac{\partial a^L}{\partial a^{L-1}} \cdot \frac{\partial a^{L-1}}{\partial a^{L-2}},
$`
</span>
<span class="fragment">
$\ldots$
</span>

<div class="fragment">
and use these to compute the partial derivatives

`$\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \bar w}$` and `$\frac{\partial \mathscr{L}_{(a,a^*)}}{\partial \bar b}$` 

with respect to each weight $\bar w$ and bias $\bar b$ for the respective activation function of the layer.
</div>

===

## Non-linear activation functions

---

So far we only considered linear activation functions of the form $f_{W,b}(a) = Wa + b$.

> [!IMPORTANT]
> A feedforward neural network with **multiple layers** and **linear activation functions** has **no advantage** over a single layer neural network.

---

### Input and output function

We can use a composed activation function $f = g \circ h$ where

- $h$ is a linear input function used to aggregate input activation values, and
- $g$ is a non-linear output function  that modifies the aggregated input to determine the output activation value.

> [!NOTE]
> Some references use the term *activation function* to describe $g$ and implicitly assume $h$ to be a linear combination of the input activation values.

---

### Backpropagation

We can also use backpropagation for $f = g \circ h$.

According to the chain rule we have

$$\frac{ \partial f}{ \partial \bar w} = \frac{ \partial g}{ \partial h } \cdot \frac{ \partial h}{ \partial \bar w}$$

and

$$\frac{ \partial f}{ \partial \bar b} = \frac{ \partial g}{ \partial h } \cdot \frac{ \partial h}{ \partial \bar b}$$

---

### Sigmoid

`$$g(x) = \frac{1}{1+e^{-x}}$$`

![Figure](03-lecture/sigmoid.svg)

<!-- 
Derivative: 
dg(x)/d = (1+e^{-x})^{-1} = -(1+e^{-x})^{-2}(-e^{-x})
-->

<!--
Usage: as an output layer activation function in binary classification
-->

---

### Rectified Linear Unit (ReLU)

`$$g(x) = \max\{0,x\}$$`

![Figure](03-lecture/ReLU.svg)

> [!NOTE]
> $\frac{ \partial g}{ \partial x }$ is not defined for $x=0$, but we can anyhow use a value of 0 or 1 in back propagation.

<!--
Usage: most commonly used activation function for hidden layers.
-->

---


#### Architecture parameters in a Neural Net

- **Number of hidden Layers:** More complex problems benefit from more layers (Deep Learning)
- **Number of nodes per hidden layer:**  used to be common to have a pyramid like structure, nowadays is fairly common to use the same 
number of neurons in all hidden layers.
- Example: input layer, 2 hidden layers, output layer → 4 total layers

---

#### Hyperparameters in a Neural Net

- **Learning rate:** Controls how much the weights are moving in the gradient descent update. On some problems using a LR that changes during training can increase the performance.
- **Batch size:** Controls how many training examples processed before updating the weights. Smaller batch size leads to slower convergence. A typical batch size is 32.
- **Optimizer:** there are more advanced optimizers besides Gradient Descent 
- **Number of iterations:** Early stopping is a way to not deal with the epochs as a hyperparameter. 
- **Activation functions:** Relu, sigmoid, Tanh...
- **Number of Epochs:** Often set based on validation loss.

---

### Data split

<img src="feedforwardnetworks/datasplit.svg" width="800"/>

| Training Set Purpose | Dev Set Purpose | Test Set Purpose |
|----------------------|-----------------|------------------|
| Train the model parameters | Tune hyperparameters | Final evaluation |
| Learn patterns from data | Identify overfitting | Unbiased assessment |

---

### Training Heuristics

- Often in practice, there is no test set and the dev set is called dev set.
- Training error informs about bias problems (underfitting).
- Dev error informs about variance problems (overfitting).


---

### Training Heuristics

<img src="feedforwardnetworks/training_guidelines.png" />

---

### Training Heuristics

- **Training a NN is highly iterative:** Evaluate the predictions on the right metric, make changes and evaluate again.

<div class=highlight-box> 
📝 A common workflow is training a model that fits the data very well (low training error)
and check whether the variance is acceptable looking at the dev set performance. Adding more data and regularization
should reduce the variance.
</div>

---

===

## Hand-written digit recognition

<br>

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

### Feedforward neuronal network with 4 layers and ReLU activation
 
<table style="table-layout: fixed!important;width:1300px;">
<tr>
<td style="vertical-align: middle;">
<canvas class="drawDigit" width="150" height="150">
</canvas>
</canvas>
</td>
<td style="vertical-align: middle;padding:0;width:900px!important;">
<div class="neuralnetwork predictions" style="height: 800px; width: 900px;">
<!--
{"type": "full" }
-->
</div>
</td>
<td style="vertical-align: middle;">
<svg class="togglePrediction" data-model="1" width="150" height="150" style="padding:0px;">
  <rect width="150" height="150" style="fill:black;" />
  <text class="predictedDigit" x="75" y="125" font-size="150" text-anchor="middle"  style="fill:white;stroke:white;">?</text>
</svg>
</td>
</tr>
</table>

===

### MNIST database

The [MNIST database](https://yann.lecun.com/exdb/mnist/) contains gray scale values of a 28x28 pixel image representing a handwritten digit and a label representing the corresponding digit.
It includes 

- a training set of 60,000 examples, and
- a test set of 10,000 examples. 

![Digits](03-lecture/digits.jpg)

---

<!-- .slide: data-fullscreen="yes"  -->

### Read MNIST data in Julia

```julia [1|3,13|5|7-10|15-16]
using MLDatasets # Provides the MNIST database

function get_data(data_type::Symbol)
  # Load MNIST data based on the specified split
  x, y = MLDatasets.MNIST(split=data_type)[:]

  # Create flattened training images -> 784×[number of images] Matrix{Float32}
  x_encoded = Flux.flatten(x) 
  # Create one-hot encoded labels -> 10×[number of images] OneHotMatrix(::Vector{UInt32})
  y_encoded = Flux.onehotbatch(y, 0:9)
    
  return ( x_encoded, y_encoded )
end

training_data = get_data(:train)
test_data = get_data(:test)
```
<!-- .element: class="fullscreen stretch" -->

---

### Machine learning libraries

There are many machine learning that do most of the work for us. In Julia, we can use Flux.jl.

<iframe class="stretch" data-src="https://fluxml.ai/Flux.jl/stable/"></iframe>

---

<!-- .slide: data-fullscreen="yes"  -->

### Create neural network in Julia

```julia [1|3-11]
using Flux # Provides machine learning library

function create_model()
    # Define the model
    model = Chain(
        Dense(28 * 28, 256, relu),
        Dense(256, 10, relu),
        softmax
    )
    return model
end
```
<!-- .element: class="fullscreen stretch" -->



===


<table style="table-layout: fixed!important;width:900px;">
<tr>
<td style="vertical-align: middle;padding:0;width:400px!important;">
<canvas class="drawDigit" width="400" height="400">
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

> [!IMPORTANT]
> The neural network will predict a digit for any input, even if no digit is provided!

