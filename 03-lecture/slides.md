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
- $F^l_{W^l,b^l}$ denote the vector of activation functions.

Then, the activation values of layer $l+1$ can be computed by
 
`$$a^{l+1} = F^l_{W^l,b^l} (a^l).$$`

---

### Computed output activations

For a given input activiation  $a^\textrm{in}$, the output activation $a^\textrm{out}$ in a feedforward neural network with $L$ layers can be computed by

`$$ a^\textrm{out} = \underbrace{F^{L-1}_{W^{L-1},b^{L-1}} ( \underbrace{\ldots \underbrace{F^2_{W^2,b^2} ( \underbrace{F^1_{W^1,b^1}( \underbrace{a^\textrm{in}}_{a^1} )}_{a^2} )}_{a^3}  \ldots}_{\unicode{x22F0}\quad})}_{a^L}.$$`


---

### Error function

For a given input/output pair  $(a^\textrm{in},a^*)$, the sum of squared errors of a feedforward neural network with $L$ layers is

`$$f^\textrm{error}_{(a^\textrm{in},a^*)}(W^1,b^1,\ldots,W^{L-1},b^{L-1}) = \sum_{i\in I^L}(a^\textrm{out}_i - a^*_i)^2.$$`

where $I^L$ denotes the set of neurons in the output layer.

---

### How does gradient descent work for feed forward networks?

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!NOTE]
> Remember, that gradient descent works by iteratively changing weights and biases in opposite direction of the gradient of the mean error $\displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a^\textrm{in},a^*) \in S}f^\textrm{error}_{(a^\textrm{in},a^*)}$.
> The gradient is the collection of all its partial derivatives.

---

### Partial derivatives for weights

For any particular weight $\bar w$, we have

`$$
\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial \bar w} = \sum_{i\in I^L} \underbrace{\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial a^\textrm{out}_i}}_{\textrm{independent of} \atop \textrm{weights and biases}} \cdot \frac{\partial a^\textrm{out}_i}{\partial \bar w} 
\class{fragment}{{= \sum_{i\in I^L}2(a^\textrm{out}_i - a^*_i) \frac{\partial a^\textrm{out}_i}{\partial \bar w}}}
$$`

> [!NOTE]
> Remember, that `$f^\textrm{error}_{(a^\textrm{in},a^*)} = \displaystyle\sum_{i\in I^L}(a^\textrm{out}_i - a^*_i)^2$`.

---

### Partial derivatives for biases

For any particular bias $\bar b$, we have

`$$
\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial \bar b} = \sum_{i\in I^L}\underbrace{\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial a^\textrm{out}_i}}_{\textrm{independent of} \atop \textrm{weights and biases}} \cdot \frac{\partial a^\textrm{out}_i}{\partial \bar b} 
\class{fragment}{{= \sum_{i\in I^L}2(a^\textrm{out}_i - a^*_i) \frac{\partial a^\textrm{out}_i}{\partial \bar b}}}
$$`

> [!NOTE]
> Remember, that `$f^\textrm{error}_{(a^\textrm{in},a^*)} = \displaystyle\sum_{i\in I^L}(a^\textrm{out}_i - a^*_i)^2$`.

---

To determine $\displaystyle\frac{\partial a^\textrm{out}_i}{\partial \bar w}$ and $\displaystyle\frac{\partial a^\textrm{out}_i}{\partial \bar b}$, we can determine

- how much a change in $\bar w$ or $\bar b$ affects the activation of the correspondig neuron, and
- how much a change in the activation of a neuron affects the activation of the next neuron.

---

### Chain rule

According to the chain rule, we have
`$$
\begin{array}{ccccccccccc}
\frac{\partial a^\textrm{out}}{\partial \bar w} & = & \frac{\partial a^L}{\partial \bar w} \\
& = &
\frac{\partial a^L}{\class{highlight}{\partial a^{L-1}}} 
&\cdot& 
\class{highlight}{\frac{\partial a^{L-1}}{\partial a^{L-2}}} 
&\cdot& \ldots &\cdot& 
\class{highlight}{\frac{\partial a^{\bar l+2}}{\partial a^{\bar l+1}}} &\cdot&
\frac{\class{highlight}{\partial a^{\bar l+1}}}{\partial \bar w}
\end{array}
$$`
and
`$$
\begin{array}{ccccccccccc}
\frac{\partial a^\textrm{out}}{\partial \bar b} & = & \frac{\partial a^L}{\partial \bar b}\\
& = &
\frac{\partial a^L}{\class{highlight}{\partial a^{L-1}}} 
&\cdot& 
\class{highlight}{\frac{\partial a^{L-1}}{\partial a^{L-2}}} 
&\cdot& \ldots &\cdot& 
\class{highlight}{\frac{\partial a^{\bar l+2}}{\partial a^{\bar l+1}}} &\cdot&
\frac{\class{highlight}{\partial a^{\bar l+1}}}{\partial \bar b}
\end{array}
$$`

> [!NOTE]
> The terms $\frac{\partial a^{l+1}}{\partial a^{l}}$ indicate how much the activation values of neurons in layer $l+1$ change if we change the activation values of neurons in layer $l$.

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
where $h=|I^l|$, $k=|I^{l+1}|$, and $f_i$ is the activation function of the $i$-th neuron in layer $l+1$. 
</div>


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

`$\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial \bar w}$` and `$\frac{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial \bar b}$` 

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

