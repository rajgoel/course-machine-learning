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
> Remember, that gradient descent works by iteratively changing weights and biases in opposite direction of the average gradient of the loss. To compute the gradient, we need the derivatives for **all** weights and biases.

---

### Derivatives for weights and biases in the last layer

- For each output neuron $i$ and each neuron $j$ in the last hidden layer, we have
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^L_{i,j} } =  \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{\partial a^{L}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{L}_i}{w^L_{i,j}  } = 2(a^L_i - a^*_i) \cdot a^{L-1}_j.$$`
- For each output neuron $i$, we have
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^L_{i} } =  \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{\partial a^{L}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{L}_i}{\partial b^{L}_j} = 2(a^L_i - a^*_i)$$`

> [!NOTE]
> Apart from notational differences, this is the same as for the single layer neural network studied in the last session.

---

### Total derivatives

Changing a weight or bias of a hidden layer ($l < L$), does not only change the activations of the layer, but also the activations of subsequent layers. 

Instead of using the partial derivative for weights and biases of hidden layers, we use the **total derivative** for gradient descent.

> [!NOTE]
> **Total derivative:** The [total derivative](https://www.geeksforgeeks.org/engineering-mathematics/total-derivative/) of a function $f$ measures the rate of change of a function with respect to one variable while considering the effect of all other variables changing as well. We write $\frac{df}{dx}$ for the total derivative of $f$ with respect to $x$.


---

### Derivatives for weights in hidden layers

For each neuron $i$ of layer $l$ and each neuron $j$ of layer $l-1$, the chain rule implies that

`$$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d w^l_{i,j} } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i }
\cdot  \displaystyle\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i }
 \cdot \displaystyle\genfrac{}{}{1pt}{1}{d z^l_i}{d w^l_{i,j} } $$` 


`$$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d w^l_{i,j} }} = \class{highlight}{\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i }} \cdot  \genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i } \cdot a^{l-1}_j $$` 

> [!NOTE]
> 
> For ReLU activation, we have `$\sigma^l_i(z^l_i) = \max \lbrace 0, z^l_i \rbrace$` and 
> `$$\genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i } = \left\lbrace \begin{align}1 \textrm{ if } z^l_i > 0 \\ 0 \textrm{ if } z^l_i \leq 0 \end{align}\right.$$`

---

### Derivatives for biases in hidden layers

For each neuron $i$ of layer $l$, the chain rule implies that

`$$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d b^l_i } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i } 
\cdot \displaystyle\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i }
\cdot \displaystyle\genfrac{}{}{1pt}{1}{d z^l_i}{d b^l_i }$$`  


`$$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d b^l_j }} = \class{highlight}{\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i }} \cdot \genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i } \cdot 1$$`  

> [!NOTE]
> 
> For ReLU activation, we have `$\sigma^l_i(z^l_i) = \max \lbrace 0, z^l_i \rbrace$` and 
> `$$\genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i } = \left\lbrace \begin{align}1 \textrm{ if } z^l_i > 0 \\ 0 \textrm{ if } z^l_i \leq 0 \end{align}\right.$$`

---

### Derivatives for activation values of hidden layers

According to the multivariable chain rule, we have

`$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l-1}_j } = \sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{d a^{l}_i}{d a^{l-1}_j }$`

`$
= \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{d a^{l}_i}{d z^l_i } \cdot \genfrac{}{}{1pt}{1}{d z^{l}_i}{d a^{l-1}_j }
$`
`$
= \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{d \sigma^{l}_i(z^l_i)}{d z^l_i } \cdot w^{l}_{i,j}
$`

> [!NOTE]
> **Multivariable chain rule:** Given a function $f( g_1(x), \ldots, g_n(x) )$, we have $\genfrac{}{}{1pt}{1}{d f}{d x }  = \displaystyle\sum_{i=1}^n \genfrac{}{}{1pt}{1}{d f}{d g_i } \cdot \genfrac{}{}{1pt}{1}{d g_i}{d x }$. 



---

## Overview

We have

- `$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^L_{i} } = 2(a^L_i - a^*_i)$`
- `$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^L_{i,j} } = 2(a^L_i - a^*_i) \cdot a^{L-1}_j$`
- `$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^L_{i} } = 2(a^L_i - a^*_i)$`
- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l-1}_j } = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{d \sigma^{l}_i(z^l_i)}{d z^l_i } \cdot w^{l}_{i,j}
$`
- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d w^l_{i,j} } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i } 
\cdot \genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i }
\cdot a^{l-1}_j$`
- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d b^l_i } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i } 
\cdot \genfrac{}{}{1pt}{1}{d \sigma^l_i(z^l_i)}{d z^l_i }$`  
- `$\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d w^l_{i,j} } = 
\displaystyle\genfrac{}{}{1pt}{1}{d \mathscr{L}_{(a,a^*)}}{d a^l_i }
\cdot  \displaystyle\genfrac{}{}{1pt}{1}{d a^l_i}{d z^l_i }
 \cdot \displaystyle\genfrac{}{}{1pt}{1}{d z^l_i}{d w^l_{i,j} } $` 


===

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

