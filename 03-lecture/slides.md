> [!CAUTION]
> Not yet complete!

---


# Feedforward neural networks

---

## Feedforward neural network with 4 layers

<div class="neuralnetwork predictions" style="height: 700px; width: 1280px!important;">
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
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^L_{i,j} } =  \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{L}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{L}_i}{w^L_{i,j}  } = 2(a^L_i - a^*_i) \cdot a^{L-1}_j.$$`
- For each output neuron $i$, we have
`$$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^L_{i} } =  \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{L}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{L}_i}{\partial b^{L}_j} = 2(a^L_i - a^*_i)$$`

> [!NOTE]
> Apart from notational differences, this is the same as for the single layer neural network studied in the last session.

---

### Derivatives for values of hidden layers

A value change in a hidden layer can change **all** activations in the final layer. 

<div class="neuralnetwork predictions" style="height: 600px; width: 1280px!important;">
<!--
{"type": "feedforward" }
-->
</div>


---

### Derivatives for weights in hidden layers

For each neuron $i$ of layer $l$ and each neuron $j$ of layer $l-1$, the chain rule implies that

`$$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{i,j} } = 
\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i }
\cdot  \displaystyle\genfrac{}{}{1pt}{1}{\partial a^l_i}{\partial z^l_i }
 \cdot \displaystyle\genfrac{}{}{1pt}{1}{\partial z^l_i}{\partial w^l_{i,j} } $$` 


`$$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{i,j} }} = \class{highlight}{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i }} \cdot  \genfrac{}{}{1pt}{1}{\partial \sigma^l_i(z^l_i)}{\partial z^l_i } \cdot a^{l-1}_j $$` 

> [!NOTE]
> 
> For ReLU activation, we have `$\sigma^l_i(z^l_i) = \max \lbrace 0, z^l_i \rbrace$` and 
> `$$\genfrac{}{}{1pt}{1}{\partial \sigma^l_i(z^l_i)}{\partial z^l_i } = \left\lbrace \begin{align}1 \textrm{ if } z^l_i > 0 \\ 0 \textrm{ if } z^l_i \leq 0 \end{align}\right.$$`

---

### Derivatives for biases in hidden layers

For each neuron $i$ of layer $l$, the chain rule implies that

`$$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_i } = 
\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } 
\cdot \displaystyle\genfrac{}{}{1pt}{1}{\partial a^l_i}{\partial z^l_i }
\cdot \displaystyle\genfrac{}{}{1pt}{1}{\partial z^l_i}{\partial b^l_i }$$`  


`$$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_j }} = \class{highlight}{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i }} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^l_i(z^l_i)}{\partial z^l_i } \cdot 1$$`  

> [!NOTE]
> 
> For ReLU activation, we have `$\sigma^l_i(z^l_i) = \max \lbrace 0, z^l_i \rbrace$` and 
> `$$\genfrac{}{}{1pt}{1}{\partial \sigma^l_i(z^l_i)}{\partial z^l_i } = \left\lbrace \begin{align}1 \textrm{ if } z^l_i > 0 \\ 0 \textrm{ if } z^l_i \leq 0 \end{align}\right.$$`

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs" style="display:inline-block;"-->
`$= \sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial a^{l-1}_j }$`

`$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }} = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial z^l_i } \cdot \genfrac{}{}{1pt}{1}{\partial z^{l}_i}{\partial a^{l-1}_j }$`

`$\hphantom{\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }}$`
`$ = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_i)}{\partial z^l_i } \cdot w^{l}_{i,j}$`<!-- .element: data-id="rhs" style="display:inline-block;"-->

> [!NOTE]
> **Multivariable chain rule:** Given a function $f( g_1(x), \ldots, g_n(x) )$, we have $\genfrac{}{}{1pt}{1}{\partial f}{\partial x }  = \displaystyle\sum_{i=1}^n \genfrac{}{}{1pt}{1}{\partial f}{\partial g_i } \cdot \genfrac{}{}{1pt}{1}{\partial g_i}{\partial x }$. 


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

We can rewrite 
`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs" style="display:inline-block;"-->
`$ = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_i)}{\partial z^l_i } \cdot w^{l}_{i,j}$`<!-- .element: data-id="rhs" style="display:inline-block;"-->

as

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j } = ( w^l_{1,j}, w^l_{2,j}, \ldots, w^l_{n^l,j} ) \cdot $` 
`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_2(z^l_2)}{\partial z^l_1 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`<!-- .element: data-id="factor" style="display:inline-block;"-->


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For
`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 }$`, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 }$`, $\ldots$, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} }$`, we have


as

<span style="white-space: nowrap;">

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =$``<!-- .element: data-id="lhs" style="display:inline-block;"-->
`$\left( \begin{array}{cccc}
w^l_{1,1} & w^l_{2,1} & \ldots & w^l_{n^l,1} \\
w^l_{1,2} & w^l_{2,2} & \ldots & w^l_{n^l,2} \\
\vdots \\
w^l_{1,n^{l-1}} & w^l_{2,n^{l-1}} & \ldots & w^l_{n^l,n^{l-1}} \\
\end{array}
\right) \cdot $` 
`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`<!-- .element: data-id="factor" style="display:inline-block;"-->

</span>

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For
`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 }$`, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 }$`, $\ldots$, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} }$`, we have

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =$``<!-- .element: data-id="lhs" style="display:inline-block;"-->
`$(W^l)^T \cdot 
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \sigma^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For
`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 }$`, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 }$`, $\ldots$, `$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} }$`, we have

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =
(W^l)^T \cdot 
\left( \left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}}
\end{array}
\right)
\odot
\left(
\begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \sigma^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \sigma^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \sigma^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right) \right)$`


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

