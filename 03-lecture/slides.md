
# Feedforward neural networks and backpropagation

---

## A neural network with 4 layers

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
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
- $\phi^l$ denote the vector of activation functions.

Then, the activation values of layer $l$ can be computed by

`$$z^l = W^l a^{l-1} + b^l,$$`
`$$a^{l} = \phi^l(z^l).$$`

---

<!-- .slide: data-fullscreen="yes"  -->

### Forward propagation in Julia

```julia[1-26|2-8|10-23|12-14|16-22|1-26]
function forwardpropagation(network::VanillaDNN, x::Vector{Float32})
    # Initialize storage for activations and z-values
    activations = OffsetVector(Vector{Float32}[], 0:-1)
    z_values = Vector{Vector{Float32}}()
    
    # Input layer: a^0 = x
    a = copy(x)
    push!(activations, a)
    
    # Forward through hidden layers and output layer
    for l in 1:network.L  # l = 1, 2, ..., L
        # Linear transformation: z^l = W^l * a^[l-1] + b^l
        z = network.W[l] * a + network.b[l]
        push!(z_values, z)
        
        # Activation: a^l = ϕ(z^l)
        if l == network.L  # Output layer
            a = z  # Linear output
        else  # Hidden layers
            a = ϕ.(z)
        end
        push!(activations, a)
    end

    return activations, z_values
end
```
<!-- .element: class="fullscreen stretch" -->

---

### Loss

For a given input/output pair  $(a,a^*)$, the sum of squared errors of a feedforward neural network is

`$$\mathscr{L}_{(a,a^*)}(W^1,b^1,\ldots,W^{L},b^{L}) = \sum_{i=1}^{n^L}(a^L_i - a^*_i)^2.$$`

where $n^L$ denotes the number of neurons in the output layer $L$.

===

### Gradient descent for feed forward networks

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!NOTE]
> Remember, that gradient descent works by iteratively changing weights and biases in opposite direction of the average gradient of the loss. To compute the gradient, we need the derivatives for **all** weights and biases.

===

### Derivatives for activations of the last layer

For each output neuron $i$, we have  

`$\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{L}_i} =  2(a^L_i - a^*_i)$`

> [!NOTE]
> Remember, the loss is `$\mathscr{L}_{(a,a^*)} = \displaystyle\sum_{i=1}^{n^L} (a^L_i - a^*_i)^2$`.


===

### Derivatives for values of hidden layers

A value change in a hidden layer can change **all** activations in the final layer. 

<div class="neuralnetwork" style="height: 600px; width: 1280px!important;">
<!--
{"type": "feedforward" }
-->
</div>

===

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For neuron $j$ in layer $l-1$, we have

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs-j" -->
`$= \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial a^{l-1}_j }$`<!-- .element: data-id="chainrule" -->

> [!NOTE]
> **Multivariable chain rule:** Given a function $f( g_1(x), \ldots, g_n(x) )$, we have $\genfrac{}{}{1pt}{1}{\partial f}{\partial x }  = \displaystyle\sum_{i=1}^n \genfrac{}{}{1pt}{1}{\partial f}{\partial g_i } \cdot \genfrac{}{}{1pt}{1}{\partial g_i}{\partial x }$. 


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For neuron $j$ in layer $l-1$, we have

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs-j" -->
`$= \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial a^{l-1}_j }$`<!-- .element: data-id="chainrule" -->

`$= \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial z^l_i } \cdot \genfrac{}{}{1pt}{1}{\partial z^{l}_i}{\partial a^{l-1}_j }$`<!-- .element: style="margin-left:150px;"-->

<span class="fragment">

`$ = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_i)}{\partial z^l_i } \cdot w^{l}_{i,j}$`<!-- .element: data-id="rhs-j" style="margin-left:150px;" -->

> [!IMPORTANT]
> To unify notation we use $\phi^{L}_i(z^L_i) = z^L_i$ for the last layer.

</span>

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For neuron $j$ in layer $l-1$, we have

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs-j" -->
`$ = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_i)}{\partial z^l_i } \cdot w^{l}_{i,j}$`<!-- .element: data-id="rhs-j"-->

> [!NOTE]
> 
> For [ReLU activation](https://en.wikipedia.org/wiki/Rectified_linear_unit), we have `$\phi^l_i(z^l_i) = \max \lbrace 0, z^l_i \rbrace$` and use
> `$$\genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i } = \begin{cases}1 \textrm{ if } z^l_i > 0 \\ \class{highlight}{0 \textrm{ if } z^l_i = 0 \textsf{ (formally undefined!)}} \\ 0 \textrm{ if } z^l_i < 0 \end{cases}$$`


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

We can rewrite 
`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j }$`<!-- .element: data-id="lhs-j" -->
`$ = \displaystyle\sum_{i=1}^{n^{l}} \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_i)}{\partial z^l_i } \cdot w^{l}_{i,j}$`<!-- .element: data-id="rhs-j"-->

as

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_j } =$`<!-- .element: data-id="lhs" --> 
`$( w^l_{1,j}, w^l_{2,j}, \ldots, w^l_{n^l,j} )\ \cdot $`<!-- .element: data-id="weights" --> 
`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_1 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`<!-- .element: data-id="factor" -->


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For layer $l - 1$, we have

<span style="white-space: nowrap;">

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =$`<!-- .element: data-id="lhs" -->
`$\left( \begin{array}{cccc}
w^l_{1,1} & w^l_{2,1} & \ldots & w^l_{n^l,1} \\
w^l_{1,2} & w^l_{2,2} & \ldots & w^l_{n^l,2} \\
\vdots \\
w^l_{1,n^{l-1}} & w^l_{2,n^{l-1}} & \ldots & w^l_{n^l,n^{l-1}} \\
\end{array}
\right) \cdot $`<!-- .element: data-id="weights" --> 
`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`<!-- .element: data-id="factor" -->

</span>

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For layer $l - 1$, we have

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =$`<!-- .element: data-id="lhs" -->
`$(W^l)^T \cdot $`<!-- .element: data-id="weights" --> 
`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}} \cdot \genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right)$`<!-- .element: data-id="factor" -->

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for activation values of hidden layers

For layer $l - 1$, we have

`$
\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_1 } \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l-1}_{n^{l-1}} } \\
\end{array}
\right) =$`<!-- .element: data-id="lhs" -->
`$(W^l)^T \cdot$`<!-- .element: data-id="weights" --> 
`$\left( \left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}}
\end{array}
\right)
\odot
\left(
\begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right) \right)$`<!-- .element: data-id="factor" -->

===

<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For each neuron $i$ of layer $l$ and each neuron $j$ of layer $l-1$, the chain rule implies that

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{i,j} }$`<!-- .element: data-id="lhs-w" -->
`$ = \displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } \cdot  \displaystyle\genfrac{}{}{1pt}{1}{\partial a^l_i}{\partial z^l_i } \cdot \displaystyle\genfrac{}{}{1pt}{1}{\partial z^l_i}{\partial w^l_{i,j} }$` 

`$= \displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i }  $`<!-- .element: class="fragment"  data-fragment-index="1" data-id="rhs-w" style="margin-left:150px;" -->
`$\cdot a^{l-1}_j $`<!-- .element: class="fragment"  data-fragment-index="1" data-id="rhs-a" -->

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For each neuron $i$ of layer $l$ and each neuron $j$ of layer $l-1$, the chain rule implies that

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{i,j} }$`<!-- .element: data-id="lhs-w" -->
`$= \displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i } $`<!-- .element:  data-id="rhs-w" -->
`$\cdot a^{l-1}_j $`<!-- .element:  data-id="rhs-a" -->

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For layer $l$ and each neuron $j$ of layer $l-1$, we have

`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,j} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,j} }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},j} }
\end{array}
\right)$`<!-- .element: data-id="lhs-w" -->
`$=\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_1 }\\ 
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_2 }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^{l}}} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_{n^{l}} } 
\end{array}
\right)$`<!-- .element:  data-id="rhs-w" -->
`$\cdot a^{l-1}_j$`<!-- .element: data-id="rhs-a" -->

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For layer $l$, we have

<span style="white-space: nowrap;">

`$\left( \begin{array}{cccc}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,n^{l-1}} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,n^{l-1}} }\\
\vdots & \vdots & \ddots& \vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},n^{l-1}} }
\end{array}
\right)$`<!-- .element: data-id="lhs-w" -->
`$=\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_1 }\\ 
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_2 }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^{l}}} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_{n^{l}} } 
\end{array}
\right)$`<!-- .element: data-id="rhs-w" -->
`$\cdot (a^{l-1}_1,\ldots,a^{l-1}_{n^{l-1}})$`<!-- .element: data-id="rhs-a" -->

</span>

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For layer $l$, we have

<span style="white-space: nowrap;">

`$\left( \begin{array}{cccc}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,n^{l-1}} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,n^{l-1}} }\\
\vdots & \vdots & \ddots& \vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},n^{l-1}} }
\end{array}
\right)$`<!-- .element: data-id="lhs-w" -->
`$=\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_1 }\\ 
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_2 }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^{l}}} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_{n^{l}} } 
\end{array}
\right)$`<!-- .element: data-id="rhs-w" -->
`$\cdot (a^{l-1})^T$`<!-- .element: data-id="rhs-a" -->

</span>

---


<!-- .slide: data-auto-animate="true" -->

### Derivatives for weights

For layer $l$, we have

<span style="white-space: nowrap;">

`$\left( \begin{array}{cccc}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{1,n^{l-1}} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{2,n^{l-1}} }\\
\vdots & \vdots & \ddots& \vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},1} } & \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},2} } & \ldots& \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w^l_{n^{l},n^{l-1}} }
\end{array}
\right)$`<!-- .element: data-id="lhs-w" -->
`$= \left( \left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}}
\end{array}
\right)
\odot
\left(
\begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right) \right)$`<!-- .element: data-id="rhs-w" -->
`$\cdot (a^{l-1})^T$`<!-- .element: data-id="rhs-a" -->

</span>

===

<!-- .slide: data-auto-animate="true" -->

### Derivatives for biases

For each neuron $i$ of layer $l$, we have

`$\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{i} }$`<!-- .element: data-id="lhs-b" --> 
`$=  \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i}  \cdot \genfrac{}{}{1pt}{1}{\partial a^{l}_i}{\partial z^l_j} \cdot \genfrac{}{}{1pt}{1}{\partial z^{l}_i}{\partial b^l_j}$`

`$= \displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i } \cdot \genfrac{}{}{1pt}{1}{\partial z^l_i}{\partial b^l_i } $`<!-- .element: class="fragment" style="margin-left:180px;"-->

`$= \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_i} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i }$`<!-- .element: data-id="rhs-b" class="fragment"  style="margin-left:100px;"-->


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for biases

For each neuron $i$ of layer $l$, we have

`$\displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_i }$`<!-- .element: data-id="lhs-b" -->
`$= \displaystyle\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^l_i } \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i } $`<!-- .element:  data-id="rhs-b" -->


---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for biases

For layer $l$, we have

`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{1} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{2} }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{n^{l}} }
\end{array}
\right)$`<!-- .element: data-id="lhs-b" -->
`$=\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_1(z^l_1)}{\partial z^l_1 }\\ 
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_2(z^l_2)}{\partial z^l_2 }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^{l}}} \cdot  \genfrac{}{}{1pt}{1}{\partial \phi^l_{n^{l}}(z^l_{n^{l}})}{\partial z^l_{n^{l}} } 
\end{array}
\right)$`<!-- .element: data-id="rhs-b" -->

---

<!-- .slide: data-auto-animate="true" -->

### Derivatives for biases

For layer $l$, we have

`$\left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{1} }\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{2} }\\
\vdots\\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b^l_{n^{l}} }
\end{array}
\right)$`<!-- .element: data-id="lhs-b" -->
`$=\left( \left( \begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_1} \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_2} \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial a^{l}_{n^l}}
\end{array}
\right)
\odot
\left(
\begin{array}{c}
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_i(z^l_1)}{\partial z^l_1 } \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_2(z^l_2)}{\partial z^l_2 } \\
\vdots \\
\genfrac{}{}{1pt}{1}{\partial \phi^{l}_{n^l}(z^l_{n^l})}{\partial z^l_{n^l} }
\end{array}
\right) \right)$`<!-- .element: data-id="rhs-b" -->


===

<!-- .slide: data-fullscreen="yes"  -->

### Backpropagation in Julia

```julia[1-31|4-13|15-28|30|1-31]
function backpropagation(network::VanillaDNN, activations::OffsetVector{Vector{Float32}}, 
                        z_values::Vector{Vector{Float32}}, y::Vector{Float32})

    ∇W = Matrix{Float32}[]
    ∇b = Vector{Float32}[]

    # Output error
    â = activations[end]
    δ = ∂ℒ_∂â(y, â)

    # Gradient for output layer
    pushfirst!(∇W, δ * activations[network.L-1]')
    pushfirst!(∇b, δ ) # ∂ℒ/∂b = δ

    # Backpropagation through hidden layers
    for l in network.L-1:-1:1 
        # Compute
        # - ∂ℒ_∂a^l = W^[l+1]' ∂ℒ/∂a^[l+1] for l = L-1 
        # - ∂ℒ_∂a^l = W^[l+1]' (∂ℒ/∂a^[l+1] ⨀ ∂ϕ^l/∂z^[l+1])  for l < L-1 
        δ = network.W[l+1]' * δ

        # Compute δ = ∂ℒ/∂a^l ⨀ ∂ϕ^l/∂z^l
        δ .*= ∂ϕ_∂z.(z_values[l])

        # Gradient for layer l
        pushfirst!(∇W, δ * activations[l-1]')
        pushfirst!(∇b, δ)
    end

    return ∇W, ∇b
end
```
<!-- .element: class="fullscreen stretch" -->

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
<canvas class="drawDigit" width="150" height="150" data-prevent-swipe>
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

---


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

> [!IMPORTANT]
> The neural network will predict a digit for any input, even if no digit is provided!


===

### MNIST database

The [MNIST database](https://yann.lecun.com/exdb/mnist/) contains gray scale values of 28x28 pixel images representing handwritten digits. 

![Digits](03-lecture/digits.jpg)

---

### Implementation

You find a full implementation for recognising handwritten digits in the [course repository](https://rajgoel.github.io/course-machine-learning/julia).

> [!TIP]
> Run:
> ```julia
> using MachineLearningCourse
> Lecture03.demo()
> ```
> Type `?Lecture03.demo()` for an overview over all (optional) parameters.

===

## Machine learning library

[Flux.jl](https://fluxml.ai/Flux.jl/stable/) is an intuitive and powerful machine learning library in Julia.

```julia
using Pkg
Pkg.add("FLux")
```

> ![NOTE]
> For Python, there is [PyTorch](https://docs.pytorch.org/docs/stable/) which is based on C++ implementation of [LibTorch](https://docs.pytorch.org/cppdocs/).

---

<!-- .slide: data-fullscreen="yes"  -->

### Minimal MNIST implementation with Flux.jl

```julia[1|16|20-27|29-34|36-49|51-57]
using Flux

"""
    flux_demo(;train_size=1000, epochs=10)

Minimal MNIST demonstration using gradient descent.

# Arguments  
- `train_size::Int`: Number of training samples (default: 5000)
- `test_size::Int`: Number of test samples (default: 1000)
- `epochs::Int`: Number of training epochs (default: 1000)

# Returns
- Trained model and final accuracy
"""
function flux_demo(; train_size::Int=5000, test_size::Int=1000, epochs::Int=1000)
    println("Minimal Flux.jl MNIST Demo")
    println("=" ^ 30)
    
    X_train, Y_train, X_test, Y_test = load_mnist_data(train_size, test_size)
    
    # Convert to Flux format (features × samples) and Float32
    X_train_flux = Float32.(reshape(X_train, 784, train_size))  # 784 × train_size
    Y_train_flux = Float32.(Y_train)     # 10 × train_size  
    X_test_flux = Float32.(reshape(X_test, 784, test_size))    # 784 × test_size
    Y_test_flux = Float32.(Y_test)       # 10 × test_size
    
    # Create simple model
    model = Flux.Chain(
        Flux.Dense(784 => 128, Flux.relu),
        Flux.Dense(128 => 64, Flux.relu),
        Flux.Dense(64 => 10)
    )
    
    # Setup training
    opt_state = Flux.setup(Flux.Adam(0.001), model)
    
    println("Training...")
    
    # Training loop using Flux.train!
    for epoch in 1:epochs
        Flux.train!(model, [(X_train_flux, Y_train_flux)], opt_state) do m, x, y
            Flux.logitcrossentropy(m(x), y)
        end
        
        train_loss = Flux.logitcrossentropy(model(X_train_flux), Y_train_flux)
        println("Epoch $epoch: Loss = $(round(train_loss, digits=4))")
    end
    
    # Evaluate
    predictions = Flux.softmax(model(X_test_flux))  # Apply softmax for predictions
    pred_classes = [argmax(predictions[:, i]) - 1 for i in 1:size(predictions, 2)]
    true_classes = [argmax(Y_test_flux[:, i]) - 1 for i in 1:size(Y_test_flux, 2)]
    
    accuracy = sum(pred_classes .== true_classes) / length(pred_classes)
    println("Test Accuracy: $(round(accuracy * 100, digits=1))%")
    
    return model, accuracy
end
```
<!-- .element: class="fullscreen stretch" -->

---

You can find the implementation in the [course repository](https://rajgoel.github.io/course-machine-learning/julia).

> [!TIP]
> Run:
> ```julia
> using MachineLearningCourse
> Lecture03.flux_demo()

