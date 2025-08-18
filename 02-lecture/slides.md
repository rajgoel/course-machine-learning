# Neural networks and gradient descent

===

## Not so simple world: Digits with 5x5 black and white pixel input

<object data="digitrecognition/5x5digits/svg/digit_0a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_1a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_2a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_3a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_4a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_5a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_6a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_7a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_8a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_9a.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<br>
<object data="digitrecognition/5x5digits/svg/digit_0b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_1b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_2b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_3b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_4b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_5b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_6b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_7b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_8b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_9b.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<br>
<object data="digitrecognition/5x5digits/svg/digit_0c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_1c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_2c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_3c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_4c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_5c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_6c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_7c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_8c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
<object data="digitrecognition/5x5digits/svg/digit_9c.svg" type="image/svg+xml" style="filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>

---

### Input encoding

The input is represented by an encoding `$(a^\textrm{in}_{1}, a^\textrm{in}_{2}, \ldots, a^\textrm{in}_{25})$`.

<table style="table-layout: fixed!important;width:600px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:80px!important;">
<object data="digitrecognition/5x5digits/svg/digit_1a.svg" type="image/svg+xml" style="transform:scale(1.5);filter: drop-shadow(4px 4px 4px rgba(0, 0, 0, 0.5));margin:6px;"></object>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:50px;width:400px!important;">
$$
\begin{align}
( & 0,0,1,0,0,  \\
&0,1,1,0,0,  \\
&0,0,1,0,0,  \\
&0,0,1,0,0,  \\
& 0,0,1,0,0 )
\end{align}
$$
</td>
</tr>
</table>

---

### Output encoding

The output is represented by an encoding  `$(a^\textrm{out}_{0}, a^\textrm{out}_{1}, \ldots, a^\textrm{out}_{9})$`.

<table style="table-layout: fixed!important;width:900px;">
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:150px!important;">
<svg width="150" height="150">
  <rect width="150" height="150" style="fill:black;" />
  <text  x="75" y="125" font-size="150" text-anchor="middle"  style="fill:white;stroke:white;">0</text>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:70px;width:400px!important;">
(1,0,0,0,0,0,0,0,0,0) 
</td>
</tr>
<tr style="border: 0;border-style:hidden;">
<td style="vertical-align: middle;padding:0;width:100px!important;">
<svg width="150" height="150">
  <rect width="150" height="150" style="fill:black;" />
  <text  x="75" y="125" font-size="150" text-anchor="middle"  style="fill:white;stroke:white;">1</text>
</svg>
</td>
<td style="vertical-align: middle;padding:0;font-size:100px;width:200px!important;">
<i class="fas fa-arrow-right" style="padding:100px;"></i>
</td>
<td style="vertical-align: middle;padding-bottom:50px;:0;font-size:70px;width:400px!important;">
(0,1,0,0,0,0,0,0,0,0) 
</td>
</tr>
</table>
and so on.

---

### A simple neural network

<table  style="width:400px;">
<tr>
<td style="vertical-align: middle;width:50px;position:relative;left:150px;">
$$
\left(
\begin{align}
a^\textrm{in}_{1} \\
a^\textrm{in}_{2} \\
\vdots \\
a^\textrm{in}_{25}
\end{align}
\right)
$$
</td>
<td style="width:600px!important;position:relative;">
<div class="neuralnetwork" style="height: 500px; width: 600px;">
<!--
{"type": "5x5" }
-->
</div>
</td>
<td style="vertical-align: middle;width:50px;position:relative;left:-150px;">
$$ \left( \begin{align} a^\textrm{out}_{0} \\  a^\textrm{out}_{1} \\ \vdots \\ a^\textrm{out}_{9}  \end{align} \right) $$
</td>
</tr>
</table>

---

### Linear activation function

Given a 

- *weight* $w_{i,j}$  for each output $i$ and input $j$, and 
- a bias $b_i$ for each output $i$, 

we can determine the output activation values as a linear combination of the input activation values by

`$$ a^\textrm{out}_i = \displaystyle\sum_{j=1}^{25} w_{i,j} a^\textrm{in}_{j} + b_i$$`

for all `$i \in \{ 0,1, \ldots, 9\}$`.

---

Assume we are given a pairs of input activation values $a^\textrm{in}$ with their expected output activation values $a^*$. 

---

If we could find weights and biases such that for each pair $(a^\textrm{in},a^*)$ we have

`$$ \underbrace{ W a^\textrm{in} + b }_{a^\textrm{out}} = a^* $$`

or 

`$$
\underbrace{
\left(
\begin{array}{cccc}
w_{0,1} & w_{0,2} & \ldots & w_{0,25} \\
w_{1,1} & w_{1,2} & \ldots & w_{1,25} \\
\vdots & \vdots & \ddots & \vdots \\
w_{9,1} &  w_{9,2} & \ldots & w_{9,25} \\
\end{array}
\right)
\left(
\begin{array}{c}
a^\textrm{in}_{1} \\
a^\textrm{in}_{2} \\
\vdots \\
a^\textrm{in}_{25}
\end{array}
\right) + \left(
\begin{array}{c}
b_{0} \\
b_{1} \\
\vdots \\
b_{9} \\
\end{array}
\right)
}_{a^\textrm{out}} = \left(
\begin{array}{c}
a^*_0 \\
a^*_1 \\
\vdots \\
a^*_9
\end{array}
\right)
$$`

our neural network would be able to perfectly recognise the given digits. 

---

### Determining weights and biases

- Our neural network has 25 $\cdot$ 10 weights and 10 bias values, thus, a total of 260 parameters.
- For each  pair $(a^\textrm{in},a^*)$ we have 10 equations for a perfect neural network.
- With 30 pairs, we would have 30 $\cdot$ 10 = 300 equations.

A system of equations with 300 equations and 260 variables (ie. parameters of the neural network) cannot be solved (unless some of the equations are redundant). <!-- .element: class="fragment" -->


---

### Error function

For each given pair $(a^\textrm{in},a^*)$ we can define an error function as the sum of the squared differences of the computed output and the expected output.

`$$f^\textrm{error}_{(a^\textrm{in},a^*)}(W,b) =
\displaystyle\sum_{i=0}^{9} 
\Big(
\underbrace{\displaystyle\sum_{j=1}^{25} w_{i,j} a^\textrm{in}_{j} + b_i
}_{a^\textrm{out}_i} - a^*_i
\Big)^2
$$`



---

Our goal is to find weights and biases that minimize the sum of squared errors for all given pairs $(a^\textrm{in},a^*)$.

===

## Gradient descent

---

## Mean error

Let $S$ denote the set of samples given. For each $(a^\textrm{in},a^*) \in S$ let  $f^\textrm{error}_{(a^\textrm{in},a^*)}(W,b)$ denote the error function for the sample.

The mean error is
`$$f^\textrm{error}(W,b) = \displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a^\textrm{in},a^*) \in S} f^\textrm{error}_{(a^\textrm{in},a^*)}(W,b).
$$`

We want to find weights and biases minimizing the mean error.
<!-- .element: class="fragment" -->

===

## Gradient descent

The gradient $\nabla f$ of a function $f$ gives the direction of the steepest ascend. 

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!NOTE]
> By changing weights and biases in opposite direction of the gradient, we can minimise the error function.

---

### Gradient of the mean error

The gradient of the mean error
`$$f^\textrm{error}(W,b) = \displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a^\textrm{in},a^*) \in S} f^\textrm{error}_{(a^\textrm{in},a^*)}(W,b).
$$`
is the mean of the gradients over all samples 
`$$
\nabla f^\textrm{error}(W,b) =
\displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a^\textrm{in},a^*) \in S} \nabla f^\textrm{error}_{(a^i,a^*)}(W,b)
$$`

---

### Partial derivatives

The gradient of $f^\textrm{error}_{(a^\textrm{in},a^*)}$ is the collection of all its partial derivatives

`$
\genfrac{}{}{1pt}{1}{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial w_{i,j} }
\quad$`
and
`$
\quad\genfrac{}{}{1pt}{1}{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial b_i }
$`

for each output neuron $i \in I$ and each input neuron $j \in J$.

---

### Partial derivative for weights

For each output neuron $i \in I$ and each input neuron $j \in J$ the partial derivative of
`$$
f^\textrm{error}_{(a^\textrm{in},a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a^\textrm{in}_{j} + b_i
}_{a^\textrm{out}_i} - a^*_i
\Big)^2
$$`
with respect to $w_{i,j}$ is

`$$
\genfrac{}{}{1pt}{1}{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial w_{i,j} }
= 2\Big(\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a^\textrm{in}_{j} + b_i
}_{a^\textrm{out}_i} - a^*_i\Big) \cdot a^\textrm{in}_j
= 2(a^\textrm{out}_i - a^*_i)a^\textrm{in}_j
$$`

---

### Partial derivative for biases

For each output neuron $i \in I$ the partial derivative of
`$$
f^\textrm{error}_{(a^\textrm{in},a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a^\textrm{in}_{j} + b_i
}_{a^\textrm{out}_i} - a^*_i
\Big)^2
$$`
with respect to $b_i$ is

`$$
\genfrac{}{}{1pt}{1}{\partial f^\textrm{error}_{(a^\textrm{in},a^*)}}{\partial b_i }
= 2\Big(\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a^\textrm{in}_{j} + b_i
}_{a^\textrm{out}_i} - a^*_i\Big) \cdot 1
= 2(a^\textrm{out}_i - a^*_i)
$$`

---

<!-- .slide: data-fullscreen="yes"  -->

### Gradient descent in Julia

```julia
function gradient_descent(initial_weights, initial_biases, samples, inputs, outputs)
    weights = copy(initial_weights)
    biases = copy(initial_biases)
    
    tolerance = 1.0e-3
    max_iterations = 1e4
    learning_rate = .1
    
    for iter in 1:max_iterations
        # Compute the average gradient
        avg_gradient_weights, avg_gradient_biases = get_average_gradient(weights, biases, samples, inputs, outputs)
        norm_gradient = get_norm(avg_gradient_weights, avg_gradient_biases)
        
        println("Iteration $iter, Norm of gradient: $norm_gradient")
        
        # Break if the gradient norm is smaller than the tolerance
        if norm_gradient < tolerance
            println("Gradient norm below tolerance. Stopping.")
            break
        end
        
        # Update weights and biases using gradient descent
        weights .-= learning_rate .* avg_gradient_weights
        biases .-= learning_rate .* avg_gradient_biases
    end
    
    return weights, biases
end
```
<!-- .element: class="fullscreen stretch" -->



===

## Digit recognition with 5x5 pixel input

Can we use the weights and biases we obtained for the 5x5 pixel input to recognise hand-written digits? 

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

### Pooling

Pooling is a technique to reduce the dimensions of an input while preserving important features or patterns.

A *filter* is used to determine groups of input values which are replaced by an output of lower dimension.
<!-- .element: class="fragment" -->

For image inputs, the filter typically determines areas of $n$ x $m$ pixels that are replaced by a single value.
<!-- .element: class="fragment" -->

---

#### Max pooling ####

A filter  moves across the input data, and for each group of values that the filter covers, the maximum value is selected to represent the group in the output.

---

#### Average pooling ####

A filter moves across the input data, and for each group of values that the filter covers, the average value is selected to represent the group in the output.

---


### Input preprocessing

First we have to create a 5x5 pixel input of our hand-written digit.

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
<svg class="preview5x5" width="400" height="400">
<rect x="0" y="0" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="80" y="0" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="160" y="0" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="240" y="0" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="320" y="0" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="80" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="80" y="80" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="160" y="80" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="240" y="80" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="320" y="80" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="160" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="80" y="160" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="160" y="160" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="240" y="160" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="320" y="160" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="240" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="80" y="240" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="160" y="240" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="240" y="240" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="320" y="240" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="320" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="80" y="320" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="160" y="320" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="240" y="320" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
<rect x="320" y="320" width="80" height="80" fill="black" stroke="gray" stroke-width="1" />
</svg>
</td>
</tr>
</table>

Here, we apply average pooling with a threshold value of 0.3 to determine a binary input.

---

## Simple digit recognition

<table style="table-layout: fixed!important;width:1300px;">
<tr>
<td style="vertical-align: middle;width:150px!important;">
<canvas class="drawDigit" width="150" height="150" data-prevent-swipe>
</canvas>
</td>
<td style="vertical-align: middle;padding:0;width:80px!important;">
<i class="fas fa-arrow-right" style="font-size:50px;padding:15px;"></i>
</td>
<td style="vertical-align: middle;padding:0;padding:0;width:150px!important;">
<svg class="preview5x5" width="150" height="150">
<rect x="0" y="0" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="30" y="0" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="60" y="0" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="90" y="0" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="120" y="0" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="30" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="30" y="30" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="60" y="30" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="90" y="30" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="120" y="30" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="60" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="30" y="60" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="60" y="60" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="90" y="60" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="120" y="60" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="90" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="30" y="90" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="60" y="90" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="90" y="90" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="120" y="90" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="0" y="120" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="30" y="120" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="60" y="120" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="90" y="120" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
<rect x="120" y="120" width="30" height="30" fill="black" stroke="gray" stroke-width="1" />
</svg>
</td>
<td style="vertical-align: middle;padding:0;width:80px!important;">
<i class="fas fa-arrow-right" style="font-size:50px;padding:15px;"></i>
</td>
<td style="vertical-align: middle;padding:0;width:600px!important;position:relative;left:-170px;">
<div class="neuralnetwork predictions" style="height: 800px; width: 800px;">
<!--
{"type": "5x5" }
-->
</div>
</td>
<td style="vertical-align: middle;">
<svg class="toggle5x5Prediction" data-model="1" width="150" height="150" style="padding:0px;">
  <rect width="150" height="150" style="fill:black;" />
  <text class="predictedDigit" x="75" y="125" font-size="150" text-anchor="middle"  style="fill:white;stroke:white;">?</text>
</svg>
</td>
</tr>
</table>

---

### Accuracy

The simple neural network can achieve **high training accuracy**, but **poor test accuracy** if used for hand-written image recognition. 

---

#### How can we improve the accuracy?

- More pixels?
- Gray values?
- More input samples?
- Better preprocessing?
- Larger neural network?
- Non-linear activation functions?

---
