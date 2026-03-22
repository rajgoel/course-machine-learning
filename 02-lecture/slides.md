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

Each input $X$ is represented by an encoding `$(a_{1}, a_{2}, \ldots, a_{25})$`.

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

and so on.

---

### Output encoding

Each output $Y$ is represented by a **one-hot encoding**  `$(\hat{a}_{0}, \hat{a}_{1}, \ldots, \hat{a}_{9})$`.

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
$$a = 
\left(
\begin{array}{c}
a_{1} \\
a_{2} \\
\vdots \\
a_{25}
\end{array}
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
$$ \left( \begin{array}{c} \hat{a}_{0} \\  \hat{a}_{1} \\ \vdots \\ \hat{a}_{9}  \end{array} \right) = \phi(Wa + b)$$
</td>
</tr>
</table>

> [!IMPORTANT]
> Again we assume a linear activation with $\phi(z) = z$, and thus, $\hat{a} = Wa + b$.

---

<!-- .slide: data-auto-animate="true" -->

Assume we are given a collection of $(X,Y)$ pairs each encoded as $(a,a^*)$. 

---

<!-- .slide: data-auto-animate="true" -->

Assume we are given a collection of $(X,Y)$ pairs each encoded as $(a,a^*)$. 

If we could find weights and biases such that for each pair $(a,a^*)$, we have

> The computed output $\hat{a} = W a + b$  matches the given output $a^*$.

our neural network would be able to perfectly recognise the given digits. 

---

<!-- .slide: data-auto-animate="true" -->

If we could find weights and biases such that for each pair $(a,a^*)$ we have

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
a_{1} \\
a_{2} \\
\vdots \\
a_{25}
\end{array}
\right) + \left(
\begin{array}{c}
b_{0} \\
b_{1} \\
\vdots \\
b_{9} \\
\end{array}
\right)
}_{\hat{a}} = \left(
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
- For each  pair $(a,a^*)$ we have 10 equations for a perfect neural network.
- With 30 pairs of training data, we would have 30 $\cdot$ 10 = 300 equations.

> [!NOTE]
> A system of equations with 300 equations and 260 variables cannot be solved in general, therefore, we need to approximate (minimise the error). 

---

### Loss

For each given pair $(a,a^*)$ we can define the loss (i.e., error function) as the sum of the squared differences of the computed output and the expected output.

`$$
\begin{align}
\mathscr{L}_{(a,a^*)}(W,b) & = ( \hat{a} - a^* )^2 \\
& = \displaystyle\sum_{i=0}^{9} \Big( \hat{a}_i - a^*_i \Big)^2 \\
& = \displaystyle\sum_{i=0}^{9} \Big( \displaystyle\sum_{j=1}^{25} w_{i,j} a_{j} + b_i - a^*_i \Big)^2
\end{align}
$$`

> [!NOTE]
> Our goal is to find weights and biases that minimize the loss **for all** given pairs $(a,a^*)$.

===

## Gradient descent

---

## Average loss

Let $S$ denote the set of samples given. Then, the **average loss** is

`$$\mathscr{L}(W,b) = \displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a,a^*) \in S} \underbrace{\mathscr{L}_{(a,a^*)}(W,b)}_{\textrm{Loss of sample}}.
$$`

> [!NOTE]
> We want to find weights and biases minimizing the average loss.

---

<!-- .slide: data-auto-animate="true" -->

## Gradient and partial derivatives

The [gradient](https://en.wikipedia.org/wiki/Gradient) $\nabla f$ of a function $f(x,y, \ldots)$ is the vector of all its partial derivatives with respect to each input variable. 

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!NOTE]
> **Partial derivative:** The [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative) measures the rate of change of a function with respect to one variable while keeping other variables constant.

---

<!-- .slide: data-auto-animate="true" -->

## Gradient and partial derivatives

The [gradient](https://en.wikipedia.org/wiki/Gradient) $\nabla f$ of a function $f$ is the vector of all its partial derivatives with respect to each input variable. 

<object data="02-lecture/gradient.svg" type="image/svg+xml" ></object>

> [!TIP]
> The gradient gives the direction of the steepest ascend. By changing weights and biases in opposite direction of the gradient, we can minimise the loss.

---

<!-- .slide: data-auto-animate="true" -->

### Gradient of the average loss

The gradient of the average loss
`$$\mathscr{L}(W,b) = \displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a,a^*) \in S} \mathscr{L}_{(a,a^*)}(W,b).
$$`
is

`$\nabla_{W,b} \mathscr{L} =$` 
`$\nabla_{W,b}$`<!-- .element: data-id="nabla" -->
`$\displaystyle\frac{1}{|S|} \cdot \displaystyle\sum_{(a,a^*) \in S}$`<!-- .element: data-id="average" -->
`$\mathscr{L}_{(a,a^*)}$`

<p></p>

---

<!-- .slide: data-auto-animate="true" -->

### Gradient of the average loss

The gradient of the average loss
`$$\mathscr{L}(W,b) = \displaystyle\frac{1}{|S|} \cdot
\displaystyle\sum_{(a,a^*) \in S} \mathscr{L}_{(a,a^*)}(W,b).
$$`
is 

`$\nabla_{W,b} \mathscr{L} =$` 
`$\displaystyle\frac{1}{|S|} \cdot \displaystyle\sum_{(a,a^*) \in S}$`<!-- .element: data-id="average" -->
`$\nabla_{W,b}$`<!-- .element: data-id="nabla" -->
`$\mathscr{L}_{(a,a^*)}$`

i.e. the average of the gradients over all samples.

---

<!-- .slide: data-auto-animate="true" -->

### Partial derivative for weights

For each output neuron $i \in I$ and each input neuron $j \in J$ the partial derivative of
`$$
\mathscr{L}_{(a,a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a_{j} + b_i
}_{\hat{a}_i} - a^*_i
\Big)^2
$$`
with respect to $w_{i,j}$ is

`$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w_{i,j} } =\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \hat{a}_i}{\partial w_{i,j} } $`<!-- .element: data-id="del-w" -->

---

<!-- .slide: data-auto-animate="true" -->

### Partial derivative for weights

For each output neuron $i \in I$ and each input neuron $j \in J$ the partial derivative of
`$$
\mathscr{L}_{(a,a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a_{j} + b_i
}_{\hat{a}_i} - a^*_i
\Big)^2
$$`
with respect to $w_{i,j}$ is

`$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial w_{i,j} } =\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \hat{a}_i}{\partial w_{i,j} } $`<!-- .element: data-id="del-w" -->
`$ = 2(\hat{a}_i - a^*_i) \cdot a_j$`

---

<!-- .slide: data-auto-animate="true" -->

### Partial derivative for biases

For each output neuron $i \in I$ the partial derivative of
`$$
\mathscr{L}_{(a,a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a_{j} + b_i
}_{\hat{a}_i} - a^*_i
\Big)^2
$$`
with respect to $b_i$ is


`$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b_i } =\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \hat{a}_i}{\partial b_i } $`<!-- .element: data-id="del-b" -->

---

<!-- .slide: data-auto-animate="true" -->

### Partial derivative for biases

For each output neuron $i \in I$ the partial derivative of
`$$
\mathscr{L}_{(a,a^*)} =
\displaystyle\sum_{i \in I} 
\Big(
\underbrace{\displaystyle\sum_{j \in J} w_{i,j} a_{j} + b_i
}_{\hat{a}_i} - a^*_i
\Big)^2
$$`
with respect to $b_i$ is


`$ \genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial b_i } =\genfrac{}{}{1pt}{1}{\partial \mathscr{L}_{(a,a^*)}}{\partial \hat{a}_i} \cdot \genfrac{}{}{1pt}{1}{\partial \hat{a}_i}{\partial b_i } $`<!-- .element: data-id="del-b" -->
`$= 2(\hat{a}_i - a^*_i) \cdot 1$`

---

<!-- .slide: data-fullscreen="yes"  -->

### Gradient descent in Julia

```julia[1-19|2-4|5-18|6-9|10-14|15-17|1-19]
function gradient_descent!(W::Matrix{Float32}, b::Vector{Float32}, X::Vector{Vector{Float32}}, Y::Vector{Vector{Float32}})
    tolerance::Float32 = 1.0e-3
    max_iterations::Int = 10000
    η::Float32 = 0.1  # learning rate
    for iter in 1:max_iterations
        # Compute the average gradients ∇W and ∇b
        ∇W, ∇b = compute_average_gradients(W, b, X, Y)
        grad_norm = gradient_norm(∇W, ∇b)
        println("Iteration $iter, ‖(∇W,∇b)‖ = $grad_norm")        
        # Break if the gradient norm is smaller than the tolerance
        if grad_norm < tolerance
            println("Gradient norm below tolerance. Stopping.")
            break
        end        
        # Parameter updates: W ← W - η * ∇W, b ← b - η * ∇b
        W .-= η .* ∇W
        b .-= η .* ∇b
    end
end
```
<!-- .element: class="fullscreen stretch" -->

---

# Implementation

You find a full implementation in the [course repository](https://rajgoel.github.io/course-machine-learning/julia).

> [!TIP]
> Run:
> ```julia
> using MachineLearningCourse
> Lecture02.demo()
> ```


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


### Input preprocessing

First we have to create a 5x5 pixel input of our hand-written digit.

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

> [!NOTE]
> Here, we calculate the average color of each area in our input and assume a white pixel if the average is above 0.3.

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

