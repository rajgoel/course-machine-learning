
# Neural networks

---

> [!CAUTION]
> Not yet complete

---

### Artificial neurons

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


---

## Regression with neural networks


When the input `$$ a^\textrm{in} \in R $$` has multiple dimensions, the output of the neural network can be written as:

`$$a^\textrm{out} = ReLU(w^\textrm{T}a^\textrm{in} + b)$$`

The term $\textit{b}$ is the bias and the vector $\textit{w}$ is referred to as the weights vector. 

The ReLU (Rectified Linear Unit) activation function will output continuous values when the input value is positive and 0 otherwise.

---

- How many output neurons would prediction of house prices need?

- How many output neurons would locating the center of an image (coordinates) need?

- How many output neurons would the bounding box of an object need?



---

## Classification with neural networks

---

### Example: Digit recognition

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


<img src="neuralnetworks/flatten.svg" width="1200px">

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

<br>

`$ a^\textrm{out}_1 = w_{1,1} a^\textrm{in}_{1} + w_{1,2} a^\textrm{in}_{2} + w_{1,3} a^\textrm{in}_{3} + w_{1,4} a^\textrm{in}_{4} + b_1$`

and 

`$ a^\textrm{out}_2 = w_{2,1} a^\textrm{in}_{1} + w_{2,2} a^\textrm{in}_{2} + w_{2,3} a^\textrm{in}_{3} + w_{2,4} a^\textrm{in}_{4} + b_2.$`

---

`$$ 
\left(
\begin{array}{c}
a^\textrm{out}_{1} \\
a^\textrm{out}_{2} 
\end{array}
\right)
=
\left(
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
\right)
+
\left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right)
$$`

---

### Matrix notation

`$$
\left(
\begin{array}{c}
a^\textrm{out}_{1} \\
a^\textrm{out}_{2} 
\end{array}
\right)
=
\left(
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
\right)
+
\left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right)
$$`

or

`$$
a^\textrm{out} = Wa^\textrm{in} + b
$$`

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

`$$
\left(
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
\right)
+
\left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right)
=
\left(
\begin{array}{c}
1 \\
0 
\end{array}
\right)
$$`

and

`$$
\left(
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
\right)
+
\left(
\begin{array}{c}
b_{1} \\
b_{2} 
\end{array}
\right)
=
\left(
\begin{array}{c}
0 \\
1 
\end{array}
\right)
$$`

our neural network can perfectly recognise both symbols. 

---

### Optimal weights and biases

Below weights and biases are optimal.

`$$
\left(
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
\right)
+
\left(
\begin{array}{c}
0 \\
0 
\end{array}
\right)
=
\left(
\begin{array}{c}
1 \\
0 
\end{array}
\right)
$$`

and

`$$
\left(
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
\right)
+
\left(
\begin{array}{c}
0 \\
0 
\end{array}
\right)
=
\left(
\begin{array}{c}
0 \\
1 
\end{array}
\right)
$$`

We can also find other weights and biases that are optimal for this example.
<!-- .element: class="fragment" -->

---

### Conclusion

A simple neural network with

- 4 input neurons
- 2 output neurons with linear activation function

can be used to reliable recognize both symbols given a 2x2 black and white pixel image.

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

<br>

`$$ a^\textrm{out}_i = \displaystyle\sum_{j=1}^{25} w_{i,j} a^\textrm{in}_{j} + b_i$$`

for all `$i \in \{ 0,1, \ldots, 9\}$`.

---

Assume we are given a pairs of input activation values $a^\textrm{in}$ with their expected output activation values $a^*$. 

---

If we could find weights and biases such that for each pair $(a^\textrm{in},a^*)$ we have

`$$
\underbrace{
W a^\textrm{in} + b
}_{a^\textrm{out}}
= a^*
$$`

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
\right)
+
\left(
\begin{array}{c}
b_{0} \\
b_{1} \\
\vdots \\
b_{9} \\
\end{array}
\right)
}_{a^\textrm{out}}
=
\left(
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

## Losses
 
The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.

### Probabilistic Losses (classification problems)

#### Binary Crossentropy: 

When the target has binary values (0,1).

$BCE = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$

where 
- N is the number of observations.

- $y_i$​ is the true label for the i_th sample (either 0 or 1).

- $\hat{y}_i$ is the predicted probability for the the i_th sample.

- log⁡(⋅) is the natural logarithm.

---

#### Categorical Crossentropy: 

When the target has more than two classes and it is hot encoded.

$CCE= -\frac{1}{C} \sum_{i=1}^{N} \log(\hat{y}_{i})$

where 
- C is the number of classes.

- $y_i$​ is the true label for the i_th sample (either 0 or 1) from the one-hot encoded.

- $\hat{y}_i$ is the predicted probability for the the i_th sample.

- log⁡(⋅) is the natural logarithm.

---

### Regression Losses 

- Mean Squared Error



- Mean Absolute Error

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

<object data="digitrecognition/gradient.svg" type="image/svg+xml" ></object>

By changing weights and biases in opposite direction of the gradient, we can quickly minimise the error function.

---

## Forward Pass (Forward Propagation)

- The input data is passed through the network, and the output is the computed output probabilities.


## Backpropagation

- The error (loss) is computed.
- Gradients of the loss w.r.t. network parameters using the chain rule are computed.
- Update the weights and biases using gradient descent (or other optimizers like Adam).

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
