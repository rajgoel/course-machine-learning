# Stochastic gradient descent

---

In gradient descent we repeatedly compute the gradient of the **average loss over all training samples** and move in the opposite direction of that gradient.

![Image](04-lecture/gradient_descent.png)<!-- .element: height="400"  -->

> [!IMPORTANT]
> We need many samples for successful learning, however, computing the average gradient over a large number of samples can become computationally expensive.

---

## Mini-batches

Stochastic gradient descent randomly draws **mini-batches** of training samples and performs gradient descent on the **average loss of the sampled mini-batch**.

> [!NOTE]
> The minibatch size is usually relatively small, ranging from 1 to a few hundred. 

---

## Local optima in gradient descent

![Image](04-lecture/gradient_descent.png)<!-- .element: height="400"  -->

<!-- \frac{1}{3}\left(1\ +\ \sin\left(.8x-1\right)\ +\frac{\left(0.7x-3\right)^{2}}{10}\ +\ 1\ +\ \sin\left(1.1x\right)\ +\frac{\left(\frac{x}{3}-5\right)^{2}}{5}+\ 1\ +\ \sin\left(0.7x-1\right)\ +\frac{\left(x-5\right)^{2}}{10}\right) -->

> [!NOTE]
> Gradient descent can get stuck in local optima of poor quality.

---

## Local optima in stochastic gradient descent

![Image](04-lecture/stochastic_gradient_descent.png)<!-- .element: height="400"  -->

> [!NOTE]
> For every mini-batch we have different local optima. Thus, stochastic gradient is less risky to be trapped in local optima of poor quality.

---

Due to its computational efficiency and the capability to produce good solutions, stochastic gradient descent is the standard algorithm for training neural networks.

---

<!-- .slide: data-fullscreen="yes"  -->

### Stochastic gradient descent with [Flux.jl](https://fluxml.ai/Flux.jl/stable/)

```julia[1-2|4-11|13-35|16-25|27-34|37]
function train!(network::FluxDNN, X_train::Matrix{Float32}, Y_train, learning_rate, epochs; 
               batch_size=128, verbose=true)
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), network.model)
    
    # Create data loader for mini-batch training
    minibatches = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
    
    losses = Float64[]
    
    for epoch in 1:epochs
        epoch_losses = Float64[]
        
        # Train on mini-batches (which are implicitly re-shuffled)
        for (x_batch, y_batch) in minibatches
            # Calculate loss and gradients for this batch
            batch_loss = loss(network.model, x_batch, y_batch)
            push!(epoch_losses, batch_loss)
            
            # Training step
            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), network.model)[1]
            Flux.update!(optimizer, network.model, grads)
        end
        
        # Average loss for this epoch
        epoch_loss = mean(epoch_losses)
        push!(losses, epoch_loss)
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Loss = $(round(epoch_loss, digits=6))")
        end
    end
    
    return losses
end
```
<!-- .element: class="fullscreen stretch" -->

---

> [!TIP]
> You find a full implementation in `Lecture04` of the [Julia repository](https://rajgoel.github.io/course-machine-learning/julia).

===

# Implementation details

---

Implementing a deep neural network involves many decisions:

- Number of layers (depth)
- Number of neurons for each layer (width)
- Activation function $\phi$
- Loss function $\mathscr{L}$
- Mini-batch size
- Number of epochs
- Learning rate $\eta$

---

### Capacity

The **number of layers (depth)** and **number of neurons of each layer (width)** of a neural network determine the **capacity**, i.e., the ability to represent complex relationships and learn from data.

---

### Universal approximation theorem

The [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that with a sufficient number of neurons, a single hidden layer is sufficient to approximate any continuous function to any desired degree of accuracy.

> [!IMPORTANT]
> The total number of neurons required may be unnecessarily high for shallow networks (i.e., networks with few layers).

---

### Depth vs. width

Despite the universal approximation theorem, deeper networks often outperform shallow ones, as they can learn hierarchical features (through aggregation from previous layers).

> [!IMPORTANT]
> In backpropagation, very deep networks may suffer from [vanishing (and exploding) gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), slowing down learning or making learning unstable.

---

## Activation functions

Non-linear [activation functions](https://fluxml.ai/Flux.jl/stable/reference/models/activation/) are crucial for learning complex relationships.

> [!IMPORTANT]
> Without non-linear activation functions, a deep neural network reduces to a linear transformation.

---

### Rectified Linear Unit (ReLU)

The [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectified_linear_unit) is `$\phi(z) = \max\lbrace 0,z \rbrace$`.

![Figure](04-lecture/ReLU.svg)

> [!NOTE]
> ReLU is commonly used for hidden layers because it is easy to compute. 

---

### Derivative of ReLU

For `$\phi(z) = \max\lbrace 0,z \rbrace$` we have
 
`$$\genfrac{}{}{1pt}{1}{\partial \phi(z)}{\partial z } = \begin{cases}1 \textrm{ if } z > 0 \\ \class{highlight}{0 \textrm{ if } z = 0 \textsf{ (formally undefined!)}} \\ 0 \textrm{ if } z < 0 \end{cases}$$`

---

#### "Dead" neurons

Whenever a neuron receives a negative input, it's gradient becomes zero. This may cause backpropagation to stop updating weights for this neuron, the neuron dies and no longer contributes to learning.

> [!NOTE]
> Theoretically, the neuron can recover as a sufficiently large shift in the bias could potentially bring the neuron back to life.

---

### Leaky ReLU

To overcome the problem of dead neurons, the **Leaky ReLU** 

`$$\phi(z) = \begin{cases}z & \text{if } z > 0 \\ \eta z& \text{if } z \leq 0\end{cases}$$`

can be used with $\eta$ being a small positive constant (typically 0.01).

![Figure](04-lecture/leakyReLU.svg)

---

### Derivative of leaky ReLU

For
`$$\phi(z) = \begin{cases}z & \text{if } z > 0 \\ \eta z& \text{if } z \leq 0\end{cases}$$`

we have 

`$$\phi'(z) = \begin{cases}1 & \text{if } z > 0 \\ \eta & \text{if } z \leq 0\end{cases}$$`

---

### Sigmoid

The [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function is

`$$\phi(z) = \frac{1}{1+e^{-z}}$$`

![Figure](04-lecture/sigmoid.svg)

> [!NOTE]
> Sigmoid is mainly used in final layers to represent probabilities. 

---


For
`$$\phi(z) = \frac{1}{1+e^{-z}}$$`

we have 

`$$\phi'(z) = \phi(z)(1-\phi(z))$$`

> [!NOTE]
> In hidden layers, sigmoid may cause the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) as gradients approach 0 for large $|z|$ values.

---

### Softmax

In **classification** with $k$ mutually exclusive alternatives, the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is used on the **full output layer** to convert activation values into probabilities

`$$\phi_i(z) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$$`

> [!NOTE]
> The sum over these probabilities is 1.

<!--
 and
`$$\frac{\partial \phi_i(z)}{\partial z_j} = \begin{cases} \phi_i(z)(1 - \phi_i(z)) & \text{if } i = j \\ -\phi_i(z) \phi_j(z) & \text{if } i \neq j \end{cases}$$`
-->

---

## Loss functions 

For **regression** with $k$ output neurons, we usually use the **squared error** for each prediction-expectation pair $(a,a^*)$:

`$$\mathscr{L} = \sum_{i=1}^{k}( \hat{a}_i - a_i^* )^2.$$`

For **classification** with $k$ mutually exclusive alternatives, we usually use the **cross-entropy** for each prediction-expectation pair $(a,a^*)$:

`$$\mathscr{L} = -\sum_{i=1}^{k} a^*_i \log(\hat{a}_i)}$$`

> [!TIP]
> In [Flux.jl](https://fluxml.ai/Flux.jl/stable/reference/models/losses/#Flux.Lossesy) it is advised to use `logitcrossentropy` rather than using `crossentropy` with `softmax` activation. 

---

## Mini-batch size

The mini-batch size determines how many samples are used to compute each gradient update.

Small batches cause more frequent updates with noisier gradients allowing to escape local optima, while larger batches provide more stable gradient estimates.

> [!TIP]
> Powers of 2 (32, 64, 128, 256) are often computationally more efficient.

---

## Number of epochs

The number of epochs determines how many times the entire training dataset is processed.

> [!IMPORTANT]
> Small numbers of epochs may not be sufficient to learn the underlying patterns, while large numbers of epochs model lead to memorisation instead of learning.

---

## Learning rate $\eta$

A crucial implementation detail for (stochastic) gradient descent is choice of the learning rate $\eta$ indicating the step size.

If the learning rate is to low, convergence will be slow and gradient descent may get stuck in local optima. If the learning rate is too high, gradient descent may overshoot, causing oscillations and divergence.

> [!NOTE]
>  Modern optimisers like [Adam](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Optimisers.Adam) automatically adapt the learning rate during training.


===

# Under- and overfitting

- When model **capacity is too small**, the underlying patterns in the data cannot be captured (**underfitting**).
- When model **capacity is large**, **intensive training** may result in memorising the training data instead of learning generalisable patterns  (**overfitting**).

![Image](04-lecture/under_and_overfitting.png")

---

## Data split


![Image](04-lecture/datasplit.svg)<!-- .element: width="800"  -->

| Training set purpose | Validation set purpose | Test set purpose |
|----------------------|-----------------|------------------|
| Train the model parameters | Preliminary evaluation | Final evaluation |
| Learn patterns from data | Identify overfitting | Unbiased assessment |

---

## Early stopping

![Image](04-lecture/training_progress.png)

> [!TIP]
> To prevent overfitting, we should stop training when the validation loss is no longer improving.


