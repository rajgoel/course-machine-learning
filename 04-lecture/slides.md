# Stochastic gradient descent

---

In gradient descent we repeatedly compute the gradient of the **average loss over all training samples** and move in the opposite direction of that gradient.

![Image](04-lecture/gradient descent.png)

> [!IMPORTANT]
> We need many samples for successful learning, however, computing the average gradient over a large number of samples can become computationally expensive.

---

## Mini-batches

Stochastic gradient descent randomly draws **mini-batches** of training samples and performs gradient descent on the **average loss of the sampled mini-batch**.

> [!NOTE]
> The minibatch size is usually relatively small, ranging from 1 to a few hundred. 

---

## Local optima in gradient descent

![Image](04-lecture/gradient descent.png)
<!-- \frac{1}{3}\left(1\ +\ \sin\left(.8x-1\right)\ +\frac{\left(0.7x-3\right)^{2}}{10}\ +\ 1\ +\ \sin\left(1.1x\right)\ +\frac{\left(\frac{x}{3}-5\right)^{2}}{5}+\ 1\ +\ \sin\left(0.7x-1\right)\ +\frac{\left(x-5\right)^{2}}{10}\right) -->

> [!NOTE]
> Gradient descent can get stuck in local optima of poor quality.

---

## Local optima in stochastic gradient descent

![Image](04-lecture/stochastic_gradient descent.png)

> [!NOTE]
> The average loss over a mini-batch has different local optima than the average loss over all samples. Using different mini-batches, stochastic gradient is less risky to be trapped in local optima of poor quality.

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
- Number of epochs
- Mini-batch size
- Step size in (stochastic) gradient descent
- Training data

---

### Capacity

The **number of layers (depth)** and **number of neurons of each layer (width)** of a neural network determine the **capacity**, i.e., the ability of representing complex relationships and learning from data.

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

The [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectified_linear_unit) is 

$$\phi(z) = \max\{0,z\}$$

![Figure](04-lecture/ReLU.svg)

We have 
`$$\genfrac{}{}{1pt}{1}{\partial \phi^l_i(z^l_i)}{\partial z^l_i } = \begin{cases}1 \textrm{ if } z^l_i > 0 \\ \class{highlight}{0 \textrm{ if } z^l_i = 0 \textsf{ (formally undefined!)}} \\ 0 \textrm{ if } z^l_i < 0 \end{cases}$$`

> [!NOTE]
> ReLU is commonly used for hidden layers because it is easy to compute. 

---

#### "Dead" neurons

Whenever a neuron receives a negative input, it's gradient becomes zero. This may cause backpropagation to stop updating weights for this neuron, the neuron dies and no longer contributes to learning.

> [!NOTE]
> Theoretically, the neuron can recover as a sufficiently large shift in the bias could potentially bring the neuron back to life.

---

### Leaky ReLU

To overcome the problem of dead neurons, the **Leaky ReLU** 

`$$\phi(z) = \begin{cases}z & \text{if } z > 0 \\ \alpha z& \text{if } z \leq 0\end{cases}$$`

can be used with $\alpha$ being a small positive constant (typically 0.01).

![Figure](04-lecture/leakyReLU.svg)

We have 

`$$\phi'(z) = \begin{cases}1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0\end{cases}$$`

---

### Sigmoid

The [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation function

$$\phi(z) = \frac{1}{1+e^{-z}}$$

is used when activation values shall be between 0 and 1.

![Figure](04-lecture/sigmoid.svg)

We have 

`$$\phi'(z) = \phi(z)(1-\phi(z))$$`

> [!NOTE]
> Sigmoid is mainly used in final layers to represent probabilities. In hidden layers, sigmoid may cause the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) as gradients approach 0 for large $|z|$ values.

---

### Softmax

In **classification** with $k$ mutually exclusive alternatives, the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is used on the full output layer to convert activation values into probabilities

`$$\phi_i(z) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$$`

> [!NOTE]
> The sum over all probabilities is 1.

<!--
 and
`$$\frac{\partial \phi_i(z)}{\partial z_j} = \begin{cases} \phi_i(z)(1 - \phi_i(z)) & \text{if } i = j \\ -\phi_i(z) \phi_j(z) & \text{if } i \neq j \end{cases}$$`
-->

===

> [!CAUTION]
> Everything below needs revision!

===

## Overfitting and underfitting

---

### Width vs. depth


===

## Regularisation

===

## Data augmentation

===

## Normalisation


===

# Architecture

> [!CAUTION]
> This session needs revision!

---


---

## Loss functions 

> [!CAUTION]
> This session needs revision!


===

---

## Capacity, Overﬁtting and Underﬁtting 

<!--
Typically, when training a machine learning model, we have access to a training
set, we can compute some error measure on the training set called the training
error, and we reduce this training error. So far, what we have described is simply
an optimization problem. What separates machine learning from optimization is
that we want the generalization error, also called the test error, to be low as
well. The generalization error is deﬁned as the expected value of the error on a
new input. Here the expectation is taken across diﬀerent possible inputs, drawn
from the distribution of inputs we expect the system to encounter in practice.
We typically estimate the generalization error of a machine learning model by
measuring its performance on a test set of examples that were collected separately
from the training set.

In our linear regression example, we trained the model by minimizing the
training error, but we actually care about the test error.

We typically make a set of assumptions
known collectively as the i.i.d. assumptions. These assumptions are that the
examples in each dataset are independent from each other, and that the train
set and test set are identically distributed.

One immediate connection we can observe between the training and test error
is that the expected training error of a randomly selected model is equal to the
expected test error of that model.

Underﬁtting occurs when the model is not able to
obtain a suﬃciently low error value on the training set. Overﬁtting occurs when
the gap between the training error and test error is too large.

We can control whether a model is more likely to overﬁt or underﬁt by altering
its capacity.

Machine learning algorithms will generally perform best when their capacity
is appropriate for the true complexity of the task they need to perform and the
amount of training data they are provided with. Models with insuﬃcient capacity
are unable to solve complex tasks. Models with high capacity can solve complex
tasks, but when their capacity is higher than needed to solve the present task they
may overﬁt.

![EXAMPLE](04-lecture/over_and_underfitting.svg)


Typically, training error decreases until it asymptotes to the minimum possible error value as model
capacity increases (assuming the error measure has a minimum value). Typically,
generalization error has a U-shaped curve as a function of model capacity.

Training and test error
behave diﬀerently.  As we increase capacity, training error
decreases, but the gap between training and test error increases. Eventually,
the size of this gap outweighs the decrease in training error, and we enter the overﬁtting
regime, where capacity is too large, above the optimal capacity.

![EXAMPLE](04-lecture/over_and_underfitting.svg)

As capacity increases (x-axis), bias (dotted) tends to decrease and variance
(dashed) tends to increase, yielding another U-shaped curve for generalization error (bold
curve). If we vary capacity along one axis, there is an optimal capacity, with underﬁtting
when the capacity is below this optimum and overﬁtting when it is above.


### Regularisation

Regularization is any modiﬁcation we make to a
learning algorithm that is intended to reduce its generalization error but not its
training error.

- L2 Parameter Regularization: weight decay. This regularization strategy drives the weights closer to the origin
by adding a regularization term Ω(θ) = ½w² to the objective function
- Dropout

### Hyperparameters and Validation Sets

It is important that the
test examples are not used in any way to make choices about the model, including
its hyperparameters. For this reason, no example from the test set can be used
in the validation set. Therefore, we always construct the validation set from the
training data. Speciﬁcally, we split the training data into two disjoint subsets. One
of these subsets is used to learn the parameters. The other subset is our validation
set, used to estimate the generalization error during or after training, allowing
for the hyperparameters to be updated accordingly. The subset of data used to
learn the parameters is still typically called the training set, even though this
may be confused with the larger pool of data used for the entire training process.
The subset of data used to guide the selection of hyperparameters is called the
validation set. Typically, one uses about 80% of the training data for training and
20% for validation. Since the validation set is used to “train” the hyperparameters,
the validation set error will underestimate the generalization error, though typically
by a smaller amount than the training error. After all hyperparameter optimization
is complete, the generalization error may be estimated using the test set.


--> 

===

# Data

> [!CAUTION]
> This session needs revision!

## Dataset Augmentation

[only on training data]

---

===

# Training

> [!CAUTION]
> This session needs revision!

---

---

<!-- Above content is sorted and complete. -->

<!--
1. Architecture Design Principles
    - Layer sizing heuristics and depth/width decisions
    - Activation function selection (ReLU, sigmoid, LeakyReLU with
  derivatives)
    - Loss function selection for different tasks (MSE, cross-entropy, 
  etc.)
    - Optimization algorithms (SGD, Adam, RMSprop with mathematical
  formulations)
    - Learning rate scheduling and convergence analysis
  2. Data Pipeline Implementation
    - Train/validation/test splits with mathematical formulation
    - Data augmentation
    - Data preprocessing (normalization, encoding)
  3. Training & Evaluation
    - Initialization strategies (Xavier/He with mathematical justification)
    - Batch size considerations and training dynamics
    - Regularization techniques (dropout, weight decay, early stopping)
    - Practical debugging (vanishing/exploding gradients)
    - Overfitting detection and validation strategies
    - Hyperparameter tuning methodologies
    - Model evaluation metrics and performance monitoring

-->
===


#### Hyperparameters in a Neural Net

- **Learning rate:** Controls how much the weights are moving in the gradient descent update. On some problems using a LR that changes during training can increase the performance.
- **Batch size:** Controls how many training examples processed before updating the weights. Smaller batch size leads to slower convergence. A typical batch size is 32.
- **Optimizer:** there are more advanced optimizers besides Gradient Descent 
- **Number of iterations:** Early stopping is a way to not deal with the epochs as a hyperparameter. 
- **Activation functions:** Relu, sigmoid, Tanh...
- **Number of Epochs:** Often set based on validation loss.

---

### Data split

<img src="04-lecture/datasplit.svg" width="800"/>

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

- **Training a NN is highly iterative:** Evaluate the predictions on the right metric, make changes and evaluate again.

<div class=highlight-box> 
📝 A common workflow is training a model that fits the data very well (low training error)
and check whether the variance is acceptable looking at the dev set performance. Adding more data and regularization
should reduce the variance.
</div>


===

---

## Data Preprocessing: Normalization

**Why normalize for neural networks?**
- Gradient descent works better with similar scales
- Prevents some features from dominating

**Common choices:**
- **Min-max scaling**: [0,1] range
- **Z-score standardization**: mean=0, std=1  
- **Robust scaling**: uses median and IQR

---

## Data Augmentation

**Purpose**: Artificially increase training data diversity

**Image data:**
- Rotation, flipping, cropping
- Color jittering, noise addition

**Text data:**  
- Synonym replacement, paraphrasing
- Back-translation

**Time series:**
- Window sliding, noise injection
- Time warping

---

## Train/Validation/Test Splits

**Traditional approach**: Random 70/15/15 split

**Deep learning considerations:**
- **Temporal data**: Use time-based splits
- **Large datasets**: Can use smaller validation sets
- **Class imbalance**: Stratified sampling essential

**Key principle**: Test set must remain untouched until final evaluation

---

## Data Bias: Sampling Bias

**Problem**: Training data doesn't represent real-world distribution

**Examples:**
- Medical data: Underrepresented demographics
- Image recognition: Biased toward certain lighting conditions
- Text data: Overrepresenting certain languages/dialects

**Solution**: Actively check and balance your dataset

---

## Data Bias: Temporal Considerations

**Data leakage**: Using future information to predict the past

**Common mistakes:**
- Normalizing entire dataset before splitting
- Using future data points in feature engineering
- Mixing temporal sequences across splits

**Rule**: Anything computed on training data cannot use validation/test data


---

## Regularization: Dropout

**How it works**: Randomly "turn off" neurons during training

**Key decisions:**
- **Placement**: After dense layers, not after conv layers usually  
- **Rate**: 0.2-0.5 for dense layers, 0.1-0.2 for conv layers
- **When**: Only during training, not inference

**Trade-off**: Prevents overfitting but slows training

---

## Regularization: Weight Decay

**L2 regularization**: Penalize large weights in loss function

**L1 regularization**: Encourage sparsity

**When to use:**
- L2: Almost always beneficial, start with 1e-4
- L1: When you want feature selection
- Both: Can combine for elastic net effect

---

## Batch Normalization

**What it does**: Normalize inputs to each layer

**Benefits:**
- Faster training, higher learning rates possible
- Less sensitive to initialization
- Acts as regularization

**Placement**: Usually after linear layer, before activation

---

## Learning Rate: The Most Important Hyperparameter

**Too high**: Training unstable, loss explodes
**Too low**: Training too slow, gets stuck in local minima

**Finding the right rate:**
- Start with 1e-3 for Adam, 1e-1 for SGD
- Use learning rate finder
- Monitor loss curves for instability

---

## Learning Rate Scheduling

**Why schedule**: High LR for fast initial learning, low LR for fine-tuning

**Common strategies:**
- **Step decay**: Reduce by factor every N epochs
- **Exponential decay**: Gradual continuous reduction  
- **Cosine annealing**: Smooth reduction with restarts
- **Warm-up**: Start low, increase, then decrease

---

## Epochs and Training Duration

**How many epochs?**
- Not a fixed number!
- Depends on dataset size, model complexity, learning rate

**Key principle**: Use validation loss to decide when to stop
- Train until validation loss stops improving
- Watch for the "hockey stick" pattern

---

## Early Stopping Strategy

**Patience**: How many epochs to wait without improvement

**Typical values**: 10-50 epochs depending on dataset size

**What to monitor**: 
- Validation loss (most common)
- Validation accuracy
- Custom metrics relevant to your problem

**Save best model**: Don't use the final epoch, use the best validation performance

---

## Recognizing Underfitting

**Signs:**
- Both training and validation loss are high
- Training loss decreases very slowly
- Large gap between desired and actual performance

**Solutions:**
- Increase model complexity (more layers/units)
- Reduce regularization (lower dropout, weight decay)
- Train longer
- Increase learning rate

---

## Recognizing Overfitting

**Signs:**
- Training loss much lower than validation loss
- Validation loss starts increasing while training loss decreases
- Good training performance, poor test performance

**This is the classic "memorization" problem**

---

## Preventing Overfitting: Data-Level

**Get more data**: Often the best solution
- Real data collection
- Data augmentation
- Synthetic data generation

**Improve data quality**:
- Remove noise and outliers
- Better feature engineering
- Cross-validation to detect issues early

---

## Preventing Overfitting: Model-Level

**Reduce model complexity:**
- Fewer layers or units
- Simpler architecture choices

**Add regularization:**
- Increase dropout rates
- Higher weight decay
- Batch normalization

**Early stopping**: Stop before memorization begins

---

## Batch Size Considerations

**Large batches (>256):**
- More stable gradients
- Better hardware utilization
- May need higher learning rates

**Small batches (<32):**
- More gradient noise (can be beneficial)
- Better generalization sometimes
- Lower memory requirements

**Sweet spot**: Often 32-128 for most problems

---

## Optimizer Selection

**SGD with momentum:**
- Reliable, well-understood
- Needs careful learning rate tuning
- Often best final performance

**Adam:**
- Adaptive learning rates
- Works well out of the box
- Can overshoot optimal solutions

**Rule of thumb**: Start with Adam, switch to SGD for final tuning

---

## Loss Function Selection

**Regression problems:**
- MSE: Standard choice, penalizes large errors heavily
- MAE: More robust to outliers
- Huber loss: Compromise between MSE and MAE

**Classification:**
- Cross-entropy: Standard for multi-class
- Focal loss: When dealing with class imbalance
- Custom losses: For specific business objectives

---

## Monitoring Training Progress

**Essential plots:**
- Training vs validation loss over epochs
- Learning rate schedule
- Gradient norms (detect vanishing/exploding gradients)

**Warning signs:**
- Loss oscillations (LR too high)
- Plateauing too early (LR too low, underfitting)
- Diverging train/val loss (overfitting)

---

## Avoiding Hyperparameter Bias

**Problem**: Overfitting to validation set through repeated hyperparameter tuning

**Solutions:**
- Hold out a true test set
- Use nested cross-validation
- Limit number of hyperparameter experiments
- Statistical significance testing

**Document everything**: Track all experiments, not just successful ones

---

## Bias Prevention in Training

**Fairness monitoring:**
- Check performance across demographic groups
- Monitor for discriminatory patterns
- Use fairness-aware metrics

**Robust evaluation:**
- Test on diverse data
- Adversarial testing
- Performance consistency across subgroups

---

===

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

