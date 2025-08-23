# Architecture and pipeline

> [!CAUTION]
> This session needs revision!

---

## Course Context

You've learned traditional ML in "Fundamentals of Data Science"
- Decision trees, SVMs, linear regression
- When to use what algorithm
- Basic evaluation and validation

**Today**: Design choices specific to deep learning

---

## Learning Objectives

By the end of this lecture, you will know how to make informed decisions about:
- Data preprocessing for neural networks
- Architecture design choices
- Training strategies that avoid common pitfalls
- Bias prevention throughout the pipeline

---

# Part I: Data Pipeline Decisions

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

## Data Preprocessing: Handling Missing Data

**Traditional ML**: Often use imputation or removal

**Deep Learning considerations:**
- Neural networks can't handle NaN values
- Large datasets → removal might be viable
- Complex patterns → learned imputation possible

**Common approaches:**
- Mean/median imputation for numerical
- Mode imputation for categorical
- Learn embeddings for "missing" as category

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

# Part II: Network Architecture Decisions

---

## Depth vs Width Trade-offs

**Deeper networks:**
- Can learn more complex representations
- Risk: vanishing gradients, harder to train
- Better for hierarchical feature learning

**Wider networks:**
- More parameters at each level
- Easier to train but less expressive
- Better for parallel feature learning

---

## Layer Types: When to Use What

**Fully Connected (Dense):**
- Final classification layers
- Small, structured data
- When relationships between all features matter

**Convolutional:**
- Spatial data (images)
- Translation invariance needed
- Local pattern detection

**Pooling:**
- Dimensionality reduction
- Translation invariance
- Computational efficiency

---

## Activation Function Selection

**ReLU**: Default choice
- Fast computation, no vanishing gradients
- Problem: "dying ReLU" issue

**Leaky ReLU**: Fixes dying ReLU
- Small negative slope prevents dead neurons

**Swish/GELU**: Modern alternatives
- Smooth, differentiable everywhere
- Often better performance, slightly more expensive

---

## When to Use Sigmoid/Tanh

**Sigmoid**: 
- Output layer for binary classification
- When you need probabilities [0,1]
- **Avoid** in hidden layers (vanishing gradients)

**Tanh**:
- Similar to sigmoid but [-1,1] range
- Zero-centered (better than sigmoid)
- Still avoid in deep hidden layers

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

## Architecture Bias Considerations

**Inductive bias**: Assumptions built into architecture

**Examples:**
- CNNs assume spatial locality and translation invariance
- RNNs assume sequential dependencies
- Fully connected assumes all features interact equally

**Fairness impact**: Some architectures may amplify demographic biases

---

# Part III: Training Strategy Design

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

# Part IV: Evaluation and Validation Framework

---

## Beyond Accuracy: Comprehensive Evaluation

**For classification:**
- Precision, recall, F1-score
- ROC curves and AUC
- Confusion matrices
- Per-class performance analysis

**For regression:**  
- R², MAE, RMSE
- Residual analysis
- Performance across value ranges

---

## Fairness Metrics

**Demographic parity**: Equal positive prediction rates across groups

**Equalized odds**: Equal true/false positive rates across groups

**Individual fairness**: Similar individuals get similar predictions

**Choose based on your application's fairness requirements**

---

## Documentation and Reproducibility

**Essential documentation:**
- Data sources and preprocessing steps
- Architecture choices and rationale
- Hyperparameter search process
- Training environment and random seeds

**Version control**: Track data, code, and model versions

**Goal**: Another researcher should be able to reproduce your results

---

## Practical Tools and Frameworks

**Bias detection:**
- Fairness toolkits (AIF360, Fairlearn)
- What-if tools for model analysis
- Dataset bias analyzers

**Experiment tracking:**
- Weights & Biases, MLflow
- TensorBoard for monitoring
- Version control for experiments

---

## Key Takeaways

**Design choices are interconnected:**
- Data preprocessing affects model requirements
- Architecture choices influence training strategies  
- Evaluation metrics should align with business goals

**Always prioritize fairness and robustness**
- Bias can enter at any stage
- Comprehensive evaluation is essential
- Document and monitor throughout

---

## Next Steps

**In upcoming lectures, we'll see these principles applied to:**
- Feed-forward networks (Lecture 3)
- Convolutional architectures (Lectures 5-6) 
- Autoencoders (Lecture 7)
- Reinforcement learning (Lectures 8-9)

**Remember**: These design principles apply to every architecture we'll study!
