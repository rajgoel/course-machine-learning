# Autoencoders

---

[Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) are neural networks that **learn compressed representations** of input data through unsupervised learning.

---

## Encoder

An encoder turns high-dimensional input into lower-dimensional output.

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "encoder" }
-->
</div>

---

## Decoder

A decoder reconstructs high-dimensional output from lower-dimensional input. 

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "decoder" }
-->
</div>

---

## Autoencoder

An autoencoder combines encoder and decoder through a **latent representation** of low dimension, also called the **bottleneck**.

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "autoencoder" }
-->
</div>

---

## Loss function

Autoencoders minimise the **reconstruction error**, i.e., the mean squared error between a given input $a$ and the reconstructed output $\hat{a}_i$.

`$$\mathscr{L}_a(\theta) = \frac{1}{n} \sum_{i=1}^{n} ( \hat{a}_i - a_i) ^2$$`

---

## Use cases

- **Dimensionality reduction**: Reduce dimensionality, e.g., for visualisation or data compression
- **Anomaly detection**: Identify outliers by reconstruction errors
- **Denoising**: Remove noise from corrupted data
- **Feature learning**: Pre-train compact representations for downstream tasks


===

## Dimensionality reduction

> [!NOTE]
> Autoencoders can be used for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), but there are also other alternatives like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), or [UMAP](https://en.wikipedia.org/wiki/Uniform_manifold_approximation_and_projection).

---

### Visualisation of distribution of MNIST images

- To visualise the distribution of MNIST images we need a 2D representation. 
- Each 28×28 pixels image is represented by a 784-dimensional feature vector. 
- We can use an autoencoder with the a **2D latent space**:  784 → 128 → 2 → 128 → 784

---

### Latent space visualisation

![Image](07-lecture/2D-MNIST.png)


===

##  Anomaly detection

[Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) or [outlier](https://en.wikipedia.org/wiki/Outlier) detection is the identification of observations which deviate significantly from other observations.

> [!NOTE]
> Anomaly detection is used for cybersecurity, fraud detection, medicine, computer vision, and other applications. 

---

### Classifiers

Classifiers assume that input belongs to one of the trained classes.

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

> [!NOTE]
> A digit classifier will always predict a digit, even for inputs that are not a digit.

---

### Combined autoencoder and classifier

Autoencoder can be used to detect outliers based on the reconstruction error.

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
  <text class="predictedDigit detectAnomaly" x="200" y="350" font-size="400" text-anchor="middle"  style="fill:white;stroke:white;">?</text>
</svg>
</td>
</tr>
</table>

> [!NOTE]
> If the reconstruction error is above a given threshold, we have an outlier and skip classification.

