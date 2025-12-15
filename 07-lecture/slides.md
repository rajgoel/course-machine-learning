# Autoencoders

> [!CAUTION]
> Incomplete

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

Autoencoders minimise the **reconstruction error**, i.e., the mean squared error between an input and the reconstructed output of a given input $a$.

$$\mathscr{L}_a(\theta) = \frac{1}{n} \sum_{i=1}^{n} ( \hat{a}_i - a_i) ^2$$

---

## Use cases

- **Data compression**: Reduce storage requirements for images, audio
- **Denoising**: Remove noise from corrupted data
- **Anomaly detection**: Identify outliers by reconstruction error
- **Feature learning**: Pre-train representations for downstream tasks


===

## MNIST example

For MNIST digit recognition:
- **Input**: 784-dimensional flattened images (28×28 pixels)
- **Architecture**: 784 → 128 → 2 → 128 → 784
- **Latent space**: 2D for visualization

The 2D latent space reveals clustering of similar digits.

> [!NOTE]
> Similar digits (6, 8, 9) cluster together in the learned 2D space, demonstrating that the autoencoder captures semantic relationships.

---

## Properties

- **Unsupervised**: No labels required during training
- **Dimensionality reduction**: Learns compressed representations
- **Non-linear**: Can capture complex data manifolds (unlike PCA)
- **Reconstruction quality**: Depends on bottleneck size and data complexity

---


> [!NOTE]
> Although auto-encoders can be used for 2D-mapping of fixed data, [t-distributed stochastic neighbor embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) is usually superior.

===

#  Anomaly detection

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
  <text class="predictedDigit anomalies" x="200" y="350" font-size="400" text-anchor="middle"  style="fill:white;stroke:white;">?</text>
</svg>
</td>
</tr>
</table>

> [!IMPORTANT]
> The model only predicts a digit if the reconstruction error is below a given threshold!

