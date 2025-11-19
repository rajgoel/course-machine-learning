# Autoencoders

> [!CAUTION]
> Incomplete

---

[Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) are neural networks that learn compressed representations of input data through unsupervised learning.

---

## Encoder

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "encoder" }
-->
</div>

---

## Decoder

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "decoder" }
-->
</div>

---


## Autoencoder

<div class="neuralnetwork" style="height: 700px; width: 1280px!important;">
<!--
{"type": "autoencoder" }
-->
</div>

---

## Architecture

An autoencoder consists of two parts:
- **Encoder**: Maps input $\mathbf{x} \in \mathbb{R}^d$ to latent representation $\mathbf{z} \in \mathbb{R}^k$ where $k < d$
- **Decoder**: Reconstructs input from latent representation $\hat{\mathbf{x}} \in \mathbb{R}^d$

$$\mathbf{z} = f_{\text{enc}}(\mathbf{x})$$
$$\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z})$$

---

## Loss function

Autoencoders minimize reconstruction error:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$

The network learns to preserve essential information while discarding noise.

---

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

## Use cases

- **Data compression**: Reduce storage requirements for images, audio
- **Denoising**: Remove noise from corrupted data
- **Anomaly detection**: Identify outliers by reconstruction error
- **Feature learning**: Pre-train representations for downstream tasks
- **Data generation**: Sample from latent space to create new data

> [!NOTE]
> Although auto-encoders can be used for 2D-mapping of fixed data, [t-distributed stochastic neighbor embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) is usually superior.

===

#  Anomaly detection
