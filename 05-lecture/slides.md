# Filtering and convolution

===

## Filtering

Filtering is the process of transforming an input to emphasize certain aspects while suppressing others.

> [!NOTE]
>  **Examples:**  
>  - Remove noise from measurements
>  - Smooth fluctuations in time series
>  - Highlight edges in images
>  - Extract trends from data

---

### Example: Moving averages

![Image](05-lecture/apple_stock.png)<!-- .element: style="width:1000px;" -->

---

### Example: Image filtering


![Image](05-lecture/containers_CC0.jpg)<!-- .element: style="width:600px;" -->
![Image](05-lecture/container_contours.png)<!-- .element: style="width:600px;" -->


---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_0.svg)


---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_1.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_2.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_3.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_4.svg)


---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_5.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_6.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_7.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_8.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_9.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_10.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_11.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_12.svg)


---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_13.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_14.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_15.svg)

---

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_16.svg)

---

How do we come up with suitable filters?

===

## Convolutional neural networks

---

### Learned filters

In convolutional neural networks, the parameters $w_{i,j}$ are learned during training.
 
![Image](05-lecture/convolution.svg)

> [!NOTE]
> The gradient for backpropagation can be determined similar as before, but care must be taken to use the right indices. 

---

### Convolutional layers

Multiple filters can be applied in parallel at each convolutional layer. Each filter can learn to detect different features of the input.

> [!IMPORTANT]
> If all weights are initialized with the same values, the filters may end up learning the same features.

---

### LeNet-5

The LeNet-5 architecture, introduced by [LeCun, Bottou, Bengio, and Haffner (1998)](https://hal.science/hal-03926082/document), is considered as a foundational model for modern CNN architectures.

![Image](05-lecture/LeNet-5_architecture.svg)
<!-- .element: style="margin-top:-100px;margin-bottom:-100px;" -->

<small >Source: [Zhang, Lipton, Li, and Smola. Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)</small>

===

<!-- .slide: data-auto-animate="true" -->

### Padding

The convolution we used so far, reduces dimensionality. This may be unwanted, in particular, when using multiple convolutional layers after each other.
 
![Image](05-lecture/no_padding.svg)


---

<!-- .slide: data-auto-animate="true" -->

### Padding

Padding can be used to maintain dimensionality.
 
![Image](05-lecture/padding.svg)

> [!NOTE]
> Instead of zero padding, we can also fill missing numbers with averages or similar. 
<!-- .element: class="fragment" -->

===

<!-- .slide: data-auto-animate="true" -->

### Striding

The convolution we used so far, assumes that we slide the input one field at a time. For high dimensional data, we may want to move the filter in larger steps.
 
![Image](05-lecture/convolution.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Convolution with a stride of 3.

![Image](05-lecture/striding_1.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Convolution with a stride of 3.

![Image](05-lecture/striding_2.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Convolution with a stride of 3.

![Image](05-lecture/striding_3.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Convolution with a stride of 3.

![Image](05-lecture/striding_4.svg)

===

## RGB images and channels

> [!CAUTION]
> This section requires revision.

---

### RGB color model

Real-world images are typically represented using the RGB color model with three channels:
- **Red channel**: Intensity of red light (0-255)
- **Green channel**: Intensity of green light (0-255) 
- **Blue channel**: Intensity of blue light (0-255)

![Image](05-lecture/rgb_channels.png)<!-- .element: style="width:800px;" -->

---

### Multi-channel representation

RGB images are 3D tensors with dimensions: **Height × Width × Channels**

```
Grayscale image: 28 × 28 × 1
RGB image:       28 × 28 × 3
```

Each pixel contains three values instead of one:
```
Grayscale: [127]
RGB:       [255, 128, 64]  # [R, G, B]
```

---

### Convolution with multiple channels

For RGB inputs, filters must have the same number of input channels:
- **Grayscale filter**: 3 × 3 × 1
- **RGB filter**: 3 × 3 × 3

![Image](05-lecture/rgb_convolution.svg)

The convolution operation sums across all channels:
$$y_{i,j} = \sum_{c=1}^{3} \sum_{u=-1}^{1} \sum_{v=-1}^{1} w_{u,v,c} \cdot x_{i+u,j+v,c}$$

---

### Multiple output channels

Convolutional layers can produce multiple output channels (feature maps):
- **Input**: Height × Width × 3 (RGB)
- **Filters**: N filters of size 3 × 3 × 3
- **Output**: Height × Width × N

Each filter learns to detect different features across all input channels.

---

### Flux.jl differences

When working with multi-channel data in Flux.jl:

```julia
# Define a convolutional layer for RGB input
conv_layer = Conv((3, 3), 3 => 32, relu)  # 3 input channels, 32 output channels

# Input tensor shape: (height, width, channels, batch_size)
x = randn(Float32, 28, 28, 3, 1)  # Single RGB image

# Forward pass
y = conv_layer(x)  # Output: (26, 26, 32, 1)
```

> [!NOTE]
> Flux uses **(height, width, channels, batch)** ordering, different from some other frameworks that use **(batch, channels, height, width)**.


