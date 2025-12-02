# Filtering, pooling, and convolution

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

<!-- .slide: data-auto-animate="true" -->

## Sliding filters

![Image](05-lecture/filtering_16.svg)

> [!NOTE]
> Above filter can be used to detect edges in images. 

---

## Handcrafted filters

In theory, we can try to identify filters helping us to determine specific features in images.

> [!WARNING]
> Handcrafting filters for specific features is time-consuming or impractical.


---

## Learned filters

We can learn filter parameters $w_{i,j}$ during training.
 
![Image](05-lecture/convolution.svg)

> [!NOTE]
> The gradient for backpropagation can be computed using the chain rule, similar to dense layers.

---

<!-- .slide: data-auto-animate="true" -->

### Padding

The filtering approach used so far, reduces dimensionality. This may be unwanted, in particular, when using multiple filters after another.
 
![Image](05-lecture/no_padding.svg)


---

<!-- .slide: data-auto-animate="true" -->

### Padding

Input padding can be used to maintain dimensionality. 
 
![Image](05-lecture/padding.svg)

> [!NOTE]
> Instead of zero padding, we can also fill missing numbers with averages or similar. 
<!-- .element: class="fragment" -->

---

<!-- .slide: data-auto-animate="true" -->

### Striding

For high dimensional data, we may want to reduce computational effort by moving the filter in larger steps.
 
![Image](05-lecture/convolution.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Filtering with a stride of 3.

![Image](05-lecture/striding_1.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Filtering with a stride of 3.

![Image](05-lecture/striding_2.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Filtering with a stride of 3.

![Image](05-lecture/striding_3.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Striding

Filtering with a stride of 3.

![Image](05-lecture/striding_4.svg)

---

## Pooling

Pooling reduces the dimensions of features while preserving important information.

Common pooling operations are:
- **Max pooling**: Takes the maximum value over a set of features.
- **Average pooling**: Takes the average value over a set of features.

> [!TIP]
> Pooling complements filtering:
> - Filtering  detects features, e.g., edges.
> - Pooling summarizes those features at lower resolution.

===

### Convolutional neural networks

**Convolutional neural networks** typically combine multiple learned filters with pooling. Each filter learns to detect different features of the input.

> [!IMPORTANT]
> If all weights are initialised with the same values, the filters may end up learning the same features.

---

### LeNet-5

The LeNet-5 architecture, introduced by [LeCun, Bottou, Bengio, and Haffner (1998)](https://hal.science/hal-03926082/document), is considered as a foundational model for modern CNN architectures.

![Image](05-lecture/LeNet-5_architecture.svg)
<!-- .element: style="margin-top:-100px;margin-bottom:-100px;" -->

<small >Source: [Zhang, Lipton, Li, and Smola. Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)</small>

