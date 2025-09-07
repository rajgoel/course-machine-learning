# Filtering and convolution

> [!CAUTION]
> Incomplete


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

### LeNet

![Image](05-lecture/LeNet-5_architecture.svg)

<small>Source: [Zhang, Lipton, Li, and Smola. Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)</small>

===

<!-- .slide: data-auto-animate="true" -->

### Padding

---

<!-- .slide: data-auto-animate="true" -->

### Padding

The convolution we used so far, reduces dimensionality. This may be unwanted, in particular, when using multiple convolutional layers after each other.
 
![Image](05-lecture/no_padding.svg)


---

<!-- .slide: data-auto-animate="true" -->

### Padding

Padding can be used to maintain dimensionality.
 
![Image](05-lecture/padding.svg)

---

<!-- .slide: data-auto-animate="true" -->

### Padding

Padding can be used to maintain dimensionality.
 
![Image](05-lecture/padding.svg)

> [!NOTE]
> Instead of zero padding, we can also fill missing numbers with averages or similar. 

---

<!-- .slide: data-auto-animate="true" -->

### Striding

---

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

