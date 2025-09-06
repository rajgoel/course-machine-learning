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


---

## Convolution

![Image](05-lecture/convolution.svg)

===

## Convolutional neural networks

---

## Convolution

![Image](05-lecture/convolution.svg)

---

### Gradient computation: Filter weights

For each filter weight $w^l_{m,n}$, the chain rule gives us:

`$$\frac{\partial \mathcal{L}}{\partial w^l_{m,n}} = \sum_{i,j}  \frac{\partial \mathcal{L}}{\partial y^l_{i,j}} \cdot \frac{\partial y^l_{i,j}}{\partial w^l_{m,n}}$$`

`$$= \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y^l_{i,j}} \cdot   x^{l-1}_{i+m,j+n}$$`

> [!NOTE]
> The filter gradient is the **convolution** of the input with the error signal from the next layer.

---

### Gradient computation: Input activations

For input activation $x^{l-1}_{i,j}$:

`$$\frac{\partial \mathcal{L}}{\partial x^{l-1}_{i,j}} = \sum_{m,n} 
\frac{\partial \mathcal{L}}{\partial y^l_{i-m,j-n}} \cdot w^l_{m,n}$$`

> [!NOTE]
> The input gradient uses **transposed convolution** (or full convolution) with the flipped filter.





---

### Padding

### Striding
