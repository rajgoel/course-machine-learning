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

![Image](05-lecture/apple_stock.png)

---

### Example: Image filtering

<div class="twocolumn" style="display: flex; align-items: center;">

<div>
<img src="05-lecture/containers_C00.svg" style="width=400px;" />
</div>

<div>
<img src="05-lecture/container_contours.svg" style="width=400px;" />
</div>

</div>

---

## Sliding filters


===

## Convolutional neural networks

---

Input: A signal (e.g., time series, image, sequence).

Filter: A function that defines what to emphasize and what to ignore.

Output: A new signal with modified characteristics (e.g., smoothed, sharpened, edge-detected).

Formally, a filter often acts as a convolution (or cross-correlation) of the input with a kernel:

y = filtered output.

---

### Examples of Filtering

---

1. Time Series Filtering

Input: Daily temperature readings.

Filter: Moving average kernel 
[1/3,1/3,1/3]
[1/3,1/3,1/3].

Effect: Smooths out fluctuations, highlighting the trend.

Example:

Signal: 
[30,32,29,31,28]
[30,32,29,31,28]

Filtered (window=3): 
[30.3,30.7,29.3]
[30.3,30.7,29.3]

---

2. Image Filtering

Input: Grayscale image (matrix of pixel intensities).

Filter: Edge detection kernel, e.g. Sobel 
[[−1,0,1],[−2,0,2],[−1,0,1]]
[[−1,0,1],[−2,0,2],[−1,0,1]].

Effect: Highlights vertical edges.

Example (conceptual): A picture of a square becomes an outline emphasizing its borders.

---

## Convolutional Neural Networks

Yann LeCun developed LeNet in 1988 as one of the very first convolutional neural networks
It was used mainly for character recognition tasks such as reading zip codes, digits, etc.

---

## A simple Multilayer Perceptron Neural Net will:

- Learn non-spatial, global features from the flattened pixel raw values.
- May struggle with local patterns, spatial dependencies, and translation invariance.
- Bigger images will require a very large input layer -> explosion of the number of parameters.

---

## Higher-level representations

- In Convolutional Neural Networks (CNNs or ConvNets), each layer is represented in 2D, which makes it easier to match neurons with their corresponding neurons.

- Neurons in the first layer are not all connected to all pixels of the input image, but only to pixels in their receptive field.
- Neurons in the second layer are connected only to neurons located in a small rectangle in the first layer.
- The CNNs concentrate on small low-level features in the first layers and higher-level features in deeper layers.

---

## Components of a ConvNet architecture:

- Convolution layer: extracts features from the input images
- Non Linearity (ReLU): Adds the non-linear activation
- Pooling or Sub Sampling: reduces the size of the input.
- Fully Connected Layer: outputs the classification prediction.

---

## Image filtering

<div class=highlight-box>
The code to guide this section can be found under traffic_signs/cnn_concepts/filtering.jl
</div>

<div class="twocolumn" style="display: flex; align-items: center;">

<div>
<img
    src="05-lecture/convolution/filter.svg"
    alt="convolution filter" />
    <figcaption>3X3 filter/Kernel</figcaption>
</div>

<div>
<figure>
  <img
    src="05-lecture/convolution/convolution_schematic.gif"
    alt="convolution operation " />
  <figcaption>Source: https://divsoni2012.medium.com/translation-invariance-in-convolutional-neural-networks-61d9b6fa03df</figcaption>
</figure>
</div>

</div>


- Every stride of the filter (orange matrix on the left) will result into one entry in the pink matrix.

---

### Applying filters to images

A filter is a 2D matrix of variable sizes. When applied to an image, will modify its pixel values and reveal features.

#### Example: Sobel filter for vertical edge detection


<div class="threecolumn" style="display: flex; align-items: center;">

<div>
<img
    src="05-lecture/filtering/sobel_x.svg"
    alt="convolution filter" />
    <figcaption>3X3 filter/Kernel</figcaption>
</div>

<div>
<img
    src="05-lecture/filtering/cameraman.png"
    alt="convolution filter" />
    <figcaption>original grayscale image</figcaption>
</div>

<div>
<figure>
  <img
    src="05-lecture/filtering/sobel_x.png"
    alt="convolution operation " />
  <figcaption>filtered image</figcaption>
</figure>
</div>

</div>

---

### Sobel filter for horizontal edge detection

<div class="threecolumn" style="display: flex; align-items: center;">

<div>
<img
    src="05-lecture/filtering/sobel_y.svg"
    alt="convolution filter" />
    <figcaption>3X3 filter/Kernel</figcaption>
</div>

<div>
<img
    src="05-lecture/filtering/cameraman.png"
    alt="convolution filter" />
    <figcaption>original grayscale image: https://testimages.juliaimages.org/stable/ </figcaption>
</div>

<div>
<figure>
  <img
    src="05-lecture/filtering/sobel_y.png"
    alt="convolution operation " />
  <figcaption>filtered image</figcaption>
</figure>
</div>

</div>

### How to combine the effects of vertical and horizontal edge detection?

---


### Filtering RGB images

Each of the RGB channels to is filtered one-by-one, and the results are stitched back into an RGB image:

<div class="twocolumn" style="display: flex; align-items: center;">

<div>
<img
    src="05-lecture/filtering/lighthouse.png"
    alt="convolution filter" />
    <figcaption>Original RGB image. Source: Images.jl </figcaption>
</div>

<div>
<img
    src="05-lecture/filtering/lighthouse_edges.png"
    alt="convolution filter" />
    <figcaption>Reconstructed image with edges detected  </figcaption>
</div>


</div>

---

Traffic signs have strong geometric boundaries (circles, triangles, rectangles) that edge detection can highlight well

<img
    src="05-lecture/filtering/edges_traffic.png"
     alt="example"
     style="width: 90%; height: auto;"/>
<figcaption>Edges filtering in a traffic scene</figcaption>

---


<div class="twocolumn" style="display: flex; align-items: center;">

<div>
<img
    src="05-lecture/filtering/orginal_scene_00025.png"
     alt="example"
     style="width: 90%; height: auto;"/>
<figcaption>Original traffic scene</figcaption>
</div>

<div>
<img
    src="05-lecture/filtering/edges_00025.png"
     alt="example"
     style="width: 90%; height: auto;"/>
<figcaption>Edges filtering in a traffic scene (missing)</figcaption>
</div>


</div>

Image → Edge Detection → Contour Finding → Shape Analysis → Feature Extraction → Classification

Limitations:

- Complex backgrounds
- Partial occlusion
- Various lighting conditions
- Multiple scales simultaneously


