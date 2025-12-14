
# Graph neural networks 

---

## Graphs

Graphs can be used to describe relationships between entities, e.g., in **social networks**, **supply chains**, **business processes**, etc.

![Graph](06-lecture/graph.svg)

> [!NOTE]
> The terms *graph* and *network* are used interchangeably.

---

## Learning with graph neural networks

Graph neural networks (GNN) can be used to learn by aggregating information from neighbouring nodes.

---

<!-- .slide: data-auto-animate="true" -->

## GNN architecture 

<svg width="1000" height="500">
<g data-id="graph">
  <text x="225" y="50" font-size="30" fill="black" text-anchor="middle"> Original graph </text>

  <line data-id="AB" x1="50" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="AC" x1="50" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BC" x1="200" y1="100" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BD" x1="200" y1="100" x2="400" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CD" x1="250" y1="400" x2="400" y2="250" stroke="lightgray" stroke-width="4" />

  <line data-id="BA" x1="200" y1="100" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CA" x1="250" y1="400" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CB" x1="250" y1="400" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DB" x1="400" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DC" x1="400" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />

  <g data-id="A">
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g data-id="B">
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g data-id="C">
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g data-id="D">
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

</svg>

---

<!-- .slide: data-auto-animate="true" -->

## GNN architecture 

<svg width="1000" height="500">
<g data-id="graph">
  <text x="225" y="50" font-size="30" fill="black" text-anchor="middle"> Original graph </text>

  <line data-id="AB" x1="50" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="AC" x1="50" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BC" x1="200" y1="100" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BD" x1="200" y1="100" x2="400" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CD" x1="250" y1="400" x2="400" y2="250" stroke="lightgray" stroke-width="4" />

  <line data-id="BA" x1="200" y1="100" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CA" x1="250" y1="400" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CB" x1="250" y1="400" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DB" x1="400" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DC" x1="400" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />

  <g>
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<g data-id="layer1">
  <text x="600" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 1 </text>

  <g data-id="A">
    <circle cx="600" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g data-id="B">
    <circle cx="600" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g data-id="C">
    <circle cx="600" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g data-id="D">
    <circle cx="600" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
</svg>

---

<!-- .slide: data-auto-animate="true" -->

## GNN architecture 

<svg width="1000" height="500">
<g data-id="graph">
  <text x="225" y="50" font-size="30" fill="black" text-anchor="middle"> Original graph </text>

  <line data-id="AB" x1="50" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="AC" x1="50" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BC" x1="200" y1="100" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line data-id="BD" x1="200" y1="100" x2="400" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CD" x1="250" y1="400" x2="400" y2="250" stroke="lightgray" stroke-width="4" />

  <line data-id="BA" x1="200" y1="100" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CA" x1="250" y1="400" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line data-id="CB" x1="250" y1="400" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DB" x1="400" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line data-id="DC" x1="400" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />

  <g>
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<g data-id="layer1">
  <text x="600" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 1 </text>

  <g>
    <circle cx="600" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="600" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="600" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="600" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
<g data-id="layer2">
  <text x="900" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 2 </text>

  <g data-id="A">
    <circle cx="900" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g data-id="B">
    <circle cx="900" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g data-id="C">
    <circle cx="900" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g data-id="D">
    <circle cx="900" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
</svg>

---

<!-- .slide: data-auto-animate="true" -->

## GNN architecture 

<svg width="1000" height="500">
<g data-id="graph">
  <text x="225" y="50" font-size="30" fill="black" text-anchor="middle"> Original graph </text>

  <line x1="50" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="50" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line x1="200" y1="100" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line x1="200" y1="100" x2="400" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="400" y2="250" stroke="lightgray" stroke-width="4" />

  <line x1="200" y1="100" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="400" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="400" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />

  <g>
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<!-- connections from layer1 -> layer2 based on graph -->
<g data-id="gnn-links" stroke="lightgray" stroke-width="3">
  <!-- A connections -->
  <line data-id="AB" x1="600" y1="100" x2="900" y2="200" />
  <line data-id="AC" x1="600" y1="100" x2="900" y2="300" />
  <!-- B connections -->
  <line data-id="BA" x1="600" y1="200" x2="900" y2="100" />
  <line data-id="BC" x1="600" y1="200" x2="900" y2="300" />
  <line data-id="BD" x1="600" y1="200" x2="900" y2="400" />
  <!-- C connections -->
  <line data-id="CA" x1="600" y1="300" x2="900" y2="100" />
  <line data-id="CB" x1="600" y1="300" x2="900" y2="200" />
  <line data-id="CD" x1="600" y1="300" x2="900" y2="400" />
  <!-- D connections -->
  <line data-id="DB" x1="600" y1="400" x2="900" y2="200" />
  <line data-id="DC" x1="600" y1="400" x2="900" y2="300" />
</g>

<g data-id="layer1">
  <text x="600" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 1 </text>

  <g>
    <circle cx="600" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="600" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="600" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="600" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
<g data-id="layer2">
  <text x="900" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 2 </text>

  <g>
    <circle cx="900" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="900" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="900" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="900" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
</svg>

---

<!-- .slide: data-auto-animate="true" -->

## GNN architecture 

<svg width="1000" height="500">
<g data-id="graph">
  <text x="225" y="50" font-size="30" fill="black" text-anchor="middle"> Original graph </text>

  <line x1="50" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="50" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line x1="200" y1="100" x2="250" y2="400" stroke="lightgray" stroke-width="4" />
  <line x1="200" y1="100" x2="400" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="400" y2="250" stroke="lightgray" stroke-width="4" />

  <line x1="200" y1="100" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="50" y2="250" stroke="lightgray" stroke-width="4" />
  <line x1="250" y1="400" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="400" y1="250" x2="200" y2="100" stroke="lightgray" stroke-width="4" />
  <line x1="400" y1="250" x2="250" y2="400" stroke="lightgray" stroke-width="4" />

  <g>
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<!-- connections from layer1 -> layer2 based on graph -->
<g data-id="gnn-links" stroke="lightgray" stroke-width="3">
  <!-- A connections -->
  <line data-id="AB" x1="600" y1="100" x2="900" y2="200" />
  <line data-id="AC" x1="600" y1="100" x2="900" y2="300" />
  <!-- B connections -->
  <line data-id="BA" x1="600" y1="200" x2="900" y2="100" />
  <line data-id="BC" x1="600" y1="200" x2="900" y2="300" />
  <line data-id="BD" x1="600" y1="200" x2="900" y2="400" />
  <!-- C connections -->
  <line data-id="CA" x1="600" y1="300" x2="900" y2="100" />
  <line data-id="CB" x1="600" y1="300" x2="900" y2="200" />
  <line data-id="CD" x1="600" y1="300" x2="900" y2="400" />
  <!-- D connections -->
  <line data-id="DB" x1="600" y1="400" x2="900" y2="200" />
  <line data-id="DC" x1="600" y1="400" x2="900" y2="300" />
</g>

<g data-id="gnn-self-links" stroke="firebrick" stroke-width="3">
  <line x1="600" y1="100" x2="900" y2="100" />
  <line x1="600" y1="200" x2="900" y2="200" />
  <line x1="600" y1="300" x2="900" y2="300" />
  <line x1="600" y1="400" x2="900" y2="400" />
</g>

<g data-id="layer1">
  <text x="600" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 1 </text>

  <g>
    <circle cx="600" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="600" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="600" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="600" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="600" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
<g data-id="layer2">
  <text x="900" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 2 </text>

  <g>
    <circle cx="900" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="900" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="900" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="900" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="900" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
</svg>

---

## GNN with $L$ layers 

![Image](06-lecture/GNN.svg)

> [!NOTE]
> In a GNN with $L$ layers, information can flow between nodes that are $L$ connections apart in the original graph.

---

## Embeddings

An [embedding](https://en.wikipedia.org/wiki/Embedding_(machine_learning)) is a mapping of a complex structure into a vector of features. 

![Image](06-lecture/embeddings.svg)

> [!NOTE]
> Every node in a GNN has an embedding.

---

## Embeddings

<!-- .slide: data-auto-animate="true" -->

An embedding for node $i$ in layer $l$ with $k^l$ features is a vector of activation values $(a^l_{i,j})_{1\leq j \leq k^l}$ denoted by $h_i^l$.

`$\blacksquare\!\square\!\square\!\blacksquare\!\square\!\blacksquare$`<!-- .element: data-id="embedding" -->
 $=$ `$( a^l_{i,1}, a^l_{i,2}, \ldots, a^l_{i,k^l} )$`<!-- .element: data-id="activations" --> $=$ `$h^l_i$`<!-- .element: data-id="h" -->

---

## Embeddings

<!-- .slide: data-auto-animate="true" -->

The matrix of all embeddings in layer $l$ is denoted by $H^l$.

`$\begin{pmatrix}
  \blacksquare\!\square\!\blacksquare\!\square\!\square\!\square \\
  \blacksquare\!\square\!\square\!\blacksquare\!\square\!\blacksquare \\
  \vdots\\
  \square\!\blacksquare\!\square\!\square\!\blacksquare\!\square \\
\end{pmatrix}$`<!-- .element: data-id="embedding" -->
$=$ `$\begin{pmatrix}
  a^l_{1,1} & a^l_{1,2} & \ldots & a^l_{1,k^l}\\
  a^l_{2,1} & a^l_{2,2} & \ldots & a^l_{2,k^l}\\
  \vdots & \vdots & \ddots & \vdots\\
  a^l_{n,1} & a^l_{n,2} & \ldots & a^l_{n,k^l}\\
\end{pmatrix}$`<!-- .element: data-id="activations" --> 
$=$ `$\begin{pmatrix} h^l_1\\ h^l_2\\ \vdots\\  h^l_n \end{pmatrix}$`<!-- .element: data-id="h" -->
`$= H^l$`

> [!IMPORTANT]
> All embeddings in layer $l$ must have exactly $k^l$ features.

---

## Forward propagation

The embedding of node $i$ in GNN layer $l$ is computed by

`$$h_i^l = f_{i,\theta^l}\big(\ (h_j^{l-1})_{j\in N_i \cup \{i\}}\ \big)$$`

where $f_{i,\theta^l}$ is a parameterised function with parameters $\theta^l$ and $N_i$ is the set of neighbours of a node $i$.

> [!IMPORTANT]
> $f_{i,\theta^l}$ only uses the embeddings of node $i$ and its neighbours. 

---

## Graph convolutional networks 

In **graph convolutional networks (GCN)** we use

`$$h_i^l = \phi^l\Big(  g_i\big(\ (h_j^{l-1})_{j\in N_i \cup \{i\}}\ \big) \cdot W^l  \Big)$$`

and

`$$g_i\big(\ (h_j^{l-1})_{j\in N_i \cup \{i\}}\ \big) = \sum_{j \in N_i\cup\{i\}} \frac{v_{i,j}}{\sqrt{(|N_i|+1) \cdot (| N_j|+1)}} h_j^{l-1}$$`

where 

- $v_{i,j}$ is the edge weight for each node $i \in N_j$ and $v_{i,i}=1$,
- $W^l \in \mathbb{R}^{k^{l-1} \times k^l}$ is a learnable weight matrix for layer $l$, and 
- $\phi^l$ is an element-wise activation function, e.g., ReLU.

---

## GCN matrix operations

To compute all embeddings of a layer simultaneously, we can use the equivalent matrix operation

`$$H^l = \phi\left( \hat{A} H^{l-1} W^l \right)$$`

where 
- `$\hat{A} = D^{-\tfrac{1}{2}} (A + I) D^{-\tfrac{1}{2}}$`,
- $A$ is the weighted [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix), i.e. using edge weights $v_{i,j}$ instead of 1s,
- $D$ is a [diagonal matrix](https://en.wikipedia.org/wiki/Diagonal_matrix) with $D_{ii} = |N_i| + 1$, and
- $I$ is the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).

===

# Recommender systems

---

[Recommender systems](https://en.wikipedia.org/wiki/Recommender_system) are systems that provide suggestions for items that may be of interest to a particular user.

> [!NOTE]
> **Examples:**
> - Netflix: Movie recommendations based on viewing history
> - Amazon: Product suggestions from purchase patterns
> - Spotify: Music recommendations from listening behavior

---

## Graph representation

User-item relationships can be represented by a [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph).


![Image](06-lecture/bipartite_graph.svg)

> [!NOTE]
> Interactions (e.g., ratings) between users and items are represented by an edge between the respective nodes with an **edge weight** (not shown) indicating the level of interaction.

---

## Collaborative filtering

[Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) uses user-item interactions collected from many users to learn similarities between

- users and users,
- items and items, and
- users and items

to make recommendations.

> [!NOTE]
> Depending on use-case the user or item with the largest similarity is recommended.

---

## Embedding space

In recommender systems, we want to obtain embeddings of objects (i.e. users and items) such that similar objects are *close* each other in the embedding space.

![Image](06-lecture/2D-movies.svg)

---

## Cosine similarity

The [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of two embeddings $h_i$ and $h_j$ is

`$$\cos(\measuredangle_{h_i,h_j}) = \frac{ h_{i} \cdot h_{j} }{\|h_{i}\| \cdot \|h_{j}\|}$$`

where $\measuredangle_{h_i,h_j}$ is the angle between the two embedding vectors. A value close to 1 indicates a large similarity, a value close to -1 a small similarity.

> [!TIP]
> **Examples**:
> - The movie *Titanic* is represented by $h_i= (0.1, -0.8)$
> - The movie *Terminator* is represented by $h_j= (0.4, 0.9)$
> - The cosine similarity of the two embeddings is
> `$$\cos(\measuredangle_{h_i,h_j}) = \displaystyle \frac{0.1 \cdot 0.4 + (-0.8) \cdot 0.9}{ \sqrt{ 0.1^2 + (-0.8)^2 } \cdot \sqrt{ 0.4^2 + 0.9^2} } \approx -0.856$$`
<!-- .element: style="font-size:26pt;" -->

---

## Learning embeddings

Graph convolutional networks with $L$ layers can be used to learn embeddings of nodes, such that for every edge $(i,j)$ with a weight $v_{i,j}$, the cosine similarity of the final layer embeddings

$$\cos(\measuredangle_{h_i^L, h_j^L})$$

approximates $v_{i,j}$

> [!IMPORTANT]
> We assume that $v_{i,j} \in [-1,1]$ and may need to shift and normalise weights accordingly. 

---

## Loss

For a GCN with $L$ layers, the mean squared error is

`$$\mathscr{L}(W) = \frac{1}{|E|} \sum_{(i,j) \in E} \left( \cos(\measuredangle_{h_i^L, h_j^L}) - v_{i,j} \right)^2$$`

where $E$ is the set of edges $(i,j)$ with a weight $v_{i,j}$.

> [!IMPORTANT]
> We learn the weights **and** the first layer embeddings simultaneously.

===

# Example: MovieLens

The [MovieLens](https://juliaml.github.io/MLDatasets.jl/dev/datasets/graphs/#MLDatasets.MovieLens) dataset contains ratings of users for a variety of movies.

---

> [!WARNING]
> Not yet complete
