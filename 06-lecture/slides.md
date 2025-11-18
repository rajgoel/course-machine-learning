# Graph neural networks 

> [!CAUTION]
> Incomplete

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

## GNN with $k$ layers 

![Image](06-lecture/GNN.svg)

> [!NOTE]
> In a GNN with $k$ layers, information can flow between nodes that are $k$ connections apart in the original graph.

---

## Forward propagation

The activations of node $j$ in GNN layer $l$ is computed by

`$$a_j^l = f_{\theta^l}( a^{l-1}, j )$$`

where 

- $a_i^l$ denotes a vector of activation values for any node $i$ in layer $l$, 
- $a^l$ contains the matrix of all activation values of all nodes in layer $l$, and 
- $f_{\theta^l}$ is a parameterised function with parameters $\theta^l$.

> [!IMPORTANT]
> $f_{\theta^l}$ only uses the activation values of node $j$ and its neighbours. 

---

## Graph convolutional networks 

In **graph convolutional networks (GCN)** we use

`$$a_j^l = \sigma\left( W^l \cdot g( a^{l-1}, j ) \right)$$`

and

`$$g( a^{l-1}, j ) = \sum_{i \in N_j\cup\{j\}} \frac{v_{i,j}}{\sqrt{|N_i| \cdot | N_j|}} a_i^{l-1}$$`

where 

- $N_i$ is the set of neighbours of any node $i$,
- $v_{i,j}$ is the edge weight for each node $i \in N_j$ and $v_{j,j}=1$,
- $W^l$ is a learnable weight matrix shared across all nodes in layer $l$, and 
- $\sigma$ is an element-wise activation function, e.g., ReLU.

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

## Graphs representation

User-item relationships can be represented by a [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph).


![Image](06-lecture/bipartite_graph.svg)

Interactions (e.g., ratings) between users and items are represented by an edge between the respective nodes with an **edge weight** indicating the level of interaction.

> [!NOTE]
> The structure of the bipartite graph can be used by a GNN to learn similarities between different nodes.

---

## Embeddings

An embedding maps entities to multi-dimensional vectors with the goal of positioning similar objects near each other.

![Image](06-lecture/2D-movies.svg)

---

## Measuring similarity

The **scalar product (or dot product)** between two vectors measures their similarity:

`$$(a_1, \ldots, a_n) \begin{pmatrix} b_1 \\ \vdots \\ b_n \end{pmatrix} = \sum_{i=1}^n a_{i} \cdot b_{i}$$`

> [!TIP]
> **Examples**: 
> - Pulp Fiction (0.3, 0.2) ↔ Terminator (0.4, 0.9): 0.12 + 0.18 = 0.30
> - Titanic (0.1, -0.8) ↔ Terminator (0.4, 0.9): 0.04 - 0.72 = -0.68
> - Titanic (0.1, -0.8) ↔ Warm Bodies (-0.5, -0.3): -0.05 + 0.24 = 0.19
<!-- .element: style="font-size:28pt;" -->

---

## Learning embeddings

Graph convolutional networks can be used to learn embeddings of nodes by aggregating embeddings from neighbouring nodes and itself.

---

## 

> [!NOTE]
> For recommender systems we learn the weights **and** the initial embeddings simultaneously.

---

## Loss

The mean squared error is

$$\mathcal{L}_{W} = \frac{1}{|E|} \sum_{(u,i) \in E} \left( r_{u,i} -
  \hat{r}_{u,i} \right)^2$$

where:

- $E$ is the set of observed user-item interactions (edges)
- $r_{u,i}$ is the true rating of user $u$ for item $i$
- $\hat{r}_{u,i} = a_u^L \cdot a_i^L$ is the predicted rating (dot product of final embeddings)
  - $L$ is the final GNN layer

