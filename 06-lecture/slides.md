# Graph neural networks 

> [!CAUTION]
> Incomplete

---

## Graphs

Graphs can be used to describe relationships between entities, e.g., in **social networks**, **supply chains**, **business processes**, etc.

<svg width="500" height="450">
<g data-id="graph">
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

> [!NOTE]
> The terms *graph* and *network* are used interchangeably.

---

## Embeddings

An embedding maps entities to multi-dimensional vectors with the goal of positioning similar objects near each other.

<svg width="600" height="600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="600" height="600" fill="white"/>
  
  <!-- Center axes -->
  <line x1="50" y1="300" x2="550" y2="300" stroke="black" stroke-width="2"/>
  <line x1="300" y1="50" x2="300" y2="550" stroke="black" stroke-width="2"/>
  
  <!-- Arrow heads -->
  <polygon points="545,295 545,305 555,300" fill="black"/>
  <polygon points="55,295 55,305 45,300" fill="black"/>
  <polygon points="295,55 305,55 300,45" fill="black"/>
  <polygon points="295,545 305,545 300,555" fill="black"/>
  
  <!-- Axis labels -->
  <text x="50" y="280" text-anchor="middle" font-size="16" font-weight="bold">Comedy</text>
  <text x="550" y="280" text-anchor="middle" font-size="16" font-weight="bold">Thriller</text>
  <text x="300" y="30" text-anchor="middle" font-size="16" font-weight="bold">Action</text>
  <text x="300" y="580" text-anchor="middle" font-size="16" font-weight="bold">Romance</text>
  
  <!-- Scale markers -->
  <text x="50" y="315" text-anchor="middle" font-size="12" fill="black">-1</text>
  <text x="550" y="315" text-anchor="middle" font-size="12" fill="black">1</text>
  
  <text x="315" y="60" text-anchor="middle" font-size="12" fill="black">1</text>
  <text x="315" y="540" text-anchor="middle" font-size="12" fill="black">-1</text>
  
  <!-- Movies positioned using vector coordinates -->
  <!-- Pulp Fiction (0.3, 0.2) -->
  <circle cx="375" cy="250" r="8" fill="black"/>
  <text x="440" y="240" text-anchor="middle" font-size="12" font-weight="bold">Pulp Fiction (0.3, 0.2)</text>
  
  <!-- Terminator (0.4, 0.9) -->
  <circle cx="400" cy="75" r="8" fill="black"/>
  <text x="460" y="62.5" text-anchor="middle" font-size="12" font-weight="bold">Terminator (0.4, 0.9) </text>
  
  <!-- Titanic (0.1, -0.8) -->
  <circle cx="325" cy="500" r="8" fill="black"/>
  <text x="385" y="502.5" text-anchor="middle" font-size="12" font-weight="bold">Titanic (0.1, -0.8)</text>
  
  <!-- Iron Sky (-0.7, 0.2) -->
  <circle cx="125" cy="250" r="8" fill="black"/>
  <text x="70" y="235" text-anchor="middle" font-size="12" font-weight="bold">Iron Sky (-0.7, 0.2)</text>
  
  <!-- Warm Bodies (-0.5, -0.3) -->
  <circle cx="175" cy="375" r="8" fill="black"/>
  <text x="95" y="377.5" text-anchor="middle" font-size="12" font-weight="bold">Warm Bodies (-0.5, -0.3)</text>
  
</svg>

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

Graph neural networks can be used to learn embeddings of nodes (or edges) by aggregating information from local neighborhoods.

===

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

<svg width="900" height="500">
<!-- connections from layer1 -> layer2 based on graph -->
<g data-id="gnn-links" stroke="lightgray" stroke-width="3">
  <!-- A connections -->
  <line data-id="AB" x1="150" y1="100" x2="450" y2="200" />
  <line data-id="AC" x1="150" y1="100" x2="450" y2="300" />
  <!-- B connections -->
  <line data-id="BA" x1="150" y1="200" x2="450" y2="100" />
  <line data-id="BC" x1="150" y1="200" x2="450" y2="300" />
  <line data-id="BD" x1="150" y1="200" x2="450" y2="400" />
  <!-- C connections -->
  <line data-id="CA" x1="150" y1="300" x2="450" y2="100" />
  <line data-id="CB" x1="150" y1="300" x2="450" y2="200" />
  <line data-id="CD" x1="150" y1="300" x2="450" y2="400" />
  <!-- D connections -->
  <line data-id="DB" x1="150" y1="400" x2="450" y2="200" />
  <line data-id="DC" x1="150" y1="400" x2="450" y2="300" />
</g>

<!-- connections from layer2 -> layer3 based on graph -->
<g data-id="gnn-links2" stroke="lightgray" stroke-width="3">
  <!-- A connections -->
  <line data-id="AB" x1="450" y1="100" x2="750" y2="200" />
  <line data-id="AC" x1="450" y1="100" x2="750" y2="300" />
  <!-- B connections -->
  <line data-id="BA" x1="450" y1="200" x2="750" y2="100" />
  <line data-id="BC" x1="450" y1="200" x2="750" y2="300" />
  <line data-id="BD" x1="450" y1="200" x2="750" y2="400" />
  <!-- C connections -->
  <line data-id="CA" x1="450" y1="300" x2="750" y2="100" />
  <line data-id="CB" x1="450" y1="300" x2="750" y2="200" />
  <line data-id="CD" x1="450" y1="300" x2="750" y2="400" />
  <!-- D connections -->
  <line data-id="DB" x1="450" y1="400" x2="750" y2="200" />
  <line data-id="DC" x1="450" y1="400" x2="750" y2="300" />
</g>

<g data-id="gnn-self-links" stroke="lightgray" stroke-width="3">
  <line x1="150" y1="100" x2="450" y2="100" />
  <line x1="150" y1="200" x2="450" y2="200" />
  <line x1="150" y1="300" x2="450" y2="300" />
  <line x1="150" y1="400" x2="450" y2="400" />
  
  <line x1="450" y1="100" x2="750" y2="100" />
  <line x1="450" y1="200" x2="750" y2="200" />
  <line x1="450" y1="300" x2="750" y2="300" />
  <line x1="450" y1="400" x2="750" y2="400" />
</g>

<g data-id="layer1">
  <text x="150" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 1 </text>

  <g>
    <circle cx="150" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="150" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="150" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="150" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="150" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="150" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="150" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="150" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<g data-id="layer2">
  <text x="450" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 2 </text>

  <g>
    <circle cx="450" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="450" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="450" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="450" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="450" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="450" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="450" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="450" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<g data-id="layer3">
  <text x="750" y="50" font-size="30" fill="black" text-anchor="middle"> GNN layer 3 </text>

  <g>
    <circle cx="750" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="750" y="112.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g>
    <circle cx="750" cy="200" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="750" y="212.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g>
    <circle cx="750" cy="300" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="750" y="312.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g>
    <circle cx="750" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="750" y="412.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>
</svg>

> [!NOTE]
> With $k$ GNN layers, information can flow between nodes that are $k$ connections apart in the original graph.
