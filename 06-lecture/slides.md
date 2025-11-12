# Graph neural networks 

> [!CAUTION]
> Incomplete

---

<!-- .slide: data-auto-animate="true" -->

<svg width="1000" height="500">
<g id="graph">
  <line data-id="AB" x1="50" y1="250" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="AC" x1="50" y1="250" x2="250" y2="400" stroke="gray" stroke-width="4" />
  <line data-id="BC" x1="200" y1="100" x2="250" y2="400" stroke="gray" stroke-width="4" />
  <line data-id="BD" x1="200" y1="100" x2="400" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="CD" x1="250" y1="400" x2="400" y2="250" stroke="gray" stroke-width="4" />

  <line data-id="BA" x1="200" y1="100" x2="50" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="CA" x1="250" y1="400" x2="50" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="CB" x1="250" y1="400" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="DB" x1="400" y1="250" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="DC" x1="400" y1="250" x2="250" y2="400" stroke="gray" stroke-width="4" />

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

<svg width="1000" height="500">
<g id="graph">
  <line data-id="xAB" x1="50" y1="250" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="xAC" x1="50" y1="250" x2="250" y2="400" stroke="gray" stroke-width="4" />
  <line data-id="xBC" x1="200" y1="100" x2="250" y2="400" stroke="gray" stroke-width="4" />
  <line data-id="xBD" x1="200" y1="100" x2="400" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="xCD" x1="250" y1="400" x2="400" y2="250" stroke="gray" stroke-width="4" />

  <line data-id="xBA" x1="200" y1="100" x2="50" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="xCA" x1="250" y1="400" x2="50" y2="250" stroke="gray" stroke-width="4" />
  <line data-id="xCB" x1="250" y1="400" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="xDB" x1="400" y1="250" x2="200" y2="100" stroke="gray" stroke-width="4" />
  <line data-id="xDC" x1="400" y1="250" x2="250" y2="400" stroke="gray" stroke-width="4" />

  <g data-id="xA">
    <circle cx="50" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="50" y="262.5" font-size="40" fill="black" text-anchor="middle"> A </text>
  </g>
  
  <g data-id="xB">
    <circle cx="200" cy="100" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="200" y="112.5" font-size="40" fill="black" text-anchor="middle"> B </text>
  </g>
  
  <g data-id="xC">
    <circle cx="250" cy="400" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="250" y="412.5" font-size="40" fill="black" text-anchor="middle"> C </text>
  </g>
  
  <g data-id="xD">
    <circle cx="400" cy="250" r="25" stroke="firebrick" stroke-width="4" fill="lightgray" />
    <text x="400" y="262.5" font-size="40" fill="black" text-anchor="middle"> D </text>
  </g>
</g>

<!-- connections from layer1 -> layer2 based on graph -->
<g id="gnn-links" stroke="gray" stroke-width="3">
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

<g id="layer1">
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
<g id="layer2">
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

