/******************************************
 * Reveal.js config
 ******************************************/

window.RevealConfig = {
  width: 1280,
  height: 1050,
  controls: true,
  navigationMode: 'linear',
  progress: true,
  history: false,
  center: true,
  mouseWheel: true,
  previewLinks: false,
  slideNumber: true,
  transition: 'slide',
  scrollActivationWidth: null,
  menu: {
    themes: false,
    transitions: false,
    markers: true,
    hideMissingTitles: true
  },
  mathjax2: {
    mathjax: 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js',
    config: 'TeX-AMS_HTML-full',
    // pass other options into `MathJax.Hub.Config()`
    tex2jax: {
      inlineMath: [
        ['$', '$'],
        ['\\(', '\\)'],
      ],
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    },
  },
  animate: { autoplay: true },
  chalkboard: { theme: "whiteboard" },
  chart: {
    defaults: {
    maintainAspectRatio: false,
    responsive: true,
    color: 'black',
    font: { size: 30 },
    scale: {
    beginAtZero: true,
    ticks: { stepSize: 1 },
    grid: { color: "darkgray" },
    }
    },
    bar: { backgroundColor: ['firebrick', 'darkred', 'red'] },
    pie: { backgroundColor: [['firebrick', 'darkred', 'red']] },
  },
  customcontrols: {
    controls: [
    { icon: '<i class="fa fa-pen-square"></i>', title: 'Toggle chalkboard (B)', action: 'RevealChalkboard.toggleChalkboard();' },
    { icon: '<i class="fa fa-pen"></i>', title: 'Toggle notes canvas (C)', action: 'RevealChalkboard.toggleNotesCanvas();' }
    ]
  },
  plugins: [
    RevealMarkdown,
    RevealMenu,
    RevealMath,
    RevealHighlight,
    RevealLoadContent,
    RevealAnimate,
    RevealAnything,
    RevealChalkboard,
    RevealChart,
    RevealCustomControls,
    RevealFullscreen
  ]
};
