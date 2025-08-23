/******************************************
 * Load Course Data
 ******************************************/
fetch('course.json')
  .then(response => response.json())
  .then(course => {
    const topics = course.topics;
    const topicId = getTopicId(topics);

    /******************************************
     * Inject Course Metadata
     ******************************************/
    document.title = getTopic();
    document.getElementById('course-title').innerHTML = course.title;
    document.getElementById('course-subtitle').innerHTML = course.subtitle;
    document.getElementById('course-author').innerHTML = course.author;
    document.getElementById('course-email').innerHTML = course.email;
    document.getElementById('course-email').href = `mailto:${course.email}`;
    document.getElementById('pdf-link').href = `${getURL(topicId)}&print-pdf`;

    document.querySelector('section[data-markdown=""]').setAttribute("data-markdown", `${getPath()}/${topics[topicId].src}`);

    if ( RevealConfig.menu && topics ) {
      RevealConfig.menu.custom = [{ title: 'Topics', icon: '<i class="fas fa-external-link-alt">', content: getTopicsMenu() }];
    }
    
    if ( (topics || []).length == 0 ) {
      document.getElementById('navigation-to-next-topic').style.display = "none";
    }
    else {
      var element = document.getElementById('next-topic');
      if ( element ) {
        document.getElementById('next-topic').addEventListener('click', function(event) {
          event.preventDefault();
          nextTopic();
        });
      }
    }
    
    /******************************************
     * Utility Functions
     ******************************************/
    function getTopicId(topics) {
      const url = new URL(window.location.href);
      const identifier = url.searchParams.get("topic");

      if (identifier !== null) {
        let id = topics.findIndex(t => t.folder === identifier);
        if (id < 0 && !isNaN(identifier)) {
          id = Number(identifier);
        }
        return Math.min(id, topics.length - 1);
      }

      return 0;
    }


    function getURL(id) {
      const baseUrl = window.location.href.split('?')[0];
      return id === undefined ? baseUrl : `${baseUrl}?topic=${topics[id].folder}`;
    }

    function getTopicsMenu() {
      return '<ul class="slide-menu-items">' +
        topics.map((t, i) => `<li class="slide-menu-item"><a href="${getURL(i)}" style="text-decoration:none">${t.topic}</a></li>`).join('') +
        '</ul>';
    }

    function nextTopic() {
      if (topicId >= topics.length - 1) {
        alert("You already reached the last slide deck!");
      } else {
        window.location.href = getURL(topicId + 1);
      }
      return false;
    }

    function getPath() {
      return topics[topicId].folder;
    }

    function getTopic() {
      return topics[topicId].topic;
    }

    function initQRCode() {
      document.querySelectorAll('#qrcode').forEach((el, i) => {
        const qrCode = new QRCodeStyling({
          type: "svg",
          width: 400,
          height: 400,
          data: getURL(topicId),
          image: course.logo,
          dotsOptions: { color: "var(--r-link-color-dark)", type: "dots" },
          backgroundOptions: { color: "white" },
          imageOptions: { crossOrigin: "anonymous", margin: 20 },
          cornersSquareOptions: { color: "#000000", type: "dots" },
          cornersDotOptions: { color: "#000000", type: "dot" }
        });
        qrCode.append(el);
        qrCode._svgDrawingPromise.then(() => fix_svg_defs(qrCode._svg.getElement(), `qrcode${i}`));
      });

      document.querySelectorAll('#url').forEach(el => {
        el.innerHTML = getURL(topicId);
        el.href = getURL(topicId);
      });
    }

    /******************************************
     * Initialize QR Code
     ******************************************/
    initQRCode();

    /******************************************
     * Initialize Reveal
     ******************************************/   
    Reveal.initialize(RevealConfig)
    .then(() => {
      // Admonitions
      document.querySelectorAll("blockquote").forEach(blockquote => {
        const firstParagraph = blockquote.querySelector("p");
        if (!firstParagraph) return;

        const match = firstParagraph.textContent.trim().match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/i);
        if (match) {
          const type = match[1].toLowerCase();
          blockquote.classList.add("admonition", type);

          // Remove the [!...] marker from the first paragraph
          firstParagraph.innerHTML = firstParagraph.innerHTML.slice(match[0].length).trimStart();
        }
      });
    });
  })
  .catch(error => console.error('Error loading course data:', error));


