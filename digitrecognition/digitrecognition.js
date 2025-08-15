//-------------------------------------
// loader for cnn model
//-------------------------------------
async function loadDigitRecognitionModels() {
  console.log("Loading models trained for digit recognition ...");

  // clear the model variable
  models = [ undefined, undefined ];
  
  // load the model using a HTTPS request (where you have stored your model files)
  models[0] = await tf.loadLayersModel("digitrecognition/models/model.json");
//  models[1] = await tf.loadLayersModel("digitrecognition/models/mnist-model.json");
  models[1] = await tf.loadLayersModel("digitrecognition/models/simple-model.json");
  
  console.log("Models trained for digit recognition loaded");
}

//-----------------------------------------------
// preprocess the canvas
//-----------------------------------------------
function preprocessCanvas(image) {
	// resize the input image to target size of (1, 28, 28)
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([28, 28])
		.mean(2)
		.expandDims(2)
		.expandDims()
		.toFloat();
//	console.log(tensor.shape);
	return tensor.div(255.0);
}


function getSection(element) {
	while ( element.tagName.toLowerCase() != "section" ) {
		element = element.parentElement;
	}
	return element;
}

function clearPrediction(section) {
	var element = section.querySelector('.predictedDigit');
	if (element) element.innerHTML = "?"
	var element = section.querySelector('.predictions');
	if (element) {
		element.updatePredictions([0,0,0,0,0,0,0,0,0,0]);
	}
;
}

function initialiseCanvas(container,options) {
//---------------
// Create canvas
//---------------
	let canvas = container;
	canvas.preview = getSection(container).querySelector('.preview5x5');

	canvas.style.backgroundColor = "black";
	if(typeof G_vmlCanvasManager != 'undefined') {
		canvas = G_vmlCanvasManager.initElement(canvas);
	}
	var ctx = canvas.getContext("2d");

	canvas.clickX = new Array();
	canvas.clickY = new Array();
	canvas.clickD = new Array();
	canvas.drawing;

	function startDrawing(canvas) {
		canvas.drawing = true;
		// clear prediction
		clearPrediction( getSection(canvas) );
	}


//---------------------
// MOUSE DOWN function
//---------------------
canvas.addEventListener("mousedown", function (e) {
	var rect = canvas.getBoundingClientRect();
	var mouseX = e.clientX- rect.left;
	var mouseY = e.clientY- rect.top;
	if ( !canvas.drawing) startDrawing(canvas);
	addUserGesture(canvas,mouseX, mouseY);
	drawOnCanvas(canvas);
});

//---------------------
// MOUSE MOVE function
//---------------------
canvas.addEventListener("mousemove", function (e) {
	if(canvas.drawing) {
		var rect = canvas.getBoundingClientRect();
		var mouseX = e.clientX- rect.left;
		var mouseY = e.clientY- rect.top;
		addUserGesture(canvas,mouseX, mouseY, true);
		drawOnCanvas(canvas);
	}
});

//-------------------
// MOUSE UP function
//-------------------
canvas.addEventListener("mouseup", function (e) {
	canvas.drawing = false;
});

//----------------------
// MOUSE LEAVE function
//----------------------
canvas.addEventListener("mouseleave", function (e) {
	canvas.drawing = false;
});


//-----------------------
// TOUCH START function
//-----------------------
canvas.addEventListener("touchstart", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	var rect = canvas.getBoundingClientRect();
	var touch = e.touches[0];

	var mouseX = touch.clientX - rect.left;
	var mouseY = touch.clientY - rect.top;

	if ( !canvas.drawing) startDrawing(canvas);
	addUserGesture(canvas,mouseX, mouseY);
	drawOnCanvas(canvas);

}, false);


//---------------------
// TOUCH MOVE function
//---------------------
canvas.addEventListener("touchmove", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	if(canvas.drawing) {
		var rect = canvas.getBoundingClientRect();
		var touch = e.touches[0];

		var mouseX = touch.clientX - rect.left;
		var mouseY = touch.clientY - rect.top;

		addUserGesture(canvas,mouseX, mouseY, true);
		drawOnCanvas(canvas);
	}
}, false);


//---------------------
// TOUCH END function
//---------------------
canvas.addEventListener("touchend", function (e) {
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	canvas.drawing = false;
}, false);


//-----------------------
// TOUCH LEAVE function
//-----------------------
canvas.addEventListener("touchleave", function (e) {
    e.stopPropagation();
	if (e.target == canvas) {
    	e.preventDefault();
  	}
	canvas.drawing = false;
}, false);

//--------------------
// ADD CLICK function
//--------------------
function addUserGesture(canvas,x, y, dragging) {
	canvas.clickX.push(x);
	canvas.clickY.push(y);
	canvas.clickD.push(dragging);
}

  // Function to calculate the average grayscale value in a rectangle
  function getAverageGrayValue(canvas, x, y, width, height) {
    const imageData = canvas.getContext("2d").getImageData(x, y, width, height);
    const data = imageData.data;  // Pixel data (RGBA values)
    
    let graySum = 0.0;
    const pixelCount = width * height;
    
    // Loop through the pixel data (RGBA)
    for (let i = 0; i < data.length; i += 4) {
      // Take the red channel (which will be the same as green and blue in grayscale)
      graySum += data[i]/255;  // or data[i+1] or data[i+2], since they're the same for grayscale
    }

    // Compute the average gray value
    const avgGray = graySum / pixelCount;
    return avgGray > 0.3 ? 1 : 0;
  }

//-------------------
// RE DRAW function
//-------------------
  function drawOnCanvas(canvas) {
	var ctx = canvas.getContext("2d");
    var rect = canvas.getBoundingClientRect();
    var scale =canvas.width / rect.width;

	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
	ctx.strokeStyle = "white";
	ctx.lineJoin    = "round";
	ctx.lineWidth   = canvas.width/10;

	for (var i = 0; i < canvas.clickX.length; i++) {
		ctx.beginPath();
		if(canvas.clickD[i] && i) {
			ctx.moveTo(canvas.clickX[i-1]*scale, canvas.clickY[i-1]*scale);
		} else {
			ctx.moveTo(canvas.clickX[i]*scale-1, canvas.clickY[i]*scale);
		}
		ctx.lineTo(canvas.clickX[i]*scale, canvas.clickY[i]*scale);
		ctx.closePath();
		ctx.stroke();

    	if ( canvas.preview ) {
     	  for (var x=0; x<5; x++) {
            for (var y=0; y<5; y++) {
              let grayValue =  getAverageGrayValue( canvas, x*canvas.width/5, y*canvas.height/5, canvas.width/5, canvas.height/5);
     	      let rgbValue = Math.round(grayValue * 255);
              // Create the color string in the form 'rgb(r, g, b)'
              let color = `rgb(${rgbValue}, ${rgbValue}, ${rgbValue})`;
              canvas.preview.children[5*y+x].value = grayValue;
              canvas.preview.children[5*y+x].setAttribute("fill",color);
            }
          }

	    }
    }
  }

}

//------------------------
// CLEAR CANVAS function
//------------------------
function clearCanvas(canvas) {
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
	canvas.clickX = new Array();
	canvas.clickY = new Array();
	canvas.clickD = new Array();
	
	if ( canvas.preview ) {
	  for (var i=0; i<25; i++) {
        canvas.preview.children[i].value = 0;
        canvas.preview.children[i].setAttribute("fill","black");
      }
    }
}

 

function initialisePredictionButton(container,options) {
    loadDigitRecognitionModels();
	container.addEventListener("click", async function () {
		var value = (getSection(container).querySelector('.predictedDigit') || {}).innerHTML;
		var section = getSection(container) 
		if ( isNaN(value) ) {
			// predict
			var canvas = section.querySelector('.drawDigit');
			// get image data from canvas
			var imageData = canvas.toDataURL();

			// preprocess canvas
			let tensor = preprocessCanvas(canvas);

			// make predictions on the preprocessed image tensor
			let predictions = await models[container.getAttribute("data-model")].predict(tensor).data();

			// get the model's prediction results
			let results = Array.from(predictions);
			let value = results.indexOf(Math.max(...results));

			// display the predictions
			var element = getSection(container).querySelector('.predictions');
			if (element) {
				element.updatePredictions(results);
			}
			var element = getSection(container).querySelector('.predictedDigit');
			if (element) element.innerHTML = value;

			console.log( results.indexOf(Math.max(...results)) );
//			console.log(results);
		}
		else {
			// clear drawing canvas 
			var canvas = section.querySelector('.drawDigit');
			var ctx = canvas.getContext("2d");
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			canvas.clickX = new Array();
			canvas.clickY = new Array();
			canvas.clickD = new Array();

			// clear prediction
			clearPrediction( section );
		}
	});
}

function initialise5x5PreviewButton(container,options) {
	container.addEventListener("click", async function () {
      var section = getSection(container) 
      var canvas = section.querySelector('.drawDigit');
      clearCanvas( canvas );
      // clear prediction
	  clearPrediction( section );
    });
}

let parameters5x5;
// Function to load the JSON file
async function load5x5Parameters() {
  try {
    // Fetch the JSON file
    const response = await fetch('digitrecognition/5x5digits/5x5parameters.json');

    // Check if the request was successful
    if (!response.ok) {
       throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON data
    parameters5x5 = await response.json();
    console.log("Parameters for 5x5 digit recognition loaded");
  } catch (error) {
    console.error('Error loading parameters for 5x5 digit recognition:', error);
  }
}

function initialise5x5PredictionButton(container,options) {
  load5x5Parameters();
  
  container.addEventListener("click", async function () {

  var value = (getSection(container).querySelector('.predictedDigit') || {}).innerHTML;
		var section = getSection(container) 
		if ( isNaN(value) ) {
            if ( !parameters5x5 ) return;
            // predict
			var preview = section.querySelector('.preview5x5');
			var input = [];
        	for (var j=0; j<25; j++) {
        	  input.push( preview.children[j].value || 0);
            }
//            console.log("Input:",input);
            var results = [];
            var value = 0;
            var best = -1e6;
        	for (var i=0; i<10; i++) {
        	  // determine result
        	  var result = parameters5x5[i].bias;
              for (var j=0; j<25; j++) {
                result += parameters5x5[i].weights[j] * input[j];
        	  }
        	  results.push(result);
        	  if ( results[i] > best ) {
        	    best = results[i];
        	    value = i;
        	  }
            }

			// display the predictions
			var element = getSection(container).querySelector('.predictions');
			if (element) {
				element.updatePredictions(results);
			}

            var element = getSection(container).querySelector('.predictedDigit');
			if (element) element.innerHTML = value;
		}
		else {
			// clear drawing canvas 
			var canvas = section.querySelector('.drawDigit');
			canvas.clickX = new Array();
			canvas.clickY = new Array();
			canvas.clickD = new Array();
			clearCanvas( canvas );

			// clear prediction
			clearPrediction( section );
		}
	});
}
