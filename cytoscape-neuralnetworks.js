function showNeuralNetwork(container,options) {
  if ( Reveal.isReady() && container.clientHeight > 0 && container.clientWidth > 0 ) {

        var nodes = [];
	var edges = [];
	if ( options.type == "neuron") {
	  for ( var i = 1; i<= 3; i++) {
		var node = JSON.parse('{ "data": { "id": "input' + i + '" }, "position": { "x": 0, "y": ' + (i-1)*50 + ' } }');
		nodes.push(node);
	  }
	  var node = JSON.parse('{ "data": { "id": "class0" }, "position": { "x": 300, "y": 50 } }');
	  nodes.push(node);
	  nodes.push(node);
	 
	  for ( var i = 1; i<= 3; i++) {
		var edge = JSON.parse('{ "data": { "source": "input' + i + '", "target": "class0' + '" } }');
		edges.push(edge);
	  }
	}
	else if ( options.type == "simple") {
	  for ( var i = 1; i<= 4; i++) {
		var node = JSON.parse('{ "data": { "id": "input' + i + '" }, "position": { "x": 0, "y": ' + (i-1)*50 + ' } }');
		nodes.push(node);
	  }
  
      for ( var j = 0; j<= 1; j++) {
	    var node = JSON.parse('{ "data": { "id": "class' + j + '" }, "position": { "x": 300, "y": ' + (25 + j*100) + ' } }');
	    nodes.push(node);
      }
	 
	  for ( var i = 1; i<= 4; i++) {
		for ( var j = 0; j<= 1; j++) {
			var edge = JSON.parse('{ "data": { "source": "input' + i + '", "target": "class' + j + '" } }');
			edges.push(edge);
		}
	  }
	}
	else if ( options.type == "5x5") {
	  for ( var i = 1; i<= 25; i++) {
		var node = JSON.parse('{ "data": { "id": "input' + i + '" }, "position": { "x": 0, "y": ' + (i-1)*50 + ' } }');
		nodes.push(node);
	  }
      for ( var j = 0; j<= 9; j++) {
	    var node = JSON.parse('{ "data": { "id": "class' + j + '" }, "position": { "x": 600, "y": ' + (150 + j*100) + ' } }');
	    nodes.push(node);
      }
	 
	  for ( var i = 1; i<= 25; i++) {
		for ( var j = 0; j<= 9; j++) {
			var edge = JSON.parse('{ "data": { "source": "input' + i + '", "target": "class' + j + '" } }');
			edges.push(edge);
		}
	  }
	}
	else if ( options.type == "feedforward" || options.type == "full") {
	  for ( var i = 1; i<= 18; i++) {
//
		var node = JSON.parse('{ "data": { "id": "input' + i + '" }, "position": { "x": 0, "y": ' + (i-1)*50 + ' } }');
		if ( i == 9 ) {
			node.data.label = "...";
			node.css = JSON.parse('{ "text-margin-y": 50, "text-margin-x": -7.5, "text-rotation": 1.57, "font-size": 40, "font-weight": 3, "color": "firebrick" }');
		}
		else if ( i > 9 ) {
			node.position.y += 100;
		}
		nodes.push(node);
	  }
	  for ( var i = 1; i<= 16; i++) {
		var node = JSON.parse('{ "data": { "id": "first' + i + '" }, "position": { "x": 300, "y": ' + (i+1)*50 + ' } }');
		nodes.push(node);
	  }
	  for ( var i = 1; i<= 16; i++) {
		var node = JSON.parse('{ "data": { "id": "second' + i + '" }, "position": { "x": 600, "y": ' + (i+1)*50 + ' } }');
		nodes.push(node);
	  }
	  if ( options.type == "feedforward" ) {
  	    for ( var i = 0; i<= 9; i++) {
		  var node = JSON.parse('{ "data": { "id": "class' + i + '" }, "position": { "x": 900, "y": ' + (i+5)*50 + ' } }');
		  nodes.push(node);
	    }
	  }
	  else if  ( options.type == "full" ) {
  	    for ( var i = 0; i<= 9; i++) {
		  var node = JSON.parse('{ "data": { "id": "class' + i + '", "label": "' + i +'    0.0000" }, "position": { "x": 900, "y": ' + (i+5)*50 + ' } }');
		  nodes.push(node);
	    }
      }
	  for ( var i = 1; i<= 18; i++) {
		for ( var j = 1; j<= 16; j++) {
			var edge = JSON.parse('{ "data": { "source": "input' + i + '", "target": "first' + j + '" } }');
			edges.push(edge);
		}
	  }
	  for ( var i = 1; i<= 16; i++) {
		for ( var j = 1; j<= 16; j++) {
			var edge = JSON.parse('{ "data": { "source": "first' + i + '", "target": "second' + j + '" } }');
			edges.push(edge);
		}
	  }
	  for ( var i = 1; i<= 16; i++) {
		for ( var j = 0; j<= 9; j++) {
			var edge = JSON.parse('{ "data": { "source": "second' + i + '", "target": "class' + j + '" } }');
			edges.push(edge);
		}
	  }
        }
	container.style.left = 0;  
	container.style.right = 0;  
	container.style.top = 0;  
	container.style.bottom = 0;  
	var cy = cytoscape({
	  container: container,
	  layout: {
	    name: 'preset',
	    padding: 5
	  },
          style: [
            	{
	              selector: 'node',
        	      css: {
		        'border-color': 'firebrick',
			'border-width': '2px',
		        'background-color': 'lightgray',
        	        'width': 35,
        	        'height': '35',
     			}
              
            	},
            	{
	              selector: 'node[label]',
        	      css: {
        	        'content': 'data(label)',
			'text-valign':'center',
                	'text-halign':'right', 
                	'color': 'black', 
                	'font-size': '40px', 
			'text-margin-x': '20px'
              		}
              
            	},
	        {
        	      selector: 'edge',
        	      css: {
        	        'curve-style': 'bezier',
		        'line-color': '#ccc',
        	        'line-width': '0.1px',
			'opacity': 0.5
        	      }
        	},
		{
		      selector: ':selected',
		      css: {
		        'background-color': 'firebrick',
		        'line-color': 'firebrick',
		        'target-arrow-color': 'firebrick',
		        'source-arrow-color': 'firebrick'
		      }
		},
		],
		elements: { nodes, edges } 
	});
//	cy.selectionType( 'additive' );
	cy.userPanningEnabled( false );
	cy.nodes().ungrabify();

	// adjust positioning of cytoscape canvasses
	var elements = container.querySelectorAll("canvas");
	for (var i = 0; i < elements.length; i++) {
		elements[i].style.left = 0;
	}


	if ( options.type == "simple" || options.type == "5x5") {
		cy.on('select', 'node', function(event){
			var links = cy.$('node:selected').connectedEdges().select();
		});
	}
	container.cy = cy;
	container.updatePredictions = function(results) {
		for (var i = 0; i < results.length; i++) {
			cy.nodes('[id = "class' + i + '"]').data('label', i + '    ' + results[i].toFixed(4) );
		}
	};


  }
  else {
	setTimeout( showNeuralNetwork, 100, container, options );

  }
}

