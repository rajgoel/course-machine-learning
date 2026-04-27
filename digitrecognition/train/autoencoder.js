import { MnistData } from './mnist_data.js';

/**
 * Create an autoencoder model for MNIST digit reconstruction
 * Architecture: 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784
 */
function createAutoencoder() {
  const model = tf.sequential();
  
  const IMAGE_SIZE = 28 * 28; // 784 pixels
  
  // Encoder layers
  model.add(tf.layers.dense({
    inputShape: [IMAGE_SIZE],
    units: 128,
    activation: 'relu',
    name: 'encoder_1'
  }));
  
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    name: 'encoder_2'
  }));
  
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    name: 'bottleneck'
  }));
  
  // Decoder layers
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    name: 'decoder_1'
  }));
  
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    name: 'decoder_2'
  }));
  
  model.add(tf.layers.dense({
    units: IMAGE_SIZE,
    activation: 'sigmoid', // Output between 0-1 for pixel values
    name: 'output'
  }));
  
  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });
  
  return model;
}

/**
 * Preprocess MNIST data for autoencoder training
 */
function preprocessData(xs) {
  return tf.tidy(() => {
    // Flatten images from 28x28 to 784
    const flattened = xs.reshape([xs.shape[0], 28 * 28]);
    // MNIST data is already normalized to [0, 1] range
    return flattened;
  });
}

/**
 * Train the autoencoder model with detailed monitoring
 */
async function trainAutoencoder(model, data) {
  const metrics = ['loss', 'val_loss', 'mae', 'val_mae'];
  const container = {
    name: 'Autoencoder Training Progress', 
    tab: 'Training', 
    styles: { height: '100%' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 256;
  const TRAIN_DATA_SIZE = 10000; // Smaller for faster debugging
  const TEST_DATA_SIZE = 2000;

  console.log('='.repeat(60));
  console.log('🚀 STARTING AUTOENCODER TRAINING');
  console.log('='.repeat(60));

  // Prepare training data with detailed logging
  console.log('📊 Preparing training data...');
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    console.log(`   Raw data shape: ${d.xs.shape}`);
    console.log(`   Raw data range: ${tf.min(d.xs).dataSync()[0]} to ${tf.max(d.xs).dataSync()[0]}`);
    
    const processedXs = preprocessData(d.xs);
    console.log(`   Processed shape: ${processedXs.shape}`);
    console.log(`   Processed range: ${tf.min(processedXs).dataSync()[0]} to ${tf.max(processedXs).dataSync()[0]}`);
    
    // Check for NaN or invalid values
    const hasNaN = tf.any(tf.isNaN(processedXs)).dataSync()[0];
    const hasInf = tf.any(tf.isInf(processedXs)).dataSync()[0];
    console.log(`   Contains NaN: ${hasNaN}, Contains Inf: ${hasInf}`);
    
    // Sample some actual values
    const sampleValues = processedXs.slice([0, 0], [1, 10]).dataSync();
    console.log(`   Sample values: [${Array.from(sampleValues).map(v => v.toFixed(3)).join(', ')}]`);
    
    return [processedXs, processedXs.clone()];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    const processedXs = preprocessData(d.xs);
    return [processedXs, processedXs.clone()];
  });

  console.log(`✅ Training data prepared: ${trainXs.shape}`);
  console.log(`✅ Test data prepared: ${testXs.shape}`);

  // Test the model with a small batch first
  console.log('🧪 Testing model with small batch...');
  const testBatch = trainXs.slice([0, 0], [4, trainXs.shape[1]]);
  const testPrediction = model.predict(testBatch);
  const testLoss = tf.losses.meanSquaredError(testBatch, testPrediction);
  console.log(`   Initial loss on 4 samples: ${testLoss.dataSync()[0]}`);
  console.log(`   Prediction range: ${tf.min(testPrediction).dataSync()[0]} to ${tf.max(testPrediction).dataSync()[0]}`);
  
  testBatch.dispose();
  testPrediction.dispose();
  testLoss.dispose();

  // Enhanced callback with detailed logging
  const detailedCallback = {
    onEpochBegin: async (epoch, logs) => {
      console.log(`\n📈 Starting Epoch ${epoch + 1}`);
    },
    
    onEpochEnd: async (epoch, logs) => {
      console.log(`\n📊 Epoch ${epoch + 1} Complete:`);
      console.log(`   Training Loss: ${logs.loss?.toFixed(6) || 'undefined'} (${typeof logs.loss})`);
      console.log(`   Validation Loss: ${logs.val_loss?.toFixed(6) || 'undefined'} (${typeof logs.val_loss})`);
      console.log(`   Training MAE: ${logs.mae?.toFixed(6) || 'undefined'}`);
      console.log(`   Validation MAE: ${logs.val_mae?.toFixed(6) || 'undefined'}`);
      
      // Check for problems
      if (logs.loss === 0 || logs.loss < 1e-10) {
        console.warn(`⚠️  Training loss suspiciously low: ${logs.loss}`);
      }
      if (isNaN(logs.loss) || isNaN(logs.val_loss)) {
        console.error(`🚨 NaN detected in losses! Stopping training.`);
        model.stopTraining = true;
      }
      if (logs.loss > 1.0) {
        console.warn(`⚠️  Training loss very high: ${logs.loss} - model may not be learning`);
      }
      
      // Early stopping if loss is very low
      if (epoch > 5 && logs.loss < 0.001) {
        console.log(`🎯 Loss very low (${logs.loss.toFixed(6)}) - early stopping!`);
        model.stopTraining = true;
      }
      
      // Memory cleanup check
      console.log(`   Memory: ${tf.memory().numTensors} tensors, ${Math.round(tf.memory().numBytes / 1024 / 1024)}MB`);
    },
    
    onBatchEnd: async (batch, logs) => {
      // Log every 20 batches to avoid spam
      if (batch % 20 === 0) {
        console.log(`   Batch ${batch}: loss=${logs.loss?.toFixed(4) || 'undefined'}, mae=${logs.mae?.toFixed(4) || 'undefined'}`);
      }
    }
  };

  console.log('\n🎓 Starting training...');
  const history = await model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 50,
    shuffle: true,
    callbacks: [fitCallbacks, detailedCallback]
  });

  console.log('\n✅ Training completed!');
  
  // Final model test
  console.log('🔬 Final model test...');
  const finalTestBatch = testXs.slice([0, 0], [5, testXs.shape[1]]);
  const finalPrediction = model.predict(finalTestBatch);
  const finalLoss = tf.losses.meanSquaredError(finalTestBatch, finalPrediction);
  console.log(`   Final reconstruction loss: ${finalLoss.dataSync()[0]}`);
  
  finalTestBatch.dispose();
  finalPrediction.dispose();
  finalLoss.dispose();

  // Clean up tensors
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();

  return history;
}

/**
 * Evaluate reconstruction quality and compute reconstruction error statistics
 */
async function evaluateAutoencoder(model, data) {
  const TEST_SIZE = 1000;
  
  console.log('Evaluating autoencoder...');
  
  const testData = data.nextTestBatch(TEST_SIZE);
  const testXs = preprocessData(testData.xs);
  
  // Get reconstructions
  const reconstructions = model.predict(testXs);
  
  // Calculate reconstruction errors (MSE per sample)
  const reconstructionErrors = tf.tidy(() => {
    const diff = testXs.sub(reconstructions);
    const squaredDiff = diff.square();
    return squaredDiff.mean(1); // Mean over features (axis 1)
  });
  
  // Get statistics
  const errorStats = await tf.tidy(() => {
    const errors = reconstructionErrors.arraySync();
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
    const sortedErrors = errors.sort((a, b) => a - b);
    const median = sortedErrors[Math.floor(sortedErrors.length / 2)];
    const q75 = sortedErrors[Math.floor(sortedErrors.length * 0.75)];
    const q95 = sortedErrors[Math.floor(sortedErrors.length * 0.95)];
    const max = Math.max(...errors);
    
    return { mean, median, q75, q95, max, errors };
  });
  
  console.log('Reconstruction Error Statistics (Normal MNIST):');
  console.log(`Mean: ${errorStats.mean.toFixed(6)}`);
  console.log(`Median: ${errorStats.median.toFixed(6)}`);
  console.log(`75th percentile: ${errorStats.q75.toFixed(6)}`);
  console.log(`95th percentile: ${errorStats.q95.toFixed(6)}`);
  console.log(`Max: ${errorStats.max.toFixed(6)}`);
  
  // Save statistics for later use in out-of-sample detection
  const autoencoderStats = {
    mean: errorStats.mean,
    median: errorStats.median,
    q75: errorStats.q75,
    q95: errorStats.q95,
    max: errorStats.max,
    threshold: errorStats.q95 // Use 95th percentile as threshold
  };
  
  // Save as JSON for use in production
  const statsJson = JSON.stringify(autoencoderStats, null, 2);
  console.log('\nAutoencoder statistics (save this for outofsample.js):');
  console.log(statsJson);
  
  // Test with 90-degree rotated images first to get thresholds
  console.log('\n🔄 Testing with 90-degree rotated images (out-of-sample test):');
  const rotatedStats = await testRotatedImages(model, data);
  
  // Now we have both thresholds
  const thresholds = {
    worstInSample: errorStats.max,
    bestOutOfSample: rotatedStats.min
  };
  
  console.log(`\n📊 Dynamic quality thresholds:`);
  console.log(`   Digit: < ${thresholds.bestOutOfSample.toFixed(6)} (better than best out-of-sample)`);
  console.log(`   Unknown: ${thresholds.bestOutOfSample.toFixed(6)} - ${thresholds.worstInSample.toFixed(6)} (between)`);
  console.log(`   No digit: > ${thresholds.worstInSample.toFixed(6)} (worse than worst in-sample)`);
  
  // Now visualize both with same thresholds
  await visualizeReconstructions(testXs, reconstructions, reconstructionErrors, 10, 'Reconstruction Quality Analysis', thresholds);
  await visualizeRotatedImages(rotatedStats.rotatedXs, rotatedStats.rotatedReconstructions, rotatedStats.rotatedErrors, 8, thresholds);
  
  // Clean up all tensors
  testXs.dispose();
  reconstructions.dispose();
  reconstructionErrors.dispose();
  rotatedStats.rotatedXs.dispose();
  rotatedStats.rotatedReconstructions.dispose();
  rotatedStats.rotatedErrors.dispose();
  
  // Update autoencoder stats
  autoencoderStats.bestOutOfSample = rotatedStats.min;
  autoencoderStats.worstInSample = errorStats.max;
  
  return autoencoderStats;
}

/**
 * Test autoencoder with 90-degree rotated images (out-of-sample detection test)
 */
async function testRotatedImages(model, data, thresholds = {}) {
  const TEST_SIZE = 50;
  
  const testData = data.nextTestBatch(TEST_SIZE);
  
  // Create 90-degree rotated versions
  const rotatedXs = tf.tidy(() => {
    const reshaped = testData.xs.reshape([TEST_SIZE, 28, 28, 1]);
    // Rotate 90 degrees clockwise by transposing and flipping
    const transposed = tf.transpose(reshaped, [0, 2, 1, 3]);
    const rotated = tf.reverse(transposed, [2]);
    return rotated.reshape([TEST_SIZE, 784]);
  });
  
  // Get reconstructions for rotated images
  const rotatedReconstructions = model.predict(rotatedXs);
  
  // Calculate reconstruction errors for rotated images
  const rotatedErrors = tf.tidy(() => {
    const diff = rotatedXs.sub(rotatedReconstructions);
    const squaredDiff = diff.square();
    return squaredDiff.mean(1);
  });
  
  const rotatedErrorValues = await rotatedErrors.data();
  
  // Statistics for rotated images
  const rotatedStats = {
    mean: rotatedErrorValues.reduce((a, b) => a + b, 0) / rotatedErrorValues.length,
    min: Math.min(...rotatedErrorValues),
    max: Math.max(...rotatedErrorValues),
    median: [...rotatedErrorValues].sort((a, b) => a - b)[Math.floor(rotatedErrorValues.length / 2)]
  };
  
  console.log(`   Rotated images reconstruction error - Mean: ${rotatedStats.mean.toFixed(6)}`);
  console.log(`   Rotated images reconstruction error - Range: ${rotatedStats.min.toFixed(6)} to ${rotatedStats.max.toFixed(6)}`);
  console.log(`   Rotated images reconstruction error - Median: ${rotatedStats.median.toFixed(6)}`);
  
  // Show a few examples
  console.log('   Sample rotated image errors:', 
    rotatedErrorValues.slice(0, 5).map(e => e.toFixed(6)).join(', '));
  
  // Return data for visualization (don't clean up here)
  return {
    min: rotatedStats.min,
    max: rotatedStats.max,
    mean: rotatedStats.mean,
    median: rotatedStats.median,
    rotatedXs,
    rotatedReconstructions,
    rotatedErrors
  };
}

/**
 * Visualize rotated images for out-of-sample detection test
 */
async function visualizeRotatedImages(rotatedXs, reconstructions, errors, numSamples = 8, thresholds = {}) {
  console.log('🎨 Creating rotated images visualization...');
  
  const surface = tfvis.visor().surface({ 
    name: 'Rotated Images (Out-of-sample evaluation)', 
    tab: 'Out-of-sample'
  });
  
  const container = document.createElement('div');
  container.style.display = 'grid';
  container.style.gridTemplateColumns = 'repeat(4, 1fr)';
  container.style.gap = '8px';
  container.style.padding = '15px';
  container.style.background = '#f9f9f9';
  container.style.borderRadius = '8px';
  
  // Add headers for rotated images
  const headers = ['Rotated', 'Reconstructed', 'Error Value', 'Result'];
  headers.forEach(header => {
    const headerEl = document.createElement('div');
    headerEl.textContent = header;
    headerEl.style.fontWeight = 'bold';
    headerEl.style.textAlign = 'center';
    headerEl.style.padding = '8px';
    headerEl.style.background = '#333';
    headerEl.style.color = 'white';
    headerEl.style.borderRadius = '4px';
    headerEl.style.fontSize = '12px';
    container.appendChild(headerEl);
  });
  
  const errorValues = await errors.data();
  console.log(`   Rotated error range: ${Math.min(...errorValues).toFixed(6)} to ${Math.max(...errorValues).toFixed(6)}`);
  
  // Sort by error to show best and worst reconstructions
  const sortedIndices = Array.from({length: numSamples}, (_, i) => i)
    .sort((a, b) => errorValues[a] - errorValues[b]);
  
  for (let idx = 0; idx < numSamples; idx++) {
    const i = sortedIndices[idx];
    
    // Original rotated image
    const rotatedImg = tf.tidy(() => {
      return rotatedXs.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
    });
    const rotatedCanvas = document.createElement('canvas');
    rotatedCanvas.width = rotatedCanvas.height = 84;
    rotatedCanvas.style.border = '2px solid #2196F3';
    rotatedCanvas.style.borderRadius = '4px';
    rotatedCanvas.title = 'Rotated input image';
    await tf.browser.toPixels(rotatedImg, rotatedCanvas);
    container.appendChild(rotatedCanvas);
    
    // Reconstructed image
    const reconstructedImg = tf.tidy(() => {
      return reconstructions.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
    });
    const reconCanvas = document.createElement('canvas');
    reconCanvas.width = reconCanvas.height = 84;
    reconCanvas.style.border = '2px solid #4CAF50';
    reconCanvas.style.borderRadius = '4px';
    reconCanvas.title = 'Autoencoder reconstruction';
    await tf.browser.toPixels(reconstructedImg, reconCanvas);
    container.appendChild(reconCanvas);
    
    // Error value with color coding
    const errorValue = errorValues[i];
    const errorDiv = document.createElement('div');
    errorDiv.textContent = errorValue.toFixed(5);
    errorDiv.style.textAlign = 'center';
    errorDiv.style.alignSelf = 'center';
    errorDiv.style.fontFamily = 'monospace';
    errorDiv.style.fontSize = '14px';
    errorDiv.style.fontWeight = 'bold';
    errorDiv.style.padding = '8px';
    errorDiv.style.borderRadius = '4px';
    
    // Color code based on error level to match result colors
    if (errorValue < thresholds.bestOutOfSample) {
      errorDiv.style.background = '#E8F5E8';
      errorDiv.style.color = '#4CAF50';
    } else if (errorValue > thresholds.worstInSample) {
      errorDiv.style.background = '#FFEBEE';
      errorDiv.style.color = '#F44336';
    } else {
      errorDiv.style.background = '#FFF3E0';
      errorDiv.style.color = '#FF9800';
    }
    container.appendChild(errorDiv);
    
    // Quality assessment based on dynamic thresholds
    const qualityDiv = document.createElement('div');
    let quality, color;
    
    // Dynamic quality based on actual data ranges
    if (errorValue < thresholds.bestOutOfSample) {
      quality = '🟢 Digit';
      color = '#4CAF50';
    } else if (errorValue > thresholds.worstInSample) {
      quality = '🔴 No digit';
      color = '#F44336';
    } else {
      quality = '🟡 Unknown';
      color = '#FF9800';
    }
    
    qualityDiv.textContent = quality;
    qualityDiv.style.textAlign = 'center';
    qualityDiv.style.alignSelf = 'center';
    qualityDiv.style.fontWeight = 'bold';
    qualityDiv.style.color = color;
    qualityDiv.style.fontSize = '12px';
    container.appendChild(qualityDiv);
    
    // Clean up tensors
    rotatedImg.dispose();
    reconstructedImg.dispose();
  }
  
  // Add summary statistics
  const statsDiv = document.createElement('div');
  statsDiv.style.gridColumn = '1 / -1';
  statsDiv.style.marginTop = '15px';
  statsDiv.style.padding = '10px';
  statsDiv.style.background = '#fff9c4';
  statsDiv.style.border = '1px solid #f57c00';
  statsDiv.style.borderRadius = '4px';
  statsDiv.innerHTML = `
    <strong>Rotated Images Statistics:</strong><br>
    Average Error: ${(errorValues.reduce((a, b) => a + b, 0) / errorValues.length).toFixed(5)}<br>
    Min Error: ${Math.min(...errorValues).toFixed(5)}<br>
    Max Error: ${Math.max(...errorValues).toFixed(5)}
  `;
  container.appendChild(statsDiv);
  
  surface.drawArea.appendChild(container);
}

/**
 * Visualize original vs reconstructed images with detailed comparison
 */
async function visualizeReconstructions(originals, reconstructions, errors, numSamples = 12, title = 'Reconstruction Quality Analysis', thresholds = {}) {
  console.log('🎨 Creating reconstruction visualization...');
  
  const surface = tfvis.visor().surface({ 
    name: title, 
    tab: 'Evaluation'
  });
  
  const container = document.createElement('div');
  container.style.display = 'grid';
  container.style.gridTemplateColumns = 'repeat(4, 1fr)';
  container.style.gap = '8px';
  container.style.padding = '15px';
  container.style.background = '#f9f9f9';
  container.style.borderRadius = '8px';
  
  // Add headers with better styling
  const firstColumn = title.includes('Rotated') ? 'Rotated' : 'Original';
  const headers = [firstColumn, 'Reconstructed', 'Error Value', 'Result'];
  headers.forEach(header => {
    const headerEl = document.createElement('div');
    headerEl.textContent = header;
    headerEl.style.fontWeight = 'bold';
    headerEl.style.textAlign = 'center';
    headerEl.style.padding = '8px';
    headerEl.style.background = '#333';
    headerEl.style.color = 'white';
    headerEl.style.borderRadius = '4px';
    headerEl.style.fontSize = '12px';
    container.appendChild(headerEl);
  });
  
  const errorValues = await errors.data();
  console.log(`   Error range: ${Math.min(...errorValues).toFixed(6)} to ${Math.max(...errorValues).toFixed(6)}`);
  
  // Sort by error to show best and worst reconstructions
  const sortedIndices = Array.from({length: numSamples}, (_, i) => i)
    .sort((a, b) => errorValues[a] - errorValues[b]);
  
  for (let idx = 0; idx < numSamples; idx++) {
    const i = sortedIndices[idx];
    
    // Original image with border
    const originalImg = tf.tidy(() => {
      return originals.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
    });
    const origCanvas = document.createElement('canvas');
    origCanvas.width = origCanvas.height = 84;
    origCanvas.style.border = '2px solid #4CAF50';
    origCanvas.style.borderRadius = '4px';
    origCanvas.title = 'Original MNIST digit';
    await tf.browser.toPixels(originalImg, origCanvas);
    container.appendChild(origCanvas);
    
    // Reconstructed image with border
    const reconstructedImg = tf.tidy(() => {
      return reconstructions.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
    });
    const reconCanvas = document.createElement('canvas');
    reconCanvas.width = reconCanvas.height = 84;
    reconCanvas.style.border = '2px solid #2196F3';
    reconCanvas.style.borderRadius = '4px';
    reconCanvas.title = 'Autoencoder reconstruction';
    await tf.browser.toPixels(reconstructedImg, reconCanvas);
    container.appendChild(reconCanvas);
    
    
    // Error value with color coding
    const errorValue = errorValues[i];
    const errorDiv = document.createElement('div');
    errorDiv.textContent = errorValue.toFixed(5);
    errorDiv.style.textAlign = 'center';
    errorDiv.style.alignSelf = 'center';
    errorDiv.style.fontFamily = 'monospace';
    errorDiv.style.fontWeight = 'bold';
    errorDiv.style.padding = '8px';
    errorDiv.style.borderRadius = '4px';
    
    // Color code based on error level to match result colors
    if (errorValue < thresholds.bestOutOfSample) {
      errorDiv.style.background = '#E8F5E8';
      errorDiv.style.color = '#4CAF50';
      errorDiv.style.border = '2px solid #4CAF50';
    } else if (errorValue > thresholds.worstInSample) {
      errorDiv.style.background = '#FFEBEE';
      errorDiv.style.color = '#F44336';
      errorDiv.style.border = '2px solid #F44336';
    } else {
      errorDiv.style.background = '#FFF3E0';
      errorDiv.style.color = '#FF9800';
      errorDiv.style.border = '2px solid #FF9800';
    }
    container.appendChild(errorDiv);
    
    // Quality assessment based on dynamic thresholds
    const qualityDiv = document.createElement('div');
    let qualityText, qualityColor;
    
    // Dynamic quality based on actual data ranges
    if (errorValue < thresholds.bestOutOfSample) {
      qualityText = '🟢 Digit';
      qualityColor = '#4CAF50';
    } else if (errorValue > thresholds.worstInSample) {
      qualityText = '🔴 No digit';
      qualityColor = '#F44336';
    } else {
      qualityText = '🟡 Unknown';
      qualityColor = '#FF9800';
    }
    
    qualityDiv.textContent = qualityText;
    qualityDiv.style.textAlign = 'center';
    qualityDiv.style.alignSelf = 'center';
    qualityDiv.style.fontWeight = 'bold';
    qualityDiv.style.color = qualityColor;
    qualityDiv.style.fontSize = '12px';
    qualityDiv.style.padding = '4px';
    container.appendChild(qualityDiv);
    
    // Clean up tensors
    originalImg.dispose();
    reconstructedImg.dispose();
  }
  
  // Add summary statistics
  const summaryDiv = document.createElement('div');
  summaryDiv.style.gridColumn = '1 / -1';
  summaryDiv.style.marginTop = '15px';
  summaryDiv.style.padding = '15px';
  summaryDiv.style.background = '#E3F2FD';
  summaryDiv.style.borderRadius = '8px';
  summaryDiv.style.borderLeft = '4px solid #2196F3';
  
  const avgError = errorValues.reduce((a, b) => a + b, 0) / errorValues.length;
  const maxError = Math.max(...errorValues);
  const minError = Math.min(...errorValues);
  
  summaryDiv.innerHTML = `
    <strong>📊 Reconstruction Statistics:</strong><br>
    • Average Error: ${avgError.toFixed(6)}<br>
    • Min Error: ${minError.toFixed(6)} (best reconstruction)<br>
    • Max Error: ${maxError.toFixed(6)} (worst reconstruction)<br>
    • Samples shown: ${numSamples} (sorted by error, best to worst)
  `;
  
  container.appendChild(summaryDiv);
  surface.drawArea.appendChild(container);
  
  console.log(`✅ Visualization complete - ${numSamples} samples displayed`);
}

/**
 * Main training function
 */
async function run() {
  console.log('Loading MNIST data...');
  const data = new MnistData();
  await data.load();
  
  console.log('Creating autoencoder model...');
  const model = createAutoencoder();
  
  // Show model summary
  tfvis.show.modelSummary({name: 'Autoencoder Architecture', tab: 'Autoencoder'}, model);
  
  // Configure TensorFlow.js Visor to use full height (after visor is created)
  setTimeout(() => {
    const dashboard = document.querySelector('.tf-dashboard');
    if (dashboard) {
      dashboard.style.height = '100vh';
    }
  }, 100);
  
  console.log('Training autoencoder...');
  await trainAutoencoder(model, data);
  
  console.log('Evaluating autoencoder...');
  const stats = await evaluateAutoencoder(model, data);
  
  console.log('Saving model...');
  await model.save('downloads://autoencoder-model');
  
  console.log('Training complete! Check the downloads folder for the model files.');
  console.log('Copy the autoencoder statistics to use in your outofsample.js file.');
}

// Start training when page loads
document.addEventListener('DOMContentLoaded', run);
