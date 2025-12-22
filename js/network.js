// Neural Network Implementation
// Architecture: 3 inputs -> 5 neurons (1차) -> 3 neurons (2차) -> 1 output

class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(rows).fill().map(() => Array(cols).fill(0));
  }

  static fromArray(arr) {
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random(); // 0 to 1 (positive weights only)
      }
    }
  }
  
  randomizeBias() {
    // Bias can be slightly negative to adjust threshold
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() - 0.5; // -0.5 to 0.5
      }
    }
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error('Columns of A must match rows of B.');
      return;
    }
    let result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  static transpose(matrix) {
    let result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static subtract(a, b) {
    let result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  multiply(n) {
    if (n instanceof Matrix) {
      // Element-wise multiplication
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n.data[i][j];
        }
      }
    } else {
      // Scalar multiplication
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n;
        }
      }
    }
  }

  add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  static map(matrix, func) {
    let result = new Matrix(matrix.rows, matrix.cols);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[i][j] = func(matrix.data[i][j]);
      }
    }
    return result;
  }

  map(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = func(this.data[i][j]);
      }
    }
  }
}

// Activation functions
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  // Derivative of sigmoid (where y is already sigmoid(x))
  return y * (1 - y);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max)); // Subtract max for numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

class NeuralNetwork {
  constructor() {
    // Architecture: 3 -> 5 -> 3 -> 3 (3-class output)
    
    // Layer 1: 1차 면접관 (5 neurons, each takes 3 inputs)
    this.weights_input_hidden1 = new Matrix(5, 3);
    this.bias_hidden1 = new Matrix(5, 1);
    this.weights_input_hidden1.randomize();
    this.bias_hidden1.randomizeBias();
    
    // Layer 2: 2차 면접관 (3 neurons, each takes 5 inputs)
    this.weights_hidden1_hidden2 = new Matrix(3, 5);
    this.bias_hidden2 = new Matrix(3, 1);
    this.weights_hidden1_hidden2.randomize();
    this.bias_hidden2.randomizeBias();
    
    // Output layer: 최종 결정 (3 neurons for softmax: 불합격/보류/합격)
    this.weights_hidden2_output = new Matrix(3, 3);
    this.bias_output = new Matrix(3, 1);
    this.weights_hidden2_output.randomize();
    this.bias_output.randomizeBias();
    
    this.learning_rate = 0.1;
    
    // Store intermediate values for visualization
    this.lastInput = null;
    this.lastHidden1 = null;
    this.lastHidden2 = null;
    this.lastOutput = null;
    this.lastHidden1Raw = null; // Before activation
    this.lastHidden2Raw = null; // Before activation
    this.lastOutputRaw = null;  // Before activation
    
    // Store gradients for backprop visualization
    this.lastGradients = {
      output: null,
      layer2: null,
      layer1: null
    };
    
    // Store weight deltas for update visualization
    this.lastWeightDeltas = {
      output_layer2: null,
      layer2_layer1: null,
      layer1_input: null
    };
    
    this.lastLoss = 0;
  }

  feedforward(input_array) {
    // Convert inputs to matrix
    let inputs = Matrix.fromArray(input_array);
    this.lastInput = inputs;
    
    // Layer 1: 1차 면접관
    let hidden1_raw = Matrix.multiply(this.weights_input_hidden1, inputs);
    hidden1_raw.add(this.bias_hidden1);
    this.lastHidden1Raw = hidden1_raw;
    let hidden1 = Matrix.map(hidden1_raw, sigmoid);
    this.lastHidden1 = hidden1;
    
    // Layer 2: 2차 면접관
    let hidden2_raw = Matrix.multiply(this.weights_hidden1_hidden2, hidden1);
    hidden2_raw.add(this.bias_hidden2);
    this.lastHidden2Raw = hidden2_raw;
    let hidden2 = Matrix.map(hidden2_raw, sigmoid);
    this.lastHidden2 = hidden2;
    
    // Output layer: 최종 결정 (Softmax for 3-class classification)
    let output_raw = Matrix.multiply(this.weights_hidden2_output, hidden2);
    output_raw.add(this.bias_output);
    this.lastOutputRaw = output_raw;
    
    // Apply softmax activation
    const output_logits = output_raw.toArray();
    const output_probs = softmax(output_logits);
    let outputs = Matrix.fromArray(output_probs);
    this.lastOutput = outputs;
    
    return output_probs;
  }

  train(input_array, target_array) {
    // Feedforward
    this.feedforward(input_array);
    
    let inputs = this.lastInput;
    let hidden1 = this.lastHidden1;
    let hidden2 = this.lastHidden2;
    let outputs = this.lastOutput;
    
    // Convert target to matrix
    let targets = Matrix.fromArray(target_array);
    
    // Calculate output errors
    let output_errors = Matrix.subtract(targets, outputs);
    
    // Calculate output gradient
    let gradients_output = Matrix.map(outputs, dsigmoid);
    gradients_output.multiply(output_errors);
    gradients_output.multiply(this.learning_rate);
    
    // Calculate hidden2 -> output deltas
    let hidden2_T = Matrix.transpose(hidden2);
    let weight_ho_deltas = Matrix.multiply(gradients_output, hidden2_T);
    
    // Adjust output weights and bias
    this.weights_hidden2_output.add(weight_ho_deltas);
    this.bias_output.add(gradients_output);
    
    // Calculate hidden2 errors
    let who_t = Matrix.transpose(this.weights_hidden2_output);
    let hidden2_errors = Matrix.multiply(who_t, output_errors);
    
    // Calculate hidden2 gradient
    let gradients_hidden2 = Matrix.map(hidden2, dsigmoid);
    gradients_hidden2.multiply(hidden2_errors);
    gradients_hidden2.multiply(this.learning_rate);
    
    // Calculate hidden1 -> hidden2 deltas
    let hidden1_T = Matrix.transpose(hidden1);
    let weight_h1h2_deltas = Matrix.multiply(gradients_hidden2, hidden1_T);
    
    // Adjust hidden2 weights and bias
    this.weights_hidden1_hidden2.add(weight_h1h2_deltas);
    this.bias_hidden2.add(gradients_hidden2);
    
    // Calculate hidden1 errors
    let wh1h2_t = Matrix.transpose(this.weights_hidden1_hidden2);
    let hidden1_errors = Matrix.multiply(wh1h2_t, hidden2_errors);
    
    // Calculate hidden1 gradient
    let gradients_hidden1 = Matrix.map(hidden1, dsigmoid);
    gradients_hidden1.multiply(hidden1_errors);
    gradients_hidden1.multiply(this.learning_rate);
    
    // Calculate input -> hidden1 deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih1_deltas = Matrix.multiply(gradients_hidden1, inputs_T);
    
    // === STORE FOR VISUALIZATION ===
    this.lastGradients.output = output_errors;
    this.lastGradients.layer2 = hidden2_errors;
    this.lastGradients.layer1 = hidden1_errors;
    
    this.lastWeightDeltas.output_layer2 = weight_ho_deltas;
    this.lastWeightDeltas.layer2_layer1 = weight_h1h2_deltas;
    this.lastWeightDeltas.layer1_input = weight_ih1_deltas;
    
    // Calculate loss (cross-entropy for softmax)
    const targetOneHot = target_array;
    this.lastLoss = -targetOneHot.reduce((sum, t, i) => 
      sum + (t > 0 ? Math.log(Math.max(outputs.data[i][0], 1e-7)) : 0), 0
    );
    
    // Adjust hidden1 weights and bias
    this.weights_input_hidden1.add(weight_ih1_deltas);
    this.bias_hidden1.add(gradients_hidden1);
  }

  getCalculationSteps() {
    if (!this.lastInput) return null;
    
    const steps = {
      input: this.lastInput.toArray(),
      layer1: [],
      layer2: [],
      output: []
    };
    
    // Layer 1 calculations (1차 면접관)
    for (let i = 0; i < 5; i++) {
      const weights = this.weights_input_hidden1.data[i];
      const bias = this.bias_hidden1.data[i][0];
      const rawValue = this.lastHidden1Raw.data[i][0];
      const activatedValue = this.lastHidden1.data[i][0];
      
      steps.layer1.push({
        neuronIndex: i,
        weights: weights,
        bias: bias,
        inputs: this.lastInput.toArray(),
        dotProduct: rawValue - bias, // Subtract bias to get pure dot product
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastInput.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    // Layer 2 calculations (2차 면접관)
    for (let i = 0; i < 3; i++) {
      const weights = this.weights_hidden1_hidden2.data[i];
      const bias = this.bias_hidden2.data[i][0];
      const rawValue = this.lastHidden2Raw.data[i][0];
      const activatedValue = this.lastHidden2.data[i][0];
      
      steps.layer2.push({
        neuronIndex: i,
        weights: weights,
        bias: bias,
        inputs: this.lastHidden1.toArray(),
        dotProduct: rawValue - bias,
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastHidden1.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    // Output calculation (3 neurons with softmax)
    const classNames = ['불합격', '보류', '합격'];
    for (let i = 0; i < 3; i++) {
      const weights = this.weights_hidden2_output.data[i];
      const bias = this.bias_output.data[i][0];
      const rawValue = this.lastOutputRaw.data[i][0];
      const activatedValue = this.lastOutput.data[i][0];
      
      steps.output.push({
        neuronIndex: i,
        className: classNames[i],
        weights: weights,
        bias: bias,
        inputs: this.lastHidden2.toArray(),
        dotProduct: rawValue - bias,
        withBias: rawValue,
        activated: activatedValue,
        calculation: `(${this.lastHidden2.toArray().map((v, j) => `${v.toFixed(2)}×${weights[j].toFixed(2)}`).join(' + ')}) + ${bias.toFixed(2)} = ${rawValue.toFixed(3)}`
      });
    }
    
    return steps;
  }
}
