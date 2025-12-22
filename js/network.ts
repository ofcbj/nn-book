// network.ts - Neural Network Implementation
// Pure TypeScript implementation without external libraries

/**
 * Matrix utility functions for neural network computations
 */
class Matrix {
    static multiply(a: number[][], b: number[][]): number[][] {
        const rows = a.length;
        const cols = b[0].length;
        const common = b.length;
        
        const result: number[][] = Array(rows).fill(0).map(() => Array(cols).fill(0));
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                for (let k = 0; k < common; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }
    
    static transpose(matrix: number[][]): number[][] {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }
    
    static add(a: number[][], b: number[][]): number[][] {
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }
    
    static subtract(a: number[][], b: number[][]): number[][] {
        return a.map((row, i) => row.map((val, j) => val - b[i][j]));
    }
    
    static scalarMultiply(matrix: number[][], scalar: number): number[][] {
        return matrix.map(row => row.map(val => val * scalar));
    }
    
    static hadamard(a: number[][], b: number[][]): number[][] {
        return a.map((row, i) => row.map((val, j) => val * b[i][j]));
    }
}

/**
 * Activation functions and their derivatives
 */
class Activation {
    static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }
    
    static sigmoidDerivative(x: number): number {
        const sig = Activation.sigmoid(x);
        return sig * (1 - sig);
    }
    
    static sigmoidMatrix(matrix: number[][]): number[][] {
        return matrix.map(row => row.map(val => Activation.sigmoid(val)));
    }
    
    static sigmoidDerivativeMatrix(matrix: number[][]): number[][] {
        return matrix.map(row => row.map(val => Activation.sigmoidDerivative(val)));
    }
}

/**
 * Layer class representing a group of neurons (interviewers)
 */
class Layer {
    weights: number[][];
    biases: number[][];
    activation: number[][] | null = null;
    weightedSum: number[][] | null = null;
    input: number[][] | null = null;
    
    constructor(
        public inputSize: number,
        public outputSize: number,
        public name: string
    ) {
        // Initialize weights with Xavier initialization
        this.weights = this.initializeWeights(inputSize, outputSize);
        this.biases = Array(1).fill(0).map(() => Array(outputSize).fill(0));
    }
    
    private initializeWeights(inputSize: number, outputSize: number): number[][] {
        const limit = Math.sqrt(6 / (inputSize + outputSize));
        const weights: number[][] = [];
        
        for (let i = 0; i < inputSize; i++) {
            weights[i] = [];
            for (let j = 0; j < outputSize; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        
        return weights;
    }
    
    forward(input: number[][]): number[][] {
        this.input = input;
        
        // z = input · weights + bias
        this.weightedSum = Matrix.add(
            Matrix.multiply(input, this.weights),
            this.biases
        );
        
        // activation = sigmoid(z)
        this.activation = Activation.sigmoidMatrix(this.weightedSum);
        
        return this.activation;
    }
    
    getWeights(): number[][] {
        return this.weights;
    }
    
    getBiases(): number[][] {
        return this.biases;
    }
}

/**
 * Neural Network class - The entire organization
 */
class NeuralNetwork {
    layers: Layer[] = [];
    learningRate: number = 0.1;
    
    // Training history
    lossHistory: number[] = [];
    epoch: number = 0;
    
    constructor(architecture: number[]) {
        // architecture = [5, 8, 6, 4, 1] for example
        // Input: 5, Hidden: 8, 6, 4, Output: 1
        
        const layerNames = [
            '入力層 → 隠れ層1',
            '隠れ層1 → 隠れ層2',
            '隠れ層2 → 隠れ層3',
            '隠れ層3 → 出力層'
        ];
        
        for (let i = 0; i < architecture.length - 1; i++) {
            const layer = new Layer(
                architecture[i],
                architecture[i + 1],
                layerNames[i] || `Layer ${i}`
            );
            this.layers.push(layer);
        }
    }
    
    /**
     * Forward propagation - The evaluation process
     */
    forward(input: number[][]): number[][] {
        let current = input;
        
        for (const layer of this.layers) {
            current = layer.forward(current);
        }
        
        return current;
    }
    
    /**
     * Backward propagation - Responsibility distribution
     */
    backward(input: number[][], target: number[][], output: number[][]): void {
        const m = input.length; // batch size (usually 1 for visualization)
        
        // Calculate output error (delta)
        // δ_output = (output - target) ⊙ σ'(z)
        const outputError = Matrix.subtract(output, target);
        const outputLayer = this.layers[this.layers.length - 1];
        const outputDelta = Matrix.hadamard(
            outputError,
            Activation.sigmoidDerivativeMatrix(outputLayer.weightedSum!)
        );
        
        // Store deltas for each layer
        const deltas: number[][][] = [outputDelta];
        
        // Backpropagate through hidden layers
        for (let i = this.layers.length - 2; i >= 0; i--) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            const nextDelta = deltas[deltas.length - 1];
            
            // δ_i = (δ_{i+1} · W_{i+1}^T) ⊙ σ'(z_i)
            const error = Matrix.multiply(nextDelta, Matrix.transpose(nextLayer.weights));
            const delta = Matrix.hadamard(
                error,
                Activation.sigmoidDerivativeMatrix(layer.weightedSum!)
            );
            
            deltas.push(delta);
        }
        
        deltas.reverse();
        
        // Update weights and biases
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const delta = deltas[i];
            const layerInput = layer.input!;
            
            // ΔW = (1/m) · input^T · δ
            const weightGradient = Matrix.scalarMultiply(
                Matrix.multiply(Matrix.transpose(layerInput), delta),
                1 / m
            );
            
            // W = W - α · ΔW
            layer.weights = Matrix.subtract(
                layer.weights,
                Matrix.scalarMultiply(weightGradient, this.learningRate)
            );
            
            // Δb = (1/m) · sum(δ)
            const biasGradient = delta[0].map((_, j) => {
                return delta.reduce((sum, row) => sum + row[j], 0) / m;
            });
            
            // b = b - α · Δb
            layer.biases[0] = layer.biases[0].map((b, j) => 
                b - this.learningRate * biasGradient[j]
            );
        }
    }
    
    /**
     * Train the network for one epoch
     */
    train(input: number[][], target: number[][]): number {
        // Forward pass
        const output = this.forward(input);
        
        // Calculate loss (MSE)
        const loss = this.calculateLoss(output, target);
        this.lossHistory.push(loss);
        
        // Backward pass
        this.backward(input, target, output);
        
        this.epoch++;
        
        return loss;
    }
    
    /**
     * Calculate Mean Squared Error
     */
    calculateLoss(output: number[][], target: number[][]): number {
        let sum = 0;
        for (let i = 0; i < output.length; i++) {
            for (let j = 0; j < output[i].length; j++) {
                const diff = output[i][j] - target[i][j];
                sum += diff * diff;
            }
        }
        return sum / (output.length * output[0].length);
    }
    
    /**
     * Predict output for given input
     */
    predict(input: number[][]): number[][] {
        return this.forward(input);
    }
    
    /**
     * Reset the network to initial state
     */
    reset(): void {
        this.epoch = 0;
        this.lossHistory = [];
        
        for (const layer of this.layers) {
            layer.weights = this.initializeWeights(layer.inputSize, layer.outputSize);
            layer.biases = Array(1).fill(0).map(() => Array(layer.outputSize).fill(0));
        }
    }
    
    private initializeWeights(inputSize: number, outputSize: number): number[][] {
        const limit = Math.sqrt(6 / (inputSize + outputSize));
        const weights: number[][] = [];
        
        for (let i = 0; i < inputSize; i++) {
            weights[i] = [];
            for (let j = 0; j < outputSize; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        
        return weights;
    }
    
    /**
     * Set learning rate
     */
    setLearningRate(rate: number): void {
        this.learningRate = rate;
    }
}

// Export for use in other files
if (typeof window !== 'undefined') {
    (window as any).NeuralNetwork = NeuralNetwork;
    (window as any).Matrix = Matrix;
    (window as any).Activation = Activation;
}
