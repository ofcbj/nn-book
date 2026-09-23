/**
 * Matrix class for neural network computations
 */
export class Matrix {
  rows: number;
  cols: number;
  data: number[][];

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  static fromArray(arr: number[]): Matrix {
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  /** Build a matrix from a 2D array (the data is copied). */
  static fromData(data: number[][]): Matrix {
    const rows = data.length;
    const cols = rows > 0 ? data[0].length : 0;
    const m = new Matrix(rows, cols);
    m.data = Matrix.cloneRows(data);
    return m;
  }

  /** Deep-copy a 2D number array. */
  static cloneRows(data: number[][]): number[][] {
    return data.map(row => [...row]);
  }

  /** Deep copy of this matrix. */
  clone(): Matrix {
    return Matrix.fromData(this.data);
  }

  toArray(): number[] {
    const arr: number[] = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  /**
   * Uniform random weights in [-2, 2). Symmetric so the sign colouring of
   * connections is meaningful from the start; wide enough that the sigmoid
   * layers are not all stuck near 0.5 (which stalls learning for hundreds of steps).
   */
  randomize(): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = (Math.random() * 2 - 1) * 2;
      }
    }
  }

  randomizeBias(): void {
    // Bias can be slightly negative to adjust threshold
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() - 0.5; // -0.5 to 0.5
      }
    }
  }

  /** Matrix product A × B */
  static multiply(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) {
      throw new Error(`Matrix.multiply: shape mismatch (${a.rows}×${a.cols}) × (${b.rows}×${b.cols})`);
    }
    const result = new Matrix(a.rows, b.cols);
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

  static transpose(matrix: Matrix): Matrix {
    const result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static subtract(a: Matrix, b: Matrix): Matrix {
    Matrix.assertSameShape(a, b, 'subtract');
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  /** In-place element-wise (Hadamard) product, or scalar product. */
  multiply(n: Matrix | number): void {
    if (n instanceof Matrix) {
      Matrix.assertSameShape(this, n, 'multiply');
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n;
        }
      }
    }
  }

  /** In-place element-wise addition, or scalar addition. */
  add(n: Matrix | number): void {
    if (n instanceof Matrix) {
      Matrix.assertSameShape(this, n, 'add');
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

  static map(matrix: Matrix, func: (val: number) => number): Matrix {
    const result = new Matrix(matrix.rows, matrix.cols);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[i][j] = func(matrix.data[i][j]);
      }
    }
    return result;
  }

  private static assertSameShape(a: Matrix, b: Matrix, op: string): void {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error(`Matrix.${op}: shape mismatch (${a.rows}×${a.cols}) vs (${b.rows}×${b.cols})`);
    }
  }
}
