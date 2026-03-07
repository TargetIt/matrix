/**
 * CPU-based dense matrix with row-major Float32 storage.
 * Used as the data structure shared between CPU and GPU code paths.
 */
export class Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float32Array;

  constructor(rows: number, cols: number, data?: Float32Array | number[]) {
    if (rows <= 0 || cols <= 0) {
      throw new RangeError(`Matrix dimensions must be positive, got ${rows}×${cols}`);
    }
    this.rows = rows;
    this.cols = cols;
    if (data !== undefined) {
      const arr = data instanceof Float32Array ? data : new Float32Array(data);
      if (arr.length !== rows * cols) {
        throw new RangeError(
          `Data length ${arr.length} does not match ${rows}×${cols} = ${rows * cols}`,
        );
      }
      this.data = arr;
    } else {
      this.data = new Float32Array(rows * cols);
    }
  }

  /** Element at row r, column c (0-indexed). */
  get(r: number, c: number): number {
    return this.data[r * this.cols + c];
  }

  /** Set element at row r, column c (0-indexed). */
  set(r: number, c: number, value: number): void {
    this.data[r * this.cols + c] = value;
  }

  /** Return a deep copy of this matrix. */
  clone(): Matrix {
    return new Matrix(this.rows, this.cols, new Float32Array(this.data));
  }

  /** Fill every element with the given scalar. */
  fill(value: number): this {
    this.data.fill(value);
    return this;
  }

  /** Create a rows×cols zero matrix. */
  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  /** Create an n×n identity matrix. */
  static identity(n: number): Matrix {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      m.set(i, i, 1);
    }
    return m;
  }

  /**
   * Create a matrix filled with random values sampled from
   * a uniform distribution in [min, max).
   */
  static random(rows: number, cols: number, min = -1, max = 1): Matrix {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = min + Math.random() * (max - min);
    }
    return m;
  }

  /**
   * Create a matrix from a nested array (rows × cols).
   */
  static fromArray(values: number[][]): Matrix {
    const rows = values.length;
    const cols = values[0]?.length ?? 0;
    const flat: number[] = [];
    for (const row of values) {
      if (row.length !== cols) {
        throw new RangeError("All rows must have the same length");
      }
      flat.push(...row);
    }
    return new Matrix(rows, cols, flat);
  }

  /** Convert to a nested number array. */
  toArray(): number[][] {
    const result: number[][] = [];
    for (let r = 0; r < this.rows; r++) {
      result.push(Array.from(this.data.slice(r * this.cols, (r + 1) * this.cols)));
    }
    return result;
  }

  toString(): string {
    return this.toArray()
      .map((row) => `[${row.map((v) => v.toFixed(4)).join(", ")}]`)
      .join("\n");
  }
}
