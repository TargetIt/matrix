import { describe, it, expect } from "vitest";
import { Matrix } from "../matrix/Matrix.js";
import * as ops from "../matrix/ops.js";

const close = (a: number, b: number, eps = 1e-5) =>
  Math.abs(a - b) < eps;

describe("CPU matrix ops", () => {
  describe("add", () => {
    it("adds element-wise", () => {
      const a = Matrix.fromArray([[1, 2], [3, 4]]);
      const b = Matrix.fromArray([[5, 6], [7, 8]]);
      const c = ops.add(a, b);
      expect(c.toArray()).toEqual([[6, 8], [10, 12]]);
    });

    it("throws on shape mismatch", () => {
      const a = Matrix.fromArray([[1, 2]]);
      const b = Matrix.fromArray([[1, 2, 3]]);
      expect(() => ops.add(a, b)).toThrow(RangeError);
    });
  });

  describe("sub", () => {
    it("subtracts element-wise", () => {
      const a = Matrix.fromArray([[5, 6], [7, 8]]);
      const b = Matrix.fromArray([[1, 2], [3, 4]]);
      const c = ops.sub(a, b);
      expect(c.toArray()).toEqual([[4, 4], [4, 4]]);
    });
  });

  describe("scale", () => {
    it("scales by a scalar", () => {
      const a = Matrix.fromArray([[1, 2], [3, 4]]);
      const c = ops.scale(a, 2);
      expect(c.toArray()).toEqual([[2, 4], [6, 8]]);
    });
  });

  describe("hadamard", () => {
    it("multiplies element-wise", () => {
      const a = Matrix.fromArray([[1, 2], [3, 4]]);
      const b = Matrix.fromArray([[2, 0], [1, 3]]);
      const c = ops.hadamard(a, b);
      expect(c.toArray()).toEqual([[2, 0], [3, 12]]);
    });
  });

  describe("matmul", () => {
    it("multiplies 2×2 matrices", () => {
      const a = Matrix.fromArray([[1, 2], [3, 4]]);
      const b = Matrix.fromArray([[5, 6], [7, 8]]);
      const c = ops.matmul(a, b);
      // [1*5+2*7, 1*6+2*8] = [19, 22]
      // [3*5+4*7, 3*6+4*8] = [43, 50]
      expect(c.toArray()).toEqual([[19, 22], [43, 50]]);
    });

    it("multiplies a 1×3 by a 3×1 (dot product)", () => {
      const a = Matrix.fromArray([[1, 2, 3]]);
      const b = Matrix.fromArray([[4], [5], [6]]);
      const c = ops.matmul(a, b);
      expect(c.rows).toBe(1);
      expect(c.cols).toBe(1);
      expect(c.get(0, 0)).toBe(32); // 1*4 + 2*5 + 3*6
    });

    it("throws on incompatible shapes", () => {
      const a = Matrix.fromArray([[1, 2, 3]]);
      const b = Matrix.fromArray([[1, 2], [3, 4]]);
      expect(() => ops.matmul(a, b)).toThrow(RangeError);
    });
  });

  describe("transpose", () => {
    it("transposes a 2×3 matrix", () => {
      const a = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
      const at = ops.transpose(a);
      expect(at.rows).toBe(3);
      expect(at.cols).toBe(2);
      expect(at.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
    });
  });

  describe("addBias", () => {
    it("adds bias to every row", () => {
      const a = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
      const bias = Matrix.fromArray([[10, 20, 30]]);
      const c = ops.addBias(a, bias);
      expect(c.toArray()).toEqual([[11, 22, 33], [14, 25, 36]]);
    });

    it("throws on wrong bias shape", () => {
      const a = Matrix.fromArray([[1, 2]]);
      const bias = Matrix.fromArray([[1, 2, 3]]);
      expect(() => ops.addBias(a, bias)).toThrow(RangeError);
    });
  });

  describe("relu", () => {
    it("zeros negative values", () => {
      const a = Matrix.fromArray([[-2, -1, 0, 1, 2]]);
      const c = ops.relu(a);
      expect(c.toArray()).toEqual([[0, 0, 0, 1, 2]]);
    });
  });

  describe("sigmoid", () => {
    it("maps 0 to 0.5", () => {
      const a = new Matrix(1, 1, [0]);
      const c = ops.sigmoid(a);
      expect(close(c.get(0, 0), 0.5)).toBe(true);
    });

    it("is bounded in [0, 1]", () => {
      const a = Matrix.fromArray([[-100, 0, 100]]);
      const c = ops.sigmoid(a);
      for (const v of c.data) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    });
  });

  describe("tanh", () => {
    it("maps 0 to 0", () => {
      const a = new Matrix(1, 1, [0]);
      const c = ops.tanh(a);
      expect(close(c.get(0, 0), 0)).toBe(true);
    });

    it("is bounded in [-1, 1]", () => {
      const a = Matrix.fromArray([[-100, 0, 100]]);
      const c = ops.tanh(a);
      for (const v of c.data) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    });
  });

  describe("softmax", () => {
    it("outputs row sums of 1", () => {
      const a = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
      const c = ops.softmax(a);
      for (let r = 0; r < 2; r++) {
        let sum = 0;
        for (let col = 0; col < 3; col++) sum += c.get(r, col);
        expect(close(sum, 1)).toBe(true);
      }
    });

    it("is numerically stable for large values", () => {
      const a = Matrix.fromArray([[1000, 1001, 1002]]);
      const c = ops.softmax(a);
      let sum = 0;
      for (const v of c.data) sum += v;
      expect(close(sum, 1)).toBe(true);
    });
  });
});
