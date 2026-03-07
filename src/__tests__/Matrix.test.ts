import { describe, it, expect } from "vitest";
import { Matrix } from "../matrix/Matrix.js";

describe("Matrix", () => {
  it("creates a zero matrix", () => {
    const m = Matrix.zeros(2, 3);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(Array.from(m.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("creates an identity matrix", () => {
    const m = Matrix.identity(3);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 1)).toBe(1);
    expect(m.get(2, 2)).toBe(1);
    expect(m.get(0, 1)).toBe(0);
    expect(m.get(1, 0)).toBe(0);
  });

  it("creates from nested array", () => {
    const m = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 2)).toBe(6);
  });

  it("round-trips through toArray", () => {
    const src = [[1, 2], [3, 4]];
    const m = Matrix.fromArray(src);
    expect(m.toArray()).toEqual(src);
  });

  it("clones independently", () => {
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = a.clone();
    b.set(0, 0, 99);
    expect(a.get(0, 0)).toBe(1);
    expect(b.get(0, 0)).toBe(99);
  });

  it("fills with a constant", () => {
    const m = new Matrix(2, 2);
    m.fill(7);
    for (let i = 0; i < 4; i++) expect(m.data[i]).toBe(7);
  });

  it("throws on mismatched data length", () => {
    expect(() => new Matrix(2, 3, [1, 2, 3])).toThrow(RangeError);
  });

  it("throws on non-positive dimensions", () => {
    expect(() => new Matrix(0, 3)).toThrow(RangeError);
    expect(() => new Matrix(3, 0)).toThrow(RangeError);
  });

  it("creates a random matrix within range", () => {
    const m = Matrix.random(10, 10, -1, 1);
    for (const v of m.data) {
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThan(1);
    }
  });

  it("fromArray throws on jagged input", () => {
    expect(() => Matrix.fromArray([[1, 2], [3]])).toThrow(RangeError);
  });
});
