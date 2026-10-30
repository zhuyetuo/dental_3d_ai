import { describe, expect, it } from "vitest";
import { controlPoints, History, insertControlPoint, removeControlPoint, resampleLoop, Vec3 } from "./margins";

const square: Vec3[] = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
];

describe("control points", () => {
  it("resamples a closed loop evenly by arc length", () => {
    const pts = resampleLoop(square, 8);
    expect(pts).toHaveLength(8);
    expect(pts[1]).toEqual([0.5, 0, 0]);
    expect(pts[7]).toEqual([0, 0.5, 0]);
  });

  it("derives control points from dense auto lines and keeps manual ones", () => {
    const dense = resampleLoop(square, 120);
    expect(controlPoints({ points: dense })).toHaveLength(24);
    expect(controlPoints({ points: dense, control: square })).toEqual(square);
  });

  it("inserts into the nearest segment, including the closing one", () => {
    expect(insertControlPoint(square, [0.5, -0.1, 0])[1]).toEqual([0.5, -0.1, 0]);
    expect(insertControlPoint(square, [-0.1, 0.5, 0])).toHaveLength(5);
    expect(insertControlPoint(square, [-0.1, 0.5, 0])[4]).toEqual([-0.1, 0.5, 0]);
  });

  it("never removes below four points", () => {
    expect(removeControlPoint(square, 0)).toBe(square);
    const five = insertControlPoint(square, [0.5, -0.1, 0]);
    expect(removeControlPoint(five, 1)).toEqual(square);
  });
});

describe("History", () => {
  it("undoes and redoes snapshots and drops redo after a new edit", () => {
    const h = new History({ a: 1 });
    h.push({ a: 2 });
    h.push({ a: 3 });
    expect(h.undo()).toEqual({ a: 2 });
    expect(h.undo()).toEqual({ a: 1 });
    expect(h.undo()).toBeNull();
    expect(h.redo()).toEqual({ a: 2 });
    h.push({ a: 9 });
    expect(h.canRedo).toBe(false);
    expect(h.undo()).toEqual({ a: 2 });
  });
});
