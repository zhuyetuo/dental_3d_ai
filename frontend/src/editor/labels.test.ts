import { describe, expect, it } from "vitest";
import {
  buildAdjacency,
  chartOrder,
  countLabels,
  floodFill,
  labelColor,
  LabelState,
  makeChange,
  splitByPlane,
  Stroke,
  verticesInSphere,
} from "./labels";

// 一条 6 个顶点的三角带（2-4、3-4、3-5 相邻）
//   0---2---4
//   | / | / |
//   1---3---5
const index = [0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5];
const positions = [0, 0, 0, 0, -1, 0, 1, 0, 0, 1, -1, 0, 2, 0, 0, 2, -1, 0];

describe("chart and palette", () => {
  it("orders teeth from patient right to left", () => {
    expect(chartOrder("lower").slice(0, 2)).toEqual([48, 47]);
    expect(chartOrder("lower").slice(7, 9)).toEqual([41, 31]);
    expect(chartOrder("upper")[15]).toBe(28);
  });

  it("gives neighbouring teeth different colors", () => {
    expect(labelColor(31)).not.toEqual(labelColor(32));
    expect(labelColor(31)).not.toEqual(labelColor(41));
  });
});

describe("adjacency and flood fill", () => {
  it("fills only the connected region with the same label", () => {
    const labels = Int32Array.from([0, 31, 31, 0, 0, 31]);
    const adj = buildAdjacency(index, 6);
    expect([...floodFill(1, labels, adj)].sort()).toEqual([1, 2]); // 5 也是 31，但不连通
    // 4 改成 31 后，2-4-5 连起来
    labels[4] = 31;
    expect([...floodFill(1, labels, adj)].sort()).toEqual([1, 2, 4, 5]);
  });

  it("finds vertices inside the brush sphere", () => {
    expect(verticesInSphere(positions, [0, 0, 0], 1.01).sort()).toEqual([0, 1, 2]);
  });
});

describe("LabelState undo / redo", () => {
  it("tracks changed vertex count through undo and redo", () => {
    const state = new LabelState(Int32Array.from([0, 31, 31, 0, 31, 31]));
    const change = makeChange(state.labels, [1, 2, 3], 32)!;
    expect(change.indices.length).toBe(3);
    state.apply(change);
    expect([...state.labels]).toEqual([0, 32, 32, 32, 31, 31]);
    expect(state.changed).toBe(3);

    state.undo();
    expect([...state.labels]).toEqual([0, 31, 31, 0, 31, 31]);
    expect(state.changed).toBe(0);
    expect(state.canRedo).toBe(true);

    state.redo();
    expect(state.changed).toBe(3);
    // 改回原值不算修改
    state.apply(makeChange(state.labels, [3], 0)!);
    expect(state.changed).toBe(2);
    // 新操作清空重做栈
    expect(state.canRedo).toBe(false);
  });

  it("skips no-op changes", () => {
    expect(makeChange(Int32Array.from([5, 5]), [0, 1], 5)).toBeNull();
  });

  it("merges a brush stroke into one undo step", () => {
    const state = new LabelState(Int32Array.from([0, 0, 0, 0, 31, 31]));
    const stroke = new Stroke(state.labels, 41);
    expect(stroke.paint([0, 1])).toEqual([0, 1]);
    expect(stroke.paint([1, 2])).toEqual([2]); // 1 已经涂过
    expect(state.labels[2]).toBe(41); // 拖动过程中实时可见
    state.apply(stroke.finish()!);
    expect([...state.labels]).toEqual([41, 41, 41, 0, 31, 31]);
    expect(state.changed).toBe(3);
    state.undo();
    expect([...state.labels]).toEqual([0, 0, 0, 0, 31, 31]);
    expect(countLabels(state.labels).get(0)).toBe(4);
  });
});

describe("splitByPlane", () => {
  it("把一块按平面切开，较小的一半改成新牙号", () => {
    // 5 个点沿 x 排开，前 4 个属于 44；平面 x = 2.5，右侧只有 1 个 44 的点
    const positions = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0];
    const labels = Int32Array.from([44, 44, 44, 44, 0]);
    const change = splitByPlane(positions, labels, 44, [2.5, 0, 0], [1, 0, 0], 45)!;
    expect(Array.from(change.indices)).toEqual([3]);
    expect(Array.from(change.after)).toEqual([45]);
    expect(splitByPlane(positions, labels, 44, [10, 0, 0], [1, 0, 0], 45)).toBeNull(); // 平面没切到
  });
});
