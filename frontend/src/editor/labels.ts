// 标注编辑的纯逻辑：配色、邻接、连通区域、画笔选点、撤销 / 重做。不依赖渲染，便于单元测试。
import { Color } from "three";

export type Jaw = "upper" | "lower";

export const GUM = 0;

/** 牙位图上的排列：从屏幕左侧（患者右侧）到右侧（患者左侧）。 */
export function chartOrder(jaw: Jaw): number[] {
  const [right, left] = jaw === "upper" ? [1, 2] : [4, 3];
  const r = [8, 7, 6, 5, 4, 3, 2, 1].map((p) => right * 10 + p);
  const l = [1, 2, 3, 4, 5, 6, 7, 8].map((p) => left * 10 + p);
  return [...r, ...l];
}

const GUM_RGB: [number, number, number] = [0.9, 0.63, 0.63];
const paletteCache = new Map<number, [number, number, number]>();

/** 每个牙号一个固定颜色：色相按牙位（同名牙同色系），左右两侧亮度不同，相邻牙颜色差异明显。 */
export function labelColor(fdi: number): [number, number, number] {
  if (fdi === GUM) return GUM_RGB;
  const cached = paletteCache.get(fdi);
  if (cached) return cached;
  const pos = fdi % 10;
  const quadrant = Math.floor(fdi / 10);
  const hue = ((pos - 1) * 0.37) % 1; // 跳着取色相，让相邻牙位颜色差别大
  const light = quadrant === 1 || quadrant === 4 ? 0.62 : 0.45;
  const c = new Color().setHSL(hue, 0.65, light);
  const rgb: [number, number, number] = [c.r, c.g, c.b];
  paletteCache.set(fdi, rgb);
  return rgb;
}

export function cssColor(fdi: number): string {
  const [r, g, b] = labelColor(fdi);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

export function fillColors(colors: Float32Array, labels: Int32Array, indices?: ArrayLike<number>) {
  const n = indices ? indices.length : labels.length;
  for (let k = 0; k < n; k++) {
    const i = indices ? indices[k] : k;
    const [r, g, b] = labelColor(labels[i]);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
}

/** 顶点邻接表（CSR 格式），由三角形索引构建。 */
export interface Adjacency {
  offsets: Uint32Array;
  neighbors: Uint32Array;
}

export function buildAdjacency(index: ArrayLike<number>, vertexCount: number): Adjacency {
  const degree = new Uint32Array(vertexCount);
  for (let f = 0; f < index.length; f += 3) {
    degree[index[f]] += 2;
    degree[index[f + 1]] += 2;
    degree[index[f + 2]] += 2;
  }
  const offsets = new Uint32Array(vertexCount + 1);
  for (let v = 0; v < vertexCount; v++) offsets[v + 1] = offsets[v] + degree[v];
  const neighbors = new Uint32Array(offsets[vertexCount]);
  const cursor = offsets.slice(0, vertexCount);
  for (let f = 0; f < index.length; f += 3) {
    const a = index[f], b = index[f + 1], c = index[f + 2];
    neighbors[cursor[a]++] = b;
    neighbors[cursor[a]++] = c;
    neighbors[cursor[b]++] = a;
    neighbors[cursor[b]++] = c;
    neighbors[cursor[c]++] = a;
    neighbors[cursor[c]++] = b;
  }
  return { offsets, neighbors };
}

/** 从 start 出发、与它标签相同且网格上连通的全部顶点（即点中的那颗牙或那块牙龈）。 */
export function floodFill(start: number, labels: Int32Array, adj: Adjacency): Uint32Array {
  const target = labels[start];
  const seen = new Uint8Array(labels.length);
  const stack = [start];
  const out: number[] = [];
  seen[start] = 1;
  while (stack.length) {
    const v = stack.pop()!;
    out.push(v);
    for (let k = adj.offsets[v]; k < adj.offsets[v + 1]; k++) {
      const u = adj.neighbors[k];
      if (!seen[u] && labels[u] === target) {
        seen[u] = 1;
        stack.push(u);
      }
    }
  }
  return Uint32Array.from(out);
}

/** 与 center 距离不超过 radius 的顶点。 */
export function verticesInSphere(positions: ArrayLike<number>, center: [number, number, number], radius: number): number[] {
  const r2 = radius * radius;
  const [cx, cy, cz] = center;
  const out: number[] = [];
  for (let i = 0, n = positions.length / 3; i < n; i++) {
    const dx = positions[i * 3] - cx, dy = positions[i * 3 + 1] - cy, dz = positions[i * 3 + 2] - cz;
    if (dx * dx + dy * dy + dz * dz <= r2) out.push(i);
  }
  return out;
}

export function countLabels(labels: Int32Array): Map<number, number> {
  const counts = new Map<number, number>();
  for (let i = 0; i < labels.length; i++) counts.set(labels[i], (counts.get(labels[i]) ?? 0) + 1);
  return counts;
}

/** 一次编辑操作：一组顶点的旧标签和新标签。 */
export interface Change {
  indices: Uint32Array;
  before: Int32Array;
  after: Int32Array;
}

export function makeChange(labels: Int32Array, indices: ArrayLike<number>, label: number): Change | null {
  const idx: number[] = [];
  for (let k = 0; k < indices.length; k++) if (labels[indices[k]] !== label) idx.push(indices[k]);
  if (!idx.length) return null;
  const indicesArr = Uint32Array.from(idx);
  return {
    indices: indicesArr,
    before: Int32Array.from(indicesArr, (i) => labels[i]),
    after: new Int32Array(idx.length).fill(label),
  };
}

/**
 * 标签数组 + 撤销栈。所有修改都通过 apply / undo / redo，
 * 同时维护“与原始标签不同的顶点数”，用于显示未保存的修改量。
 */
export class LabelState {
  readonly labels: Int32Array;
  private readonly original: Int32Array;
  private done: Change[] = [];
  private undone: Change[] = [];
  changed = 0;

  constructor(labels: Int32Array) {
    this.labels = labels;
    this.original = labels.slice();
  }

  get canUndo() {
    return this.done.length > 0;
  }

  get canRedo() {
    return this.undone.length > 0;
  }

  /** 写入一组标签，返回被改动的顶点，供调用方刷新颜色。 */
  private write(indices: Uint32Array, values: Int32Array) {
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const wasChanged = this.labels[i] !== this.original[i];
      this.labels[i] = values[k];
      const isChanged = values[k] !== this.original[i];
      this.changed += Number(isChanged) - Number(wasChanged);
    }
    return indices;
  }

  apply(change: Change) {
    this.done.push(change);
    this.undone = [];
    return this.write(change.indices, change.after);
  }

  undo(): Uint32Array | null {
    const c = this.done.pop();
    if (!c) return null;
    this.undone.push(c);
    return this.write(c.indices, c.before);
  }

  redo(): Uint32Array | null {
    const c = this.undone.pop();
    if (!c) return null;
    this.done.push(c);
    return this.write(c.indices, c.after);
  }
}

/** 一次画笔拖动合并成一个撤销步骤：记录每个顶点第一次被涂时的旧值。 */
export class Stroke {
  private readonly before = new Map<number, number>();

  constructor(private readonly labels: Int32Array, private readonly label: number) {}

  /** 涂一批顶点，返回实际改变的顶点。 */
  paint(indices: number[]): number[] {
    const changed: number[] = [];
    for (const i of indices) {
      if (this.labels[i] === this.label) continue;
      if (!this.before.has(i)) this.before.set(i, this.labels[i]);
      this.labels[i] = this.label;
      changed.push(i);
    }
    return changed;
  }

  /** 结束拖动：把标签恢复成涂之前的状态，返回等价的 Change，交给 LabelState.apply 统一记账。 */
  finish(): Change | null {
    if (!this.before.size) return null;
    const indices = Uint32Array.from(this.before.keys());
    const before = Int32Array.from(indices, (i) => this.before.get(i)!);
    for (let k = 0; k < indices.length; k++) this.labels[indices[k]] = before[k];
    return { indices, before, after: new Int32Array(indices.length).fill(this.label) };
  }
}
