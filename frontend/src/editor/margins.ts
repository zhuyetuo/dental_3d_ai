// 颈缘线编辑的纯逻辑：控制点的抽取、插入、删除，以及撤销栈。不依赖渲染，便于单元测试。

export type Vec3 = [number, number, number];

/** 一颗牙的颈缘线。points 是贴在网格表面上的闭合折线；control 是编辑用的控制点。 */
export interface MarginLine {
  points: Vec3[];
  control?: Vec3[] | null;
  source?: "auto" | "manual";
  confidence?: number;
}

export type MarginMap = Record<string, MarginLine>;

export const MIN_CONTROL_POINTS = 4;
export const DEFAULT_CONTROL_POINTS = 24;

function dist(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** 沿闭合折线按弧长均匀取 n 个点，作为初始控制点。 */
export function resampleLoop(points: Vec3[], n: number): Vec3[] {
  const m = points.length;
  const cum = [0];
  for (let i = 0; i < m; i++) cum.push(cum[i] + dist(points[i], points[(i + 1) % m]));
  const total = cum[m];
  const out: Vec3[] = [];
  let seg = 0;
  for (let k = 0; k < n; k++) {
    const t = (total * k) / n;
    while (cum[seg + 1] < t) seg++;
    const a = points[seg];
    const b = points[(seg + 1) % m];
    const f = cum[seg + 1] > cum[seg] ? (t - cum[seg]) / (cum[seg + 1] - cum[seg]) : 0;
    out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
  }
  return out;
}

export function controlPoints(line: MarginLine): Vec3[] {
  if (line.control && line.control.length >= MIN_CONTROL_POINTS) return line.control.map((p) => [...p] as Vec3);
  return resampleLoop(line.points, Math.min(DEFAULT_CONTROL_POINTS, line.points.length));
}

function pointToSegment(p: Vec3, a: Vec3, b: Vec3): number {
  const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const len2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
  const t = len2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / len2)) : 0;
  return dist(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]);
}

/** 在离 p 最近的那一段中间插入新控制点。 */
export function insertControlPoint(control: Vec3[], p: Vec3): Vec3[] {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < control.length; i++) {
    const d = pointToSegment(p, control[i], control[(i + 1) % control.length]);
    if (d < bestDist) [best, bestDist] = [i, d];
  }
  return [...control.slice(0, best + 1), p, ...control.slice(best + 1)];
}

/** 删除控制点；少于 MIN_CONTROL_POINTS 个时不删。 */
export function removeControlPoint(control: Vec3[], index: number): Vec3[] {
  if (control.length <= MIN_CONTROL_POINTS) return control;
  return control.filter((_, i) => i !== index);
}

/** 快照式撤销栈：每次编辑完成后 push 当前状态。 */
export class History<T> {
  private states: string[] = [];
  private cursor = -1;

  constructor(initial: T) {
    this.push(initial);
  }

  push(state: T) {
    this.states = this.states.slice(0, this.cursor + 1);
    this.states.push(JSON.stringify(state));
    this.cursor = this.states.length - 1;
  }

  get canUndo() {
    return this.cursor > 0;
  }

  get canRedo() {
    return this.cursor < this.states.length - 1;
  }

  undo(): T | null {
    if (!this.canUndo) return null;
    return JSON.parse(this.states[--this.cursor]);
  }

  redo(): T | null {
    if (!this.canRedo) return null;
    return JSON.parse(this.states[++this.cursor]);
  }
}
