import type { MarginMap } from "./editor/margins";

export type Jaw = "upper" | "lower";

export interface MissingTooth {
  fdi: number;
  kind: "space" | "closed" | "free_end";
  mesial: number | null;
  distal: number | null;
  gap_mm: number | null;
  expected_mm: number;
  description: string;
}

export interface GeneratedTooth {
  fdi: number;
  file: string;
  method: string;
  watertight: boolean;
  source?: string;
  scale?: number;
  /** 预备体牙冠 */
  min_thickness_mm?: number;
  blockout_max_mm?: number;
}

export interface NumberingFix {
  applied: boolean;
  changes: { from: number; to: number; vertices: number }[];
  warnings: string[];
}

export interface Report {
  segmenter: string;
  margins?: Record<string, { source: "auto" | "manual"; confidence?: number | null; method?: string | null }>;
  numbering?: NumberingFix | null;
  jaw: Jaw;
  teeth: number[];
  warnings: string[];
  missing: MissingTooth[];
  generated: GeneratedTooth[];
}

export interface Case {
  id: string;
  jaw: Jaw;
  status: "uploaded" | "queued" | "running" | "done" | "failed";
  created_at: string;
  files: Record<string, string>;
  report: Report | null;
  error: string | null;
  /** 最近一次处理结束的时间，用作结果文件的版本号（避免浏览器缓存旧文件） */
  finished_at?: string;
  label_edits?: { version: number; file: string; created_at: string; changed_vertices: number }[];
  prepared_teeth?: number[];
  margin_edits?: { version: number; file: string; created_at: string; teeth: number[] }[];
  /** 从示例病例创建 */
  demo?: { id: string; title: string };
}

/** 网页演示用的示例病例（仓库 demo/ 目录） */
export interface Demo {
  id: string;
  title: string;
  description: string;
  jaw: Jaw;
  available: boolean;
  unavailable_reason?: string | null;
  /** 本地示例（demo/local，不在仓库里） */
  local?: boolean;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? res.statusText);
  }
  return res.json();
}

export const api = {
  listCases: () => fetch("/api/cases").then((r) => json<Case[]>(r)),
  listDemos: () => fetch("/api/demos").then((r) => json<Demo[]>(r)),
  createDemoCase: (id: string) => fetch(`/api/demos/${id}`, { method: "POST" }).then((r) => json<Case>(r)),
  getCase: (id: string) => fetch(`/api/cases/${id}`).then((r) => json<Case>(r)),
  createCase: (form: FormData) => fetch("/api/cases", { method: "POST", body: form }).then((r) => json<Case>(r)),
  runCase: (id: string) => fetch(`/api/cases/${id}/run`, { method: "POST" }).then((r) => json<Case>(r)),
  saveLabels: (id: string, labels: number[]) =>
    fetch(`/api/cases/${id}/labels`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    }).then((r) => json<Case>(r)),
  setPrepared: (id: string, teeth: number[]) =>
    fetch(`/api/cases/${id}/prepared`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ teeth }),
    }).then((r) => json<Case>(r)),
  saveMargins: (id: string, margins: MarginMap) =>
    fetch(`/api/cases/${id}/margins`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ margins }),
    }).then((r) => json<Case>(r)),
  fileUrl: (id: string, path: string, version?: string) =>
    `/api/cases/${id}/files/${path}${version ? `?v=${encodeURIComponent(version)}` : ""}`,
};
