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
}

export interface Report {
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
  getCase: (id: string) => fetch(`/api/cases/${id}`).then((r) => json<Case>(r)),
  createCase: (form: FormData) => fetch("/api/cases", { method: "POST", body: form }).then((r) => json<Case>(r)),
  runCase: (id: string) => fetch(`/api/cases/${id}/run`, { method: "POST" }).then((r) => json<Case>(r)),
  fileUrl: (id: string, path: string) => `/api/cases/${id}/files/${path}`,
};
