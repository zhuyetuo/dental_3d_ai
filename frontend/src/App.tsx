import { useCallback, useEffect, useMemo, useState } from "react";
import { api, Case } from "./api";
import { LabelEditor } from "./editor/LabelEditor";
import { Viewer } from "./Viewer";

const KIND_TEXT = { space: "有缺隙，可修复", closed: "间隙已关闭", free_end: "游离端缺失" } as const;

// 设计参数（mm），默认值与后端 dental_ai.preferences.DesignPreferences 一致
const PREFERENCE_FIELDS = [
  { key: "contacts_distance", label: "邻接距离", value: -0.03 },
  { key: "occlusion_distance", label: "咬合距离", value: 0.3 },
  { key: "gum_relief", label: "底面到牙龈距离", value: 0.05 },
  { key: "emergence_taper", label: "穿龈收拢比例", value: 0.85 },
] as const;

export function App() {
  const [cases, setCases] = useState<Case[]>([]);
  const [current, setCurrent] = useState<Case | null>(null);
  const [cervical, setCervical] = useState<Record<string, number[][]>>();
  const [showCervical, setShowCervical] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const onDirtyChange = useCallback((d: boolean) => setDirty(d), []);
  const version = current?.finished_at;

  // 有未保存的修改时，切换病例或关闭页面前确认
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  function select(c: Case) {
    if (c.id === current?.id) return;
    if (dirty && !window.confirm("当前病例有未保存的标注修改，确定要离开吗？")) return;
    setDirty(false);
    setCurrent(c);
  }

  async function saveLabels(labels: number[]) {
    if (!current) return;
    await api.saveLabels(current.id, labels);
    setDirty(false);
    setCurrent(await api.runCase(current.id));
    refresh();
  }

  const refresh = () => api.listCases().then(setCases).catch((e) => setError(String(e)));
  useEffect(() => {
    refresh();
  }, []);

  // 任务运行中时轮询状态；生产环境可以换成 WebSocket 推送
  useEffect(() => {
    if (!current || !["queued", "running"].includes(current.status)) return;
    const timer = setInterval(async () => {
      const c = await api.getCase(current.id);
      setCurrent(c);
      if (!["queued", "running"].includes(c.status)) refresh();
    }, 1500);
    return () => clearInterval(timer);
  }, [current]);

  useEffect(() => {
    setCervical(undefined);
    if (current?.status !== "done") return;
    fetch(api.fileUrl(current.id, "result/cervical_lines.json", current.finished_at))
      .then((r) => r.json())
      .then(setCervical)
      .catch(() => setCervical(undefined));
  }, [current?.id, current?.status, current?.finished_at]);

  async function upload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const form = new FormData(e.currentTarget);
    const prefs: Record<string, number> = {};
    for (const { key } of PREFERENCE_FIELDS) {
      prefs[key] = Number(form.get(key));
      form.delete(key);
    }
    form.set("preferences", JSON.stringify(prefs));
    for (const key of ["labels", "antagonist"]) {
      const f = form.get(key);
      if (f instanceof File && f.size === 0) form.delete(key);
    }
    try {
      const c = await api.createCase(form);
      setCurrent(await api.runCase(c.id));
      refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  const generatedUrls = useMemo(
    () =>
      current?.status === "done" ? current.report!.generated.map((g) => api.fileUrl(current.id, `result/${g.file}`, version)) : [],
    [current?.id, current?.status, version],
  );

  const report = current?.report;
  return (
    <div className="layout">
      <aside>
        <h1>Dental 3D AI</h1>
        <form onSubmit={upload}>
          <label>
            颌位
            <select name="jaw" defaultValue="lower">
              <option value="upper">上颌</option>
              <option value="lower">下颌</option>
            </select>
          </label>
          <label>
            口扫网格 (STL/PLY/OBJ)
            <input type="file" name="mesh" accept=".stl,.ply,.obj" required />
          </label>
          <label>
            标签 JSON（可选，Teeth3DS 格式）
            <input type="file" name="labels" accept=".json" />
          </label>
          <label>
            对颌网格（可选，用于咬合约束）
            <input type="file" name="antagonist" accept=".stl,.ply,.obj" />
          </label>
          <fieldset>
            <legend>设计参数 (mm)</legend>
            {PREFERENCE_FIELDS.map((f) => (
              <label key={f.key} className="pref">
                {f.label}
                <input type="number" name={f.key} step="0.01" defaultValue={f.value} />
              </label>
            ))}
          </fieldset>
          <button type="submit">上传并处理</button>
        </form>
        {error && <p className="error">{error}</p>}

        <h2>病例</h2>
        <ul className="cases">
          {cases.map((c) => (
            <li key={c.id} className={c.id === current?.id ? "active" : ""} onClick={() => select(c)}>
              {c.jaw === "upper" ? "上颌" : "下颌"} · {new Date(c.created_at).toLocaleString()} · {c.status}
            </li>
          ))}
        </ul>

        {current && (
          <section>
            <h2>结果</h2>
            <p>状态：{current.status}</p>
            {current.error && <p className="error">{current.error}</p>}
            {report && (
              <>
                <p>识别到的牙齿：{report.teeth.join(", ")}</p>
                <p>
                  分割来源：{report.segmenter === "manual" ? "人工修正" : report.segmenter}
                  {current.label_edits?.length ? `（已修正 ${current.label_edits.length} 次）` : ""}
                </p>
                <h3>缺失牙</h3>
                <ul>
                  {report.missing.map((m) => (
                    <li key={m.fdi}>
                      {m.description}：{KIND_TEXT[m.kind]}
                      {m.gap_mm != null && `，缺隙 ${m.gap_mm.toFixed(1)} mm / 期望 ${m.expected_mm} mm`}
                    </li>
                  ))}
                  {report.missing.length === 0 && <li>无</li>}
                </ul>
                <h3>生成</h3>
                <ul>
                  {report.generated.map((g) => (
                    <li key={g.fdi}>
                      {g.fdi}：{g.method}（{g.source}）{g.watertight ? "" : " ⚠ 非封闭"} ·{" "}
                      <a href={api.fileUrl(current.id, `result/${g.file}`)} download>
                        下载 STL
                      </a>
                    </li>
                  ))}
                </ul>
                {report.warnings.length > 0 && (
                  <>
                    <h3>警告</h3>
                    <ul className="error">
                      {report.warnings.map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  </>
                )}
                <label className="inline">
                  <input type="checkbox" checked={showCervical} onChange={(e) => setShowCervical(e.target.checked)} />
                  显示牙颈线
                </label>
              </>
            )}
          </section>
        )}
      </aside>
      <main>
        {current?.status === "done" ? (
          <LabelEditor
            key={current.id}
            plyUrl={api.fileUrl(current.id, "result/segmentation.ply", version)}
            labelsUrl={api.fileUrl(current.id, "result/labels.json", version)}
            jaw={current.jaw}
            onSave={saveLabels}
            onDirtyChange={onDirtyChange}
            generatedUrls={generatedUrls}
            cervicalLines={cervical}
            showCervical={showCervical}
          />
        ) : (
          <Viewer meshUrl={current ? api.fileUrl(current.id, current.files.mesh) : undefined} />
        )}
      </main>
    </div>
  );
}
