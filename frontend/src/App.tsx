import { useCallback, useEffect, useMemo, useState } from "react";
import { api, Case, Demo } from "./api";
import type { MarginMap } from "./editor/margins";
import { LabelEditor } from "./editor/LabelEditor";
import { Viewer } from "./Viewer";

const KIND_TEXT = { space: "有缺隙，可修复", closed: "间隙已关闭", free_end: "游离端缺失" } as const;

// 设计参数（mm），默认值与后端 dental_ai.preferences.DesignPreferences 一致
const PREFERENCE_FIELDS = [
  { key: "contacts_distance", label: "邻接距离", value: -0.03 },
  { key: "occlusion_distance", label: "咬合距离", value: 0.3 },
  { key: "gum_relief", label: "底面到牙龈距离", value: 0.05 },
  { key: "emergence_taper", label: "穿龈收拢比例", value: 0.85 },
  { key: "cement_gap", label: "粘接间隙（预备体冠）", value: 0.03 },
  { key: "crown_thickness", label: "最小冠厚度（预备体冠）", value: 0.5 },
] as const;

export function App() {
  const [cases, setCases] = useState<Case[]>([]);
  const [demos, setDemos] = useState<Demo[]>([]);
  const [current, setCurrent] = useState<Case | null>(null);
  const [cervical, setCervical] = useState<Record<string, number[][]>>();
  const [showCervical, setShowCervical] = useState(true);
  /** 展开的二级面板：示例 id，或 "__uploads"（自己上传的） */
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const [antagonistOpacity, setAntagonistOpacity] = useState(0.32);
  const [compare, setCompare] = useState<"off" | "overlay" | "heat">("off");
  const [showTags, setShowTags] = useState(true);
  const [isolate, setIsolate] = useState(false);
  /** 第 2 步里勾选的、要生成的缺失牙 */
  const [pick, setPick] = useState<Set<number>>(new Set());
  /** 正在提交“生成 / 重新分割”：按钮立刻给反馈，网络慢时也不会重复点 */
  const [busy, setBusy] = useState(false);
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
    setOpenGroup(null);
  }

  async function saveMargins(margins: MarginMap) {
    if (!current) return;
    setCurrent(await api.saveMargins(current.id, margins));
  }

  async function setPrepared(teeth: number[]) {
    if (!current) return;
    await api.setPrepared(current.id, teeth);
    setDirty(false);
    setCurrent(await api.runCase(current.id));
    refresh();
  }

  async function saveTransform(fdi: number, matrix: number[]) {
    if (!current) return;
    setCurrent(await api.saveTransform(current.id, fdi, matrix));
  }

  async function saveLabels(labels: number[]) {
    if (!current) return;
    await api.saveLabels(current.id, labels);
    setDirty(false);
    setCurrent(await api.runCase(current.id));
    refresh();
  }

  // 换病例或重新计算后，勾选状态跟着结果走：指定过就用指定的，否则是检测到的全部缺失牙
  useEffect(() => {
    const r = current?.report;
    if (!r) return setPick(new Set());
    const chosen = current?.generate_teeth;
    setPick(new Set(chosen ?? r.missing.filter((m) => m.kind !== "closed").map((m) => m.fdi)));
  }, [current?.id, current?.finished_at]);

  async function generateSelected() {
    if (!current) return;
    if (dirty && !window.confirm("当前病例有未保存的标注修改，重新生成会丢掉这些修改，确定吗？")) return;
    setError(null);
    setBusy(true);
    try {
      await api.setGenerate(current.id, [...pick].sort());
      setDirty(false);
      setCurrent(await api.runCase(current.id));
      refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function resegment() {
    if (!current) return;
    if ((dirty || current.label_edits?.length) && !window.confirm("重新自动分割会丢掉人工修正的牙号（历史版本仍保留），确定吗？")) return;
    setError(null);
    setBusy(true);
    try {
      setDirty(false);
      setCurrent(await api.resegment(current.id));
      refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const refresh = () => api.listCases().then(setCases).catch((e) => setError(String(e)));
  useEffect(() => {
    refresh();
    api.listDemos().then(setDemos).catch(() => setDemos([]));
  }, []);

  async function openDemo(d: Demo) {
    if (dirty && !window.confirm("当前病例有未保存的标注修改，确定要离开吗？")) return;
    setError(null);
    try {
      setDirty(false);
      setCurrent(await api.createDemoCase(d.id));
      setOpenGroup(null);
      refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  // 任务运行中时轮询状态；生产环境可以换成 WebSocket 推送
  useEffect(() => {
    if (!current || !["queued", "running"].includes(current.status)) return;
    // 上一次返回后再发下一次：网络慢时请求不会越堆越多
    let stopped = false;
    const timer = setTimeout(async () => {
      try {
        const c = await api.getCase(current.id);
        if (stopped) return;
        setCurrent(c);
        if (!["queued", "running"].includes(c.status)) refresh();
      } catch {
        if (!stopped) setCurrent({ ...current }); // 出错了过一会儿再试
      }
    }, 1500);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
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

  const originalUrls = useMemo(
    () =>
      current
        ? Object.entries(current.files)
            .filter(([key]) => key.startsWith("original_"))
            .map(([, name]) => api.fileUrl(current.id, name))
        : [],
    [current?.id],
  );
  const report = current?.report;
  const running = current ? ["queued", "running"].includes(current.status) : false;
  return (
    <div className="layout">
      <aside>
        <h1>
          Dental <b>3D</b> AI
          <small>口扫 → 分割 → 缺牙检测 → 牙冠设计</small>
        </h1>
        <details className="fold upload">
          <summary>上传自己的口扫</summary>
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
            预备体牙号（可选，逗号分隔，如 46,36）
            <input type="text" name="prepared" placeholder="自动检测这些牙的颈缘线" />
          </label>
          <label>
            对颌网格（可选，用于咬合约束）
            <input type="file" name="antagonist" accept=".stl,.ply,.obj" />
          </label>
          <label>
            上传后
            <select name="generate" defaultValue="auto">
              <option value="auto">分割，并生成检测到的全部缺失牙</option>
              <option value="none">只分割（之后在第 2 步选要生成的牙）</option>
            </select>
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
        </details>
        {error && <p className="error">{error}</p>}

        {demos.length > 0 && (
          <>
            <details open className="fold">
              <summary>病例</summary>
            <ul className="demos">
              {demos.map((d) => {
                const runs = cases.filter((c) => c.demo?.id === d.id).length;
                return (
                  <li key={d.id} className={openGroup === d.id ? "open" : ""}>
                    <button type="button" disabled={!d.available} onClick={() => setOpenGroup(openGroup === d.id ? null : d.id)} title={d.description}>
                      {d.title}
                      {runs > 0 && <em>{runs}</em>}
                    </button>
                    <small>{d.available ? d.description : d.unavailable_reason}</small>
                  </li>
                );
              })}
              {cases.some((c) => !c.demo) && (
                <li className={openGroup === "__uploads" ? "open" : ""}>
                  <button type="button" onClick={() => setOpenGroup(openGroup === "__uploads" ? null : "__uploads")}>
                    我上传的口扫
                    <em>{cases.filter((c) => !c.demo).length}</em>
                  </button>
                  <small>自己上传并处理过的病例</small>
                </li>
              )}
            </ul>
            <p className="note">示例数据：Teeth3DS（CC BY-NC-ND 4.0），仅用于非商业演示</p>
            </details>
          </>
        )}


        {current && (
          <section>
            <h2>结果</h2>
            <p>状态：{current.status}</p>
            {current.error && <p className="error">{current.error}</p>}
            {report && (
              <>
                <div className="steps">
                  <div className="step">
                    <h3><i>1</i>分割</h3>
                    <p>
                      识别到 {report.teeth.length} 颗牙 ·{" "}
                      {report.segmenter === "manual" ? "人工修正" : report.segmenter === "label_file" ? "上传的标签" : `模型 ${report.segmenter}`}
                    </p>
                    <p className="note">要改：上方工具栏 改牙号 / 画笔 / 拆牙，改完点“保存并重新计算”。</p>
                    <button type="button" className="ghost" disabled={busy || running} onClick={resegment}>
                      {busy ? "提交中…" : running ? "处理中…" : "重新自动分割"}
                    </button>
                  </div>
                  <div className="step">
                    <h3><i>2</i>生成牙</h3>
                    {report.missing.length === 0 ? (
                      <p>没有检测到缺失牙</p>
                    ) : (
                      <ul className="pick">
                        {report.missing.map((m) => (
                          <li key={m.fdi}>
                            <label className="inline">
                              <input
                                type="checkbox"
                                disabled={m.kind === "closed"}
                                checked={pick.has(m.fdi)}
                                onChange={(e) => {
                                  const next = new Set(pick);
                                  if (e.target.checked) next.add(m.fdi);
                                  else next.delete(m.fdi);
                                  setPick(next);
                                }}
                              />
                              {m.description}
                              <small>
                                {KIND_TEXT[m.kind]}
                                {report.generated.some((g) => g.fdi === m.fdi) && " · 已生成"}
                              </small>
                            </label>
                          </li>
                        ))}
                      </ul>
                    )}
                    <button type="button" disabled={report.missing.length === 0 || busy || running} onClick={generateSelected}>
                      {busy ? "提交中…" : running ? "处理中…" : pick.size > 0 ? `生成选中的牙（${pick.size}）` : "不生成（清除生成的牙）"}
                    </button>
                  </div>
                </div>
                {report.generated.length > 0 && (
                  <p className="generated-list">
                    生成的牙：
                    {report.generated.map((g) => (
                      <b key={g.fdi}>{g.fdi}</b>
                    ))}
                  </p>
                )}
                <ul className="legend">
                  {report.generated.length > 0 && <li><i style={{ background: "#19d3e6" }} />生成的牙（牙上有标签）</li>}
                  {originalUrls.length > 0 && <li><i style={{ background: "#ffc857" }} />原始牙</li>}
                  {current.files.antagonist && <li><i style={{ background: "#d5dde8" }} />对颌</li>}
                </ul>
                <p>识别到的牙齿：{report.teeth.join(", ")}</p>
                <p>
                  分割来源：{report.segmenter === "manual" ? "人工修正" : report.segmenter}
                  {current.label_edits?.length ? `（已修正 ${current.label_edits.length} 次）` : ""}
                </p>
                {report.margins && Object.keys(report.margins).length > 0 && (
                  <>
                    <h3>颈缘线</h3>
                    <ul>
                      {Object.entries(report.margins).map(([fdi, m]) => {
                        // 人工编辑保存后不用重新计算，报告里的来源可能还是上一次的“自动”
                        const manual = m.source === "manual" || current.margin_edits?.some((e) => e.teeth.includes(Number(fdi)));
                        return (
                          <li key={fdi}>
                            {fdi}：{manual ? "人工编辑" : `自动检测，置信度 ${m.confidence?.toFixed(2) ?? "-"}`}
                          </li>
                        );
                      })}
                    </ul>
                    <a href={api.fileUrl(current.id, current.files.margins ?? "result/margins.json", version)} download>
                      下载颈缘线（JSON）
                    </a>
                  </>
                )}
                {report.numbering?.applied && (
                  <p>
                    自动纠正编号：
                    {report.numbering.changes.map((c) => `${c.from}→${c.to}`).join("，")}
                    （请在编辑器中确认）
                  </p>
                )}
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
                      {g.fdi}：{g.method === "prep_crown" ? "预备体牙冠" : "缺失牙"}（{g.source}
                      {g.min_thickness_mm != null && `，最小厚度 ${g.min_thickness_mm} mm`}）{g.watertight ? "" : " ⚠ 非封闭"}
                      {current.adjustments?.[String(g.fdi)] && "（已手动调整）"} ·{" "}
                      <a href={api.fileUrl(current.id, `result/${g.file}`, version)} download>
                        下载 STL
                      </a>
                    </li>
                  ))}
                </ul>
                {report.generated.length > 1 && (
                  <p>
                    <a href={api.exportUrl(current.id, version)} download>
                      下载全部（合并成一个 STL）
                    </a>
                  </p>
                )}
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
                {report.generated.length > 0 && (
                  <>
                    <label className="inline">
                      <input type="checkbox" checked={showTags} onChange={(e) => setShowTags(e.target.checked)} />
                      显示“生成 牙号”标签
                    </label>
                    <label className="inline">
                      <input type="checkbox" checked={isolate} onChange={(e) => setIsolate(e.target.checked)} />
                      只看生成的牙（隐藏牙弓和对颌）
                    </label>
                  </>
                )}
                {compare === "heat" && (
                  <ul className="legend">
                    <li><i style={{ background: "#3ddc97" }} />偏差 ≤ 0.2 mm</li>
                    <li><i style={{ background: "#ffdb4d" }} />0.5 mm</li>
                    <li><i style={{ background: "#f24038" }} />≥ 1 mm</li>
                    <li><i style={{ background: "#737f94" }} />封底部分（原始牙没有）</li>
                  </ul>
                )}
                {report.comparison && Object.keys(report.comparison).length > 0 && (
                  <table className="compare">
                    <thead>
                      <tr><th>牙号</th><th>平均</th><th>中位数</th><th>95%</th><th>最大</th><th>0.3 mm 内</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(report.comparison).map(([fdi, c]) => (
                        <tr key={fdi}>
                          <td>{fdi}</td><td>{c.mean_mm.toFixed(2)}</td><td>{c.median_mm.toFixed(2)}</td>
                          <td>{c.p95_mm.toFixed(2)}</td><td>{c.max_mm.toFixed(2)}</td><td>{Math.round(c.within_03 * 100)}%</td>
                        </tr>
                      ))}
                    </tbody>
                    <caption>原始牙表面到生成牙的距离（mm）</caption>
                  </table>
                )}
                {originalUrls.length > 0 && (
                  <label className="inline">
                    和原始牙对比
                    <select value={compare} onChange={(e) => setCompare(e.target.value as typeof compare)}>
                      <option value="off">不对比</option>
                      <option value="overlay">叠加原始牙（金色半透明）</option>
                      <option value="heat">偏差色图（生成的牙按偏差上色）</option>
                    </select>
                  </label>
                )}
                {current.files.antagonist && (
                  <label className="inline">
                    对颌
                    <select value={antagonistOpacity} onChange={(e) => setAntagonistOpacity(Number(e.target.value))}>
                      <option value={0}>隐藏</option>
                      <option value={0.32}>半透明</option>
                      <option value={1}>不透明（看对颌的凹凸面）</option>
                    </select>
                  </label>
                )}
              </>
            )}
          </section>
        )}
      </aside>
      {openGroup &&
        (() => {
          const demo = demos.find((d) => d.id === openGroup);
          const runs = cases.filter((c) => (demo ? c.demo?.id === demo.id : !c.demo));
          if (current && !runs.some((c) => c.id === current.id) && (demo ? current.demo?.id === demo.id : !current.demo)) runs.unshift(current);
          const STATUS: Record<string, string> = { uploaded: "已上传", queued: "排队中", running: "处理中", done: "完成", failed: "失败" };
          return (
            <div className="flyout">
              <header>
                <b>{demo ? demo.title : "我上传的口扫"}</b>
                <button type="button" onClick={() => setOpenGroup(null)} aria-label="关闭">
                  ×
                </button>
              </header>
              {demo && (
                <button type="button" className="run" onClick={() => openDemo(demo)}>
                  新建一次运行
                </button>
              )}
              <h3>{runs.length > 0 ? `历史运行（${runs.length}）` : "还没有运行过"}</h3>
              <ul className="cases">
                {runs.map((c) => (
                  <li key={c.id} className={c.id === current?.id ? "active" : ""} onClick={() => select(c)}>
                    <span>{new Date(c.created_at).toLocaleString()}</span>
                    <i className={`st ${c.status}`}>{STATUS[c.status] ?? c.status}</i>
                  </li>
                ))}
              </ul>
            </div>
          );
        })()}
      <main>
        {current?.status === "done" ? (
          <LabelEditor
            key={current.id}
            plyUrl={api.fileUrl(current.id, "result/segmentation.ply", version)}
            labelsUrl={api.fileUrl(current.id, "result/labels.json", version)}
            jaw={current.jaw}
            onSave={saveLabels}
            onDirtyChange={onDirtyChange}
            marginsUrl={
              current.files.margins
                ? api.fileUrl(current.id, current.files.margins)
                : api.fileUrl(current.id, "result/margins.json", version)
            }
            prepared={current.prepared_teeth ?? []}
            onSaveMargins={saveMargins}
            onSetPrepared={setPrepared}
            onSaveTransform={saveTransform}
            generatedUrls={generatedUrls}
            cervicalLines={cervical}
            showCervical={showCervical}
            antagonistUrl={current.files.antagonist ? api.fileUrl(current.id, current.files.antagonist) : undefined}
            antagonistOpacity={antagonistOpacity}
            originalUrls={originalUrls}
            compare={compare}
            showTags={showTags}
            isolate={isolate}
          />
        ) : (
          <Viewer meshUrl={current ? api.fileUrl(current.id, current.files.mesh) : undefined} />
        )}
        {current?.status === "done" && report && report.generated.length > 0 && (
          <div className="viewbar" role="toolbar" aria-label="显示选项">
            <button type="button" className={showTags ? "on" : ""} onClick={() => setShowTags(!showTags)} title="牙上的“生成 牙号”标签">
              牙号标签：{showTags ? "显示" : "隐藏"}
            </button>
            {current.files.antagonist && (
              <button
                type="button"
                className={antagonistOpacity > 0 ? "on" : ""}
                onClick={() => setAntagonistOpacity(antagonistOpacity === 0 ? 0.32 : antagonistOpacity < 1 ? 1 : 0)}
                title="点一下在 隐藏 / 半透明 / 不透明 之间切换"
              >
                对颌：{antagonistOpacity === 0 ? "隐藏" : antagonistOpacity < 1 ? "半透明" : "不透明"}
              </button>
            )}
            <button type="button" className={isolate ? "on" : ""} onClick={() => setIsolate(!isolate)} title="隐藏牙弓和对颌，只留生成的牙">
              只看生成牙：{isolate ? "开" : "关"}
            </button>
            {originalUrls.length > 0 && (
              <button
                type="button"
                className={compare !== "off" ? "on" : ""}
                onClick={() => setCompare(compare === "off" ? "overlay" : compare === "overlay" ? "heat" : "off")}
                title="点一下在 不对比 / 叠加原始牙 / 偏差色图 之间切换"
              >
                对比原始牙：{compare === "off" ? "关" : compare === "overlay" ? "叠加" : "偏差色图"}
              </button>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
