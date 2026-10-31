import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from "three-mesh-bvh";
import { frameCamera, loadGeometry, Overlays, useOverlays, useScene } from "../three/scene";
import {
  Adjacency,
  buildAdjacency,
  countLabels,
  fillColors,
  floodFill,
  GUM,
  Jaw,
  LabelState,
  makeChange,
  splitByPlane,
  Stroke,
  verticesInSphere,
} from "./labels";
import { MarginMap } from "./margins";
import { Odontogram } from "./Odontogram";
import { useMarginEditor } from "./useMarginEditor";

// BVH 加速射线拾取：几十万面的口扫也能实时响应
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

type Mode = "view" | "select" | "brush" | "split" | "measure" | "adjust" | "margin";

const MODES: { mode: Mode; label: string; key: string; hint: string }[] = [
  { mode: "view", label: "查看", key: "v", hint: "左键旋转、右键平移、滚轮缩放" },
  { mode: "select", label: "改牙号", key: "s", hint: "点击一颗牙（或一块牙龈），整块改成选中的牙号" },
  { mode: "brush", label: "画笔", key: "b", hint: "按住左键涂抹，修正牙齿和牙龈的边界；在空白处拖动可以旋转" },
  { mode: "split", label: "拆牙", key: "x", hint: "两颗牙连成一块时：在牙位图选新牙号，在这块牙上点两下画一条切线，较小的一边改成新牙号" },
  { mode: "measure", label: "测量", key: "d", hint: "在模型上点两下，显示两点之间的距离；再点一下重新开始" },
  { mode: "adjust", label: "调整生成牙", key: "t", hint: "点击蓝色的生成牙，拖动手柄调整；W 移动、E 旋转、R 缩放，调好后点“保存调整”" },
  {
    mode: "margin",
    label: "颈缘线",
    key: "m",
    hint: "在牙位图选择预备体；拖动青色控制点移动，Shift+点击模型添加控制点，Alt+点击控制点删除",
  },
];

export interface LabelEditorProps extends Overlays {
  /** 分割结果网格（PLY，顶点顺序与标签一致） */
  plyUrl: string;
  /** Teeth3DS 格式的标签文件 */
  labelsUrl: string;
  jaw: Jaw;
  onSave: (labels: number[]) => Promise<void>;
  onDirtyChange?: (dirty: boolean) => void;
  /** 颈缘线文件（人工编辑的最新版本，或自动检测结果） */
  marginsUrl?: string;
  /** 预备体牙号 */
  prepared: number[];
  onSaveMargins: (margins: MarginMap) => Promise<void>;
  onSetPrepared: (teeth: number[]) => Promise<void>;
  /** 保存生成牙的手动调整：4x4 矩阵（行优先），作用在原始生成结果上 */
  onSaveTransform?: (fdi: number, matrix: number[]) => Promise<void>;
}

interface Loaded {
  mesh: THREE.Mesh;
  colors: THREE.BufferAttribute;
  positions: ArrayLike<number>;
  state: LabelState;
  adjacency?: Adjacency;
}

export function LabelEditor({
  plyUrl,
  labelsUrl,
  jaw,
  onSave,
  onDirtyChange,
  marginsUrl,
  prepared,
  onSaveMargins,
  onSetPrepared,
  onSaveTransform,
  ...overlays
}: LabelEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const ctx = useScene(host);
  useOverlays(ctx, overlays);

  const data = useRef<Loaded | null>(null);
  const cursor = useRef<THREE.Mesh | null>(null);
  const [mode, setMode] = useState<Mode>("view");
  const [active, setActive] = useState<number>(jaw === "upper" ? 11 : 31);
  const [radius, setRadius] = useState(1.5);
  const [hover, setHover] = useState<number | null>(null);
  const [status, setStatus] = useState({ changed: 0, canUndo: false, canRedo: false, counts: new Map<number, number>() });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState(0);
  const [measure, setMeasure] = useState<number | null>(null);
  const [adjusting, setAdjusting] = useState<{ fdi: number; dirty: boolean } | null>(null);
  const marks = useRef<THREE.Group | null>(null); // 拆牙切线、测量线的临时标记
  const clicks = useRef<{ point: THREE.Vector3; vertex: number }[]>([]);
  const gizmo = useRef<TransformControls | null>(null);
  const meshGetter = useCallback(() => data.current?.mesh ?? null, []);
  const margin = useMarginEditor({ ctx, mesh: meshGetter, loadedKey, marginsUrl, enabled: mode === "margin", active });

  // 事件处理里读取最新值
  const latest = useRef({ mode, active, radius });
  latest.current = { mode, active, radius };

  const refresh = useCallback(() => {
    const d = data.current;
    if (!d) return;
    setStatus({ changed: d.state.changed, canUndo: d.state.canUndo, canRedo: d.state.canRedo, counts: countLabels(d.state.labels) });
  }, []);

  const recolor = useCallback((indices: ArrayLike<number>) => {
    const d = data.current;
    if (!d || !indices.length) return;
    fillColors(d.colors.array as Float32Array, d.state.labels, indices);
    d.colors.needsUpdate = true;
  }, []);

  useEffect(() => {
    onDirtyChange?.(status.changed > 0 || margin.dirty);
  }, [status.changed, margin.dirty, onDirtyChange]);

  // 加载网格和标签
  useEffect(() => {
    const scene = ctx.current;
    if (!scene) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([loadGeometry(plyUrl), fetch(labelsUrl, { cache: "no-store" }).then((r) => r.json())])
      .then(([geometry, json]) => {
        if (cancelled) return;
        const labels = Int32Array.from(json.labels as number[]);
        const n = geometry.getAttribute("position").count;
        if (labels.length !== n || !geometry.index) {
          throw new Error(`标签数量 ${labels.length} 与网格顶点数 ${n} 不一致`);
        }
        geometry.deleteAttribute("color");
        const colors = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
        colors.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute("color", colors);
        geometry.computeVertexNormals();
        geometry.computeBoundsTree();
        const state = new LabelState(labels);
        fillColors(colors.array as Float32Array, labels);

        const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }));
        scene.content.clear();
        scene.content.add(mesh);
        const ring = new THREE.Mesh(
          new THREE.SphereGeometry(1, 16, 12),
          new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.6 }),
        );
        ring.visible = false;
        scene.content.add(ring);
        cursor.current = ring;
        frameCamera(scene, geometry);
        data.current = { mesh, colors, positions: geometry.getAttribute("position").array, state };
        refresh();
        setLoadedKey((k) => k + 1);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
      const d = data.current;
      if (d) {
        d.mesh.geometry.disposeBoundsTree();
        d.mesh.geometry.dispose();
        (d.mesh.material as THREE.Material).dispose();
      }
      data.current = null;
    };
  }, [ctx, plyUrl, labelsUrl, refresh]);

  const undo = useCallback(() => {
    const idx = data.current?.state.undo();
    if (idx) {
      recolor(idx);
      refresh();
    }
  }, [recolor, refresh]);

  const redo = useCallback(() => {
    const idx = data.current?.state.redo();
    if (idx) {
      recolor(idx);
      refresh();
    }
  }, [recolor, refresh]);

  // 鼠标交互：拾取、改牙号、画笔
  useEffect(() => {
    const scene = ctx.current;
    if (!scene) return;
    const canvas = scene.renderer.domElement;
    const raycaster = new THREE.Raycaster();
    raycaster.firstHitOnly = true;
    let stroke: Stroke | null = null;
    let pending = false;

    const pick = (e: PointerEvent) => {
      const d = data.current;
      if (!d) return null;
      const rect = canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, scene.camera);
      const hit = raycaster.intersectObject(d.mesh)[0];
      if (!hit?.face) return null;
      // 三角形三个顶点中离交点最近的那个
      const p = d.positions;
      let best = hit.face.a;
      let bestDist = Infinity;
      for (const v of [hit.face.a, hit.face.b, hit.face.c]) {
        const dist = hit.point.distanceToSquared(new THREE.Vector3(p[v * 3], p[v * 3 + 1], p[v * 3 + 2]));
        if (dist < bestDist) [best, bestDist] = [v, dist];
      }
      return { point: hit.point, vertex: best };
    };

    const paint = (point: THREE.Vector3) => {
      const d = data.current;
      if (!d || !stroke) return;
      recolor(stroke.paint(verticesInSphere(d.positions, [point.x, point.y, point.z], latest.current.radius)));
    };

    const showMarks = (points: THREE.Vector3[]) => {
      const group = marks.current;
      if (!group) return;
      group.clear();
      const material = new THREE.MeshBasicMaterial({ color: 0xff4081, depthTest: false });
      for (const pt of points) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 8), material);
        dot.position.copy(pt);
        dot.renderOrder = 10;
        group.add(dot);
      }
      if (points.length === 2) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0xff4081, depthTest: false }));
        line.renderOrder = 10;
        group.add(line);
      }
    };

    const onDown = (e: PointerEvent) => {
      const { mode: m, active: label } = latest.current;
      if (e.button !== 0) return;
      if (m === "adjust") {
        if (gizmo.current?.dragging) return;
        const rect = canvas.getBoundingClientRect();
        const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
        raycaster.setFromCamera(ndc, scene.camera);
        const target = raycaster.intersectObjects(scene.overlays.children.filter((o) => o.userData.generated), false)[0]?.object;
        if (target && gizmo.current) {
          e.stopImmediatePropagation();
          gizmo.current.attach(target);
          setAdjusting({ fdi: target.userData.fdi, dirty: false });
        }
        return;
      }
      if (m === "split" || m === "measure") {
        const hit = pick(e);
        if (!hit) return;
        e.stopImmediatePropagation();
        const click = { point: hit.point.clone(), vertex: hit.vertex };
        clicks.current = clicks.current.length >= 2 ? [click] : [...clicks.current, click];
        showMarks(clicks.current.map((c) => c.point));
        if (clicks.current.length < 2) {
          setMeasure(null);
          return;
        }
        const [a, b] = clicks.current.map((c) => c.point);
        if (m === "measure") {
          setMeasure(a.distanceTo(b));
          return;
        }
        // 拆牙：切平面过两个点、包含视线方向；切的是第一个点所在的那颗牙
        const d = data.current!;
        const toothLabel = d.state.labels[clicks.current[0].vertex];
        const view = new THREE.Vector3();
        scene.camera.getWorldDirection(view);
        const normal = new THREE.Vector3().subVectors(b, a).cross(view).normalize();
        if (toothLabel === GUM || toothLabel === label) {
          setError(toothLabel === GUM ? "第一下要点在要拆的那颗牙上" : "新牙号不能和这颗牙一样：先在牙位图选另一个牙号");
          return;
        }
        const change = splitByPlane(d.positions, d.state.labels, toothLabel, [a.x, a.y, a.z], [normal.x, normal.y, normal.z], label);
        if (change) {
          recolor(d.state.apply(change));
          refresh();
        }
        clicks.current = [];
        showMarks([]);
        return;
      }
      if (m !== "select" && m !== "brush") return;
      const hit = pick(e);
      if (!hit) return; // 没点中模型：交给轨道控制旋转
      e.stopImmediatePropagation();
      const d = data.current!;
      if (m === "select") {
        d.adjacency ??= buildAdjacency(d.mesh.geometry.index!.array, d.state.labels.length);
        const change = makeChange(d.state.labels, floodFill(hit.vertex, d.state.labels, d.adjacency), label);
        if (change) {
          recolor(d.state.apply(change));
          refresh();
        }
      } else {
        stroke = new Stroke(d.state.labels, label);
        canvas.setPointerCapture(e.pointerId);
        paint(hit.point);
      }
    };

    const onMove = (e: PointerEvent) => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const hit = pick(e);
        const d = data.current;
        if (stroke && hit) paint(hit.point);
        setHover(hit && d ? d.state.labels[hit.vertex] : null);
        const ring = cursor.current;
        if (ring) {
          ring.visible = Boolean(hit) && latest.current.mode === "brush";
          if (hit) {
            ring.position.copy(hit.point);
            ring.scale.setScalar(latest.current.radius);
          }
        }
      });
    };

    const onUp = () => {
      if (!stroke) return;
      const change = stroke.finish();
      stroke = null;
      if (change) {
        data.current?.state.apply(change);
        refresh();
      }
    };

    canvas.addEventListener("pointerdown", onDown, { capture: true });
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    return () => {
      canvas.removeEventListener("pointerdown", onDown, { capture: true });
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
    };
  }, [ctx, recolor, refresh]);

  // 临时标记（拆牙切线、测量线）和生成牙的调整手柄
  useEffect(() => {
    const scene = ctx.current;
    if (!scene) return;
    const group = new THREE.Group();
    scene.scene.add(group);
    marks.current = group;
    const tc = new TransformControls(scene.camera, scene.renderer.domElement);
    tc.setSize(0.8);
    const helper = tc.getHelper();
    scene.scene.add(helper);
    // 拖动手柄时不要转动视角
    tc.addEventListener("dragging-changed", (e) => {
      scene.controls.enabled = !(e as unknown as { value: boolean }).value;
    });
    tc.addEventListener("objectChange", () => setAdjusting((a) => (a ? { ...a, dirty: true } : a)));
    gizmo.current = tc;
    return () => {
      tc.detach();
      tc.dispose();
      scene.scene.remove(helper, group);
      marks.current = null;
      gizmo.current = null;
    };
  }, [ctx]);

  // 切换模式时清掉临时标记，退出“调整”时收起手柄
  useEffect(() => {
    clicks.current = [];
    marks.current?.clear();
    setMeasure(null);
    if (mode !== "adjust") {
      gizmo.current?.detach();
      setAdjusting(null);
    }
  }, [mode]);

  // 快捷键
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      const inMargin = latest.current.mode === "margin";
      if (mod && key === "z") {
        e.preventDefault();
        if (e.shiftKey) (inMargin ? margin.redo : redo)();
        else (inMargin ? margin.undo : undo)();
      } else if (mod && key === "y") {
        e.preventDefault();
        (inMargin ? margin.redo : redo)();
      } else if (!mod && latest.current.mode === "adjust" && gizmo.current?.object && ["w", "e", "r"].includes(key)) {
        gizmo.current.setMode(key === "w" ? "translate" : key === "e" ? "rotate" : "scale");
      } else if (!mod) {
        const m = MODES.find((x) => x.key === key);
        if (m) setMode(m.mode);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, margin]);

  async function busy(task: () => Promise<void>) {
    setSaving(true);
    setError(null);
    try {
      await task();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  const save = () =>
    busy(async () => {
      const d = data.current;
      if (d) await onSave(Array.from(d.state.labels));
    });

  const saveMargins = () =>
    busy(async () => {
      await onSaveMargins(margin.edited());
      margin.markSaved();
    });

  const saveTransform = () =>
    busy(async () => {
      const obj = gizmo.current?.object;
      if (!obj || !adjusting || !onSaveTransform) return;
      // 网格的几何中心移到了原点（见 useOverlays），保存的是作用在原始生成结果上的矩阵：M · T(-center)
      obj.updateMatrix();
      const c = obj.userData.center as THREE.Vector3;
      const m = new THREE.Matrix4().multiplyMatrices(obj.matrix, new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
      await onSaveTransform(adjusting.fdi, m.clone().transpose().toArray()); // three.js 是列优先，转成行优先
      setAdjusting({ ...adjusting, dirty: false });
    });

  const togglePrepared = () =>
    busy(() => onSetPrepared(prepared.includes(active) ? prepared.filter((t) => t !== active) : [...prepared, active].sort()));

  const inMargin = mode === "margin";
  const activeMargin = margin.summary[String(active)];

  const hint = MODES.find((m) => m.mode === mode)!.hint;
  return (
    <div className="editor">
      <div ref={host} className="viewer" />
      <div className="toolbar">
        <div className="row">
          {MODES.map((m) => (
            <button key={m.mode} type="button" className={m.mode === mode ? "on" : ""} onClick={() => setMode(m.mode)} title={`快捷键 ${m.key.toUpperCase()}`}>
              {m.label}
            </button>
          ))}
          <span className="sep" />
          <button type="button" onClick={inMargin ? margin.undo : undo} disabled={inMargin ? !margin.canUndo : !status.canUndo} title="Ctrl+Z">
            撤销
          </button>
          <button type="button" onClick={inMargin ? margin.redo : redo} disabled={inMargin ? !margin.canRedo : !status.canRedo} title="Ctrl+Shift+Z">
            重做
          </button>
          <span className="sep" />
          {inMargin ? (
            <button type="button" className="primary" onClick={saveMargins} disabled={saving || !margin.dirty}>
              {saving ? "保存中…" : "保存颈缘线"}
            </button>
          ) : mode === "adjust" ? (
            <button type="button" className="primary" onClick={saveTransform} disabled={saving || !adjusting?.dirty || !onSaveTransform}>
              {saving ? "保存中…" : adjusting ? `保存 ${adjusting.fdi} 的调整` : "保存调整"}
            </button>
          ) : (
            <button type="button" className="primary" onClick={save} disabled={saving || status.changed === 0}>
              {saving ? "保存中…" : "保存并重新计算"}
            </button>
          )}
        </div>
        {mode === "brush" && (
          <label className="row">
            画笔半径 {radius.toFixed(1)} mm
            <input type="range" min={0.3} max={5} step={0.1} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          </label>
        )}
        {!["view", "measure", "adjust"].includes(mode) && <Odontogram jaw={jaw} counts={status.counts} active={active} onPick={setActive} />}
        {inMargin && (
          <div className="row">
            预备体：{prepared.length ? prepared.join("、") : "无"}
            <button type="button" onClick={togglePrepared} disabled={saving || active === GUM}>
              {prepared.includes(active) ? `取消 ${active} 预备体` : `把 ${active} 设为预备体并检测颈缘线`}
            </button>
            {activeMargin ? (
              <span>
                {active} 颈缘线：{activeMargin.source === "manual" ? "人工" : "自动"}
                {activeMargin.confidence != null && `，置信度 ${activeMargin.confidence.toFixed(2)}`}，{activeMargin.controls} 个控制点
              </span>
            ) : (
              prepared.includes(active) && <span>{active} 还没有颈缘线，可以 Shift+点击逐点画出</span>
            )}
          </div>
        )}
        <div className="status">
          {hint}
          {!["view", "measure", "adjust"].includes(mode) && <> · 当前牙号：<b>{active === GUM ? "牙龈" : active}</b></>}
          {measure !== null && <> · 距离：<b>{measure.toFixed(2)} mm</b></>}
          {adjusting && <> · 正在调整 {adjusting.fdi}{adjusting.dirty ? "（未保存）" : ""}</>}
          {margin.dirty && <> · 颈缘线有未保存的修改</>}
          {hover !== null && <> · 鼠标处：<b>{hover === GUM ? "牙龈" : hover}</b></>}
          {status.changed > 0 && <> · 未保存的修改：{status.changed} 个顶点</>}
        </div>
        {loading && <div className="status">加载中…</div>}
        {error && <div className="status error">{error}</div>}
      </div>
    </div>
  );
}
