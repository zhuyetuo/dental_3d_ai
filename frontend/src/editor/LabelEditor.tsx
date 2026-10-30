import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
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
  Stroke,
  verticesInSphere,
} from "./labels";
import { Odontogram } from "./Odontogram";

// BVH 加速射线拾取：几十万面的口扫也能实时响应
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

type Mode = "view" | "select" | "brush";

const MODES: { mode: Mode; label: string; key: string; hint: string }[] = [
  { mode: "view", label: "查看", key: "v", hint: "左键旋转、右键平移、滚轮缩放" },
  { mode: "select", label: "改牙号", key: "s", hint: "点击一颗牙（或一块牙龈），整块改成选中的牙号" },
  { mode: "brush", label: "画笔", key: "b", hint: "按住左键涂抹，修正牙齿和牙龈的边界；在空白处拖动可以旋转" },
];

export interface LabelEditorProps extends Overlays {
  /** 分割结果网格（PLY，顶点顺序与标签一致） */
  plyUrl: string;
  /** Teeth3DS 格式的标签文件 */
  labelsUrl: string;
  jaw: Jaw;
  onSave: (labels: number[]) => Promise<void>;
  onDirtyChange?: (dirty: boolean) => void;
}

interface Loaded {
  mesh: THREE.Mesh;
  colors: THREE.BufferAttribute;
  positions: ArrayLike<number>;
  state: LabelState;
  adjacency?: Adjacency;
}

export function LabelEditor({ plyUrl, labelsUrl, jaw, onSave, onDirtyChange, ...overlays }: LabelEditorProps) {
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
    onDirtyChange?.(status.changed > 0);
  }, [status.changed, onDirtyChange]);

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

    const onDown = (e: PointerEvent) => {
      const { mode: m, active: label } = latest.current;
      if (e.button !== 0 || m === "view") return;
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

  // 快捷键
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      if (mod && key === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (mod && key === "y") {
        e.preventDefault();
        redo();
      } else if (!mod) {
        const m = MODES.find((x) => x.key === key);
        if (m) setMode(m.mode);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

  async function save() {
    const d = data.current;
    if (!d) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(Array.from(d.state.labels));
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

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
          <button type="button" onClick={undo} disabled={!status.canUndo} title="Ctrl+Z">
            撤销
          </button>
          <button type="button" onClick={redo} disabled={!status.canRedo} title="Ctrl+Shift+Z">
            重做
          </button>
          <span className="sep" />
          <button type="button" className="primary" onClick={save} disabled={saving || status.changed === 0}>
            {saving ? "保存中…" : "保存并重新计算"}
          </button>
        </div>
        {mode === "brush" && (
          <label className="row">
            画笔半径 {radius.toFixed(1)} mm
            <input type="range" min={0.3} max={5} step={0.1} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          </label>
        )}
        {mode !== "view" && <Odontogram jaw={jaw} counts={status.counts} active={active} onPick={setActive} />}
        <div className="status">
          {hint}
          {mode !== "view" && <> · 当前牙号：<b>{active === GUM ? "牙龈" : active}</b></>}
          {hover !== null && <> · 鼠标处：<b>{hover === GUM ? "牙龈" : hover}</b></>}
          {status.changed > 0 && <> · 未保存的修改：{status.changed} 个顶点</>}
        </div>
        {loading && <div className="status">加载中…</div>}
        {error && <div className="status error">{error}</div>}
      </div>
    </div>
  );
}
