// 颈缘线编辑：显示所有预备体的颈缘线，拖动当前牙的控制点，样条曲线实时贴合到网格表面。
import { RefObject, useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { MeshBVH } from "three-mesh-bvh";
import { SceneContext } from "../three/scene";
import { controlPoints, History, insertControlPoint, MarginMap, MIN_CONTROL_POINTS, removeControlPoint, Vec3 } from "./margins";

interface Entry {
  control: Vec3[];
  points: Vec3[];
  source: "auto" | "manual";
  confidence?: number;
}

type Snapshot = Record<string, Vec3[]>;

export interface MarginEditor {
  /** 每颗牙的来源和置信度，用于显示 */
  summary: Record<string, { source: string; confidence?: number; controls: number }>;
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  /** 本次编辑过的牙，提交给后端 */
  edited: () => MarginMap;
  markSaved: () => void;
}

const LINE_COLOR = 0xff7043;
const ACTIVE_COLOR = 0xffd54f;
const HANDLE_COLOR = 0x26c6da;

export function useMarginEditor(opts: {
  ctx: RefObject<SceneContext | null>;
  mesh: () => THREE.Mesh | null;
  /** 网格加载完成后变化，触发颈缘线重新加载 */
  loadedKey: number;
  marginsUrl?: string;
  enabled: boolean;
  active: number;
}): MarginEditor {
  const { ctx, mesh, loadedKey, marginsUrl, enabled, active } = opts;
  const entries = useRef<Record<string, Entry>>({});
  const history = useRef(new History<Snapshot>({}));
  const editedTeeth = useRef(new Set<string>());
  const group = useRef(new THREE.Group());
  const handles = useRef<THREE.Mesh[]>([]);
  const [, setTick] = useState(0);
  const [dirty, setDirty] = useState(false);
  const rerender = () => setTick((t) => t + 1);

  const latest = useRef({ enabled, active });
  latest.current = { enabled, active };

  /** Catmull-Rom 闭合样条，逐点吸附到网格表面最近处。 */
  const project = useCallback(
    (control: Vec3[]): Vec3[] => {
      const m = mesh();
      if (!m || control.length < MIN_CONTROL_POINTS) return control;
      const curve = new THREE.CatmullRomCurve3(
        control.map((p) => new THREE.Vector3(...p)),
        true,
        "centripetal",
      );
      const bvh = m.geometry.boundsTree as MeshBVH;
      const target = { point: new THREE.Vector3(), distance: 0, faceIndex: 0 };
      return curve.getSpacedPoints(control.length * 10).slice(0, -1).map((p) => {
        bvh.closestPointToPoint(p, target);
        return [target.point.x, target.point.y, target.point.z] as Vec3;
      });
    },
    [mesh],
  );

  const redraw = useCallback(() => {
    const g = group.current;
    g.children.forEach((c) => (c as THREE.Mesh).geometry?.dispose());
    g.clear();
    handles.current = [];
    const handleGeometry = new THREE.SphereGeometry(0.22, 12, 8);
    const handleMaterial = new THREE.MeshBasicMaterial({ color: HANDLE_COLOR, depthTest: false });
    for (const [fdi, e] of Object.entries(entries.current)) {
      const isActive = Number(fdi) === latest.current.active && latest.current.enabled;
      if (e.points.length >= 2) {
        const geometry = new THREE.BufferGeometry().setFromPoints(e.points.map((p) => new THREE.Vector3(...p)));
        const line = new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color: isActive ? ACTIVE_COLOR : LINE_COLOR, depthTest: false }));
        line.renderOrder = 2;
        g.add(line);
      }
      if (isActive) {
        e.control.forEach((p, i) => {
          const h = new THREE.Mesh(handleGeometry, handleMaterial);
          h.position.set(...p);
          h.renderOrder = 3;
          h.userData.index = i;
          g.add(h);
          handles.current.push(h);
        });
      }
    }
  }, []);

  const snapshot = (): Snapshot => Object.fromEntries(Object.entries(entries.current).map(([k, e]) => [k, e.control]));

  const restore = useCallback(
    (s: Snapshot) => {
      for (const [k, control] of Object.entries(s)) {
        const e = entries.current[k];
        if (e && JSON.stringify(e.control) === JSON.stringify(control)) continue;
        entries.current[k] = { control, points: project(control), source: e?.source ?? "manual", confidence: e?.confidence };
        editedTeeth.current.add(k);
      }
      for (const k of Object.keys(entries.current)) if (!(k in s)) delete entries.current[k];
      setDirty(true);
      redraw();
      rerender();
    },
    [project, redraw],
  );

  // 加载颈缘线
  useEffect(() => {
    const scene = ctx.current;
    if (!scene || !mesh()) return;
    scene.content.add(group.current);
    entries.current = {};
    editedTeeth.current.clear();
    setDirty(false);
    if (!marginsUrl) {
      history.current = new History<Snapshot>({});
      redraw();
      rerender();
      return;
    }
    let cancelled = false;
    fetch(marginsUrl, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : {}))
      .then((data: MarginMap) => {
        if (cancelled) return;
        for (const [fdi, line] of Object.entries(data)) {
          const control = controlPoints(line);
          entries.current[fdi] = {
            control,
            points: line.points?.length ? line.points : project(control),
            source: line.source ?? "auto",
            confidence: line.confidence,
          };
        }
        history.current = new History<Snapshot>(snapshot());
        redraw();
        rerender();
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctx, marginsUrl, loadedKey, project, redraw]);

  // 切换模式或当前牙时刷新高亮和控制点
  useEffect(() => redraw(), [enabled, active, redraw]);

  const commit = useCallback((fdi: string) => {
    editedTeeth.current.add(fdi);
    history.current.push(snapshot());
    setDirty(true);
    rerender();
  }, []);

  // 鼠标交互：拖动控制点、Shift+点击添加、Alt+点击删除
  useEffect(() => {
    const scene = ctx.current;
    if (!scene) return;
    const canvas = scene.renderer.domElement;
    const raycaster = new THREE.Raycaster();
    raycaster.firstHitOnly = true;
    let drag: { fdi: string; index: number } | null = null;
    let pending = false;

    const ray = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      raycaster.setFromCamera(
        new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1),
        scene.camera,
      );
    };
    const meshHit = () => {
      const m = mesh();
      return m ? raycaster.intersectObject(m)[0]?.point ?? null : null;
    };

    const onDown = (e: PointerEvent) => {
      if (!latest.current.enabled || e.button !== 0) return;
      const fdi = String(latest.current.active);
      ray(e);
      const handle = raycaster.intersectObjects(handles.current)[0];
      if (handle) {
        e.stopImmediatePropagation();
        const index = handle.object.userData.index as number;
        const entry = entries.current[fdi];
        if (e.altKey) {
          const next = removeControlPoint(entry.control, index);
          if (next !== entry.control) {
            entries.current[fdi] = { ...entry, control: next, points: project(next) };
            redraw();
            commit(fdi);
          }
          return;
        }
        drag = { fdi, index };
        canvas.setPointerCapture(e.pointerId);
        return;
      }
      if (e.shiftKey) {
        const p = meshHit();
        if (!p) return;
        e.stopImmediatePropagation();
        const point: Vec3 = [p.x, p.y, p.z];
        const entry = entries.current[fdi];
        // 没有颈缘线的牙：Shift+点击依次放点，从头画一条
        const control = entry ? (entry.control.length >= 2 ? insertControlPoint(entry.control, point) : [...entry.control, point]) : [point];
        entries.current[fdi] = { control, points: project(control), source: "manual", confidence: entry?.confidence };
        redraw();
        commit(fdi);
      }
    };

    const onMove = (e: PointerEvent) => {
      if (!drag || pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (!drag) return;
        ray(e);
        const p = meshHit();
        if (!p) return;
        const entry = entries.current[drag.fdi];
        const control = entry.control.slice();
        control[drag.index] = [p.x, p.y, p.z];
        entries.current[drag.fdi] = { ...entry, control, points: project(control) };
        redraw();
      });
    };

    const onUp = () => {
      if (!drag) return;
      const { fdi } = drag;
      drag = null;
      commit(fdi);
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
  }, [ctx, mesh, project, redraw, commit]);

  const summary = Object.fromEntries(
    Object.entries(entries.current).map(([k, e]) => [k, { source: editedTeeth.current.has(k) ? "manual" : e.source, confidence: e.confidence, controls: e.control.length }]),
  );

  return {
    summary,
    dirty,
    canUndo: history.current.canUndo,
    canRedo: history.current.canRedo,
    undo: () => {
      const s = history.current.undo();
      if (s) restore(s);
    },
    redo: () => {
      const s = history.current.redo();
      if (s) restore(s);
    },
    edited: () =>
      Object.fromEntries(
        [...editedTeeth.current]
          .filter((k) => entries.current[k]?.control.length >= MIN_CONTROL_POINTS)
          .map((k) => [k, { points: entries.current[k].points, control: entries.current[k].control }]),
      ),
    markSaved: () => setDirty(false),
  };
}
