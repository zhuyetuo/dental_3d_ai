// 查看器和编辑器共用的 Three.js 场景、加载和叠加层。
import { RefObject, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

export interface SceneContext {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  /** 主模型放这里 */
  content: THREE.Group;
  /** 生成牙、牙颈线等叠加层放这里 */
  overlays: THREE.Group;
}

/** 创建渲染器、灯光、相机和轨道控制，挂到 host 上；组件卸载时释放。 */
export function useScene(host: RefObject<HTMLDivElement>): RefObject<SceneContext | null> {
  const ctx = useRef<SceneContext | null>(null);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1d2127);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.2));
    const light = new THREE.DirectionalLight(0xffffff, 1.5);
    light.position.set(30, -40, 80);
    scene.add(light);
    const content = new THREE.Group();
    const overlays = new THREE.Group();
    scene.add(content, overlays);

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -80, 80);
    const controls = new OrbitControls(camera, renderer.domElement);
    ctx.current = { renderer, scene, camera, controls, content, overlays };

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    let frame = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      ctx.current = null;
    };
  }, [host]);

  return ctx;
}

/** 按扩展名选择解析器。 */
export function loadGeometry(url: string): Promise<THREE.BufferGeometry> {
  const ext = url.split("?")[0].split(".").pop()?.toLowerCase();
  return new Promise((resolve, reject) => {
    if (ext === "ply") new PLYLoader().load(url, resolve, undefined, reject);
    else if (ext === "obj")
      new OBJLoader().load(
        url,
        (group) => {
          const mesh = group.children.find((c): c is THREE.Mesh => c instanceof THREE.Mesh);
          if (mesh) resolve(mesh.geometry);
          else reject(new Error(`${url} 中没有网格`));
        },
        undefined,
        reject,
      );
    else new STLLoader().load(url, resolve, undefined, reject);
  });
}

/** 相机对准模型。 */
export function frameCamera(ctx: SceneContext, geometry: THREE.BufferGeometry) {
  geometry.computeBoundingSphere();
  const { center, radius } = geometry.boundingSphere!;
  ctx.controls.target.copy(center);
  ctx.camera.position.copy(center).add(new THREE.Vector3(0, -radius * 1.6, radius * 1.6));
}

export interface Overlays {
  generatedUrls: string[];
  cervicalLines?: Record<string, number[][]>;
  showCervical: boolean;
}

/** 生成牙（高亮蓝色）和牙颈线（黄色线框）。返回取消函数，防止旧请求晚到时加进场景。 */
export function useOverlays(ctx: RefObject<SceneContext | null>, { generatedUrls, cervicalLines, showCervical }: Overlays) {
  useEffect(() => {
    const group = ctx.current?.overlays;
    if (!group) return;
    group.clear();
    let cancelled = false;

    for (const url of generatedUrls) {
      loadGeometry(url).then((geometry) => {
        if (cancelled) return;
        geometry.computeVertexNormals();
        // 几何中心移到原点、网格放回原位：手动调整（旋转、缩放）时绕牙自己的中心转
        geometry.computeBoundingBox();
        const center = geometry.boundingBox!.getCenter(new THREE.Vector3());
        geometry.translate(-center.x, -center.y, -center.z);
        const material = new THREE.MeshStandardMaterial({ color: 0x4fc3f7, roughness: 0.35, metalness: 0.05 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(center);
        const fdi = Number(/(?:tooth|crown)_(\d+)\.stl/.exec(url)?.[1] ?? NaN);
        mesh.userData = { generated: true, fdi, url, center: center.clone() };
        group.add(mesh);
      });
    }
    if (showCervical && cervicalLines) {
      const material = new THREE.LineBasicMaterial({ color: 0xffeb3b });
      for (const pts of Object.values(cervicalLines)) {
        const geometry = new THREE.BufferGeometry().setFromPoints(pts.map(([x, y, z]) => new THREE.Vector3(x, y, z)));
        group.add(new THREE.LineLoop(geometry, material));
      }
    }
    return () => {
      cancelled = true;
    };
  }, [ctx, generatedUrls, cervicalLines, showCervical]);
}
