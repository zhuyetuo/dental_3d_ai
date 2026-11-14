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
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = null; // 透明：背景由页面的渐变给出（style.css 里的 main）
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
  antagonistUrl?: string;
  showAntagonist?: boolean;
  originalUrls?: string[];
  showOriginal?: boolean;
}

export const GENERATED_COLOR = 0x19d3e6;
export const ORIGINAL_COLOR = 0xffc857;

/** 贴在牙上的小标签（始终朝向相机、不被遮挡）：一眼看出哪颗是生成的。 */
function makeTag(text: string, at: THREE.Vector3, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 96;
  const g = canvas.getContext("2d")!;
  g.fillStyle = "rgba(7, 13, 26, 0.9)";
  g.strokeStyle = color;
  g.lineWidth = 6;
  g.beginPath();
  g.roundRect(4, 4, 248, 88, 40);
  g.fill();
  g.stroke();
  g.fillStyle = "#ffffff";
  g.font = "bold 44px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 128, 50);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, transparent: true }));
  sprite.scale.set(6.4, 2.4, 1); // mm
  sprite.position.copy(at);
  sprite.renderOrder = 10;
  sprite.raycast = () => {};
  sprite.userData = { tag: true };
  return sprite;
}

/** 生成牙（高亮蓝色）和牙颈线（黄色线框）。返回取消函数，防止旧请求晚到时加进场景。 */
export function useOverlays(ctx: RefObject<SceneContext | null>, { generatedUrls, cervicalLines, showCervical, antagonistUrl, showAntagonist, originalUrls, showOriginal }: Overlays) {
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
        const material = new THREE.MeshStandardMaterial({ color: GENERATED_COLOR, emissive: 0x0a4a55, roughness: 0.3, metalness: 0.05 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(center);
        const fdi = Number(/(?:tooth|crown)_(\d+)\.stl/.exec(url)?.[1] ?? NaN);
        mesh.userData = { generated: true, fdi, url, center: center.clone() };
        group.add(mesh);
        if (Number.isFinite(fdi)) group.add(makeTag(`生成 ${fdi}`, center, "#19d3e6"));
      });
    }
    if (showOriginal) {
      // 示例里被去掉的真牙：半透明金色叠在生成的牙上，直接看差在哪
      for (const url of originalUrls ?? []) {
        loadGeometry(url).then((geometry) => {
          if (cancelled) return;
          geometry.computeVertexNormals();
          const material = new THREE.MeshStandardMaterial({ color: ORIGINAL_COLOR, roughness: 0.4, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.userData = { original: true };
          mesh.raycast = () => {};
          group.add(mesh);
        });
      }
    }
    if (showAntagonist && antagonistUrl) {
      // 对颌：半透明，不参与拾取，只用来看生成的牙冠和它的咬合关系
      loadGeometry(antagonistUrl).then((geometry) => {
        if (cancelled) return;
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          color: 0xd5dde8, roughness: 0.6, transparent: true, opacity: 0.32, depthWrite: false, side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData = { antagonist: true };
        mesh.raycast = () => {};
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
  }, [ctx, generatedUrls, cervicalLines, showCervical, antagonistUrl, showAntagonist, originalUrls, showOriginal]);
}
