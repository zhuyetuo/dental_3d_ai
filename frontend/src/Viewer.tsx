import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

export interface ViewerProps {
  /** 带逐顶点颜色的分割结果（PLY） */
  segmentationUrl?: string;
  /** 原始网格，分割结果还没出来时显示 */
  meshUrl?: string;
  /** 生成的缺失牙（STL） */
  generatedUrls: string[];
  /** FDI -> 牙颈线折线坐标 */
  cervicalLines?: Record<string, number[][]>;
  showCervical: boolean;
}

export function Viewer({ segmentationUrl, meshUrl, generatedUrls, cervicalLines, showCervical }: ViewerProps) {
  const host = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls } | null>(null);
  const content = useRef<THREE.Group>(new THREE.Group());

  // 初始化渲染器，只执行一次
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
    scene.add(content.current);

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -80, 80);
    const controls = new OrbitControls(camera, renderer.domElement);
    sceneRef.current = { scene, camera, controls };

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
    };
  }, []);

  // 加载模型
  useEffect(() => {
    const group = content.current;
    group.clear();
    let cancelled = false;
    const add = (obj: THREE.Object3D) => {
      if (!cancelled) group.add(obj);
    };

    const baseUrl = segmentationUrl ?? meshUrl;
    if (baseUrl) {
      const isPly = baseUrl.endsWith(".ply") || baseUrl === segmentationUrl;
      const loader = isPly ? new PLYLoader() : new STLLoader();
      loader.load(baseUrl, (geometry) => {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          vertexColors: geometry.hasAttribute("color"),
          color: geometry.hasAttribute("color") ? 0xffffff : 0xe8d9c8,
          roughness: 0.6,
        });
        add(new THREE.Mesh(geometry, material));
        frame(geometry);
      });
    }

    for (const url of generatedUrls) {
      new STLLoader().load(url, (geometry) => {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({ color: 0x4fc3f7, roughness: 0.35, metalness: 0.05 });
        add(new THREE.Mesh(geometry, material));
      });
    }

    if (showCervical && cervicalLines) {
      const material = new THREE.LineBasicMaterial({ color: 0xffeb3b });
      for (const pts of Object.values(cervicalLines)) {
        const geometry = new THREE.BufferGeometry().setFromPoints(pts.map(([x, y, z]) => new THREE.Vector3(x, y, z)));
        add(new THREE.LineLoop(geometry, material));
      }
    }
    return () => {
      cancelled = true;
    };
  }, [segmentationUrl, meshUrl, generatedUrls, cervicalLines, showCervical]);

  // 相机对准模型
  function frame(geometry: THREE.BufferGeometry) {
    const ctx = sceneRef.current;
    if (!ctx) return;
    geometry.computeBoundingSphere();
    const { center, radius } = geometry.boundingSphere!;
    ctx.controls.target.copy(center);
    ctx.camera.position.copy(center).add(new THREE.Vector3(0, -radius * 1.6, radius * 1.6));
  }

  return <div ref={host} className="viewer" />;
}
