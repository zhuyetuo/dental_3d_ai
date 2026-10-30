import { useEffect, useRef } from "react";
import * as THREE from "three";
import { frameCamera, loadGeometry, useScene } from "./three/scene";

export interface ViewerProps {
  /** 原始网格（分割结果出来之前显示） */
  meshUrl?: string;
}

/** 只读查看器：显示上传的原始网格。有分割结果后由 LabelEditor 接管。 */
export function Viewer({ meshUrl }: ViewerProps) {
  const host = useRef<HTMLDivElement>(null);
  const ctx = useScene(host);

  useEffect(() => {
    const scene = ctx.current;
    if (!scene || !meshUrl) return;
    scene.content.clear();
    let cancelled = false;
    loadGeometry(meshUrl).then((geometry) => {
      if (cancelled) return;
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({ color: 0xe8d9c8, roughness: 0.6 });
      scene.content.add(new THREE.Mesh(geometry, material));
      frameCamera(scene, geometry);
    });
    return () => {
      cancelled = true;
    };
  }, [ctx, meshUrl]);

  return <div ref={host} className="viewer" />;
}
