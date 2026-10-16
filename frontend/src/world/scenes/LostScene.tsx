// 404, "an unlinked node": one node drifting free, its thread snapped. Focusing or hovering the link back to the
// overview ties the thread to it again. One sphere, one short thread, two draw calls.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, Color, IcosahedronGeometry, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial } from "three";
import { palette } from "../palettes";
import { live } from "../store";
import { useCompiled } from "../useCompiled";

const SEGMENTS = 28;

export default function LostScene() {
  const rig = useMemo(() => {
    const nodeMat = new MeshBasicMaterial();
    const node = new Mesh(new IcosahedronGeometry(0.34, 2), nodeMat);
    const geo = new BufferGeometry();
    geo.setAttribute("position", new BufferAttribute(new Float32Array(SEGMENTS * 2 * 2 * 3), 3));
    const lineMat = new LineBasicMaterial({ transparent: true, opacity: 0.9 });
    const thread = new LineSegments(geo, lineMat);
    thread.frustumCulled = false;
    return { node, nodeMat, geo, lineMat, thread };
  }, []);

  const tied = useRef(0);
  const tmp = useMemo(() => new Color(), []);

  useEffect(() => {
    live.continuous = !live.reduced;
    live.fpsCap = 30;
    return () => {
      live.continuous = false;
      live.fpsCap = 0;
      rig.node.geometry.dispose();
      rig.nodeMat.dispose();
      rig.geo.dispose();
      rig.lineMat.dispose();
    };
  }, [rig]);

  useFrame((state, dt) => {
    const want = live.data.lost && (live.data.lost as { tied: boolean }).tied ? 1 : 0;
    tied.current += (want - tied.current) * (live.reduced ? 1 : 1 - Math.exp(-dt * 4));
    if (Math.abs(want - tied.current) > 1 / 200) live.dirty = true;
    const t = live.reduced ? 1.2 : live.time;
    // the node drifts on a slow loop; when the thread is tied it settles toward the middle of the anchor line
    const nx = (1 - tied.current * 0.8) * Math.sin(t * 0.37) * 3.2, ny = (1 - tied.current * 0.8) * Math.cos(t * 0.51) * 1.6 + 0.4;
    rig.node.position.set(nx, ny, 0);
    const pos = rig.geo.attributes.position.array as Float32Array;
    // two halves of the thread: left anchor to the node, node to the right anchor. Each end hangs loose when not tied.
    const halves: [number, number, number][] = [[-9, -1.2, 0], [9, -1.6, 0]];
    let k = 0;
    halves.forEach((a, h) => {
      const reach = 0.46 + tied.current * 0.54; // how much of the way from the anchor the thread gets before it snaps
      for (let s = 0; s < SEGMENTS; s++) {
        for (const e of [s, s + 1]) {
          const u = (e / SEGMENTS) * reach;
          const sag = Math.sin(u * Math.PI) * -1.1 * (1 - tied.current * 0.6);
          const x = a[0] + (nx - a[0]) * u + (h ? 0.14 : -0.14) * Math.sin(t * 2 + e * 0.4) * (1 - tied.current);
          pos.set([x, a[1] + (ny - a[1]) * u + sag, 0], k);
          k += 3;
        }
      }
    });
    rig.geo.attributes.position.needsUpdate = true;
    rig.nodeMat.color.copy(palette.accent);
    rig.lineMat.color.copy(tmp.copy(palette.thread));
    state.camera.position.set(0, 0, 10);
    state.camera.lookAt(0, 0, 0);
  });

  const ready = useCompiled(() => [rig.thread, rig.node]);
  if (!ready) return null;
  return (
    <>
      <primitive object={rig.thread} />
      <primitive object={rig.node} />
    </>
  );
}
