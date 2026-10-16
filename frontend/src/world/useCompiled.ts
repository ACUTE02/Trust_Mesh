// Compile a scene's shaders without freezing the page. A scene hands over the objects it is about to show; they are
// compiled with the browser's parallel shader compilation where it exists (renderer.compileAsync), and only then
// does the scene put them on screen. Compiling on the first draw instead blocks the page for the whole compile,
// which on a slow device is a long task during the route change.
import { useThree } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import { Group, type Object3D } from "three";
import { markDirty } from "./store";

export function useCompiled(make: () => Object3D[]): boolean {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const [ready, setReady] = useState(false);
  const makeRef = useRef(make);
  useEffect(() => {
    let alive = true;
    const done = () => {
      if (!alive) return;
      setReady(true);
      markDirty();
    };
    // the objects wait in a temporary group; the scene's own <primitive> moves them out when it shows them
    const bag = new Group();
    for (const o of makeRef.current()) bag.add(o);
    const compile = (gl as unknown as { compileAsync?: (o: Object3D, c: typeof camera) => Promise<unknown> }).compileAsync;
    if (compile) compile.call(gl, bag, camera).then(done, done);
    else done();
    return () => {
      alive = false;
    };
  }, [gl, camera]);
  return ready;
}
