// A solid "engine" node for the Home scene: a glowing core, a wireframe shell that turns the other way, two tilted
// rings, and a soft halo that always faces the camera. Four draw calls, no textures, no lights. The Visual Engine stays
// a hollow dashed ghost; these two are its solid, measured counterparts.
import {
  AdditiveBlending, BufferAttribute, BufferGeometry, Camera, Color, Group, IcosahedronGeometry, LineBasicMaterial, LineSegments,
  Mesh, PlaneGeometry, ShaderMaterial, WireframeGeometry,
} from "three";
import { engineFragment, engineVertex } from "./shaders/engine";

const haloVertex = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const haloFragment = /* glsl */ `
  uniform vec3 uColor; uniform float uAlpha; uniform float uPulse; varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    float a = pow(clamp(1.0 - r, 0.0, 1.0), 2.6) * uAlpha * (0.8 + 0.2 * uPulse);
    gl_FragColor = vec4(uColor, a);
  }
`;

function ring(radius: number, segments = 96): Float32Array {
  const out = new Float32Array(segments * 2 * 3);
  for (let i = 0; i < segments; i++) {
    for (const [k, j] of [[0, i], [1, i + 1]]) {
      const a = (j / segments) * Math.PI * 2;
      out.set([Math.cos(a) * radius, 0, Math.sin(a) * radius], (i * 2 + k) * 3);
    }
  }
  return out;
}

export interface Engine {
  root: Group;
  update(time: number, camera: Camera, color: Color, bg: Color, alpha: number, still: boolean): void;
  dispose(): void;
}

export function createEngine(): Engine {
  const root = new Group();
  const spin = new Group();
  const tilt = new Group();

  const coreGeo = new IcosahedronGeometry(0.5, 3);
  const coreMat = new ShaderMaterial({
    vertexShader: engineVertex, fragmentShader: engineFragment, transparent: true, depthWrite: false, blending: AdditiveBlending,
    uniforms: { uBody: { value: new Color() }, uRim: { value: new Color() }, uAlpha: { value: 1 } },
  });
  const core = new Mesh(coreGeo, coreMat);

  const shellSource = new IcosahedronGeometry(0.78, 1);
  const shellGeo = new WireframeGeometry(shellSource);
  shellSource.dispose();
  const shellMat = new LineBasicMaterial({ transparent: true, blending: AdditiveBlending, depthWrite: false });
  const shell = new LineSegments(shellGeo, shellMat);

  // two rings in one line list, one wide and one narrow, set at an angle to each other
  const a = ring(1.05), b = ring(1.3);
  const ringPos = new Float32Array(a.length + b.length);
  ringPos.set(a, 0);
  for (let i = 0; i < b.length; i += 3) ringPos.set([b[i] * 0.96, b[i + 1] + b[i] * 0.55, b[i + 2]], a.length + i); // the second ring is tipped
  const ringGeo = new BufferGeometry();
  ringGeo.setAttribute("position", new BufferAttribute(ringPos, 3));
  const ringMat = new LineBasicMaterial({ transparent: true, blending: AdditiveBlending, depthWrite: false });
  const rings = new LineSegments(ringGeo, ringMat);

  const haloGeo = new PlaneGeometry(4.4, 4.4);
  const haloMat = new ShaderMaterial({
    vertexShader: haloVertex, fragmentShader: haloFragment, transparent: true, depthWrite: false, blending: AdditiveBlending,
    uniforms: { uColor: { value: new Color() }, uAlpha: { value: 0.6 }, uPulse: { value: 0 } },
  });
  const halo = new Mesh(haloGeo, haloMat);
  halo.position.z = -0.2;

  spin.add(core, shell);
  tilt.add(rings);
  tilt.rotation.x = 0.5;
  root.add(halo, spin, tilt);

  const lighter = new Color();
  return {
    root,
    update(time, camera, color, bg, alpha, still) {
      const t = still ? 1.2 : time;
      spin.rotation.y = t * 0.45;
      spin.rotation.x = t * 0.18;
      tilt.rotation.y = -t * 0.3;
      shell.rotation.y = -t * 0.9;
      const pulse = 0.5 + 0.5 * Math.sin(t * 1.6);
      core.scale.setScalar(1 + 0.05 * pulse);
      coreMat.uniforms.uBody.value.copy(bg).lerp(color, 0.45);
      coreMat.uniforms.uRim.value.copy(color).lerp(lighter.set(0xffffff), 0.25);
      coreMat.uniforms.uAlpha.value = alpha;
      shellMat.color.copy(color).lerp(lighter.set(0xffffff), 0.2);
      shellMat.opacity = 0.75 * alpha;
      ringMat.color.copy(color).lerp(lighter.set(0xffffff), 0.45);
      ringMat.opacity = 0.9 * alpha;
      haloMat.uniforms.uColor.value.copy(color);
      haloMat.uniforms.uAlpha.value = 0.7 * alpha;
      haloMat.uniforms.uPulse.value = pulse;
      halo.quaternion.copy(camera.quaternion); // the halo faces the camera (root itself never turns)
    },
    dispose() {
      for (const g of [coreGeo, shellGeo, ringGeo, haloGeo]) g.dispose();
      for (const m of [coreMat, shellMat, ringMat, haloMat]) m.dispose();
    },
  };
}
