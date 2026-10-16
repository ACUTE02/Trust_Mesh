// Shaders for the Home mesh. Everything animates on the GPU from a few uniforms: the CPU never touches the
// per-node data after the geometry is built. Each node carries its position in every chapter (aT0..aT4); the
// vertex shader blends between neighbouring chapters by `uStage`, then adds a slow drift and a pointer pull.
const common = /* glsl */ `
  attribute vec3 aT0; attribute vec3 aT1; attribute vec3 aT2; attribute vec3 aT3; attribute vec3 aT4;
  attribute float aSeed;
  uniform float uTime; uniform float uStage; uniform float uDrift; uniform vec2 uPointer; uniform vec2 uHalf;

  vec3 chapter(float k) {
    if (k < 0.5) return aT0;
    if (k < 1.5) return aT1;
    if (k < 2.5) return aT2;
    if (k < 3.5) return aT3;
    if (k < 4.5) return aT4;
    return aT0; // chapter 5 is the calm mesh again
  }

  vec3 place() {
    float s = clamp(uStage, 0.0, 5.0);
    float k = floor(s);
    float f = smoothstep(0.0, 1.0, s - k);
    vec3 p = mix(chapter(k), chapter(k + 1.0), f);
    float a = aSeed * 6.2831 + uTime * 0.35;
    p += uDrift * vec3(sin(a), cos(a * 1.3), sin(a * 0.7)) * 0.16;
    // the pointer pulls nearby threads toward it, like fabric
    vec2 d = uPointer * uHalf - p.xy;
    float pull = exp(-dot(d, d) * 0.35) * uDrift;
    p.xy += d * pull * 0.22;
    return p;
  }
`;

export const meshVertex = /* glsl */ `
  ${common}
  attribute float aKind;
  uniform float uSize; uniform float uPixel;
  varying float vKind; varying float vSeed; varying float vFade;
  void main() {
    vec4 mv = modelViewMatrix * vec4(place(), 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * uPixel * (0.6 + aSeed * 0.8) * (9.0 / -mv.z);
    vKind = aKind; vSeed = aSeed;
    // chapter-4 nodes above the cut-off line fade out: the missing band
    float inFour = 1.0 - clamp(abs(uStage - 4.0), 0.0, 1.0);
    vFade = mix(1.0, aKind > 2.5 ? 0.12 : 1.0, inFour);
  }
`;

export const meshFragment = /* glsl */ `
  uniform vec3 uThread; uniform vec3 uAccent; uniform vec3 uFlag; uniform float uAlpha;
  varying float vKind; varying float vSeed; varying float vFade;
  void main() {
    float r = length(gl_PointCoord - 0.5);
    if (r > 0.5) discard;
    float edge = smoothstep(0.5, 0.2, r);
    vec3 col = mix(uThread, uAccent, step(0.82, vSeed));
    float flagged = step(0.5, vKind) * (1.0 - step(2.5, vKind)); // leaked (1) and shortcut (2)
    col = mix(col, uFlag, flagged);
    gl_FragColor = vec4(col, edge * uAlpha * vFade);
  }
`;

export const lineVertex = /* glsl */ `
  ${common}
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(place(), 1.0);
  }
`;

export const lineFragment = /* glsl */ `
  uniform vec3 uThread; uniform float uAlpha; uniform float uStage;
  void main() {
    // threads are strongest in the resting mesh and the web chapter, and fade out in the data chapters
    float strength = 1.0 - smoothstep(1.0, 2.0, uStage) * 0.85;
    strength = mix(strength, 1.0, smoothstep(4.0, 5.0, uStage));
    // while the nodes are mid-move between two chapters the threads would be long streaks: fade them out
    float settled = abs(fract(uStage) - 0.5) * 2.0;
    strength *= mix(0.12, 1.0, smoothstep(0.0, 0.8, settled));
    gl_FragColor = vec4(uThread, uAlpha * strength);
  }
`;
