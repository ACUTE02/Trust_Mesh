// Shaders for the Graph world: beads as lit-sphere point sprites (one draw call, no geometry per bead) and the
// copper threads as one line list. Positions blend between four chapters on the GPU.
export const beadVertex = /* glsl */ `
  attribute vec3 aT0; attribute vec3 aT1; attribute vec3 aT2; attribute vec3 aT3; attribute vec3 aLeak;
  attribute vec4 aVis; attribute float aTint; attribute float aKind; attribute float aSeed;
  uniform float uStage; uniform float uLeak; uniform float uTime; uniform float uSway; uniform float uSize; uniform float uPixel;
  uniform float uLiftY; uniform float uLift;
  varying float vTint; varying float vKind; varying float vAlpha; varying float vSeed;

  vec3 chapter(float k) {
    if (k < 0.5) return aT0;
    if (k < 1.5) return aT1;
    if (k < 2.5) return aT2;
    if (k < 3.5) return aT3;
    return aT0; // the last chapter is the calm loom again
  }
  float visAt(float k) {
    if (k < 0.5) return aVis.x;
    if (k < 1.5) return aVis.y;
    if (k < 2.5) return aVis.z;
    if (k < 3.5) return aVis.w;
    return aVis.x;
  }

  void main() {
    float s = clamp(uStage, 0.0, 4.0);
    float k = floor(min(s, 3.0));
    float f = smoothstep(0.0, 1.0, s - k);
    vec3 p = mix(chapter(k), chapter(k + 1.0), f);
    float vis = mix(visAt(k), visAt(k + 1.0), f);
    // chapter 1: a leaking future bead slides into the past sheet while uLeak is up
    float in1 = 1.0 - clamp(abs(s - 1.0), 0.0, 1.0);
    float leaker = step(2.5, aKind);
    p = mix(p, aLeak, leaker * uLeak * in1);
    // slow sway, like beads on strings, and the pointer lifts the thread nearest to it
    p.y += sin(uTime * 0.6 + aSeed * 6.28 + p.x * 0.4) * 0.05 * uSway;
    p.y += uLift * exp(-pow((p.y - uLiftY) * 2.2, 2.0));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = uSize * (0.75 + aSeed * 0.5);
    gl_PointSize = size * uPixel * (9.0 / -mv.z) * (0.2 + 0.8 * vis);
    float late = smoothstep(1.5, 2.5, s) * (1.0 - smoothstep(3.0, 4.0, s));        // in the data chapters the tint comes from the data, not from the bead
    vAlpha = vis;
    vTint = mix(0.0, aTint, late);
    vKind = mix(aKind, 0.0, late);
    vSeed = aSeed;
  }
`;

export const beadFragment = /* glsl */ `
  uniform vec3 uBase; uniform vec3 uAccent; uniform vec3 uFlag; uniform vec3 uLight; uniform float uLeak; uniform float uStage;
  varying float vTint; varying float vKind; varying float vAlpha; varying float vSeed;
  void main() {
    vec2 c = (gl_PointCoord - 0.5) * 2.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 n = vec3(c.x, -c.y, sqrt(1.0 - r2));
    float diff = clamp(dot(n, normalize(vec3(-0.5, 0.7, 0.8))), 0.0, 1.0);
    float spec = pow(clamp(dot(reflect(vec3(0.5, -0.7, -0.8), n), vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 14.0);
    vec3 col = uBase;
    float in1 = 1.0 - clamp(abs(uStage - 1.0), 0.0, 1.0);
    float flagged = step(0.5, vKind) * (1.0 - step(2.5, vKind));          // an illustrative flagged bead
    float leaking = step(2.5, vKind) * uLeak * in1;                     // a future bead inside the past sheet
    col = mix(col, uAccent, step(0.5, vTint) * (1.0 - step(1.5, vTint)));
    col = mix(col, uFlag, max(max(flagged, leaking), step(1.5, vTint)));
    col = col * (0.45 + 0.75 * diff) + uLight * spec * 0.55;
    gl_FragColor = vec4(col, vAlpha * smoothstep(1.0, 0.85, r2));
  }
`;

export const threadVertex = /* glsl */ `
  attribute float aX; attribute float aRow;
  uniform float uStage; uniform float uTime; uniform float uSway; uniform float uLiftY; uniform float uLift;
  varying float vShow;
  void main() {
    float y0 = (aRow - 10.5) * 0.32;
    float sag = -0.22 * (1.0 - min(1.0, pow(aX / 8.6, 2.0)));
    vec3 loom = vec3(aX, y0 + sag, (aRow - 11.0) * 0.06);
    bool future = aRow >= 11.0;
    float sheetRow = future ? aRow - 11.0 : aRow;
    vec3 sheet = vec3((future ? 4.9 : -4.9) + aX * 0.42, ((sheetRow - 10.5) * 0.32) * 1.45 + sag * 0.3, (aRow - 11.0) * 0.12);
    float s = clamp(uStage, 0.0, 4.0);
    float f1 = smoothstep(0.0, 1.0, clamp(s, 0.0, 1.0));
    float f2 = smoothstep(1.0, 2.0, s);
    vec3 p = mix(loom, sheet, f1);
    p = mix(p, loom, smoothstep(3.0, 4.0, s));
    p.y += sin(uTime * 0.6 + aRow + aX * 0.4) * 0.05 * uSway;
    p.y += uLift * exp(-pow((p.y - uLiftY) * 2.2, 2.0));
    vShow = (1.0 - f2) + smoothstep(3.0, 4.0, s) * f2;      // the threads are drawn in the loom and sheet chapters and again at the end
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

export const threadFragment = /* glsl */ `
  uniform vec3 uThread; uniform float uAlpha;
  varying float vShow;
  void main() { gl_FragColor = vec4(uThread, uAlpha * clamp(vShow, 0.0, 1.0)); }
`;
