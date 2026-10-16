// The Voice terrain. One grid, heights from a vertex attribute (written once per clip), everything else in the shaders:
// breathing, the score light sweeping along time, the collapse for "not speech", and a fog bank at the cut-off
// frequency for "low-quality audio". Dark theme uses a thermal colour map (as a real spectrogram does); the light
// theme draws the same heights as contour lines on paper.
export const terrainVertex = /* glsl */ `
  attribute float aH;
  uniform float uAmp; uniform float uCollapse; uniform float uFog; uniform float uCut; uniform float uTime; uniform float uBreath;
  varying float vH; varying float vFreq; varying float vTime; varying float vFogF;
  void main() {
    vTime = uv.x;
    vFreq = uv.y;
    float breathe = 1.0 + uBreath * 0.06 * sin(uTime * 0.9 + uv.x * 5.0);
    // above the cut-off frequency the terrain drops away into the fog: the cliff
    float fogF = uFog * smoothstep(uCut - 0.03, uCut + 0.03, uv.y);
    vFogF = fogF;
    float h = aH * breathe * (1.0 - 0.9 * fogF) * (1.0 - uCollapse);
    vH = h;
    vec3 p = position + vec3(0.0, h * uAmp, 0.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

export const terrainFragment = /* glsl */ `
  uniform float uContour; uniform vec3 uPaper; uniform vec3 uInk; uniform vec3 uFogColor; uniform vec3 uGlow;
  uniform float uScoreX; uniform float uScoreOn; uniform float uAlpha;
  varying float vH; varying float vFreq; varying float vTime; varying float vFogF;

  // thermal colour map: night purple, red, orange, yellow, white
  vec3 thermal(float t) {
    vec3 a = vec3(0.02, 0.0, 0.08);
    vec3 b = vec3(0.45, 0.04, 0.35);
    vec3 c = vec3(0.95, 0.25, 0.1);
    vec3 d = vec3(1.0, 0.75, 0.15);
    vec3 e = vec3(1.0, 1.0, 0.85);
    if (t < 0.25) return mix(a, b, t / 0.25);
    if (t < 0.5) return mix(b, c, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c, d, (t - 0.5) / 0.25);
    return mix(d, e, (t - 0.75) / 0.25);
  }

  void main() {
    vec3 col;
    if (uContour > 0.5) {
      // contour lines every 1/14 of the height, anti-aliased by how fast the height changes on screen
      float g = vH * 14.0;
      float w = fwidth(g) * 1.2 + 0.001;
      float line = 1.0 - smoothstep(0.0, w, abs(fract(g - 0.5) - 0.5));
      col = mix(uPaper, uInk, clamp(line * (0.35 + 0.65 * smoothstep(0.02, 0.2, vH)), 0.0, 1.0));
    } else {
      col = thermal(clamp(vH * 1.15, 0.0, 1.0));
      col *= 0.5 + 0.5 * smoothstep(0.0, 0.2, vH);
    }
    // the score arrives as a light moving along the time axis
    float glow = exp(-pow((vTime - uScoreX) * 7.0, 2.0)) * uScoreOn;
    col = mix(col, uGlow, clamp(glow * 0.75, 0.0, 1.0));
    col = mix(col, uFogColor, vFogF * 0.78);
    gl_FragColor = vec4(col, uAlpha);
  }
`;
