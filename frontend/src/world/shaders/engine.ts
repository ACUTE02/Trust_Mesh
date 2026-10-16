// A small fresnel shader for the solid engine nodes: bright rim, darker body. No lights, no shadows, no
// environment map: depth is faked from the angle between the surface and the eye, which costs almost nothing.
export const engineVertex = /* glsl */ `
  varying vec3 vNormal; varying vec3 vView;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

export const engineFragment = /* glsl */ `
  uniform vec3 uBody; uniform vec3 uRim; uniform float uAlpha;
  varying vec3 vNormal; varying vec3 vView;
  void main() {
    float f = pow(1.0 - clamp(dot(normalize(vNormal), normalize(vView)), 0.0, 1.0), 2.2);
    gl_FragColor = vec4(mix(uBody, uRim, f), uAlpha);
  }
`;
