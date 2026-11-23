#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec3 vLocalPos;

uniform vec3 uColor;
uniform vec3 uCamPos;
uniform float uTime;
uniform int uPass;       // 0 = Fresnel translucent core, 1 = Neon wireframe
uniform float uSelected; // 0..1 selection highlight
uniform float uHovered;  // 0..1 hover highlight
uniform float uSpawn;    // 0..1 materialization progress

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCamPos - vWorldPos);

    // Fresnel rim lighting (hard-light projection edge glow)
    float ndotv = clamp( abs(dot(N, V)), 0.0, 1.0 );
    float fresnel = pow(1.0 - ndotv, 2.4);

    // Animated horizontal scanlines + subtle traveling energy wave
    float scanFine = 0.85 + 0.15 * sin(vWorldPos.y * 52.0 - uTime * 7.5);
    float scanBand = smoothstep(0.4, 0.95, 0.5 + 0.5 * sin(vWorldPos.y * 5.5 - uTime * 2.4));
    float flicker  = 0.96 + 0.04 * sin(uTime * 37.0 + vWorldPos.x * 3.0);

    // Boost brightness when hovered or selected
    float boost = 1.0 + 0.40 * uSelected + 0.22 * uHovered;
    vec3 baseCol = mix(uColor, vec3(1.0), 0.18 * uSelected);

    if (uPass == 0) {
        // PASS 0: Translucent Fresnel-glow core
        vec3 core = baseCol * (0.26 + 0.95 * fresnel + 0.25 * scanBand) * scanFine * flicker * boost;
        // Subtle specular highlight from above-right virtual light
        vec3 L = normalize(vec3(0.45, 0.8, 0.6));
        float spec = pow(max(dot(reflect(-L, N), V), 0.0), 20.0) * 0.45;
        core += (baseCol + vec3(0.35)) * spec;

        float alpha = (0.14 + 0.52 * fresnel + 0.10 * scanBand) * uSpawn;
        alpha = clamp(alpha + 0.08 * uSelected, 0.0, 0.88);
        FragColor = vec4(core, alpha);
    } else {
        // PASS 1: Bright neon wireframe overlay
        float pulse = 0.82 + 0.18 * sin(uTime * 5.0 + (vLocalPos.x + vLocalPos.y) * 6.0);
        vec3 wireCol = mix(baseCol, vec3(1.0), 0.35 + 0.25 * uSelected) * (1.25 + 0.45 * fresnel) * pulse * boost;
        float alpha = clamp((0.68 + 0.25 * fresnel + 0.20 * uSelected) * scanFine * uSpawn, 0.0, 1.0);
        FragColor = vec4(wireCol, alpha);
    }
}
