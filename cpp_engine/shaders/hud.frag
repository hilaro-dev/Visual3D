#version 330 core
in vec2 vUV;
in vec4 vColor;

uniform int uMode;        // 0 = solid/glow vertex color, 1 = font atlas, 2 = holographic camera bg
uniform sampler2D uTex;
uniform float uTime;

out vec4 FragColor;

void main() {
    if (uMode == 1) {
        float cov = texture(uTex, vUV).r;
        FragColor = vec4(vColor.rgb, vColor.a * cov);
    } else if (uMode == 2) {
        vec3 tex = texture(uTex, vUV).rgb;
        float luma = dot(tex, vec3(0.299, 0.587, 0.114));
        // Live camera feed with subtle sci-fi holographic scanline & cyan edge grade
        vec3 holo = mix(tex, vec3(luma) * vec3(0.25, 0.85, 1.0), 0.15);
        float scan = 0.94 + 0.06 * sin(vUV.y * 360.0 + uTime * 8.0);
        FragColor = vec4(holo * scan * vColor.rgb, vColor.a);
    } else {
        FragColor = vColor;
    }
}
