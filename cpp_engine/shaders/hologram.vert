#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uTime;
uniform int uPass; // 0 = Fresnel core, 1 = Neon wireframe overlay

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec3 vLocalPos;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    // Slight normal offset on wireframe pass so lines never z-fight the core
    vec3 nrm = normalize(mat3(uModel) * aNormal);
    if (uPass == 1) {
        world.xyz += nrm * 0.004;
    }
    vWorldPos = world.xyz;
    vNormal = nrm;
    vUV = aUV;
    vLocalPos = aPos;
    gl_Position = uProj * uView * world;
}
