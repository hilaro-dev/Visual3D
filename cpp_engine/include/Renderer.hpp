// Renderer.hpp -- Dual-Pass Holographic 3D Renderer + Sci-Fi AR/HUD Engine.
//
// Responsibilities:
//   * Compiles GLSL 3.30 shaders from cpp_engine/shaders/ (with embedded
//     fallback strings so the binary also runs standalone from any directory).
//   * Renders the 3D background perspective grid at Z = -2.0.
//   * Renders all SceneObjects with a Dual-Pass Hologram shader:
//       Pass 0 -> translucent Fresnel-glow core + scanlines + rim light
//       Pass 1 -> bright neon wireframe overlay
//   * Renders the AR Hand Skeleton Overlay (21 landmarks, glowing cyan/neon
//     bones, circular targeting reticles on active fingertips, dynamic
//     HSV color-wheel + mixed-RGB ring around the thumb, and two-hand scaling
//     beam).
//   * Renders the Futuristic Telemetry HUD (corner brackets, FPS/UDP latency
//     counters, active gesture state readouts, shape selector bar, and a live
//     RGB additive channel mixer panel).
#pragma once

#include "GLLoader.hpp"
#include "MathUtils.hpp"
#include "MeshGenerator.hpp"
#include "Protocol.hpp"
#include "UDPReceiver.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace v3d {

// ---------------------------------------------------------------------------
// Hand landmark topology (21 MediaPipe landmarks)
// ---------------------------------------------------------------------------
inline constexpr std::pair<int, int> kHandBones[21] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},    // Thumb
    {0, 5},   {5, 6},   {6, 7},   {7, 8},    // Index
    {5, 9},   {9, 10},  {10, 11}, {11, 12},  // Middle
    {9, 13},  {13, 14}, {14, 15}, {15, 16},  // Ring
    {13, 17}, {0, 17},  {17, 18}, {18, 19}, {19, 20}  // Pinky + palm arch
};

// ---------------------------------------------------------------------------
// Embedded 5x7 ASCII bitmap font (printable ASCII 32..95, uppercase sci-fi)
// Each glyph is 5 columns of 7-bit vertical bitmasks (bit0 = top row).
// ---------------------------------------------------------------------------
inline const uint8_t* glyph5x7(char c) {
    static const uint8_t kTable[64][5] = {
        {0x00, 0x00, 0x00, 0x00, 0x00},  // 32 ' '
        {0x00, 0x00, 0x5F, 0x00, 0x00},  // 33 '!'
        {0x00, 0x07, 0x00, 0x07, 0x00},  // 34 '"'
        {0x14, 0x7F, 0x14, 0x7F, 0x14},  // 35 '#'
        {0x24, 0x2A, 0x7F, 0x2A, 0x12},  // 36 '$'
        {0x23, 0x13, 0x08, 0x64, 0x62},  // 37 '%'
        {0x36, 0x49, 0x55, 0x22, 0x50},  // 38 '&'
        {0x00, 0x05, 0x03, 0x00, 0x00},  // 39 '''
        {0x00, 0x1C, 0x22, 0x41, 0x00},  // 40 '('
        {0x00, 0x41, 0x22, 0x1C, 0x00},  // 41 ')'
        {0x14, 0x08, 0x3E, 0x08, 0x14},  // 42 '*'
        {0x08, 0x08, 0x3E, 0x08, 0x08},  // 43 '+'
        {0x00, 0x50, 0x30, 0x00, 0x00},  // 44 ','
        {0x08, 0x08, 0x08, 0x08, 0x08},  // 45 '-'
        {0x00, 0x60, 0x60, 0x00, 0x00},  // 46 '.'
        {0x20, 0x10, 0x08, 0x04, 0x02},  // 47 '/'
        {0x3E, 0x51, 0x49, 0x45, 0x3E},  // 48 '0'
        {0x00, 0x42, 0x7F, 0x40, 0x00},  // 49 '1'
        {0x42, 0x61, 0x51, 0x49, 0x46},  // 50 '2'
        {0x21, 0x41, 0x45, 0x4B, 0x31},  // 51 '3'
        {0x18, 0x14, 0x12, 0x7F, 0x10},  // 52 '4'
        {0x27, 0x45, 0x45, 0x45, 0x39},  // 53 '5'
        {0x3C, 0x4A, 0x49, 0x49, 0x30},  // 54 '6'
        {0x01, 0x71, 0x09, 0x05, 0x03},  // 55 '7'
        {0x36, 0x49, 0x49, 0x49, 0x36},  // 56 '8'
        {0x06, 0x49, 0x49, 0x29, 0x1E},  // 57 '9'
        {0x00, 0x36, 0x36, 0x00, 0x00},  // 58 ':'
        {0x00, 0x56, 0x36, 0x00, 0x00},  // 59 ';'
        {0x08, 0x14, 0x22, 0x41, 0x00},  // 60 '<'
        {0x14, 0x14, 0x14, 0x14, 0x14},  // 61 '='
        {0x00, 0x41, 0x22, 0x14, 0x08},  // 62 '>'
        {0x02, 0x01, 0x51, 0x09, 0x06},  // 63 '?'
        {0x32, 0x49, 0x79, 0x41, 0x3E},  // 64 '@'
        {0x7E, 0x11, 0x11, 0x11, 0x7E},  // 65 'A'
        {0x7F, 0x49, 0x49, 0x49, 0x36},  // 66 'B'
        {0x3E, 0x41, 0x41, 0x41, 0x22},  // 67 'C'
        {0x7F, 0x41, 0x41, 0x22, 0x1C},  // 68 'D'
        {0x7F, 0x49, 0x49, 0x49, 0x41},  // 69 'E'
        {0x7F, 0x09, 0x09, 0x09, 0x01},  // 70 'F'
        {0x3E, 0x41, 0x49, 0x49, 0x7A},  // 71 'G'
        {0x7F, 0x08, 0x08, 0x08, 0x7F},  // 72 'H'
        {0x00, 0x41, 0x7F, 0x41, 0x00},  // 73 'I'
        {0x20, 0x40, 0x41, 0x3F, 0x01},  // 74 'J'
        {0x7F, 0x08, 0x14, 0x22, 0x41},  // 75 'K'
        {0x7F, 0x40, 0x40, 0x40, 0x40},  // 76 'L'
        {0x7F, 0x02, 0x0C, 0x02, 0x7F},  // 77 'M'
        {0x7F, 0x04, 0x08, 0x10, 0x7F},  // 78 'N'
        {0x3E, 0x41, 0x41, 0x41, 0x3E},  // 79 'O'
        {0x7F, 0x09, 0x09, 0x09, 0x06},  // 80 'P'
        {0x3E, 0x41, 0x51, 0x21, 0x5E},  // 81 'Q'
        {0x7F, 0x09, 0x19, 0x29, 0x46},  // 82 'R'
        {0x46, 0x49, 0x49, 0x49, 0x31},  // 83 'S'
        {0x01, 0x01, 0x7F, 0x01, 0x01},  // 84 'T'
        {0x3F, 0x40, 0x40, 0x40, 0x3F},  // 85 'U'
        {0x1F, 0x20, 0x40, 0x20, 0x1F},  // 86 'V'
        {0x3F, 0x40, 0x38, 0x40, 0x3F},  // 87 'W'
        {0x63, 0x14, 0x08, 0x14, 0x63},  // 88 'X'
        {0x07, 0x08, 0x70, 0x08, 0x07},  // 89 'Y'
        {0x61, 0x51, 0x49, 0x45, 0x43},  // 90 'Z'
        {0x00, 0x7F, 0x41, 0x41, 0x00},  // 91 '['
        {0x02, 0x04, 0x08, 0x10, 0x20},  // 92 '\'
        {0x00, 0x41, 0x41, 0x7F, 0x00},  // 93 ']'
        {0x04, 0x02, 0x01, 0x02, 0x04},  // 94 '^'
        {0x40, 0x40, 0x40, 0x40, 0x40},  // 95 '_'
    };
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    if (c < 32 || c > 95) c = '?';
    return kTable[c - 32];
}

// ---------------------------------------------------------------------------
// Embedded fallback GLSL sources (identical to cpp_engine/shaders/*)
// ---------------------------------------------------------------------------
namespace shaders {

inline const char* kHoloVert = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uTime;
uniform int uPass;
out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec3 vLocalPos;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vec3 nrm = normalize(mat3(uModel) * aNormal);
    if (uPass == 1) world.xyz += nrm * 0.004;
    vWorldPos = world.xyz;
    vNormal = nrm;
    vUV = aUV;
    vLocalPos = aPos;
    gl_Position = uProj * uView * world;
}
)GLSL";

inline const char* kHoloFrag = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec3 vLocalPos;
uniform vec3 uColor;
uniform vec3 uCamPos;
uniform float uTime;
uniform int uPass;
uniform float uSelected;
uniform float uHovered;
uniform float uSpawn;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCamPos - vWorldPos);
    float ndotv = clamp(abs(dot(N, V)), 0.0, 1.0);
    float fresnel = pow(1.0 - ndotv, 2.4);
    float scanFine = 0.85 + 0.15 * sin(vWorldPos.y * 52.0 - uTime * 7.5);
    float scanBand = smoothstep(0.4, 0.95, 0.5 + 0.5 * sin(vWorldPos.y * 5.5 - uTime * 2.4));
    float flicker  = 0.96 + 0.04 * sin(uTime * 37.0 + vWorldPos.x * 3.0);
    float boost = 1.0 + 0.40 * uSelected + 0.22 * uHovered;
    vec3 baseCol = mix(uColor, vec3(1.0), 0.18 * uSelected);
    if (uPass == 0) {
        vec3 core = baseCol * (0.26 + 0.95 * fresnel + 0.25 * scanBand) * scanFine * flicker * boost;
        vec3 L = normalize(vec3(0.45, 0.8, 0.6));
        float spec = pow(max(dot(reflect(-L, N), V), 0.0), 20.0) * 0.45;
        core += (baseCol + vec3(0.35)) * spec;
        float alpha = (0.14 + 0.52 * fresnel + 0.10 * scanBand) * uSpawn;
        alpha = clamp(alpha + 0.08 * uSelected, 0.0, 0.88);
        FragColor = vec4(core, alpha);
    } else {
        float pulse = 0.82 + 0.18 * sin(uTime * 5.0 + (vLocalPos.x + vLocalPos.y) * 6.0);
        vec3 wireCol = mix(baseCol, vec3(1.0), 0.35 + 0.25 * uSelected) * (1.25 + 0.45 * fresnel) * pulse * boost;
        float alpha = clamp((0.68 + 0.25 * fresnel + 0.20 * uSelected) * scanFine * uSpawn, 0.0, 1.0);
        FragColor = vec4(wireCol, alpha);
    }
}
)GLSL";

inline const char* kHudVert = R"GLSL(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
layout (location = 2) in vec4 aColor;
uniform mat4 uProj;
out vec2 vUV;
out vec4 vColor;
void main() {
    vUV = aUV;
    vColor = aColor;
    gl_Position = uProj * vec4(aPos, 0.0, 1.0);
}
)GLSL";

inline const char* kHudFrag = R"GLSL(
#version 330 core
in vec2 vUV;
in vec4 vColor;
uniform int uMode;
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
        vec3 holo = mix(tex, vec3(luma) * vec3(0.25, 0.85, 1.0), 0.15);
        float scan = 0.94 + 0.06 * sin(vUV.y * 360.0 + uTime * 8.0);
        FragColor = vec4(holo * scan * vColor.rgb, vColor.a);
    } else {
        FragColor = vColor;
    }
}
)GLSL";

inline const char* kGridVert = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
uniform mat4 uVP;
out vec3 vWorldPos;
out vec4 vColor;
void main() {
    vWorldPos = aPos;
    vColor = aColor;
    gl_Position = uVP * vec4(aPos, 1.0);
}
)GLSL";

inline const char* kGridFrag = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in vec4 vColor;
uniform float uTime;
out vec4 FragColor;
void main() {
    float dist = length(vWorldPos.xy * vec2(0.55, 0.75));
    float vignette = clamp(1.0 - dist * 0.16, 0.0, 1.0);
    float wave = 0.80 + 0.20 * sin(vWorldPos.x * 1.4 + vWorldPos.y * 1.4 - uTime * 2.2);
    FragColor = vec4(vColor.rgb * wave, vColor.a * vignette);
}
)GLSL";

}  // namespace shaders

// ---------------------------------------------------------------------------
// Smoothed hand state for rendering & interaction
// ---------------------------------------------------------------------------
struct SmoothedHand {
    bool present = false;
    uint8_t handedness = 0;  // 0 = Left, 1 = Right
    uint8_t fingersMask = 0;
    uint8_t fingerCount = 0;
    uint8_t gesture = 0;
    uint8_t contactMask = 0;
    bool pinch = false;
    bool fist = false;
    bool open = false;
    bool thumbFolded = false;   // Thumb folded inward across palm (dist(4,17) < dist(3,17) or dist(2,17))
    bool rotationMode = false;  // Left Hand ROTATION_MODE (Left Thumb folded + 4 fingers extended)
    bool spawnBurst = false;    // Left Hand fast Fist -> Open-Palm spawner trigger
    float pinchDist = 1.f;
    Vec3 color{0.f, 0.f, 0.f};
    std::array<Vec3, 21> lmNorm{};   // Normalized [0, 1] smoothed landmarks
    std::array<Vec3, 21> lmWorld{};  // Mapped to OpenGL world space
    float visibility = 0.f;          // 0..1 fade-in/out
};

// ---------------------------------------------------------------------------
// HUD state passed from the interaction controller to the renderer
// ---------------------------------------------------------------------------
struct HudTelemetry {
    float renderFps = 0.f;
    float trackerFps = 0.f;
    LinkStats link;
    PrimitiveType activeBlueprint = PrimitiveType::Cube;
    bool rightBlueprintMode = false;   // True when Right Thumb is folded & 1..4 fingers extended
    bool rotationModeActive = false;   // True when Right Hand holds object + Left Hand ROTATION_MODE active
    float crushProgress = 0.f;         // 0.0 -> 1.0 (0.4s Right-Hand Fist Crush Delete indicator)
    Vec2 crushWorldPos{0.f, 0.f};
    Vec3 activeColor{0.0f, 0.90f, 1.0f};
    uint8_t activeContactMask = 0;
    bool twoHandScaling = false;
    float twoHandDistance = 0.f;
    bool demoMode = false;
    bool showCameraBg = true;
    std::string statusBanner;
    float spawnFlash = 0.f;  // 0..1 flash pulse on burst spawn
    Vec2 spawnFlashWorld{0.f, 0.f};
};

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
class Renderer {
public:
    // Camera sits at +Z looking toward the Z = 0 manipulation plane and
    // Z = -2.0 perspective background grid.
    static constexpr float kCameraZ = 5.0f;
    static constexpr float kFovYDeg = 45.0f;
    static constexpr float kGridZ = -2.0f;

    bool init(const std::string& shaderDir = "shaders") {
        holoProg_ = buildProgram(shaderDir + "/hologram.vert", shaderDir + "/hologram.frag",
                                 shaders::kHoloVert, shaders::kHoloFrag);
        hudProg_ = buildProgram(shaderDir + "/hud.vert", shaderDir + "/hud.frag",
                                shaders::kHudVert, shaders::kHudFrag);
        gridProg_ = buildProgram("", "", shaders::kGridVert, shaders::kGridFrag);
        if (!holoProg_ || !hudProg_ || !gridProg_) return false;

        // Upload all 4 procedural primitive meshes
        meshes_[static_cast<int>(PrimitiveType::Cube)].upload(MeshGenerator::makeCube());
        meshes_[static_cast<int>(PrimitiveType::Capsule)].upload(MeshGenerator::makeCapsule());
        meshes_[static_cast<int>(PrimitiveType::Cylinder)].upload(MeshGenerator::makeCylinder());
        meshes_[static_cast<int>(PrimitiveType::Torus)].upload(MeshGenerator::makeTorus());

        // Dynamic 2D HUD VAO/VBO
        glGenVertexArrays(1, &hudVao_);
        glGenBuffers(1, &hudVbo_);
        glBindVertexArray(hudVao_);
        glBindBuffer(GL_ARRAY_BUFFER, hudVbo_);
        glBufferData(GL_ARRAY_BUFFER, 65536 * sizeof(HudVertex), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(HudVertex),
                              reinterpret_cast<const void*>(offsetof(HudVertex, pos)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(HudVertex),
                              reinterpret_cast<const void*>(offsetof(HudVertex, uv)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(HudVertex),
                              reinterpret_cast<const void*>(offsetof(HudVertex, col)));
        glBindVertexArray(0);

        buildBackgroundGrid();
        buildCameraTexture();
        return true;
    }

    void shutdown() {
        for (auto& m : meshes_) m.destroy();
        if (hudVao_) glDeleteVertexArrays(1, &hudVao_);
        if (hudVbo_) glDeleteBuffers(1, &hudVbo_);
        if (gridVao_) glDeleteVertexArrays(1, &gridVao_);
        if (gridVbo_) glDeleteBuffers(1, &gridVbo_);
        if (camTex_) glDeleteTextures(1, &camTex_);
        if (holoProg_) glDeleteProgram(holoProg_);
        if (hudProg_) glDeleteProgram(hudProg_);
        if (gridProg_) glDeleteProgram(gridProg_);
    }

    // Coordinate mapping: Normalized MediaPipe (x, y) in [0, 1] -> OpenGL
    // world space (X, Y) on the Z = 0.0 plane.
    static Vec3 normalizedToWorld(float nx, float ny, float aspect, float z = 0.0f) {
        const float halfH = (kCameraZ - z) * std::tan(kFovYDeg * 0.5f * kPi / 180.0f);
        const float halfW = halfH * aspect;
        const float wx = (nx * 2.0f - 1.0f) * halfW;
        const float wy = (1.0f - ny * 2.0f) * halfH;
        return {wx, wy, z};
    }

    static Vec2 worldToScreen(Vec3 worldPos, int vpW, int vpH) {
        const float aspect = vpH > 0 ? static_cast<float>(vpW) / vpH : 16.f / 9.f;
        const Mat4 vp = Mat4::perspective(kFovYDeg * kPi / 180.f, aspect, 0.1f, 50.f) *
                        Mat4::lookAt({0.f, 0.f, kCameraZ}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});
        const Vec4 clip = vp * Vec4(worldPos, 1.0f);
        if (std::abs(clip.w) < 1e-5f) return {0.f, 0.f};
        const float ndcX = clip.x / clip.w;
        const float ndcY = clip.y / clip.w;
        return {(ndcX * 0.5f + 0.5f) * vpW, (1.0f - (ndcY * 0.5f + 0.5f)) * vpH};
    }

    void updateCameraFeed(const VideoFrame& vf) {
        if (vf.width <= 0 || vf.height <= 0 || vf.pixels.empty()) return;
        glBindTexture(GL_TEXTURE_2D, camTex_);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        const GLenum fmt = (vf.channels == 1) ? GL_RED : GL_RGB;
        const GLint internalFmt = (vf.channels == 1) ? GL_R8 : GL_RGB8;
        if (vf.width != camW_ || vf.height != camH_ || vf.channels != camC_) {
            camW_ = vf.width;
            camH_ = vf.height;
            camC_ = vf.channels;
            glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, camW_, camH_, 0, fmt, GL_UNSIGNED_BYTE,
                         vf.pixels.data());
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camW_, camH_, fmt, GL_UNSIGNED_BYTE,
                            vf.pixels.data());
        }
        hasCamFrame_ = true;
    }

    void renderFrame(int vpW, int vpH, float timeSec, const std::vector<SceneObject>& objects,
                     const std::array<SmoothedHand, 2>& hands, const HudTelemetry& hud) {
        if (vpW <= 0 || vpH <= 0) return;
        const float aspect = static_cast<float>(vpW) / static_cast<float>(vpH);

        glViewport(0, 0, vpW, vpH);
        glClearColor(0.012f, 0.020f, 0.042f, 0.94f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const Mat4 proj3D = Mat4::perspective(kFovYDeg * kPi / 180.f, aspect, 0.1f, 50.f);
        const Mat4 view3D = Mat4::lookAt({0.f, 0.f, kCameraZ}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});
        const Mat4 vp3D = proj3D * view3D;
        const Mat4 ortho2D = Mat4::ortho(0.f, static_cast<float>(vpW), static_cast<float>(vpH), 0.f, -1.f, 1.f);

        // 0. Optional holographic camera feed in the background
        if (hud.showCameraBg && hasCamFrame_ && hud.link.secondsSinceVideo < 1.5) {
            drawCameraBackground(vpW, vpH, timeSec, ortho2D);
        }

        // 1. 3D Perspective Grid at Z = -2.0
        drawBackgroundGrid(vp3D, timeSec);

        // 2. 3D Scene Objects (Dual-Pass Hologram Shader)
        drawSceneObjects(objects, view3D, proj3D, timeSec);

        // 3. 2D AR Hand Skeleton Overlay + Futuristic Telemetry HUD
        hudVerts_.clear();
        drawSceneObjectAnnotations(objects, vpW, vpH, timeSec);
        drawArHandOverlays(hands, objects, hud, vpW, vpH, timeSec);
        drawTelemetryHud(hands, objects, hud, vpW, vpH, timeSec);
        flushHud(ortho2D, timeSec);
    }

private:
    struct HudVertex {
        Vec2 pos;
        Vec2 uv;
        Vec4 col;
    };
    struct GridVertex {
        Vec3 pos;
        Vec4 col;
    };

    GLuint holoProg_ = 0, hudProg_ = 0, gridProg_ = 0;
    std::array<GpuMesh, static_cast<size_t>(PrimitiveType::Count)> meshes_{};
    GLuint hudVao_ = 0, hudVbo_ = 0;
    GLuint gridVao_ = 0, gridVbo_ = 0;
    GLsizei gridVertexCount_ = 0;
    GLuint camTex_ = 0;
    int camW_ = 0, camH_ = 0, camC_ = 0;
    bool hasCamFrame_ = false;
    std::vector<HudVertex> hudVerts_;

    // --- Shader compiler ----------------------------------------------------
    static std::string readFileOrFallback(const std::string& path, const char* fallback) {
        if (!path.empty()) {
            std::ifstream f(path);
            if (f.good()) {
                std::ostringstream ss;
                ss << f.rdbuf();
                const std::string s = ss.str();
                if (!s.empty()) return s;
            }
        }
        return fallback ? std::string(fallback) : std::string();
    }

    static GLuint compileStage(GLenum type, const std::string& src) {
        GLuint sh = glCreateShader(type);
        const char* p = src.c_str();
        glShaderSource(sh, 1, &p, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024]{};
            glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            std::cerr << "[shader] compile error: " << log << "\n";
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    static GLuint buildProgram(const std::string& vertPath, const std::string& fragPath,
                               const char* vertFallback, const char* fragFallback) {
        const std::string vs = readFileOrFallback(vertPath, vertFallback);
        const std::string fs = readFileOrFallback(fragPath, fragFallback);
        GLuint v = compileStage(GL_VERTEX_SHADER, vs);
        GLuint f = compileStage(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) return 0;
        GLuint prog = glCreateProgram();
        glAttachShader(prog, v);
        glAttachShader(prog, f);
        glLinkProgram(prog);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024]{};
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            std::cerr << "[shader] link error: " << log << "\n";
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }

    // --- 3D Background Grid at Z = -2.0 + Floor Horizon Plane ---------------
    void buildBackgroundGrid() {
        std::vector<GridVertex> v;
        auto pushLine = [&](Vec3 a, Vec3 b, Vec4 c) {
            v.push_back({a, c});
            v.push_back({b, c});
        };

        // Main vertical plane grid at Z = -2.0 (per spec)
        const int halfN = 16;
        const float step = 0.55f;
        const float extent = halfN * step;
        for (int i = -halfN; i <= halfN; ++i) {
            const float d = i * step;
            const bool major = (i % 4 == 0);
            const Vec4 col = major ? Vec4{0.05f, 0.55f, 0.78f, 0.34f}
                                   : Vec4{0.02f, 0.32f, 0.52f, 0.16f};
            pushLine({-extent, d, kGridZ}, {extent, d, kGridZ}, col);
            pushLine({d, -extent, kGridZ}, {d, extent, kGridZ}, col);
        }

        // Receding perspective floor & ceiling rails from Z = +1.5 to Z = -2.0
        const float floorY = -2.35f;
        const float ceilY = 2.35f;
        for (int i = -12; i <= 12; ++i) {
            const float x = i * 0.65f;
            const Vec4 c = (i % 3 == 0) ? Vec4{0.06f, 0.62f, 0.88f, 0.28f}
                                        : Vec4{0.02f, 0.35f, 0.55f, 0.13f};
            pushLine({x, floorY, kGridZ}, {x, floorY, 1.5f}, c);
            pushLine({x, ceilY, kGridZ}, {x, ceilY, 1.5f}, c * 0.6f);
        }
        for (int iz = 0; iz <= 10; ++iz) {
            const float z = lerp(kGridZ, 1.5f, iz / 10.0f);
            const Vec4 c{0.04f, 0.48f, 0.72f, 0.20f};
            pushLine({-7.8f, floorY, z}, {7.8f, floorY, z}, c);
            pushLine({-7.8f, ceilY, z}, {7.8f, ceilY, z}, c * 0.5f);
        }

        gridVertexCount_ = static_cast<GLsizei>(v.size());
        glGenVertexArrays(1, &gridVao_);
        glGenBuffers(1, &gridVbo_);
        glBindVertexArray(gridVao_);
        glBindBuffer(GL_ARRAY_BUFFER, gridVbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(v.size() * sizeof(GridVertex)),
                     v.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GridVertex),
                              reinterpret_cast<const void*>(offsetof(GridVertex, pos)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(GridVertex),
                              reinterpret_cast<const void*>(offsetof(GridVertex, col)));
        glBindVertexArray(0);
    }

    void buildCameraTexture() {
        glGenTextures(1, &camTex_);
        glBindTexture(GL_TEXTURE_2D, camTex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void drawCameraBackground(int vpW, int vpH, float timeSec, const Mat4& ortho2D) {
        std::vector<HudVertex> bg;
        const Vec4 tint{0.92f, 0.97f, 1.00f, 0.62f};
        const float w = static_cast<float>(vpW), h = static_cast<float>(vpH);
        bg.push_back({{0, 0}, {0, 0}, tint});
        bg.push_back({{w, 0}, {1, 0}, tint});
        bg.push_back({{w, h}, {1, 1}, tint});
        bg.push_back({{0, 0}, {0, 0}, tint});
        bg.push_back({{w, h}, {1, 1}, tint});
        bg.push_back({{0, h}, {0, 1}, tint});

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(hudProg_);
        glUniformMatrix4fv(glGetUniformLocation(hudProg_, "uProj"), 1, GL_FALSE, ortho2D.data());
        glUniform1i(glGetUniformLocation(hudProg_, "uMode"), 2);
        glUniform1f(glGetUniformLocation(hudProg_, "uTime"), timeSec);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, camTex_);
        glUniform1i(glGetUniformLocation(hudProg_, "uTex"), 0);

        glBindVertexArray(hudVao_);
        glBindBuffer(GL_ARRAY_BUFFER, hudVbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(bg.size() * sizeof(HudVertex)),
                        bg.data());
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(bg.size()));
        glBindVertexArray(0);
    }

    void drawBackgroundGrid(const Mat4& vp3D, float timeSec) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glUseProgram(gridProg_);
        glUniformMatrix4fv(glGetUniformLocation(gridProg_, "uVP"), 1, GL_FALSE, vp3D.data());
        glUniform1f(glGetUniformLocation(gridProg_, "uTime"), timeSec);

        glBindVertexArray(gridVao_);
        glLineWidth(1.0f);
        glDrawArrays(GL_LINES, 0, gridVertexCount_);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);
    }

    // --- Dual-Pass 3D Hologram Rendering ------------------------------------
    void drawSceneObjects(const std::vector<SceneObject>& objects, const Mat4& view,
                          const Mat4& proj, float timeSec) {
        if (objects.empty()) return;

        glUseProgram(holoProg_);
        glUniformMatrix4fv(glGetUniformLocation(holoProg_, "uView"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(glGetUniformLocation(holoProg_, "uProj"), 1, GL_FALSE, proj.data());
        glUniform3f(glGetUniformLocation(holoProg_, "uCamPos"), 0.f, 0.f, kCameraZ);
        glUniform1f(glGetUniformLocation(holoProg_, "uTime"), timeSec);

        const GLint locModel = glGetUniformLocation(holoProg_, "uModel");
        const GLint locColor = glGetUniformLocation(holoProg_, "uColor");
        const GLint locPass = glGetUniformLocation(holoProg_, "uPass");
        const GLint locSel = glGetUniformLocation(holoProg_, "uSelected");
        const GLint locHov = glGetUniformLocation(holoProg_, "uHovered");
        const GLint locSpw = glGetUniformLocation(holoProg_, "uSpawn");

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);

        // PASS 0: Translucent Fresnel-glow core (alpha + additive hybrid)
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUniform1i(locPass, 0);
        for (const SceneObject& obj : objects) {
            const GpuMesh& mesh = meshes_[static_cast<int>(obj.type)];
            const Mat4 model = obj.modelMatrix();
            glUniformMatrix4fv(locModel, 1, GL_FALSE, model.data());
            glUniform3f(locColor, obj.color.x, obj.color.y, obj.color.z);
            glUniform1f(locSel, obj.selected ? 1.f : 0.f);
            glUniform1f(locHov, obj.hovered ? 1.f : 0.f);
            glUniform1f(locSpw, obj.spawnAnim);
            mesh.drawTriangles();
        }

        // PASS 1: Bright neon additive wireframe overlay
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glUniform1i(locPass, 1);
        glLineWidth(1.5f);
        for (const SceneObject& obj : objects) {
            const GpuMesh& mesh = meshes_[static_cast<int>(obj.type)];
            const Mat4 model = obj.modelMatrix();
            glUniformMatrix4fv(locModel, 1, GL_FALSE, model.data());
            glUniform3f(locColor, obj.color.x, obj.color.y, obj.color.z);
            glUniform1f(locSel, obj.selected ? 1.f : 0.f);
            glUniform1f(locHov, obj.hovered ? 1.f : 0.f);
            glUniform1f(locSpw, obj.spawnAnim);
            mesh.drawLines();
        }

        glLineWidth(1.0f);
        glDepthMask(GL_TRUE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    // --- 2D Primitives Batch Builder ----------------------------------------
    void pushTri(Vec2 a, Vec2 b, Vec2 c, Vec4 col) {
        hudVerts_.push_back({a, {0, 0}, col});
        hudVerts_.push_back({b, {0, 0}, col});
        hudVerts_.push_back({c, {0, 0}, col});
    }

    void pushQuad(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 c0, Vec4 c1, Vec4 c2, Vec4 c3) {
        hudVerts_.push_back({p0, {0, 0}, c0});
        hudVerts_.push_back({p1, {0, 0}, c1});
        hudVerts_.push_back({p2, {0, 0}, c2});
        hudVerts_.push_back({p0, {0, 0}, c0});
        hudVerts_.push_back({p2, {0, 0}, c2});
        hudVerts_.push_back({p3, {0, 0}, c3});
    }

    void drawRect(float x, float y, float w, float h, Vec4 col) {
        pushQuad({x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}, col, col, col, col);
    }

    void drawLine(Vec2 a, Vec2 b, float thickness, Vec4 col, float glowRadius = 0.f) {
        const Vec2 dir = b - a;
        const float len = length(dir);
        if (len < 0.01f) return;
        const Vec2 n = Vec2{-dir.y / len, dir.x / len};
        if (glowRadius > 0.f) {
            const Vec2 g = n * (thickness * 0.5f + glowRadius);
            const Vec4 gc{col.x, col.y, col.z, col.w * 0.22f};
            pushQuad(a - g, b - g, b + g, a + g, gc, gc, gc, gc);
        }
        const Vec2 o = n * (thickness * 0.5f);
        pushQuad(a - o, b - o, b + o, a + o, col, col, col, col);
    }

    void drawRectStroke(float x, float y, float w, float h, float t, Vec4 col) {
        drawLine({x, y}, {x + w, y}, t, col);
        drawLine({x + w, y}, {x + w, y + h}, t, col);
        drawLine({x + w, y + h}, {x, y + h}, t, col);
        drawLine({x, y + h}, {x, y}, t, col);
    }

    void drawCornerBrackets(float x, float y, float w, float h, float len, float t, Vec4 col) {
        // Top-left
        drawLine({x, y}, {x + len, y}, t, col);
        drawLine({x, y}, {x, y + len}, t, col);
        // Top-right
        drawLine({x + w - len, y}, {x + w, y}, t, col);
        drawLine({x + w, y}, {x + w, y + len}, t, col);
        // Bottom-left
        drawLine({x, y + h - len}, {x, y + h}, t, col);
        drawLine({x, y + h}, {x + len, y + h}, t, col);
        // Bottom-right
        drawLine({x + w - len, y + h}, {x + w, y + h}, t, col);
        drawLine({x + w, y + h - len}, {x + w, y + h}, t, col);
    }

    void drawCircleFill(Vec2 center, float radius, Vec4 centerCol, Vec4 edgeCol, int segs = 24) {
        for (int i = 0; i < segs; ++i) {
            const float a0 = (static_cast<float>(i) / segs) * kTau;
            const float a1 = (static_cast<float>(i + 1) / segs) * kTau;
            const Vec2 p0{center.x + std::cos(a0) * radius, center.y + std::sin(a0) * radius};
            const Vec2 p1{center.x + std::cos(a1) * radius, center.y + std::sin(a1) * radius};
            hudVerts_.push_back({center, {0, 0}, centerCol});
            hudVerts_.push_back({p0, {0, 0}, edgeCol});
            hudVerts_.push_back({p1, {0, 0}, edgeCol});
        }
    }

    void drawRingArc(Vec2 center, float rInner, float rOuter, float aStart, float aEnd,
                     Vec4 col, int segs = 28) {
        for (int i = 0; i < segs; ++i) {
            const float t0 = static_cast<float>(i) / segs;
            const float t1 = static_cast<float>(i + 1) / segs;
            const float a0 = lerp(aStart, aEnd, t0);
            const float a1 = lerp(aStart, aEnd, t1);
            const Vec2 c0{std::cos(a0), std::sin(a0)};
            const Vec2 c1{std::cos(a1), std::sin(a1)};
            pushQuad(center + c0 * rInner, center + c0 * rOuter, center + c1 * rOuter,
                     center + c1 * rInner, col, col, col, col);
        }
    }

    // Dynamic HSV color-wheel ring + inner mixed-RGB indicator around the thumb
    void drawThumbColorWheel(Vec2 center, float radius, Vec3 mixedRgb, uint8_t contactMask,
                             float timeSec, float alpha) {
        const int segs = 36;
        const float rIn = radius * 0.76f;
        const float rOut = radius;
        for (int i = 0; i < segs; ++i) {
            const float t0 = static_cast<float>(i) / segs;
            const float t1 = static_cast<float>(i + 1) / segs;
            const float a0 = t0 * kTau + timeSec * 0.9f;
            const float a1 = t1 * kTau + timeSec * 0.9f;
            const Vec3 rgb0 = hsv2rgb(t0, 0.95f, 1.0f);
            const Vec3 rgb1 = hsv2rgb(t1, 0.95f, 1.0f);
            const Vec4 c0{rgb0.x, rgb0.y, rgb0.z, 0.72f * alpha};
            const Vec4 c1{rgb1.x, rgb1.y, rgb1.z, 0.72f * alpha};
            const Vec2 d0{std::cos(a0), std::sin(a0)};
            const Vec2 d1{std::cos(a1), std::sin(a1)};
            pushQuad(center + d0 * rIn, center + d0 * rOut, center + d1 * rOut, center + d1 * rIn,
                     c0, c0, c1, c1);
        }

        // Inner ring showing the currently mixed additive RGB color
        const Vec3 ringRgb = (contactMask != 0) ? mixedRgb : Vec3{0.0f, 0.85f, 1.0f};
        const float pulse = (contactMask != 0) ? (1.0f + 0.14f * std::sin(timeSec * 10.f)) : 1.0f;
        drawRingArc(center, radius * 0.42f * pulse, radius * 0.64f * pulse, 0.f, kTau,
                    {ringRgb.x, ringRgb.y, ringRgb.z, (contactMask ? 0.92f : 0.42f) * alpha}, 28);
        if (contactMask != 0) {
            drawCircleFill(center, radius * 1.45f,
                           {ringRgb.x, ringRgb.y, ringRgb.z, 0.25f * alpha},
                           {ringRgb.x, ringRgb.y, ringRgb.z, 0.0f}, 24);
        }
    }

    // Sci-Fi targeting reticle on active fingertips
    void drawTargetReticle(Vec2 center, float radius, Vec4 col, float timeSec, bool active) {
        drawCircleFill(center, active ? 4.2f : 2.8f, {1.f, 1.f, 1.f, col.w}, col, 12);
        const int arcs = 3;
        for (int i = 0; i < arcs; ++i) {
            const float base = timeSec * (active ? 2.6f : 1.1f) + i * (kTau / arcs);
            drawRingArc(center, radius - 1.8f, radius + 0.8f, base, base + 1.15f, col, 10);
        }
        if (active) {
            const float r2 = radius + 6.0f;
            drawLine({center.x - r2, center.y}, {center.x - radius + 2.f, center.y}, 1.4f, col);
            drawLine({center.x + radius - 2.f, center.y}, {center.x + r2, center.y}, 1.4f, col);
            drawLine({center.x, center.y - r2}, {center.x, center.y - radius + 2.f}, 1.4f, col);
            drawLine({center.x, center.y + radius - 2.f}, {center.x, center.y + r2}, 1.4f, col);
        }
    }

    void drawText(float x, float y, float pixelScale, const std::string& text, Vec4 col) {
        float cx = x;
        for (char ch : text) {
            if (ch == '\n') {
                y += 9.f * pixelScale;
                cx = x;
                continue;
            }
            const uint8_t* rows = glyph5x7(ch);
            for (int colIdx = 0; colIdx < 5; ++colIdx) {
                const uint8_t mask = rows[colIdx];
                if (!mask) continue;
                for (int rowIdx = 0; rowIdx < 7; ++rowIdx) {
                    if (mask & (1u << rowIdx)) {
                        drawRect(cx + colIdx * pixelScale, y + rowIdx * pixelScale,
                                 pixelScale * 0.95f, pixelScale * 0.95f, col);
                    }
                }
            }
            cx += 6.0f * pixelScale;
        }
    }

    // --- Object Selection Brackets & Labels in 2D Screen Space --------------
    void drawSceneObjectAnnotations(const std::vector<SceneObject>& objects, int vpW, int vpH,
                                    float timeSec) {
        for (const SceneObject& obj : objects) {
            if (!obj.selected && !obj.hovered && !obj.grabbed) continue;
            const Vec2 sc = worldToScreen(obj.position, vpW, vpH);
            const Vec2 edge = worldToScreen(obj.position + Vec3{obj.scale * 0.72f, 0.f, 0.f}, vpW, vpH);
            const float r = std::max(28.f, std::abs(edge.x - sc.x));

            const Vec4 col = obj.grabbed
                                 ? Vec4{1.0f, 0.85f, 0.20f, 0.92f}
                                 : (obj.selected ? Vec4{0.15f, 0.95f, 1.0f, 0.85f}
                                                 : Vec4{0.15f, 0.85f, 1.0f, 0.50f});
            drawCornerBrackets(sc.x - r, sc.y - r, r * 2.f, r * 2.f, 12.f, 2.0f, col);

            if (obj.selected || obj.grabbed) {
                drawRingArc(sc, r + 6.f, r + 8.f, timeSec * 1.4f, timeSec * 1.4f + 1.8f, col, 16);
                drawRingArc(sc, r + 6.f, r + 8.f, timeSec * 1.4f + kPi,
                            timeSec * 1.4f + kPi + 1.8f, col, 16);
                char buf[96];
                std::snprintf(buf, sizeof(buf), "%s #%u  S:%.2f", primitiveName(obj.type), obj.id,
                              obj.scale);
                drawText(sc.x - r, sc.y - r - 14.f, 1.55f, buf, col);
            }

            // Right-Hand Fist Crush Deletion Progress Indicator (shrinking red ring + charge bar)
            if (obj.crushProgress > 0.01f) {
                const float p = clampf(obj.crushProgress, 0.f, 1.f);
                const float shrinkR = lerp(r * 1.35f, r * 0.28f, p);
                const Vec4 danger{1.0f, 0.15f, 0.20f, 0.95f};
                drawRingArc(sc, shrinkR - 3.5f, shrinkR + 3.5f, 0.f, kTau * p, danger, 36);
                drawRingArc(sc, r + 12.f, r + 15.f, 0.f, kTau * p, {1.0f, 0.35f, 0.15f, 0.85f}, 32);

                // Charging bar directly below the object
                const float bw = r * 2.0f;
                const float bx = sc.x - r;
                const float by = sc.y + r + 10.f;
                drawRect(bx, by, bw, 10.f, {0.18f, 0.02f, 0.04f, 0.85f});
                drawRect(bx, by, bw * p, 10.f, danger);
                drawRectStroke(bx, by, bw, 10.f, 1.4f, {1.0f, 0.55f, 0.55f, 0.95f});
                char cbuf[64];
                std::snprintf(cbuf, sizeof(cbuf), "CRUSH DELETE %3.0f%%", p * 100.f);
                drawText(bx, by + 14.f, 1.45f, cbuf, danger);
            }
        }
    }

    // --- AR Hand Skeleton Overlay -------------------------------------------
    void drawArHandOverlays(const std::array<SmoothedHand, 2>& hands,
                            const std::vector<SceneObject>& /*objects*/, const HudTelemetry& hud,
                            int vpW, int vpH, float timeSec) {
        // Spawn burst shockwave ring
        if (hud.spawnFlash > 0.01f) {
            const Vec2 sp = worldToScreen({hud.spawnFlashWorld.x, hud.spawnFlashWorld.y, 0.f}, vpW, vpH);
            const float t = 1.0f - hud.spawnFlash;
            const float rad = 18.f + t * 140.f;
            drawRingArc(sp, rad - 3.f, rad + 3.f, 0.f, kTau,
                        {0.25f, 0.98f, 1.0f, hud.spawnFlash * 0.85f}, 40);
        }

        // Two-Handed Rotation Link Beam (Right Hand Grabbing + Left Hand Thumb-Down ROTATION_MODE)
        if (hud.rotationModeActive && hands[0].visibility > 0.2f && hands[1].visibility > 0.2f) {
            const Vec2 pL{hands[0].lmNorm[9].x * vpW, hands[0].lmNorm[9].y * vpH};
            const Vec2 pR{hands[1].lmNorm[8].x * vpW, hands[1].lmNorm[8].y * vpH};
            drawLine(pL, pR, 2.2f, {0.95f, 0.45f, 1.0f, 0.88f}, 5.5f);
            drawRingArc(pL, 28.f, 32.f, timeSec * 3.0f, timeSec * 3.0f + 4.2f,
                        {0.95f, 0.45f, 1.0f, 0.92f}, 24);
            const Vec2 mid = (pL + pR) * 0.5f;
            drawRect(mid.x - 84.f, mid.y - 22.f, 168.f, 18.f, {0.08f, 0.02f, 0.14f, 0.85f});
            drawText(mid.x - 78.f, mid.y - 18.f, 1.45f, "ROTATION MODE [DELTA]",
                     {0.98f, 0.60f, 1.0f, 1.0f});
        }

        // Two-hand dynamic scaling beam between Left Index Tip (8) and Right Index Tip (8)
        if (hud.twoHandScaling && hands[0].visibility > 0.2f && hands[1].visibility > 0.2f) {
            const Vec2 pL{hands[0].lmNorm[8].x * vpW, hands[0].lmNorm[8].y * vpH};
            const Vec2 pR{hands[1].lmNorm[8].x * vpW, hands[1].lmNorm[8].y * vpH};
            drawLine(pL, pR, 2.4f, {1.0f, 0.88f, 0.25f, 0.90f}, 6.0f);
            const Vec2 mid = (pL + pR) * 0.5f;
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "SCALE SPAN %.2f", hud.twoHandDistance);
            drawRect(mid.x - 56.f, mid.y - 22.f, 112.f, 18.f, {0.02f, 0.08f, 0.14f, 0.82f});
            drawText(mid.x - 50.f, mid.y - 18.f, 1.5f, lbl, {1.0f, 0.92f, 0.30f, 1.0f});
        }

        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            const SmoothedHand& h = hands[hIdx];
            if (h.visibility < 0.02f) continue;
            const float a = h.visibility;

            std::array<Vec2, 21> pts{};
            for (int i = 0; i < 21; ++i) {
                pts[i] = {h.lmNorm[i].x * vpW, h.lmNorm[i].y * vpH};
            }

            const Vec4 boneCol = (hIdx == 0) ? Vec4{0.05f, 0.92f, 1.0f, 0.88f * a}
                                             : Vec4{0.20f, 1.0f, 0.78f, 0.88f * a};

            // Bone connections with outer neon glow
            for (const auto& [i0, i1] : kHandBones) {
                drawLine(pts[i0], pts[i1], 2.2f, boneCol, 4.8f);
            }

            // Joints
            for (int i = 0; i < 21; ++i) {
                const bool isTip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
                if (!isTip) {
                    drawCircleFill(pts[i], 3.8f, {1.f, 1.f, 1.f, 0.95f * a},
                                   {boneCol.x, boneCol.y, boneCol.z, 0.35f * a}, 10);
                }
            }

            // Asymmetric Hand Reticles:
            //   * Left Hand (hIdx == 0):
            //       - Index(8)  = Spawn / Cycle / 2-Hand Scale Reticle (Cyan/Gold)
            //       - Middle(12)= RED   (1, 0, 0) [bit 0: kMiddleRed]
            //       - Ring(16)  = GREEN (0, 1, 0) [bit 1: kRingGreen]
            //       - Pinky(20) = BLUE  (0, 0, 1) [bit 2: kPinkyBlue]
            //       - Thumb(4)  = Dynamic HSV Color-Wheel + Mixed RGB Core
            //   * Right Hand (hIdx == 1):
            //       - Thumb(4) + Index(8) = Spatial Grab / Translate Reticles (NO color mixing)
            if (hIdx == 0) {
                const struct LeftTipSpec {
                    int idx;
                    uint8_t bit;
                    Vec4 col;
                    const char* tag;
                } leftTips[4] = {
                    {8,  0x00,              {0.25f, 0.96f, 1.00f, 0.92f * a}, "SPWN"},
                    {12, proto::kMiddleRed, {1.00f, 0.22f, 0.28f, 0.95f * a}, "R"},
                    {16, proto::kRingGreen, {0.20f, 1.00f, 0.38f, 0.95f * a}, "G"},
                    {20, proto::kPinkyBlue, {0.25f, 0.55f, 1.00f, 0.95f * a}, "B"},
                };
                for (const auto& tc : leftTips) {
                    const bool touching = (tc.bit != 0) && ((h.contactMask & tc.bit) != 0);
                    const bool pinchActive = (tc.idx == 8 && h.pinch);
                    const float rad = (touching || pinchActive) ? 16.0f : 11.5f;
                    drawTargetReticle(pts[tc.idx], rad, tc.col, timeSec, touching || pinchActive);
                    if (tc.tag[0] != '\0') {
                        drawText(pts[tc.idx].x + rad + 4.f, pts[tc.idx].y - 5.f, 1.4f, tc.tag, tc.col);
                    }
                    if (touching || pinchActive) {
                        drawLine(pts[4], pts[tc.idx], 2.5f, tc.col, 5.5f);
                    }
                }
                // Dynamic HSV Color-Wheel Ring + Mixed RGB Core ONLY on Left Thumb Tip (4)
                drawThumbColorWheel(pts[4], 22.0f, h.color, h.contactMask, timeSec, a);
                if (h.rotationMode) {
                    const Vec2 palm = (pts[0] + pts[5] + pts[9] + pts[13] + pts[17]) * 0.2f;
                    drawRingArc(palm, 22.f, 26.f, timeSec * 2.5f, timeSec * 2.5f + 4.5f,
                                {0.95f, 0.45f, 1.0f, 0.92f * a}, 24);
                    drawText(palm.x - 54.f, palm.y - 34.f, 1.45f, "ROTATION MODE [4F]",
                             {0.98f, 0.55f, 1.0f, 1.0f * a});
                }
                drawText(pts[0].x - 78.f, pts[0].y + 18.f, 1.45f,
                         h.rotationMode ? "L-HAND [ROTATION MODE 4F]"
                                        : "L-HAND [SPAWN / ROT / RGB]",
                         boneCol);
            } else {
                const Vec4 grabCol = h.fist
                                         ? Vec4{1.0f, 0.20f, 0.25f, 0.96f * a}
                                         : (h.pinch ? Vec4{1.0f, 0.88f, 0.25f, 0.96f * a}
                                                    : Vec4{0.20f, 1.00f, 0.78f, 0.90f * a});
                drawTargetReticle(pts[4], h.pinch ? 15.f : 11.f, grabCol, timeSec,
                                  h.pinch || h.thumbFolded);
                drawTargetReticle(pts[8], h.pinch ? 16.f : 12.f, grabCol, timeSec,
                                  h.pinch || h.thumbFolded);
                const char* rTag = h.fist ? "CRUSH!"
                                          : (h.pinch ? "GRAB"
                                                     : (h.thumbFolded ? "BLUEPRINT" : "MOVE"));
                drawText(pts[8].x + 16.f, pts[8].y - 5.f, 1.4f, rTag, grabCol);
                if (h.pinch) {
                    drawLine(pts[4], pts[8], 2.6f, grabCol, 5.5f);
                }
                drawText(pts[0].x - 78.f, pts[0].y + 18.f, 1.45f,
                         h.thumbFolded ? "R-HAND [BLUEPRINT MODE]"
                                       : "R-HAND [GRAB / BLUEPRINT / CRUSH]",
                         boneCol);
            }
        }
    }

    // --- Futuristic Telemetry HUD -------------------------------------------
    static const char* gestureName(uint8_t g) {
        switch (static_cast<proto::Gesture>(g)) {
            case proto::Gesture::Open: return "OPEN PALM";
            case proto::Gesture::Fist: return "FIST [CRUSH/ARM]";
            case proto::Gesture::Pinch: return "PINCH [HOLD]";
            case proto::Gesture::Color: return "CHROMA MIX";
            case proto::Gesture::Count: return "BLUEPRINT SELECT";
            case proto::Gesture::Rotate: return "ROTATION MODE [4F]";
            default: return "IDLE";
        }
    }

    static const char* colorMixName(uint8_t mask) {
        switch (mask & 0x07) {
            case 0b001: return "RED [L-MID]";
            case 0b010: return "GREEN [L-RING]";
            case 0b100: return "BLUE [L-PINKY]";
            case 0b011: return "YELLOW [M+R]";
            case 0b110: return "CYAN [R+P]";
            case 0b101: return "MAGENTA [M+P]";
            case 0b111: return "WHITE [M+R+P]";
            default: return "HOLD COLOUR";
        }
    }

    void drawPanel(float x, float y, float w, float h, const char* title) {
        drawRect(x, y, w, h, {0.015f, 0.045f, 0.085f, 0.72f});
        drawRect(x, y, w, 22.f, {0.03f, 0.16f, 0.26f, 0.85f});
        drawRectStroke(x, y, w, h, 1.2f, {0.06f, 0.55f, 0.78f, 0.45f});
        drawCornerBrackets(x, y, w, h, 10.f, 2.0f, {0.15f, 0.92f, 1.0f, 0.95f});
        drawText(x + 10.f, y + 6.f, 1.55f, title, {0.55f, 0.96f, 1.0f, 1.0f});
    }

    void drawTelemetryHud(const std::array<SmoothedHand, 2>& hands,
                          const std::vector<SceneObject>& objects, const HudTelemetry& hud,
                          int vpW, int vpH, float /*timeSec*/) {
        const float W = static_cast<float>(vpW);
        const float H = static_cast<float>(vpH);
        const Vec4 cyan{0.15f, 0.92f, 1.0f, 0.95f};
        const Vec4 dimCyan{0.35f, 0.78f, 0.92f, 0.80f};
        const Vec4 white{0.92f, 0.98f, 1.0f, 0.95f};

        // Outer viewport corner brackets
        drawCornerBrackets(14.f, 14.f, W - 28.f, H - 28.f, 28.f, 2.2f,
                           {0.10f, 0.88f, 1.0f, 0.80f});

        // 1. TOP-LEFT: System & UDP Bridge Telemetry
        drawPanel(26.f, 26.f, 330.f, 142.f, "SYS // TELEMETRY & IPC BRIDGE");
        char line[128];
        std::snprintf(line, sizeof(line), "ENGINE FPS  : %5.1f FPS", hud.renderFps);
        drawText(38.f, 56.f, 1.55f, line, white);
        std::snprintf(line, sizeof(line), "VISION FPS  : %5.1f FPS  [%s]", hud.trackerFps,
                      hud.link.connected ? "UDP ONLINE" : (hud.demoMode ? "SYNTH DEMO" : "WAITING"));
        drawText(38.f, 74.f, 1.55f, line,
                 hud.link.connected ? Vec4{0.25f, 1.0f, 0.55f, 1.f}
                                    : Vec4{1.0f, 0.78f, 0.25f, 1.f});
        std::snprintf(line, sizeof(line), "UDP LATENCY : %5.2f MS  (E2E %4.1f MS)",
                      hud.link.udpLatencyMs, hud.link.pipelineLatencyMs);
        drawText(38.f, 92.f, 1.55f, line, dimCyan);
        std::snprintf(line, sizeof(line), "PKT STREAM  : %5.1f PKT/S  LOSS %.1f%%",
                      hud.link.packetsPerSec, hud.link.lossPct);
        drawText(38.f, 110.f, 1.55f, line, dimCyan);
        std::snprintf(line, sizeof(line), "SCENE GRAPH : %zu OBJECTS  (Z = 0.00)",
                      objects.size());
        drawText(38.f, 128.f, 1.55f, line, cyan);
        drawText(38.f, 146.f, 1.35f, hud.statusBanner, {0.80f, 0.95f, 1.0f, 0.90f});

        // 2. TOP-RIGHT: Asymmetric Hand Roles & Gesture State Readouts
        const float rx = W - 366.f;
        drawPanel(rx, 26.f, 340.f, 142.f, "AR // MULTI-HAND GESTURE TELEMETRY");
        for (int s = 0; s < 2; ++s) {
            const SmoothedHand& h = hands[s];
            const float y0 = 56.f + s * 52.f;
            std::snprintf(line, sizeof(line), "%s : %s",
                          s == 0 ? "LEFT  [SPAWN/ROT]" : "RIGHT [GRAB/BP]",
                          h.present ? gestureName(h.gesture) : "NOT DETECTED");
            drawText(rx + 12.f, y0, 1.55f, line,
                     h.present ? Vec4{0.25f, 0.98f, 1.0f, 1.f} : Vec4{0.45f, 0.58f, 0.68f, 0.8f});
            if (s == 0) {
                std::snprintf(line, sizeof(line), "ROT-MODE:%s  MIX:%s",
                              h.rotationMode ? "ON(4F+TFOLD)" : "OFF",
                              h.rotationMode ? "LOCKED" : colorMixName(h.contactMask));
            } else {
                std::snprintf(line, sizeof(line), "THUMB-FOLD:%s  FINGERS:%u  CRUSH:%2.0f%%",
                              h.thumbFolded ? "YES" : "NO", h.fingerCount,
                              hud.crushProgress * 100.f);
            }
            drawText(rx + 12.f, y0 + 18.f, 1.35f, line, dimCyan);
        }

        // 3. BOTTOM-LEFT: Shape Blueprint Selector (Right Thumb Folded + 1..4 Fingers)
        const float by = H - 122.f;
        drawPanel(26.f, by, 460.f, 96.f,
                  hud.rightBlueprintMode
                      ? "BLUEPRINT [ARMED] // L-FIST->PALM TO SPAWN"
                      : "BLUEPRINT // FOLD R-THUMB + 1-4 R-FINGERS");
        const char* labels[4] = {"1:CUBE", "2:CAPSULE", "3:CYLINDER", "4:TORUS"};
        for (int i = 0; i < 4; ++i) {
            const float bx = 38.f + i * 110.f;
            const bool active = (static_cast<int>(hud.activeBlueprint) == i);
            drawRect(bx, by + 32.f, 102.f, 50.f,
                     active ? Vec4{0.06f, 0.42f, 0.62f, 0.85f} : Vec4{0.02f, 0.08f, 0.14f, 0.65f});
            drawRectStroke(bx, by + 32.f, 102.f, 50.f, active ? 2.0f : 1.0f,
                           active ? cyan : Vec4{0.08f, 0.40f, 0.55f, 0.55f});
            drawText(bx + 10.f, by + 52.f, 1.45f, labels[i], active ? white : dimCyan);
        }

        // 4. BOTTOM-RIGHT: Live Left-Hand Additive RGB Channel Mixer (Middle/Ring/Pinky)
        drawPanel(rx, by, 340.f, 96.f, "L-HAND CHROMA // MID:R  RING:G  PNK:B");
        const struct BarSpec {
            const char* name;
            float val;
            bool touched;
            Vec4 col;
        } bars[3] = {
            {"R [L-MID 12]", hud.activeColor.x, (hud.activeContactMask & proto::kMiddleRed) != 0,
             {1.0f, 0.22f, 0.28f, 0.95f}},
            {"G [L-RNG 16]", hud.activeColor.y, (hud.activeContactMask & proto::kRingGreen) != 0,
             {0.20f, 1.00f, 0.38f, 0.95f}},
            {"B [L-PNK 20]", hud.activeColor.z, (hud.activeContactMask & proto::kPinkyBlue) != 0,
             {0.25f, 0.55f, 1.00f, 0.95f}},
        };
        for (int i = 0; i < 3; ++i) {
            const float ry = by + 32.f + i * 19.f;
            drawText(rx + 12.f, ry + 2.f, 1.32f, bars[i].name, bars[i].col);
            const float barX = rx + 114.f, barW = 142.f;
            drawRect(barX, ry, barW, 12.f, {0.04f, 0.10f, 0.18f, 0.85f});
            drawRect(barX, ry, barW * clampf(bars[i].val, 0.f, 1.f), 12.f, bars[i].col);
            drawRectStroke(barX, ry, barW, 12.f, bars[i].touched ? 1.8f : 1.0f,
                           bars[i].touched ? white : Vec4{0.15f, 0.60f, 0.80f, 0.60f});
        }
        // Combined RGB swatch
        drawRect(rx + 268.f, by + 32.f, 58.f, 50.f,
                 {hud.activeColor.x, hud.activeColor.y, hud.activeColor.z, 0.95f});
        drawRectStroke(rx + 268.f, by + 32.f, 58.f, 50.f, 1.8f, white);

        // Controls footer hint
        drawText(W * 0.5f - 235.f, H - 22.f, 1.35f,
                 "[1-4] SHAPE  [SPACE] SPAWN  [D] DEMO HANDS  [C] CLEAR  [V] CAM BG  [ESC] QUIT",
                 {0.40f, 0.78f, 0.92f, 0.75f});
    }

    void flushHud(const Mat4& ortho2D, float timeSec) {
        if (hudVerts_.empty()) return;
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(hudProg_);
        glUniformMatrix4fv(glGetUniformLocation(hudProg_, "uProj"), 1, GL_FALSE, ortho2D.data());
        glUniform1i(glGetUniformLocation(hudProg_, "uMode"), 0);
        glUniform1f(glGetUniformLocation(hudProg_, "uTime"), timeSec);

        glBindVertexArray(hudVao_);
        glBindBuffer(GL_ARRAY_BUFFER, hudVbo_);
        const GLsizeiptr bytes = static_cast<GLsizeiptr>(hudVerts_.size() * sizeof(HudVertex));
        glBufferData(GL_ARRAY_BUFFER, bytes, hudVerts_.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(hudVerts_.size()));
        glBindVertexArray(0);
    }
};

}  // namespace v3d
