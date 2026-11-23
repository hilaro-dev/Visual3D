// MeshGenerator.hpp -- Procedural 3D primitive & scene-graph definitions.
//
// Generates indexed triangle + clean wireframe (GL_LINES) VAOs/VBOs/EBOs for:
//   * Cube      (subdivided faces + crisp outer/grid edges)
//   * Capsule   (cylinder body + top/bottom hemispherical caps)
//   * Cylinder  (smooth barrel + top/bottom discs with radial rings)
//   * Torus     (parametric ring with meridian & parallel wire loops)
//
// Also defines SceneObject (std::vector<SceneObject> scene graph) with position
// on the Z = 0.0 plane, uniform scale, RGB color, continuous slow-rotation
// matrix, and selection/hover states.
#pragma once

#include "GLLoader.hpp"
#include "MathUtils.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace v3d {

enum class PrimitiveType : int {
    Cube = 0,
    Capsule = 1,
    Cylinder = 2,
    Torus = 3,
    Count = 4
};

inline const char* primitiveName(PrimitiveType t) {
    switch (t) {
        case PrimitiveType::Cube: return "CUBE";
        case PrimitiveType::Capsule: return "CAPSULE";
        case PrimitiveType::Cylinder: return "CYLINDER";
        case PrimitiveType::Torus: return "TORUS / RING";
        default: return "UNKNOWN";
    }
}

struct Vertex {
    Vec3 pos;
    Vec3 normal;
    Vec2 uv;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> triIndices;   // GL_TRIANGLES
    std::vector<uint32_t> lineIndices;  // GL_LINES (clean architectural wireframe)
    float boundingRadius = 1.0f;
};

struct GpuMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint triEbo = 0;
    GLuint lineVao = 0;
    GLuint lineEbo = 0;
    GLsizei triCount = 0;
    GLsizei lineCount = 0;
    float boundingRadius = 1.0f;

    void upload(const MeshData& m) {
        destroy();
        boundingRadius = m.boundingRadius;
        triCount = static_cast<GLsizei>(m.triIndices.size());
        lineCount = static_cast<GLsizei>(m.lineIndices.size());

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m.vertices.size() * sizeof(Vertex)),
                     m.vertices.data(), GL_STATIC_DRAW);

        // Triangle VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        setupAttribs();
        glGenBuffers(1, &triEbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m.triIndices.size() * sizeof(uint32_t)),
                     m.triIndices.data(), GL_STATIC_DRAW);

        // Wireframe Line VAO (shares the same vertex buffer)
        glGenVertexArrays(1, &lineVao);
        glBindVertexArray(lineVao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        setupAttribs();
        glGenBuffers(1, &lineEbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m.lineIndices.size() * sizeof(uint32_t)),
                     m.lineIndices.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void drawTriangles() const {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, triCount, GL_UNSIGNED_INT, nullptr);
    }

    void drawLines() const {
        glBindVertexArray(lineVao);
        glDrawElements(GL_LINES, lineCount, GL_UNSIGNED_INT, nullptr);
    }

    void destroy() {
        if (vao) glDeleteVertexArrays(1, &vao);
        if (lineVao) glDeleteVertexArrays(1, &lineVao);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (triEbo) glDeleteBuffers(1, &triEbo);
        if (lineEbo) glDeleteBuffers(1, &lineEbo);
        vao = lineVao = vbo = triEbo = lineEbo = 0;
        triCount = lineCount = 0;
    }

private:
    static void setupAttribs() {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<const void*>(offsetof(Vertex, pos)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<const void*>(offsetof(Vertex, normal)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              reinterpret_cast<const void*>(offsetof(Vertex, uv)));
    }
};

// ---------------------------------------------------------------------------
// SceneObject -- element of std::vector<SceneObject> scene graph
// ---------------------------------------------------------------------------
struct SceneObject {
    uint32_t id = 0;
    PrimitiveType type = PrimitiveType::Cube;

    Vec3 position{0.f, 0.f, 0.f};       // Locked to Z = 0.0 plane
    Vec3 targetPosition{0.f, 0.f, 0.f};

    float scale = 0.55f;                // Uniform scale factor
    float targetScale = 0.55f;

    Vec3 color{0.0f, 0.88f, 1.0f};      // Current RGB color
    Vec3 targetColor{0.0f, 0.88f, 1.0f};

    Mat4 rotation = Mat4::identity();   // Static 3D orientation (rotated ONLY by Left-Hand Claw)
    float crushProgress = 0.f;          // 0.0 -> 1.0 (0.4s Right-Hand Fist Crush Delete progress)

    bool selected = false;
    bool hovered = false;
    bool grabbed = false;
    float spawnAnim = 0.f;              // 0 -> 1 holographic materialization

    void update(float dt) {
        // Smooth exponential interpolation toward targets
        const float kPos = damp(18.0f, dt);
        const float kScl = damp(14.0f, dt);
        const float kCol = damp(10.0f, dt);
        position = lerp(position, targetPosition, kPos);
        position.z = 0.0f;  // Strictly lock to Z = 0.0 plane
        scale = lerp(scale, targetScale, kScl);
        color = lerp(color, targetColor, kCol);

        // No continuous auto-spin: objects sit completely still unless rotated by Left-Hand Claw
        if (spawnAnim < 1.0f) spawnAnim = std::min(1.0f, spawnAnim + dt * 3.2f);
    }

    Mat4 modelMatrix() const {
        const float ease = smoothstep(0.f, 1.f, spawnAnim);
        // Subtle compression as the object is being crushed right before deletion
        const float crushShrink = 1.0f - 0.25f * crushProgress;
        const Vec3 p{position.x, position.y, 0.0f};
        return Mat4::translate(p) * rotation * Mat4::scale(scale * ease * crushShrink);
    }
};

// ---------------------------------------------------------------------------
// Procedural MeshGenerator
// ---------------------------------------------------------------------------
class MeshGenerator {
public:
    static void addTri(MeshData& m, uint32_t a, uint32_t b, uint32_t c) {
        m.triIndices.push_back(a);
        m.triIndices.push_back(b);
        m.triIndices.push_back(c);
    }
    static void addLine(MeshData& m, uint32_t a, uint32_t b) {
        m.lineIndices.push_back(a);
        m.lineIndices.push_back(b);
    }

    // 1. CUBE: Subdivided faces (subdiv=3) for rich hologram lattice + crisp edges
    static MeshData makeCube(float halfExtent = 0.62f, int subdiv = 3) {
        MeshData m;
        m.boundingRadius = halfExtent * 1.45f;

        struct Face {
            Vec3 normal, uAxis, vAxis;
        };
        const Face faces[6] = {
            {{ 0,  0,  1}, { 1, 0, 0}, {0, 1, 0}},  // +Z
            {{ 0,  0, -1}, {-1, 0, 0}, {0, 1, 0}},  // -Z
            {{ 1,  0,  0}, { 0, 0,-1}, {0, 1, 0}},  // +X
            {{-1,  0,  0}, { 0, 0, 1}, {0, 1, 0}},  // -X
            {{ 0,  1,  0}, { 1, 0, 0}, {0, 0,-1}},  // +Y
            {{ 0, -1,  0}, { 1, 0, 0}, {0, 0, 1}},  // -Y
        };

        const int stride = subdiv + 1;
        for (const Face& f : faces) {
            const uint32_t base = static_cast<uint32_t>(m.vertices.size());
            for (int iy = 0; iy <= subdiv; ++iy) {
                const float v = static_cast<float>(iy) / subdiv;
                const float sy = (v * 2.f - 1.f) * halfExtent;
                for (int ix = 0; ix <= subdiv; ++ix) {
                    const float u = static_cast<float>(ix) / subdiv;
                    const float sx = (u * 2.f - 1.f) * halfExtent;
                    const Vec3 pos = f.normal * halfExtent + f.uAxis * sx + f.vAxis * sy;
                    m.vertices.push_back({pos, f.normal, {u, v}});
                }
            }
            for (int iy = 0; iy < subdiv; ++iy) {
                for (int ix = 0; ix < subdiv; ++ix) {
                    const uint32_t i0 = base + iy * stride + ix;
                    const uint32_t i1 = i0 + 1;
                    const uint32_t i2 = i0 + stride;
                    const uint32_t i3 = i2 + 1;
                    addTri(m, i0, i1, i3);
                    addTri(m, i0, i3, i2);
                }
            }
            for (int iy = 0; iy <= subdiv; ++iy)
                for (int ix = 0; ix < subdiv; ++ix)
                    addLine(m, base + iy * stride + ix, base + iy * stride + ix + 1);
            for (int ix = 0; ix <= subdiv; ++ix)
                for (int iy = 0; iy < subdiv; ++iy)
                    addLine(m, base + iy * stride + ix, base + (iy + 1) * stride + ix);
        }
        return m;
    }

    // 2. CAPSULE: Cylinder body + top & bottom hemispherical caps
    static MeshData makeCapsule(float radius = 0.44f, float cylHalfH = 0.42f,
                                int slices = 28, int capRings = 8, int bodyRings = 4) {
        MeshData m;
        m.boundingRadius = cylHalfH + radius;

        // Build profile of (yCenter, ringRadius, ny, nr, vCoord)
        struct RingSpec {
            float yCenter;
            float yOffset;
            float ringR;
            float ny;
            float v;
            bool wireRing;
        };
        std::vector<RingSpec> rings;

        // Bottom hemisphere: phi from -pi/2 to 0
        for (int i = 0; i <= capRings; ++i) {
            const float t = static_cast<float>(i) / capRings;
            const float phi = -0.5f * kPi + t * (0.5f * kPi);
            rings.push_back({-cylHalfH, radius * std::sin(phi), radius * std::cos(phi),
                             std::sin(phi), 0.25f * t, (i % 2 == 0) || i == capRings});
        }
        // Cylinder interior rings
        for (int i = 1; i < bodyRings; ++i) {
            const float t = static_cast<float>(i) / bodyRings;
            const float y = lerp(-cylHalfH, cylHalfH, t);
            rings.push_back({y, 0.f, radius, 0.f, 0.25f + 0.5f * t, true});
        }
        // Top hemisphere: phi from 0 to +pi/2
        for (int i = 0; i <= capRings; ++i) {
            const float t = static_cast<float>(i) / capRings;
            const float phi = t * (0.5f * kPi);
            rings.push_back({cylHalfH, radius * std::sin(phi), radius * std::cos(phi),
                             std::sin(phi), 0.75f + 0.25f * t, (i % 2 == 0) || i == 0});
        }

        const int numRings = static_cast<int>(rings.size());
        const int stride = slices + 1;
        for (int r = 0; r < numRings; ++r) {
            const auto& rs = rings[r];
            const float nr = std::sqrt(std::max(0.f, 1.f - rs.ny * rs.ny));
            for (int s = 0; s <= slices; ++s) {
                const float u = static_cast<float>(s) / slices;
                const float theta = u * kTau;
                const float cs = std::cos(theta), sn = std::sin(theta);
                const Vec3 pos{rs.ringR * cs, rs.yCenter + rs.yOffset, rs.ringR * sn};
                const Vec3 nrm = normalize(Vec3{nr * cs, rs.ny, nr * sn});
                m.vertices.push_back({pos, nrm, {u, rs.v}});
            }
        }

        for (int r = 0; r < numRings - 1; ++r) {
            for (int s = 0; s < slices; ++s) {
                const uint32_t i0 = r * stride + s;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = i0 + stride;
                const uint32_t i3 = i2 + 1;
                addTri(m, i0, i2, i1);
                addTri(m, i1, i2, i3);
                if (rings[r].wireRing) addLine(m, i0, i1);
                if (s % 2 == 0) addLine(m, i0, i2);  // 14 clean meridian ribs
            }
        }
        return m;
    }

    // 3. CYLINDER: Smooth barrel + flat top/bottom caps with concentric rings
    static MeshData makeCylinder(float radius = 0.54f, float halfH = 0.66f,
                                 int slices = 28, int stacks = 4) {
        MeshData m;
        m.boundingRadius = std::sqrt(radius * radius + halfH * halfH);
        const int stride = slices + 1;

        // Barrel
        for (int iy = 0; iy <= stacks; ++iy) {
            const float v = static_cast<float>(iy) / stacks;
            const float y = lerp(-halfH, halfH, v);
            for (int s = 0; s <= slices; ++s) {
                const float u = static_cast<float>(s) / slices;
                const float th = u * kTau;
                const float cs = std::cos(th), sn = std::sin(th);
                m.vertices.push_back({{radius * cs, y, radius * sn}, {cs, 0.f, sn}, {u, v}});
            }
        }
        for (int iy = 0; iy < stacks; ++iy) {
            for (int s = 0; s < slices; ++s) {
                const uint32_t i0 = iy * stride + s;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = i0 + stride;
                const uint32_t i3 = i2 + 1;
                addTri(m, i0, i2, i1);
                addTri(m, i1, i2, i3);
                addLine(m, i0, i1);
                if (s % 2 == 0) addLine(m, i0, i2);
            }
        }
        for (int s = 0; s < slices; ++s) {
            const uint32_t top = stacks * stride + s;
            addLine(m, top, top + 1);
        }

        // Top and bottom caps (2 concentric rings for holographic disc pattern)
        for (int cap = 0; cap < 2; ++cap) {
            const float y = cap == 0 ? halfH : -halfH;
            const Vec3 nrm{0.f, cap == 0 ? 1.f : -1.f, 0.f};
            const uint32_t centerIdx = static_cast<uint32_t>(m.vertices.size());
            m.vertices.push_back({{0.f, y, 0.f}, nrm, {0.5f, 0.5f}});

            const uint32_t midRing = static_cast<uint32_t>(m.vertices.size());
            for (int s = 0; s <= slices; ++s) {
                const float th = (static_cast<float>(s) / slices) * kTau;
                const float r = radius * 0.52f;
                m.vertices.push_back({{r * std::cos(th), y, r * std::sin(th)}, nrm, {0.5f, 0.5f}});
            }
            const uint32_t outerRing = static_cast<uint32_t>(m.vertices.size());
            for (int s = 0; s <= slices; ++s) {
                const float th = (static_cast<float>(s) / slices) * kTau;
                m.vertices.push_back(
                    {{radius * std::cos(th), y, radius * std::sin(th)}, nrm, {0.5f, 0.5f}});
            }

            for (int s = 0; s < slices; ++s) {
                if (cap == 0) {
                    addTri(m, centerIdx, midRing + s + 1, midRing + s);
                    addTri(m, midRing + s, midRing + s + 1, outerRing + s + 1);
                    addTri(m, midRing + s, outerRing + s + 1, outerRing + s);
                } else {
                    addTri(m, centerIdx, midRing + s, midRing + s + 1);
                    addTri(m, midRing + s, outerRing + s + 1, midRing + s + 1);
                    addTri(m, midRing + s, outerRing + s, outerRing + s + 1);
                }
                addLine(m, midRing + s, midRing + s + 1);
                if (s % 4 == 0) addLine(m, centerIdx, outerRing + s);
            }
        }
        return m;
    }

    // 4. TORUS / RING: Major radius R, tube radius r
    static MeshData makeTorus(float majorR = 0.62f, float minorR = 0.23f,
                              int majorSegs = 36, int minorSegs = 18) {
        MeshData m;
        m.boundingRadius = majorR + minorR;
        const int stride = minorSegs + 1;

        for (int i = 0; i <= majorSegs; ++i) {
            const float u = static_cast<float>(i) / majorSegs;
            const float theta = u * kTau;
            const float ct = std::cos(theta), st = std::sin(theta);

            for (int j = 0; j <= minorSegs; ++j) {
                const float v = static_cast<float>(j) / minorSegs;
                const float phi = v * kTau;
                const float cp = std::cos(phi), sp = std::sin(phi);

                const float ring = majorR + minorR * cp;
                const Vec3 pos{ring * ct, ring * st, minorR * sp};
                const Vec3 nrm{cp * ct, cp * st, sp};
                m.vertices.push_back({pos, nrm, {u, v}});
            }
        }

        for (int i = 0; i < majorSegs; ++i) {
            for (int j = 0; j < minorSegs; ++j) {
                const uint32_t i0 = i * stride + j;
                const uint32_t i1 = i0 + 1;
                const uint32_t i2 = i0 + stride;
                const uint32_t i3 = i2 + 1;
                addTri(m, i0, i2, i1);
                addTri(m, i1, i2, i3);
                if (j % 2 == 0) addLine(m, i0, i2);  // Parallel rings
                if (i % 2 == 0) addLine(m, i0, i1);  // Meridian loops
            }
        }
        return m;
    }
};

}  // namespace v3d
