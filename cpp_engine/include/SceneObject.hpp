// SceneObject.hpp -- Raylib 3D Scene Graph, Grouping, Snapped Transforms & .OBJ Exporter.
//
// Features:
//   * World Scale: `1.0f` world unit = `10.0f cm`.
//   * `ObjectTransform` stores translation (`position`), non-uniform/uniform
//     `scale`, and Euler `rotationDeg` (in degrees, snapped to 15.0 deg steps)
//     and builds a Raylib `Matrix` using `MatrixMultiply` from `raymath.h`.
//   * `SceneObject` includes `int groupId = -1` and `bool isSelected = false`
//     for multi-object grouping and synchronized group transformations.
//   * `ExportSceneToOBJ(filename, objects, meshes)` iterates through all
//     `SceneObjects`, multiplies their base mesh vertices by each object's
//     current Raylib transform `Matrix` (using `MatrixMultiply` /
//     `Vector3Transform` from `raymath.h`), and writes standard Wavefront
//     `.obj` `v`, `vn`, and `f` records.
#pragma once

#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace v3d {

// World scale conversion: 1.0f world unit = 10.0 cm
constexpr float kCmPerWorldUnit = 10.0f;
constexpr float kSnapMoveStepUnits = 0.5f;    // 0.5 world units = 5.0 cm (X/Y)
constexpr float kSnapMoveStepZCm = 1.0f;      // 1.0 cm snapped increments along Z-axis
constexpr float kZSensitivityCm = 320.0f;     // Apparent 2D palm scale -> Z displacement (cm)
constexpr float kSnapScaleStepUnits = 0.5f;   // 0.5 world units = 5.0 cm
constexpr float kSnapRotateStepDeg = 15.0f;   // 15.0 degree rotation increments

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
        case PrimitiveType::Torus: return "RING";
        default: return "OBJECT";
    }
}

inline float snapToStep(float value, float step) {
    if (step <= 1e-6f) return value;
    return std::round(value / step) * step;
}

struct ObjectTransform {
    Vector3 position{0.0f, 0.0f, 0.0f};      // World units (1.0f = 10 cm)
    Vector3 scale{1.0f, 1.0f, 1.0f};         // Scale factors (1.0f = 10 cm base size)
    Vector3 rotationDeg{0.0f, 0.0f, 0.0f};   // Tracked Euler angle readouts (degrees)
    Matrix rotMatrix = MatrixIdentity();     // World-space rotation matrix (1-to-1 X/Y/Z axes)

    /// Builds the composite Raylib transform Matrix: T * R * S
    /// using `MatrixMultiply` from `raymath.h`.
    Matrix toMatrix() const {
        const Matrix matScale = MatrixScale(scale.x, scale.y, scale.z);
        const Matrix matTrans = MatrixTranslate(position.x, position.y, position.z);
        return MatrixMultiply(MatrixMultiply(matScale, rotMatrix), matTrans);
    }
};

struct SceneObject {
    uint32_t id = 0;
    PrimitiveType type = PrimitiveType::Cube;
    ObjectTransform transform{};
    ObjectTransform anchorTransform{};  // Captured at rising edge of pinch clutch
    Color color{38, 230, 255, 255};
    int groupId = -1;                   // -1 = ungrouped; >= 0 = shared group ID
    bool isSelected = false;
    bool isHovered = false;

    Matrix matrix() const { return transform.toMatrix(); }
};

// ---------------------------------------------------------------------------
// CPU-backed Base Mesh Cache (for both Raylib `Model` rendering & `.OBJ` export)
// ---------------------------------------------------------------------------
struct CpuTriangleMesh {
    std::vector<Vector3> vertices;
    std::vector<Vector3> normals;
    std::vector<uint32_t> indices;  // 3 indices per triangle
};

class PrimitiveLibrary {
public:
    void init() {
        // Generate Raylib procedural meshes for all 4 primitives (1.0f unit = 10 cm base size)
        Mesh cubeMesh = GenMeshCube(1.0f, 1.0f, 1.0f);
        Mesh sphereCapMesh = GenMeshHemiSphere(0.5f, 16, 16);
        Mesh capsuleMesh = buildCapsuleMesh(0.45f, 0.65f, 20, 8);
        Mesh cylinderMesh = GenMeshCylinder(0.5f, 1.1f, 24);
        Mesh torusMesh = GenMeshTorus(0.32f, 0.95f, 20, 28);
        UnloadMesh(sphereCapMesh);

        Mesh rawMeshes[4] = {cubeMesh, capsuleMesh, cylinderMesh, torusMesh};
        for (int i = 0; i < 4; ++i) {
            cpuMeshes_[i] = extractCpuMesh(rawMeshes[i]);
            models_[i] = LoadModelFromMesh(rawMeshes[i]);
        }
        initialized_ = true;
    }

    void shutdown() {
        if (!initialized_) return;
        for (auto& m : models_) UnloadModel(m);
        initialized_ = false;
    }

    Model& model(PrimitiveType t) { return models_[static_cast<int>(t)]; }
    const Model& model(PrimitiveType t) const { return models_[static_cast<int>(t)]; }
    const CpuTriangleMesh& cpuMesh(PrimitiveType t) const { return cpuMeshes_[static_cast<int>(t)]; }

private:
    bool initialized_ = false;
    Model models_[4]{};
    CpuTriangleMesh cpuMeshes_[4]{};

    static CpuTriangleMesh extractCpuMesh(const Mesh& m) {
        CpuTriangleMesh out;
        out.vertices.resize(m.vertexCount);
        out.normals.resize(m.vertexCount, {0.f, 1.f, 0.f});
        for (int i = 0; i < m.vertexCount; ++i) {
            out.vertices[i] = {m.vertices[i * 3 + 0], m.vertices[i * 3 + 1], m.vertices[i * 3 + 2]};
            if (m.normals) {
                out.normals[i] = {m.normals[i * 3 + 0], m.normals[i * 3 + 1], m.normals[i * 3 + 2]};
            }
        }
        if (m.indices) {
            out.indices.resize(m.triangleCount * 3);
            for (int i = 0; i < m.triangleCount * 3; ++i) {
                out.indices[i] = m.indices[i];
            }
        } else {
            out.indices.resize(m.vertexCount);
            for (int i = 0; i < m.vertexCount; ++i) {
                out.indices[i] = static_cast<uint32_t>(i);
            }
        }
        return out;
    }

    static Mesh buildCapsuleMesh(float radius, float cylHalfH, int slices, int rings) {
        struct ProfileRing {
            float y;
            float r;
            float ny;
        };
        std::vector<ProfileRing> prof;
        for (int i = 0; i <= rings; ++i) {
            const float t = static_cast<float>(i) / rings;
            const float phi = -0.5f * PI + t * (0.5f * PI);
            prof.push_back({-cylHalfH + radius * std::sin(phi), radius * std::cos(phi), std::sin(phi)});
        }
        for (int i = 0; i <= rings; ++i) {
            const float t = static_cast<float>(i) / rings;
            const float phi = t * (0.5f * PI);
            prof.push_back({cylHalfH + radius * std::sin(phi), radius * std::cos(phi), std::sin(phi)});
        }

        const int numR = static_cast<int>(prof.size());
        const int stride = slices + 1;
        const int vCount = numR * stride;
        const int triCount = (numR - 1) * slices * 2;

        Mesh mesh{};
        mesh.vertexCount = vCount;
        mesh.triangleCount = triCount;
        mesh.vertices = static_cast<float*>(MemAlloc(vCount * 3 * sizeof(float)));
        mesh.normals = static_cast<float*>(MemAlloc(vCount * 3 * sizeof(float)));
        mesh.texcoords = static_cast<float*>(MemAlloc(vCount * 2 * sizeof(float)));
        mesh.indices = static_cast<unsigned short*>(MemAlloc(triCount * 3 * sizeof(unsigned short)));

        int vIdx = 0;
        for (int r = 0; r < numR; ++r) {
            const float ny = prof[r].ny;
            const float nr = std::sqrt(std::max(0.f, 1.f - ny * ny));
            for (int s = 0; s <= slices; ++s) {
                const float u = static_cast<float>(s) / slices;
                const float th = u * 2.0f * PI;
                const float cs = std::cos(th), sn = std::sin(th);
                mesh.vertices[vIdx * 3 + 0] = prof[r].r * cs;
                mesh.vertices[vIdx * 3 + 1] = prof[r].y;
                mesh.vertices[vIdx * 3 + 2] = prof[r].r * sn;
                mesh.normals[vIdx * 3 + 0] = nr * cs;
                mesh.normals[vIdx * 3 + 1] = ny;
                mesh.normals[vIdx * 3 + 2] = nr * sn;
                mesh.texcoords[vIdx * 2 + 0] = u;
                mesh.texcoords[vIdx * 2 + 1] = static_cast<float>(r) / (numR - 1);
                ++vIdx;
            }
        }

        int iIdx = 0;
        for (int r = 0; r < numR - 1; ++r) {
            for (int s = 0; s < slices; ++s) {
                const unsigned short i0 = static_cast<unsigned short>(r * stride + s);
                const unsigned short i1 = static_cast<unsigned short>(i0 + 1);
                const unsigned short i2 = static_cast<unsigned short>(i0 + stride);
                const unsigned short i3 = static_cast<unsigned short>(i2 + 1);
                mesh.indices[iIdx++] = i0;
                mesh.indices[iIdx++] = i2;
                mesh.indices[iIdx++] = i1;
                mesh.indices[iIdx++] = i1;
                mesh.indices[iIdx++] = i2;
                mesh.indices[iIdx++] = i3;
            }
        }
        UploadMesh(&mesh, false);
        return mesh;
    }
};

// ---------------------------------------------------------------------------
// Wavefront .OBJ Exporter (`ExportSceneToOBJ`)
// ---------------------------------------------------------------------------
inline bool ExportSceneToOBJ(const std::string& filename,
                             const std::vector<SceneObject>& objects,
                             const PrimitiveLibrary& lib) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "# Visual3D Wavefront .OBJ Export\n";
    out << "# Scale: 1.0 world unit = 10.0 cm\n";
    out << "# Objects exported: " << objects.size() << "\n\n";
    out << std::fixed << std::setprecision(6);

    uint32_t vertexOffset = 1;  // Wavefront .obj indices are 1-based
    for (const SceneObject& obj : objects) {
        out << "o " << primitiveName(obj.type) << "_" << obj.id;
        if (obj.groupId >= 0) out << "_Group" << obj.groupId;
        out << "\n";
        if (obj.groupId >= 0) {
            out << "g Group_" << obj.groupId << "\n";
        }

        // Compose transform matrix using MatrixMultiply from raymath.h:
        // M = Scale * Rotation * Translation
        const Matrix matScale = MatrixScale(obj.transform.scale.x,
                                            obj.transform.scale.y,
                                            obj.transform.scale.z);
        const Matrix matRot = obj.transform.rotMatrix;
        const Matrix matTrans = MatrixTranslate(obj.transform.position.x,
                                                obj.transform.position.y,
                                                obj.transform.position.z);
        const Matrix worldMat = MatrixMultiply(MatrixMultiply(matScale, matRot), matTrans);

        const CpuTriangleMesh& base = lib.cpuMesh(obj.type);
        for (const Vector3& v : base.vertices) {
            // Multiply base vertex by current Raylib transform matrix
            const Vector3 tv = Vector3Transform(v, worldMat);
            out << "v " << tv.x << " " << tv.y << " " << tv.z << "\n";
        }
        for (const Vector3& n : base.normals) {
            const Vector3 tn = Vector3Normalize(Vector3Transform(n, matRot));
            out << "vn " << tn.x << " " << tn.y << " " << tn.z << "\n";
        }
        for (size_t i = 0; i + 2 < base.indices.size(); i += 3) {
            const uint32_t i0 = vertexOffset + base.indices[i + 0];
            const uint32_t i1 = vertexOffset + base.indices[i + 1];
            const uint32_t i2 = vertexOffset + base.indices[i + 2];
            out << "f " << i0 << "//" << i0 << " "
                        << i1 << "//" << i1 << " "
                        << i2 << "//" << i2 << "\n";
        }
        out << "\n";
        vertexOffset += static_cast<uint32_t>(base.vertices.size());
    }
    return true;
}

}  // namespace v3d
