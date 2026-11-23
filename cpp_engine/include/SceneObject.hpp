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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
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
    ImportedObj = 4,
    Count = 5
};

inline constexpr PrimitiveType PRIMITIVE_IMPORTED_OBJ = PrimitiveType::ImportedObj;

inline const char* primitiveName(PrimitiveType t) {
    switch (t) {
        case PrimitiveType::Cube: return "CUBE";
        case PrimitiveType::Capsule: return "CAPSULE";
        case PrimitiveType::Cylinder: return "CYLINDER";
        case PrimitiveType::Torus: return "RING";
        case PrimitiveType::ImportedObj: return "OBJ";
        default: return "OBJECT";
    }
}

inline std::string extractFilename(const std::string& path) {
    if (path.empty()) return "";
    const size_t pos = path.find_last_of("\\/");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

inline std::string sanitizeObjIdentifier(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("Object") : out;
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

// ---------------------------------------------------------------------------
// CPU-backed Base Mesh Cache & Normalized Imported .OBJ Asset Container
// ---------------------------------------------------------------------------
struct CpuTriangleMesh {
    std::vector<Vector3> vertices;
    std::vector<Vector3> normals;
    std::vector<uint32_t> indices;  // 3 indices per triangle
};

struct ImportedModelAsset {
    Model model{};
    bool hasModel = false;
    std::string displayName;
    std::string sourcePath;
    Vector3 pivotOffset{0.0f, 0.0f, 0.0f};
    float normScale = 1.0f;
    BoundingBox normalizedLocalBox{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};
    CpuTriangleMesh cpuMesh{};

    ImportedModelAsset() = default;
    ~ImportedModelAsset() {
        if (hasModel) {
            UnloadModel(model);
            hasModel = false;
        }
    }
    ImportedModelAsset(const ImportedModelAsset&) = delete;
    ImportedModelAsset& operator=(const ImportedModelAsset&) = delete;

    Matrix normMatrix() const {
        return MatrixMultiply(
            MatrixTranslate(pivotOffset.x, pivotOffset.y, pivotOffset.z),
            MatrixScale(normScale, normScale, normScale));
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
    std::shared_ptr<ImportedModelAsset> importedModel;

    Matrix matrix() const { return transform.toMatrix(); }

    std::string displayLabel() const {
        if (type == PrimitiveType::ImportedObj && importedModel && !importedModel->displayName.empty()) {
            return importedModel->displayName;
        }
        return primitiveName(type);
    }
};

inline void ComputeNormalizationParams(const BoundingBox& rawBox,
                                       float targetExtent,
                                       Vector3& outPivot,
                                       float& outScale,
                                       BoundingBox& outNormBox) {
    const Vector3 rawCenter{
        0.5f * (rawBox.min.x + rawBox.max.x),
        0.5f * (rawBox.min.y + rawBox.max.y),
        0.5f * (rawBox.min.z + rawBox.max.z),
    };
    const float extX = std::max(0.0f, rawBox.max.x - rawBox.min.x);
    const float extY = std::max(0.0f, rawBox.max.y - rawBox.min.y);
    const float extZ = std::max(0.0f, rawBox.max.z - rawBox.min.z);
    const float maxExtent = std::max({extX, extY, extZ, 1e-5f});

    outPivot = {-rawCenter.x, -rawCenter.y, -rawCenter.z};
    outScale = targetExtent / maxExtent;
    outNormBox.min = {
        (rawBox.min.x + outPivot.x) * outScale,
        (rawBox.min.y + outPivot.y) * outScale,
        (rawBox.min.z + outPivot.z) * outScale,
    };
    outNormBox.max = {
        (rawBox.max.x + outPivot.x) * outScale,
        (rawBox.max.y + outPivot.y) * outScale,
        (rawBox.max.z + outPivot.z) * outScale,
    };
}

inline CpuTriangleMesh ExtractNormalizedCpuMesh(const Model& model, const Matrix& normMat) {
    CpuTriangleMesh out;
    for (int m = 0; m < model.meshCount; ++m) {
        const Mesh& mesh = model.meshes[m];
        if (!mesh.vertices || mesh.vertexCount <= 0) continue;

        const uint32_t baseVertex = static_cast<uint32_t>(out.vertices.size());
        const size_t triStartIdx = out.indices.size();

        for (int v = 0; v < mesh.vertexCount; ++v) {
            const Vector3 rawV{
                mesh.vertices[v * 3 + 0],
                mesh.vertices[v * 3 + 1],
                mesh.vertices[v * 3 + 2],
            };
            out.vertices.push_back(Vector3Transform(rawV, normMat));
            if (mesh.normals) {
                Vector3 rawN{
                    mesh.normals[v * 3 + 0],
                    mesh.normals[v * 3 + 1],
                    mesh.normals[v * 3 + 2],
                };
                const float nLen = Vector3Length(rawN);
                out.normals.push_back(nLen > 1e-6f ? Vector3Scale(rawN, 1.0f / nLen)
                                                   : Vector3{0.0f, 1.0f, 0.0f});
            } else {
                out.normals.push_back({0.0f, 1.0f, 0.0f});
            }
        }

        if (mesh.indices && mesh.triangleCount > 0) {
            for (int k = 0; k < mesh.triangleCount * 3; ++k) {
                out.indices.push_back(baseVertex + mesh.indices[k]);
            }
        } else {
            const int usableVerts = (mesh.vertexCount / 3) * 3;
            for (int k = 0; k < usableVerts; ++k) {
                out.indices.push_back(baseVertex + static_cast<uint32_t>(k));
            }
        }

        // Synthesize flat face normals when the source .obj lacked `vn` entries
        if (!mesh.normals) {
            for (size_t i = triStartIdx; i + 2 < out.indices.size(); i += 3) {
                const uint32_t i0 = out.indices[i + 0];
                const uint32_t i1 = out.indices[i + 1];
                const uint32_t i2 = out.indices[i + 2];
                const Vector3 e1 = Vector3Subtract(out.vertices[i1], out.vertices[i0]);
                const Vector3 e2 = Vector3Subtract(out.vertices[i2], out.vertices[i0]);
                Vector3 fn = Vector3CrossProduct(e1, e2);
                const float len = Vector3Length(fn);
                if (len > 1e-6f) {
                    fn = Vector3Scale(fn, 1.0f / len);
                    out.normals[i0] = fn;
                    out.normals[i1] = fn;
                    out.normals[i2] = fn;
                }
            }
        }
    }
    return out;
}

inline std::shared_ptr<ImportedModelAsset> LoadNormalizedOBJAsset(const std::string& filepath,
                                                                  float targetExtent = 2.0f) {
    if (filepath.empty()) return nullptr;
    Model mdl = LoadModel(filepath.c_str());
    if (mdl.meshCount <= 0) {
        if (mdl.materialCount > 0 || mdl.meshes != nullptr) UnloadModel(mdl);
        return nullptr;
    }

    const BoundingBox rawBox = GetModelBoundingBox(mdl);
    auto asset = std::make_shared<ImportedModelAsset>();
    asset->model = mdl;
    asset->hasModel = true;
    asset->sourcePath = filepath;
    asset->displayName = extractFilename(filepath);
    ComputeNormalizationParams(rawBox, targetExtent,
                               asset->pivotOffset, asset->normScale, asset->normalizedLocalBox);
    asset->cpuMesh = ExtractNormalizedCpuMesh(mdl, asset->normMatrix());

    if (asset->cpuMesh.vertices.empty() || asset->cpuMesh.indices.size() < 3) {
        return nullptr;
    }

    for (int i = 0; i < asset->model.materialCount; ++i) {
        asset->model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    }
    return asset;
}

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
// Wavefront .OBJ Exporter (`ExportObjectsToOBJ` & `ExportSceneToOBJ`)
// ---------------------------------------------------------------------------
inline bool ExportObjectsToOBJ(const std::string& filename,
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
        const CpuTriangleMesh* base = nullptr;
        if (obj.type == PrimitiveType::ImportedObj) {
            if (obj.importedModel && !obj.importedModel->cpuMesh.vertices.empty()) {
                base = &obj.importedModel->cpuMesh;
            }
        } else {
            base = &lib.cpuMesh(obj.type);
        }
        if (!base || base->vertices.empty()) continue;

        out << "o " << sanitizeObjIdentifier(obj.displayLabel()) << "_" << obj.id;
        if (obj.groupId >= 0) out << "_Group" << obj.groupId;
        out << "\n";
        if (obj.groupId >= 0) {
            out << "g Group_" << obj.groupId << "\n";
        }

        // Compose world transform matrix: M = Scale * Rotation * Translation
        const Matrix worldMat = obj.matrix();
        const Matrix matRot = obj.transform.rotMatrix;
        const float invSx = 1.0f / ((std::abs(obj.transform.scale.x) > 1e-5f) ? obj.transform.scale.x : 1.0f);
        const float invSy = 1.0f / ((std::abs(obj.transform.scale.y) > 1e-5f) ? obj.transform.scale.y : 1.0f);
        const float invSz = 1.0f / ((std::abs(obj.transform.scale.z) > 1e-5f) ? obj.transform.scale.z : 1.0f);

        for (const Vector3& v : base->vertices) {
            const Vector3 tv = Vector3Transform(v, worldMat);
            out << "v " << tv.x << " " << tv.y << " " << tv.z << "\n";
        }
        for (const Vector3& n : base->normals) {
            const Vector3 scaledN{n.x * invSx, n.y * invSy, n.z * invSz};
            Vector3 tn = Vector3Transform(scaledN, matRot);
            const float nLen = Vector3Length(tn);
            tn = (nLen > 1e-6f) ? Vector3Scale(tn, 1.0f / nLen) : Vector3{0.0f, 1.0f, 0.0f};
            out << "vn " << tn.x << " " << tn.y << " " << tn.z << "\n";
        }
        for (size_t i = 0; i + 2 < base->indices.size(); i += 3) {
            const uint32_t i0 = vertexOffset + base->indices[i + 0];
            const uint32_t i1 = vertexOffset + base->indices[i + 1];
            const uint32_t i2 = vertexOffset + base->indices[i + 2];
            out << "f " << i0 << "//" << i0 << " "
                        << i1 << "//" << i1 << " "
                        << i2 << "//" << i2 << "\n";
        }
        out << "\n";
        vertexOffset += static_cast<uint32_t>(base->vertices.size());
    }
    return out.good();
}

inline bool ExportSceneToOBJ(const std::string& filename,
                             const std::vector<SceneObject>& objects,
                             const PrimitiveLibrary& lib) {
    return ExportObjectsToOBJ(filename, objects, lib);
}

// ---------------------------------------------------------------------------
// Visual3D Project Persistence (`.3dmproj` Save / Load)
// ---------------------------------------------------------------------------
inline const char* projectTypeToken(PrimitiveType t) {
    switch (t) {
        case PrimitiveType::Cube: return "CUBE";
        case PrimitiveType::Capsule: return "CAPSULE";
        case PrimitiveType::Cylinder: return "CYLINDER";
        case PrimitiveType::Torus: return "RING";
        case PrimitiveType::ImportedObj: return "IMPORTED_OBJ";
        default: return "CUBE";
    }
}

inline bool parseProjectTypeToken(const std::string& tok, PrimitiveType& outType) {
    if (tok == "CUBE") { outType = PrimitiveType::Cube; return true; }
    if (tok == "CAPSULE") { outType = PrimitiveType::Capsule; return true; }
    if (tok == "CYLINDER") { outType = PrimitiveType::Cylinder; return true; }
    if (tok == "RING") { outType = PrimitiveType::Torus; return true; }
    if (tok == "IMPORTED_OBJ") { outType = PrimitiveType::ImportedObj; return true; }
    return false;
}

inline std::string escapeProjectPath(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 4);
    out.push_back('"');
    for (char c : raw) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

inline bool parseQuotedProjectPath(const std::string& rest, std::string& outPath) {
    const size_t openPos = rest.find('"');
    if (openPos == std::string::npos) return false;
    outPath.clear();
    bool closed = false;
    for (size_t i = openPos + 1; i < rest.size(); ++i) {
        const char c = rest[i];
        if (c == '\\' && i + 1 < rest.size()) {
            const char nxt = rest[i + 1];
            if (nxt == '\\' || nxt == '"') {
                outPath.push_back(nxt);
                ++i;
            } else {
                // Tolerate unescaped Windows drive separators in hand-edited files
                outPath.push_back(c);
            }
        } else if (c == '"') {
            closed = true;
            break;
        } else {
            outPath.push_back(c);
        }
    }
    return closed;
}

inline bool SaveProjectToFile(const std::string& filename,
                              const std::vector<SceneObject>& objects,
                              int nextGroupId) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    const int safeNextGroupId = std::max(1, nextGroupId);
    out << "NEXT_GROUP_ID " << safeNextGroupId << "\n";
    out << std::fixed << std::setprecision(6);

    for (const SceneObject& obj : objects) {
        const Quaternion q = QuaternionFromMatrix(obj.transform.rotMatrix);
        const Vector3 eulerRad = QuaternionToEuler(q);
        const float rx = eulerRad.x * RAD2DEG;
        const float ry = eulerRad.y * RAD2DEG;
        const float rz = eulerRad.z * RAD2DEG;

        const std::string srcPath =
            (obj.type == PrimitiveType::ImportedObj && obj.importedModel)
                ? obj.importedModel->sourcePath
                : std::string();

        out << projectTypeToken(obj.type) << " "
            << obj.transform.position.x << " " << obj.transform.position.y << " " << obj.transform.position.z << " "
            << rx << " " << ry << " " << rz << " "
            << obj.transform.scale.x << " " << obj.transform.scale.y << " " << obj.transform.scale.z << " "
            << static_cast<unsigned int>(obj.color.r) << " "
            << static_cast<unsigned int>(obj.color.g) << " "
            << static_cast<unsigned int>(obj.color.b) << " "
            << static_cast<unsigned int>(obj.color.a) << " "
            << obj.groupId << " "
            << (obj.isSelected ? 1 : 0) << " "
            << escapeProjectPath(srcPath) << "\n";
    }
    return out.good();
}

using ObjAssetLoaderFn = std::function<std::shared_ptr<ImportedModelAsset>(const std::string&)>;

inline bool LoadProjectFromFile(
    const std::string& filename,
    std::vector<SceneObject>& outObjects,
    int& outNextGroupId,
    int& outMissingObjCount,
    ObjAssetLoaderFn objLoader = [](const std::string& p) {
        return LoadNormalizedOBJAsset(p, 2.0f);
    }) {
    outMissingObjCount = 0;
    std::ifstream in(filename);
    if (!in.is_open()) return false;

    std::vector<SceneObject> parsed;
    int headerNextGroupId = -1;
    int maxGroupId = 0;
    uint32_t nextLocalId = 1;
    bool parsedHeader = false;

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t firstNonWs = line.find_first_not_of(" \t");
        if (firstNonWs == std::string::npos) continue;
        if (line[firstNonWs] == '#') continue;

        std::istringstream iss(line.substr(firstNonWs));
        if (!parsedHeader) {
            std::string firstTok;
            if (!(iss >> firstTok)) return false;
            if (firstTok == "NEXT_GROUP_ID") {
                if (!(iss >> headerNextGroupId) || headerNextGroupId < 1) return false;
            } else {
                try {
                    size_t consumed = 0;
                    headerNextGroupId = std::stoi(firstTok, &consumed);
                    if (consumed != firstTok.size() || headerNextGroupId < 1) return false;
                } catch (...) {
                    return false;
                }
            }
            parsedHeader = true;
            continue;
        }

        std::string typeTok;
        float px = 0.f, py = 0.f, pz = 0.f;
        float rx = 0.f, ry = 0.f, rz = 0.f;
        float sx = 1.f, sy = 1.f, sz = 1.f;
        int cr = 255, cg = 255, cb = 255, ca = 255;
        int groupId = -1;
        int isSel = 0;
        if (!(iss >> typeTok >> px >> py >> pz >> rx >> ry >> rz >> sx >> sy >> sz
                  >> cr >> cg >> cb >> ca >> groupId >> isSel)) {
            return false;
        }

        PrimitiveType pType = PrimitiveType::Cube;
        if (!parseProjectTypeToken(typeTok, pType)) return false;

        std::string remainder;
        std::getline(iss, remainder);
        std::string srcPath;
        if (!parseQuotedProjectPath(remainder, srcPath)) {
            if (pType == PrimitiveType::ImportedObj) return false;
            srcPath.clear();
        }

        SceneObject obj{};
        obj.id = nextLocalId++;
        obj.type = pType;
        obj.transform.position = {px, py, pz};
        obj.transform.rotationDeg = {rx, ry, rz};
        const Quaternion q = QuaternionFromEuler(rx * DEG2RAD, ry * DEG2RAD, rz * DEG2RAD);
        obj.transform.rotMatrix = QuaternionToMatrix(q);
        obj.transform.scale = {
            std::max(0.05f, sx),
            std::max(0.05f, sy),
            std::max(0.05f, sz),
        };
        obj.anchorTransform = obj.transform;
        obj.color = Color{
            static_cast<unsigned char>(std::clamp(cr, 0, 255)),
            static_cast<unsigned char>(std::clamp(cg, 0, 255)),
            static_cast<unsigned char>(std::clamp(cb, 0, 255)),
            static_cast<unsigned char>(std::clamp(ca, 0, 255)),
        };
        obj.groupId = (groupId >= 0) ? groupId : -1;
        obj.isSelected = (isSel != 0);

        if (pType == PrimitiveType::ImportedObj) {
            obj.importedModel = objLoader ? objLoader(srcPath) : nullptr;
            if (!obj.importedModel) {
                ++outMissingObjCount;
                continue;
            }
        }

        if (obj.groupId >= 0) {
            maxGroupId = std::max(maxGroupId, obj.groupId);
        }
        parsed.push_back(std::move(obj));
    }

    if (!parsedHeader) return false;

    outObjects = std::move(parsed);
    outNextGroupId = std::max({1, headerNextGroupId, maxGroupId + 1});
    return true;
}

}  // namespace v3d
