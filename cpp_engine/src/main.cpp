// main.cpp -- Visual3D C++ Raylib "Mode + Clutch" Spatial 3D Modeler & HUD (Process 2)
//
// Architecture & Feature Summary:
//   1. Non-Blocking Binary UDP Receiver (`127.0.0.1:5005`):
//      Drains all queued packets in a `while (recvfrom(...) > 0)` loop and parses
//      `#pragma pack(push, 1) TelemetryPacket` (`left_mode`, `right_axis`,
//      `is_pinching`, 3D cursor coordinates, and 2D/3D hand landmarks).
//   2. World Scale & Snapped "Anchor + Delta" Transformations:
//      * World Scale: `1.0f` world unit = `10.0 cm`.
//      * Rising Edge of Pinch (`is_pinching` 0 -> 1): records `anchor_hand_pos`
//        and every selected/grouped object's `anchorTransform`.
//      * Held Pinch (`is_pinching == 1`): computes hand delta from `anchor_hand_pos`
//        along the locked axis (`X-Axis`, `Y-Axis`, or `Z-Axis`) ONLY:
//          - MOVE   : snaps delta to `0.5f` world units (`5.0 cm` steps).
//          - RESIZE : snaps scale delta to `0.5f` world units (`5.0 cm` steps).
//          - ROTATE : snaps angular delta to `15.0f` degree increments.
//          - RECOLOR: applies the active chromatic RGB color to the target object/group.
//          - SPAWN / GROUP: renders a wireframe ghost preview (`DrawModelWires`) at
//            the 3D cursor; confirming the pinch spawns a permanent `SceneObject`
//            (or groups selected objects when hovering an existing object).
//      * Falling Edge of Pinch (`is_pinching` 1 -> 0): bakes the snapped transform
//        into `SceneObject::transform` so the object stays frozen in place.
//   3. Dual-Panel View Toggle (`TAB` key):
//      * Mode A (AR Camera): Draws the live webcam `Texture2D` fullscreen with 3D
//        models and AR hand skeleton overlayed.
//      * Mode B (Studio 3D): Clears background to dark gray `{24, 26, 32, 255}`,
//        draws `DrawGrid(20, 0.5f)`, projects a 3D cursor sphere + axis gizmo into
//        the 3D scene, and renders the webcam feed + 2D hand skeleton as a
//        Picture-in-Picture (PiP) panel in the bottom-right corner.
//   4. Top-Center Pill-Shaped Measurement HUD & `.OBJ` Export (`E` key):
//      * Displays live snapped telemetry (e.g., `MOVE [X-AXIS]: +15.0 cm` or
//        `ROTATE [Y-AXIS]: +45 deg`).
//      * `ExportSceneToOBJ("scene_export.obj", ...)` bakes each object's Raylib
//        transform matrix (`MatrixMultiply` from `raymath.h`) into world-space
//        `v` and `f` records.

#include "UDPReceiver.hpp"
#include "ProcessManager.hpp"
#include "SceneObject.hpp"
#include "FileDialog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace v3d {

inline constexpr std::pair<int, int> kHandBones[21] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},
    {0, 5},   {5, 6},   {6, 7},   {7, 8},
    {5, 9},   {9, 10},  {10, 11}, {11, 12},
    {9, 13},  {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {0, 17},  {17, 18}, {18, 19}, {19, 20}
};

enum class ViewportMode : int {
    Studio3D = 0,   // Mode B: Dark gray {24,26,32,255} + DrawGrid(20, 0.5f) + bottom-right PiP
    ArCamera = 1    // Mode A: Fullscreen webcam Texture2D background + 3D overlay
};

inline const char* modeName(proto::LeftMode m) {
    switch (m) {
        case proto::LeftMode::Idle: return "IDLE";
        case proto::LeftMode::Move: return "MOVE";
        case proto::LeftMode::Resize: return "RESIZE";
        case proto::LeftMode::Rotate: return "ROTATE";
        case proto::LeftMode::Recolor: return "RECOLOR";
        case proto::LeftMode::Group: return "GROUP";
        case proto::LeftMode::Spawn: return "SPAWN";
        case proto::LeftMode::Delete: return "DELETE";
        default: return "IDLE";
    }
}

inline const char* axisName(proto::RightAxis a) {
    switch (a) {
        case proto::RightAxis::X: return "X-AXIS";
        case proto::RightAxis::Y: return "Y-AXIS";
        case proto::RightAxis::Z: return "Z-AXIS";
        default: return "X-AXIS";
    }
}

inline const char* axisFingerName(proto::RightAxis a) {
    switch (a) {
        case proto::RightAxis::X: return "Thumb+Index";
        case proto::RightAxis::Y: return "Thumb+Middle";
        case proto::RightAxis::Z: return "Thumb+Ring";
        default: return "Thumb+Index";
    }
}

inline const char* spawnFingerName(PrimitiveType t) {
    switch (t) {
        case PrimitiveType::Cube: return "Thumb+Index";
        case PrimitiveType::Capsule: return "Thumb+Middle";
        case PrimitiveType::Cylinder: return "Thumb+Ring";
        case PrimitiveType::Torus: return "Thumb+Pinky";
        default: return "Thumb+Index";
    }
}

inline const char* groupActionName(proto::GroupAction a) {
    switch (a) {
        case proto::GroupAction::SelectToggle: return "SELECT";
        case proto::GroupAction::Group: return "GROUP";
        case proto::GroupAction::Ungroup: return "UNGROUP";
        case proto::GroupAction::ClearSelection: return "CLEAR";
        default: return "SELECT";
    }
}

inline const char* groupActionFingerName(proto::GroupAction a) {
    switch (a) {
        case proto::GroupAction::SelectToggle: return "Thumb+Index";
        case proto::GroupAction::Group: return "Thumb+Middle";
        case proto::GroupAction::Ungroup: return "Thumb+Ring";
        case proto::GroupAction::ClearSelection: return "Thumb+Pinky";
        default: return "Thumb+Index";
    }
}

inline Color axisColor(proto::RightAxis a) {
    switch (a) {
        case proto::RightAxis::X: return Color{255, 75, 85, 255};   // Red X
        case proto::RightAxis::Y: return Color{75, 245, 115, 255};  // Green Y
        case proto::RightAxis::Z: return Color{75, 165, 255, 255};  // Blue Z
        default: return Color{80, 225, 255, 255};
    }
}

inline Color recolorColorFromMask(uint8_t mask) {
    const unsigned char r = (mask & proto::kMiddleRed) ? 255 : 0;  // bit 0: Right Index (R)
    const unsigned char g = (mask & proto::kRingGreen) ? 255 : 0;  // bit 1: Right Middle (G)
    const unsigned char b = (mask & proto::kPinkyBlue) ? 255 : 0;  // bit 2: Right Ring (B)
    return Color{r, g, b, 255};
}

inline const char* recolorNameFromMask(uint8_t mask) {
    switch (mask & 0x07) {
        case 0b001: return "RED";
        case 0b010: return "GREEN";
        case 0b100: return "BLUE";
        case 0b011: return "YELLOW";
        case 0b110: return "CYAN";
        case 0b101: return "MAGENTA";
        case 0b111: return "WHITE";
        default: return "CYAN";
    }
}

// ---------------------------------------------------------------------------
// Mode + Clutch Spatial Modeler Application
// ---------------------------------------------------------------------------
class ModeClutchApp {
public:
    void init() {
        primitives_.init();
        initFallbackCameraTexture();
        resetDefaultScene();
    }

    void shutdown() {
        // Release imported Raylib Model GPU resources before CloseWindow() invalidates GL.
        objects_.clear();
        activeObjectIdx_ = -1;
        if (camTex_.id != 0) UnloadTexture(camTex_);
        primitives_.shutdown();
    }

    void resetDefaultScene() {
        objects_.clear();
        nextId_ = 1;
        nextGroupId_ = 1;

        spawnObject(PrimitiveType::Cube,     {-2.0f, 0.5f,  0.0f}, Color{45, 225, 255, 255});
        spawnObject(PrimitiveType::Capsule,  {-0.5f, 0.6f,  0.0f}, Color{80, 245, 145, 255});
        spawnObject(PrimitiveType::Cylinder, { 1.0f, 0.55f, 0.0f}, Color{255, 95, 210, 255});
        spawnObject(PrimitiveType::Torus,    { 2.5f, 0.6f,  0.0f}, Color{255, 215, 65, 255});

        if (!objects_.empty()) {
            activeObjectIdx_ = 0;
        }
        deleteHoldTimer_ = 0.0f;
        deleteCooldownTimer_ = 0.0f;
        measurementBadge_ = "IDLE [SAFE] // 1:MOVE 2:SIZE 3:ROT 4:CLR 5:GRP TU:SPAWN TD:DEL";
        statusLine_ = "TAB: AR/3D View | S: Export Scene .OBJ | Drag & Drop .3dmproj/.obj | G: Group | R: Reset";
    }

    SceneObject& spawnObject(PrimitiveType type, Vector3 pos, Color col) {
        SceneObject obj{};
        obj.id = nextId_++;
        obj.type = type;
        obj.transform.position = {
            snapToStep(pos.x, kSnapMoveStepUnits),
            snapToStep(pos.y, kSnapMoveStepUnits),
            snapToStep(pos.z, kSnapMoveStepUnits),
        };
        obj.transform.scale = {1.0f, 1.0f, 1.0f};
        obj.transform.rotationDeg = {0.0f, 0.0f, 0.0f};
        obj.transform.rotMatrix = MatrixIdentity();
        obj.anchorTransform = obj.transform;
        obj.color = col;
        obj.groupId = -1;
        obj.isSelected = false;
        objects_.push_back(obj);
        return objects_.back();
    }

    void handleKeyboardShortcuts(UDPReceiver& udp) {
        if (IsKeyPressed(KEY_TAB)) {
            viewportMode_ = (viewportMode_ == ViewportMode::Studio3D) ? ViewportMode::ArCamera
                                                                      : ViewportMode::Studio3D;
        }
        if (IsKeyPressed(KEY_ONE)) activeShape_ = PrimitiveType::Cube;
        if (IsKeyPressed(KEY_TWO)) activeShape_ = PrimitiveType::Capsule;
        if (IsKeyPressed(KEY_THREE)) activeShape_ = PrimitiveType::Cylinder;
        if (IsKeyPressed(KEY_FOUR)) activeShape_ = PrimitiveType::Torus;
        if (IsKeyPressed(KEY_X)) {
            rightAxis_ = proto::RightAxis::X;
            hasKeyboardAxisOverride_ = true;
            statusLine_ = "AXIS OVERRIDE: X-AXIS [KEY X]";
        }
        if (IsKeyPressed(KEY_Y)) {
            rightAxis_ = proto::RightAxis::Y;
            hasKeyboardAxisOverride_ = true;
            statusLine_ = "AXIS OVERRIDE: Y-AXIS [KEY Y]";
        }
        if (IsKeyPressed(KEY_Z)) {
            rightAxis_ = proto::RightAxis::Z;
            hasKeyboardAxisOverride_ = true;
            statusLine_ = "AXIS OVERRIDE: Z-AXIS [KEY Z]";
        }

        if (IsKeyPressed(KEY_G)) {
            groupSelectedObjects();
        }
        if (IsKeyPressed(KEY_U)) {
            for (auto& o : objects_) o.groupId = -1;
            statusLine_ = "UNGROUPED ALL OBJECTS";
        }
        if (IsKeyPressed(KEY_SPACE)) {
            spawnObject(activeShape_, cursor3D_, activeColor_);
            activeObjectIdx_ = static_cast<int>(objects_.size()) - 1;
            statusLine_ = std::string("SPAWNED ") + primitiveName(activeShape_);
        }
        const bool ctrlHeld = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
        if (IsKeyPressed(KEY_P) || (ctrlHeld && IsKeyPressed(KEY_S))) {
            std::string savePath;
            runModalDialogGuard(udp, [&]() {
                savePath = SaveProjectFileDialog("untitled.3dmproj", GetWindowHandle());
            });
            if (!savePath.empty()) {
                saveProjectFile(savePath);
            }
        }
        if (IsKeyPressed(KEY_O)) {
            std::string openPath;
            runModalDialogGuard(udp, [&]() {
                openPath = OpenProjectFileDialog(GetWindowHandle());
            });
            if (!openPath.empty()) {
                loadProjectFile(openPath);
            }
        }
        if (IsKeyPressed(KEY_I)) {
            std::string openPath;
            runModalDialogGuard(udp, [&]() {
                openPath = OpenOBJFileDialog(GetWindowHandle());
            });
            if (!openPath.empty()) {
                importOBJFile(openPath);
            }
        }
        if (IsKeyPressed(KEY_E)) {
            bool isGroup = false;
            const std::vector<int> exportIdx = collectGroupExportTargets(isGroup);
            if (exportIdx.empty()) {
                showToast("NOTHING TO EXPORT: SELECT AN OBJECT OR GROUP FIRST", true);
            } else {
                std::string savePath;
                runModalDialogGuard(udp, [&]() {
                    savePath = SaveOBJFileDialog("group_export.obj", GetWindowHandle());
                });
                if (!savePath.empty()) {
                    std::vector<SceneObject> subset;
                    subset.reserve(exportIdx.size());
                    for (int idx : exportIdx) subset.push_back(objects_[idx]);
                    const std::string base = extractFilename(savePath);
                    if (ExportObjectsToOBJ(savePath, subset, primitives_)) {
                        if (isGroup || subset.size() > 1) {
                            showToast("EXPORTED GROUP (" + std::to_string(subset.size()) +
                                      " objects) -> " + base, false);
                        } else {
                            showToast("EXPORTED OBJECT (1 object) -> " + base, false);
                        }
                    } else {
                        showToast("ERROR: FAILED TO WRITE " + base, true);
                    }
                }
            }
        }
        if (!ctrlHeld && IsKeyPressed(KEY_S)) {
            if (objects_.empty()) {
                showToast("SCENE IS EMPTY: NOTHING TO EXPORT", true);
            } else {
                std::string savePath;
                runModalDialogGuard(udp, [&]() {
                    savePath = SaveOBJFileDialog("scene_export.obj", GetWindowHandle());
                });
                if (!savePath.empty()) {
                    const std::string base = extractFilename(savePath);
                    if (ExportObjectsToOBJ(savePath, objects_, primitives_)) {
                        showToast("EXPORTED SCENE (" + std::to_string(objects_.size()) +
                                  " objects) -> " + base, false);
                    } else {
                        showToast("ERROR: FAILED TO WRITE " + base, true);
                    }
                }
            }
        }
        if (IsKeyPressed(KEY_C)) {
            objects_.clear();
            activeObjectIdx_ = -1;
            deleteHoldTimer_ = 0.0f;
            deleteCooldownTimer_ = 0.0f;
            statusLine_ = "SCENE CLEARED";
        }
        if (IsKeyPressed(KEY_R)) {
            resetDefaultScene();
        }
    }

    void groupSelectedObjects() {
        int count = 0;
        for (const auto& o : objects_) {
            if (o.isSelected) ++count;
        }
        if (count < 2 && objects_.size() >= 2) {
            // If fewer than 2 are flagged, group all current objects into a single assembly
            const int gid = nextGroupId_++;
            for (auto& o : objects_) {
                o.groupId = gid;
                o.isSelected = true;
            }
            statusLine_ = "GROUPED ALL " + std::to_string(objects_.size()) +
                          " OBJECTS INTO GROUP #" + std::to_string(gid);
            return;
        }
        const int gid = nextGroupId_++;
        for (auto& o : objects_) {
            if (o.isSelected) o.groupId = gid;
        }
        statusLine_ = "ASSIGNED GROUP #" + std::to_string(gid) + " TO " +
                      std::to_string(count) + " SELECTED OBJECTS";
    }

    void update(float dt, UDPReceiver& udp) {
        if (toastTimer_ > 0.0f) {
            toastTimer_ = std::max(0.0f, toastTimer_ - dt);
        }

        handleKeyboardShortcuts(udp);

        // Drag-and-drop .3dmproj projects and external .obj meshes onto the application window
        if (!dialogOpen_ && IsFileDropped()) {
            FilePathList dropped = LoadDroppedFiles();
            bool anyUnsupported = false;
            for (unsigned int i = 0; i < dropped.count; ++i) {
                const char* p = dropped.paths[i];
                if (!p) continue;
                if (hasExtensionCI(p, ".3dmproj")) {
                    loadProjectFile(p);
                } else if (hasExtensionCI(p, ".obj")) {
                    importOBJFile(p);
                } else {
                    anyUnsupported = true;
                }
            }
            if (anyUnsupported) {
                showToast("DRAG & DROP: ONLY .3DMPROJ AND .OBJ FILES SUPPORTED", true);
            }
            UnloadDroppedFiles(dropped);
        }

        // 1. Update live webcam Texture2D if a new video frame arrived over UDP
        VideoFrame vf;
        if (udp.latestVideo(vf, lastVideoVersion_)) {
            lastVideoVersion_ = vf.version;
            uploadCameraTexture(vf);
        }

        // 2. Drain UDP socket queue (`while (recvfrom(...) > 0)`) and read newest packet
        proto::TelemetryPacket pkt{};
        const bool hasPkt = udp.latest(pkt);
        linkStats_ = udp.stats();

        if (hasPkt && linkStats_.connected) {
            trackerFps_ = pkt.trackerFps;
            leftMode_ = static_cast<proto::LeftMode>(pkt.left_mode);
            if (pkt.right_axis != lastTrackerRightAxis_) {
                lastTrackerRightAxis_ = pkt.right_axis;
                hasKeyboardAxisOverride_ = false;
            }
            if (!hasKeyboardAxisOverride_) {
                rightAxis_ = static_cast<proto::RightAxis>(pkt.right_axis);
            }
            isPinching_ = (pkt.is_pinching != 0);
            spawnTriggerFromPkt_ = ((pkt.flags & proto::kSpawnTrigger) != 0);
            groupTriggerFromPkt_ = ((pkt.flags & proto::kGroupTrigger) != 0);
            activeGroupAction_ = pkt.hands[1].isPresent() ? pkt.hands[1].groupAction()
                                                          : proto::GroupAction::SelectToggle;
            pinchPx_ = pkt.pinch_px;
            if (pkt.active_shape < 4) {
                activeShape_ = static_cast<PrimitiveType>(pkt.active_shape);
            }

            // Smooth 3D cursor X/Y and apparent 2D palm size (sent in pkt.cursor_z)
            const float alpha = std::min(1.0f, dt * 18.0f);
            cursor3D_.x = cursor3D_.x + (pkt.cursor_x - cursor3D_.x) * alpha;
            cursor3D_.y = cursor3D_.y + (pkt.cursor_y - cursor3D_.y) * alpha;
            currentPalmSize_ = currentPalmSize_ + (pkt.cursor_z - currentPalmSize_) * alpha;
            cursor3D_.z = std::clamp(
                -(currentPalmSize_ - 0.18f) * (kZSensitivityCm / kCmPerWorldUnit), -4.5f, 4.5f);

            for (int s = 0; s < 2; ++s) {
                hands_[s] = pkt.hands[s];
            }
            if (leftMode_ != proto::LeftMode::Recolor && hands_[0].isPresent()) {
                activeColor_ = Color{
                    static_cast<unsigned char>( std::clamp(hands_[0].color[0] * 255.f, 0.f, 255.f) ),
                    static_cast<unsigned char>( std::clamp(hands_[0].color[1] * 255.f, 0.f, 255.f) ),
                    static_cast<unsigned char>( std::clamp(hands_[0].color[2] * 255.f, 0.f, 255.f) ),
                    255,
                };
            }
        } else {
            isPinching_ = false;
            spawnTriggerFromPkt_ = false;
            groupTriggerFromPkt_ = false;
            recolorSessionMask_ = 0;
            hands_[0].present = 0;
            hands_[1].present = 0;
        }

        // 3. Execute "Mode + Clutch" Anchor & Snapped Delta State Machine
        if (skipFrameAfterDialog_) {
            skipFrameAfterDialog_ = false;
            return;
        }
        runModeClutchStateMachine(dt);
    }

    void render(const Camera3D& cam) {
        const int sw = GetScreenWidth();
        const int sh = GetScreenHeight();

        BeginDrawing();
        if (viewportMode_ == ViewportMode::ArCamera) {
            // ---------------------------------------------------------------
            // MODE A: Fullscreen AR Camera Background + 3D Overlay
            // ---------------------------------------------------------------
            ClearBackground(Color{12, 16, 22, 255});
            drawCameraTextureFullscreen(sw, sh);
            draw2DHandSkeletonOverlay(0, 0, sw, sh);

            BeginMode3D(cam);
            draw3DScene(false);
            EndMode3D();
        } else {
            // ---------------------------------------------------------------
            // MODE B: Studio 3D Viewport ({24, 26, 32, 255}) + DrawGrid(20, 0.5f)
            //         + 3D Cursor Sphere + Bottom-Right Webcam PiP
            // ---------------------------------------------------------------
            ClearBackground(Color{24, 26, 32, 255});

            BeginMode3D(cam);
            DrawGrid(20, 0.5f);  // 20 slices, 0.5f unit (5 cm) spacing
            draw3DAxisOriginLines();
            draw3DScene(true);
            EndMode3D();

            // Picture-in-Picture (PiP) webcam + 2D hand skeleton in bottom-right corner
            const int pipW = 320;
            const int pipH = 240;
            const int pipX = sw - pipW - 20;
            const int pipY = sh - pipH - 20;
            drawStudioPiP(pipX, pipY, pipW, pipH);
        }

        // Top-Center Pill-Shaped Measurement HUD + Mode/Axis Telemetry Panels
        drawHudOverlays(cam, sw, sh);
        EndDrawing();
    }

private:
    PrimitiveLibrary primitives_{};
    std::vector<SceneObject> objects_;
    uint32_t nextId_ = 1;
    int nextGroupId_ = 1;
    int activeObjectIdx_ = -1;

    ViewportMode viewportMode_ = ViewportMode::Studio3D;
    proto::LeftMode leftMode_ = proto::LeftMode::Idle;
    proto::RightAxis rightAxis_ = proto::RightAxis::X;
    uint8_t lastTrackerRightAxis_ = 1;
    bool hasKeyboardAxisOverride_ = false;
    PrimitiveType activeShape_ = PrimitiveType::Cube;
    proto::GroupAction activeGroupAction_ = proto::GroupAction::SelectToggle;
    Color activeColor_{38, 230, 255, 255};
    uint8_t activeRecolorMask_ = 0;
    uint8_t recolorSessionMask_ = 0;
    std::string activeColorName_ = "CYAN";

    static constexpr float kDeleteHoldDurationSec = 1.50f;
    static constexpr float kDeleteCooldownDurationSec = 1.00f;

    bool isPinching_ = false;
    bool wasPinching_ = false;
    bool spawnTriggerFromPkt_ = false;
    bool groupTriggerFromPkt_ = false;
    float pinchPx_ = 999.f;
    float deleteHoldTimer_ = 0.0f;
    float deleteCooldownTimer_ = 0.0f;
    float currentPalmSize_ = 0.18f;
    float anchorPalmSize_ = 0.18f;
    Vector3 cursor3D_{0.0f, 0.5f, 0.0f};
    Vector3 anchorHandPos_{0.0f, 0.5f, 0.0f};
    std::array<proto::HandPacket, 2> hands_{};

    LinkStats linkStats_{};
    float trackerFps_ = 0.f;
    Texture2D camTex_{};
    uint64_t lastVideoVersion_ = 0;

    std::string measurementBadge_ = "IDLE [SAFE]";
    std::string statusLine_;
    bool dialogOpen_ = false;
    bool skipFrameAfterDialog_ = false;
    std::string toastMessage_;
    float toastTimer_ = 0.0f;
    bool toastIsError_ = false;

    void showToast(const std::string& msg, bool isError = false, float duration = 4.0f) {
        toastMessage_ = msg;
        toastIsError_ = isError;
        toastTimer_ = duration;
        statusLine_ = msg;
    }

    static bool hasExtensionCI(const std::string& path, const char* ext) {
        const std::string suffix(ext);
        if (path.size() < suffix.size()) return false;
        const size_t start = path.size() - suffix.size();
        for (size_t i = 0; i < suffix.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(path[start + i])) !=
                std::tolower(static_cast<unsigned char>(suffix[i]))) {
                return false;
            }
        }
        return true;
    }

    bool saveProjectFile(const std::string& filepath) {
        if (SaveProjectToFile(filepath, objects_, nextGroupId_)) {
            showToast("PROJECT SAVED -> " + extractFilename(filepath), false);
            return true;
        }
        showToast("ERROR: FAILED TO SAVE " + extractFilename(filepath), true);
        return false;
    }

    bool loadProjectFile(const std::string& filepath) {
        std::vector<SceneObject> loaded;
        int loadedNextGroupId = 1;
        int missingObjCount = 0;
        if (!LoadProjectFromFile(filepath, loaded, loadedNextGroupId, missingObjCount)) {
            showToast("LOAD FAILED: " + extractFilename(filepath), true);
            return false;
        }
        objects_.clear();
        objects_ = std::move(loaded);
        nextGroupId_ = loadedNextGroupId;
        nextId_ = static_cast<uint32_t>(objects_.size()) + 1;
        deleteHoldTimer_ = 0.0f;
        deleteCooldownTimer_ = 0.0f;
        activeObjectIdx_ = -1;
        for (size_t i = 0; i < objects_.size(); ++i) {
            if (objects_[i].isSelected) {
                activeObjectIdx_ = static_cast<int>(i);
                break;
            }
        }
        if (activeObjectIdx_ < 0 && !objects_.empty()) {
            activeObjectIdx_ = 0;
        }
        std::string msg = "PROJECT LOADED -> " + extractFilename(filepath) +
                          " (" + std::to_string(objects_.size()) + " objects)";
        if (missingObjCount > 0) {
            msg += " [" + std::to_string(missingObjCount) + " missing .OBJ skipped]";
        }
        showToast(msg, false);
        return true;
    }

    template <typename Fn>
    void runModalDialogGuard(UDPReceiver& udp, Fn&& fn) {
        dialogOpen_ = true;
        isPinching_ = false;
        wasPinching_ = false;
        spawnTriggerFromPkt_ = false;
        groupTriggerFromPkt_ = false;
        deleteHoldTimer_ = 0.0f;
        fn();
        proto::TelemetryPacket discard{};
        udp.latest(discard);
        skipFrameAfterDialog_ = true;
        dialogOpen_ = false;
    }

    bool importOBJFile(const std::string& filepath) {
        auto asset = LoadNormalizedOBJAsset(filepath, 2.0f);
        if (!asset) {
            showToast("IMPORT FAILED: " + extractFilename(filepath), true);
            return false;
        }
        SceneObject obj{};
        obj.id = nextId_++;
        obj.type = PRIMITIVE_IMPORTED_OBJ;
        obj.transform.position = {0.0f, 0.0f, 0.0f};
        obj.transform.scale = {1.0f, 1.0f, 1.0f};
        obj.transform.rotationDeg = {0.0f, 0.0f, 0.0f};
        obj.transform.rotMatrix = MatrixIdentity();
        obj.anchorTransform = obj.transform;
        obj.color = Color{215, 225, 240, 255};
        obj.groupId = -1;
        obj.isSelected = false;
        obj.importedModel = asset;
        objects_.push_back(std::move(obj));
        activeObjectIdx_ = static_cast<int>(objects_.size()) - 1;
        showToast("IMPORTED: " + asset->displayName, false);
        return true;
    }

    std::vector<int> collectGroupExportTargets(bool& outIsGroup) const {
        std::vector<int> indices;
        outIsGroup = false;
        if (activeObjectIdx_ >= 0 && activeObjectIdx_ < static_cast<int>(objects_.size()) &&
            objects_[activeObjectIdx_].groupId >= 0) {
            outIsGroup = true;
            return collectGroupMembers(activeObjectIdx_);
        }
        const int selCount = countSelectedObjects();
        if (selCount >= 2) {
            outIsGroup = true;
            for (size_t i = 0; i < objects_.size(); ++i) {
                if (objects_[i].isSelected) {
                    for (int m : collectGroupMembers(static_cast<int>(i))) indices.push_back(m);
                }
            }
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
            return indices;
        }
        if (activeObjectIdx_ >= 0 && activeObjectIdx_ < static_cast<int>(objects_.size())) {
            indices.push_back(activeObjectIdx_);
            return indices;
        }
        if (selCount == 1) {
            for (size_t i = 0; i < objects_.size(); ++i) {
                if (objects_[i].isSelected) return collectGroupMembers(static_cast<int>(i));
            }
        }
        return indices;
    }

    // --- Helper: find object closest to 3D cursor --------------------------
    // Measures 2D (X, Y) distance + 0.25-weighted Z distance so objects shifted
    // forward/backward along the Z-axis remain easy to hover and select.
    int findClosestObjectToCursor(Vector3 pos, float maxDist = 1.35f) const {
        int best = -1;
        float bestD = maxDist;
        for (size_t i = 0; i < objects_.size(); ++i) {
            const Vector3 p = objects_[i].transform.position;
            const float dx = p.x - pos.x;
            const float dy = p.y - pos.y;
            const float dz = p.z - pos.z;
            const float d = std::hypot(dx, dy) + 0.25f * std::abs(dz);
            if (d < bestD) {
                bestD = d;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    static bool isTargetLockedMode(proto::LeftMode m) {
        return m == proto::LeftMode::Move ||
               m == proto::LeftMode::Resize ||
               m == proto::LeftMode::Rotate ||
               m == proto::LeftMode::Recolor ||
               m == proto::LeftMode::Delete;
    }

    // Returns either `{idx}` (if ungrouped) or all members sharing `objects_[idx].groupId`.
    // Never scans `isSelected` across unrelated objects.
    std::vector<int> collectGroupMembers(int idx) const {
        std::vector<int> indices;
        if (idx < 0 || idx >= static_cast<int>(objects_.size())) return indices;
        const int gid = objects_[idx].groupId;
        if (gid >= 0) {
            for (size_t i = 0; i < objects_.size(); ++i) {
                if (objects_[i].groupId == gid) indices.push_back(static_cast<int>(i));
            }
        } else {
            indices.push_back(idx);
        }
        return indices;
    }

    // Returns all target object indices for transform/recolor/delete operations,
    // propagating cleanly across shared `groupId` or active `GROUP` multi-selection.
    std::vector<int> collectTransformTargets() const {
        std::vector<int> indices;
        auto addWithGroup = [&](int baseIdx) {
            for (int m : collectGroupMembers(baseIdx)) indices.push_back(m);
        };
        if (activeObjectIdx_ >= 0 && activeObjectIdx_ < static_cast<int>(objects_.size())) {
            addWithGroup(activeObjectIdx_);
            if (objects_[activeObjectIdx_].isSelected) {
                for (size_t i = 0; i < objects_.size(); ++i) {
                    if (objects_[i].isSelected) addWithGroup(static_cast<int>(i));
                }
            }
        } else {
            for (size_t i = 0; i < objects_.size(); ++i) {
                if (objects_[i].isSelected) addWithGroup(static_cast<int>(i));
            }
        }
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        return indices;
    }

    bool isObjectInActiveTarget(size_t idx) const {
        const std::vector<int> targets = collectTransformTargets();
        return std::find(targets.begin(), targets.end(), static_cast<int>(idx)) != targets.end();
    }

    bool isObjectInActiveHover(size_t idx) const {
        if (activeObjectIdx_ < 0 || activeObjectIdx_ >= static_cast<int>(objects_.size()) ||
            idx >= objects_.size()) {
            return false;
        }
        const std::vector<int> hovered = collectGroupMembers(activeObjectIdx_);
        return std::find(hovered.begin(), hovered.end(), static_cast<int>(idx)) != hovered.end();
    }

    static void drawHoverReticle2D(Vector2 center, Color col) {
        constexpr float kArcSpans[4][2] = {
            {15.0f, 75.0f}, {105.0f, 165.0f}, {195.0f, 255.0f}, {285.0f, 345.0f}
        };
        for (const auto& span : kArcSpans) {
            DrawRing(center, 25.0f, 28.0f, span[0], span[1], 12, col);
        }
        constexpr float kTickRad = 30.0f;
        constexpr float kTickOuter = 36.0f;
        DrawLineEx({center.x - kTickOuter, center.y}, {center.x - kTickRad, center.y}, 2.0f, col);
        DrawLineEx({center.x + kTickRad, center.y},   {center.x + kTickOuter, center.y}, 2.0f, col);
        DrawLineEx({center.x, center.y - kTickOuter}, {center.x, center.y - kTickRad}, 2.0f, col);
        DrawLineEx({center.x, center.y + kTickRad},   {center.x, center.y + kTickOuter}, 2.0f, col);
    }

    static void drawLockedHighlight2D(Vector2 center, Color col, bool activePinch) {
        DrawRing(center, 27.0f, activePinch ? 32.5f : 31.0f, 0.0f, 360.0f, 48, col);
        constexpr float kOffset = 35.0f;
        constexpr float kArm = 11.0f;
        const float thick = activePinch ? 3.2f : 2.6f;
        for (int sx : {-1, 1}) {
            for (int sy : {-1, 1}) {
                const Vector2 corner{center.x + sx * kOffset, center.y + sy * kOffset};
                DrawLineEx(corner, {corner.x - sx * kArm, corner.y}, thick, col);
                DrawLineEx(corner, {corner.x, corner.y - sy * kArm}, thick, col);
            }
        }
    }

    int countSelectedObjects() const {
        return static_cast<int>(std::count_if(
            objects_.begin(), objects_.end(),
            [](const SceneObject& o) { return o.isSelected; }));
    }

    BoundingBox computeWorldBoundingBox(const SceneObject& obj, float pad = 0.08f) const {
        BoundingBox localBox{{-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}};
        if (obj.type == PrimitiveType::ImportedObj && obj.importedModel) {
            localBox = obj.importedModel->normalizedLocalBox;
        } else {
            const Model& mdl = primitives_.model(obj.type);
            if (mdl.meshCount > 0) localBox = GetMeshBoundingBox(mdl.meshes[0]);
        }
        const Matrix mat = obj.matrix();
        const Vector3 corners[8] = {
            {localBox.min.x, localBox.min.y, localBox.min.z},
            {localBox.max.x, localBox.min.y, localBox.min.z},
            {localBox.min.x, localBox.max.y, localBox.min.z},
            {localBox.max.x, localBox.max.y, localBox.min.z},
            {localBox.min.x, localBox.min.y, localBox.max.z},
            {localBox.max.x, localBox.min.y, localBox.max.z},
            {localBox.min.x, localBox.max.y, localBox.max.z},
            {localBox.max.x, localBox.max.y, localBox.max.z},
        };
        const Vector3 first = Vector3Transform(corners[0], mat);
        BoundingBox world{first, first};
        for (int c = 1; c < 8; ++c) {
            const Vector3 p = Vector3Transform(corners[c], mat);
            world.min.x = std::min(world.min.x, p.x);
            world.min.y = std::min(world.min.y, p.y);
            world.min.z = std::min(world.min.z, p.z);
            world.max.x = std::max(world.max.x, p.x);
            world.max.y = std::max(world.max.y, p.y);
            world.max.z = std::max(world.max.z, p.z);
        }
        world.min.x -= pad;
        world.min.y -= pad;
        world.min.z -= pad;
        world.max.x += pad;
        world.max.y += pad;
        world.max.z += pad;
        return world;
    }

    void executeGroupModeAction(proto::GroupAction action, int hovIdx) {
        switch (action) {
            case proto::GroupAction::SelectToggle: {
                const int targetIdx =
                    (hovIdx >= 0) ? hovIdx : findClosestObjectToCursor(cursor3D_);
                if (targetIdx < 0 || targetIdx >= static_cast<int>(objects_.size())) {
                    statusLine_ = "GROUP SELECT: HOVER CURSOR NEAR AN OBJECT FIRST";
                    return;
                }
                const bool nextSel = !objects_[targetIdx].isSelected;
                const std::vector<int> toggleSet = collectGroupMembers(targetIdx);
                for (int idx : toggleSet) {
                    objects_[idx].isSelected = nextSel;
                }
                activeObjectIdx_ = targetIdx;
                statusLine_ = std::string(nextSel ? "SELECTED " : "DESELECTED ") +
                              objects_[targetIdx].displayLabel() + " #" +
                              std::to_string(objects_[targetIdx].id) + " (" +
                              std::to_string(countSelectedObjects()) + " SELECTED)";
                break;
            }
            case proto::GroupAction::Group: {
                const int selCount = countSelectedObjects();
                if (selCount < 2) {
                    statusLine_ = "GROUP NEEDS >= 2 SELECTED OBJECTS (INDEX PINCH TO SELECT)";
                    return;
                }
                const int gid = nextGroupId_++;
                for (size_t i = 0; i < objects_.size(); ++i) {
                    if (objects_[i].isSelected) {
                        objects_[i].groupId = gid;
                        if (activeObjectIdx_ < 0 ||
                            activeObjectIdx_ >= static_cast<int>(objects_.size()) ||
                            !objects_[activeObjectIdx_].isSelected) {
                            activeObjectIdx_ = static_cast<int>(i);
                        }
                    }
                }
                statusLine_ = "GROUPED " + std::to_string(selCount) +
                              " OBJECTS INTO GROUP #" + std::to_string(gid);
                break;
            }
            case proto::GroupAction::Ungroup: {
                std::vector<int> gids;
                for (const auto& o : objects_) {
                    if (o.isSelected && o.groupId >= 0) {
                        if (std::find(gids.begin(), gids.end(), o.groupId) == gids.end()) {
                            gids.push_back(o.groupId);
                        }
                    }
                }
                int ungrouped = 0;
                if (!gids.empty()) {
                    for (auto& o : objects_) {
                        const bool inTargetGid =
                            (o.groupId >= 0) &&
                            (std::find(gids.begin(), gids.end(), o.groupId) != gids.end());
                        if (o.isSelected || inTargetGid) {
                            if (o.groupId >= 0) {
                                o.groupId = -1;
                                ++ungrouped;
                            }
                        }
                    }
                } else {
                    const int targetIdx =
                        (hovIdx >= 0) ? hovIdx : findClosestObjectToCursor(cursor3D_);
                    if (targetIdx >= 0 && targetIdx < static_cast<int>(objects_.size()) &&
                        objects_[targetIdx].groupId >= 0) {
                        const int hovGid = objects_[targetIdx].groupId;
                        for (auto& o : objects_) {
                            if (o.groupId == hovGid) {
                                o.groupId = -1;
                                ++ungrouped;
                            }
                        }
                    }
                }
                statusLine_ = (ungrouped > 0)
                    ? ("UNGROUPED " + std::to_string(ungrouped) + " OBJECTS")
                    : "NO GROUPED OBJECTS TO UNGROUP";
                break;
            }
            case proto::GroupAction::ClearSelection: {
                for (auto& o : objects_) o.isSelected = false;
                statusLine_ = "CLEARED SELECTION";
                break;
            }
        }
    }

    // --- Core "Mode + Clutch" Anchor & Snapped Delta State Machine ---------
    void runModeClutchStateMachine(float dt) {
        if (deleteCooldownTimer_ > 0.0f) {
            deleteCooldownTimer_ = std::max(0.0f, deleteCooldownTimer_ - dt);
        }

        for (auto& o : objects_) o.isHovered = false;
        int hovIdx = -1;
        if (leftMode_ == proto::LeftMode::Idle && hands_[1].isPresent()) {
            hovIdx = findClosestObjectToCursor(cursor3D_);
            if (hovIdx >= 0) {
                activeObjectIdx_ = hovIdx;
                for (int idx : collectGroupMembers(activeObjectIdx_)) {
                    objects_[idx].isHovered = true;
                }
            }
        } else if (leftMode_ == proto::LeftMode::Group && hands_[1].isPresent()) {
            hovIdx = findClosestObjectToCursor(cursor3D_);
            if (hovIdx >= 0) {
                for (int idx : collectGroupMembers(hovIdx)) {
                    objects_[idx].isHovered = true;
                }
            }
        }

        const bool wasNotPinching = !wasPinching_;
        const bool risingEdge = isPinching_ && wasNotPinching;
        const bool fallingEdge = !isPinching_ && wasPinching_;
        const bool spawnTrigger =
            (leftMode_ == proto::LeftMode::Spawn) &&
            (risingEdge || (spawnTriggerFromPkt_ && wasNotPinching));
        const bool groupTrigger =
            (leftMode_ == proto::LeftMode::Group) &&
            (risingEdge || (groupTriggerFromPkt_ && wasNotPinching));
        wasPinching_ = isPinching_;

        if (leftMode_ != proto::LeftMode::Recolor || !isPinching_) {
            recolorSessionMask_ = 0;
        }

        const char* axStr = axisName(rightAxis_);

        // 1. Single-Shot Rising-Edge Spawn/Group OR Anchor Snapshot on Pinch Clutch
        if (spawnTrigger) {
            SceneObject& created = spawnObject(activeShape_, cursor3D_, activeColor_);
            activeObjectIdx_ = static_cast<int>(objects_.size()) - 1;
            spawnTriggerFromPkt_ = false;
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "SPAWNED [%s] @ (%.0f, %.0f, %.0f) cm // RELEASE (>48px) FOR NEXT",
                          primitiveName(activeShape_),
                          created.transform.position.x * kCmPerWorldUnit,
                          created.transform.position.y * kCmPerWorldUnit,
                          created.transform.position.z * kCmPerWorldUnit);
            measurementBadge_ = buf;
        } else if (groupTrigger) {
            executeGroupModeAction(activeGroupAction_, hovIdx);
            groupTriggerFromPkt_ = false;
        } else if (risingEdge) {
            anchorHandPos_ = cursor3D_;
            anchorPalmSize_ = currentPalmSize_;

            // Target switching is strictly disabled outside IDLE/GROUP/SPAWN;
            // snapshot every object's `anchorTransform` at pinch engagement.
            for (auto& o : objects_) {
                o.anchorTransform = o.transform;
            }
        }

        // 2. While `isPinching_` is Held: Compute Axis-Locked Snapped Delta from Anchor
        const std::vector<int> targets = collectTransformTargets();
        if (isPinching_ && !targets.empty()) {
            const Vector3 rawDelta = Vector3Subtract(cursor3D_, anchorHandPos_);
            const float rawZDeltaCm = -(currentPalmSize_ - anchorPalmSize_) * kZSensitivityCm;
            float axisRawDelta = 0.0f;
            if (rightAxis_ == proto::RightAxis::X) axisRawDelta = rawDelta.x;
            else if (rightAxis_ == proto::RightAxis::Y) axisRawDelta = rawDelta.y;
            else if (rightAxis_ == proto::RightAxis::Z) axisRawDelta = rawZDeltaCm / kCmPerWorldUnit;

            char badge[128];

            switch (leftMode_) {
                case proto::LeftMode::Idle: {
                    std::snprintf(badge, sizeof(badge), "IDLE [LOCKED] // PINCH HELD");
                    measurementBadge_ = badge;
                    break;
                }
                case proto::LeftMode::Move: {
                    // X/Y snap to 0.5 world units (5.0 cm); Z snaps to 1.0 cm from apparent palm scale
                    const float snappedCm = (rightAxis_ == proto::RightAxis::Z)
                                                ? snapToStep(rawZDeltaCm, kSnapMoveStepZCm)
                                                : (snapToStep(axisRawDelta, kSnapMoveStepUnits) * kCmPerWorldUnit);
                    const float snappedUnits = snappedCm / kCmPerWorldUnit;
                    for (int idx : targets) {
                        objects_[idx].transform = objects_[idx].anchorTransform;
                        if (rightAxis_ == proto::RightAxis::X)
                            objects_[idx].transform.position.x += snappedUnits;
                        else if (rightAxis_ == proto::RightAxis::Y)
                            objects_[idx].transform.position.y += snappedUnits;
                        else if (rightAxis_ == proto::RightAxis::Z)
                            objects_[idx].transform.position.z += snappedUnits;
                    }
                    std::snprintf(badge, sizeof(badge), "MOVE [%s]: %+.1f cm", axStr, snappedCm);
                    measurementBadge_ = badge;
                    break;
                }
                case proto::LeftMode::Resize: {
                    // Snap scale delta to nearest 0.5 world units (= 5.0 cm)
                    const float snappedUnits = snapToStep(axisRawDelta, kSnapScaleStepUnits);
                    const float snappedCm = snappedUnits * kCmPerWorldUnit;
                    for (int idx : targets) {
                        objects_[idx].transform = objects_[idx].anchorTransform;
                        if (rightAxis_ == proto::RightAxis::X) {
                            objects_[idx].transform.scale.x = std::max(
                                0.5f, objects_[idx].anchorTransform.scale.x + snappedUnits);
                        } else if (rightAxis_ == proto::RightAxis::Y) {
                            objects_[idx].transform.scale.y = std::max(
                                0.5f, objects_[idx].anchorTransform.scale.y + snappedUnits);
                        } else if (rightAxis_ == proto::RightAxis::Z) {
                            objects_[idx].transform.scale.z = std::max(
                                0.5f, objects_[idx].anchorTransform.scale.z + snappedUnits);
                        }
                    }
                    std::snprintf(badge, sizeof(badge), "RESIZE [%s]: %+.1f cm", axStr, snappedCm);
                    measurementBadge_ = badge;
                    break;
                }
                case proto::LeftMode::Rotate: {
                    // Use intuitive hand axis displacement mapped 1-to-1 to World X / Y / Z
                    float rotDriver = axisRawDelta;
                    if (rightAxis_ == proto::RightAxis::X) {
                        rotDriver = (std::abs(rawDelta.y) > std::abs(rawDelta.x)) ? -rawDelta.y : rawDelta.x;
                    } else if (rightAxis_ == proto::RightAxis::Y) {
                        rotDriver = (std::abs(rawDelta.x) > std::abs(rawDelta.y)) ? rawDelta.x : rawDelta.y;
                    } else if (rightAxis_ == proto::RightAxis::Z) {
                        rotDriver = (std::abs(rawDelta.x) > std::abs(axisRawDelta)) ? -rawDelta.x : axisRawDelta;
                    }
                    const float rawDeg = rotDriver * 45.0f;
                    const float snappedDeg = snapToStep(rawDeg, kSnapRotateStepDeg);
                    Vector3 axisVec{1.0f, 0.0f, 0.0f};
                    if (rightAxis_ == proto::RightAxis::Y) axisVec = {0.0f, 1.0f, 0.0f};
                    else if (rightAxis_ == proto::RightAxis::Z) axisVec = {0.0f, 0.0f, 1.0f};
                    const Matrix deltaRot = MatrixRotate(axisVec, snappedDeg * DEG2RAD);
                    for (int idx : targets) {
                        objects_[idx].transform = objects_[idx].anchorTransform;
                        objects_[idx].transform.rotMatrix =
                            MatrixMultiply(objects_[idx].anchorTransform.rotMatrix, deltaRot);
                        if (rightAxis_ == proto::RightAxis::X)
                            objects_[idx].transform.rotationDeg.x += snappedDeg;
                        else if (rightAxis_ == proto::RightAxis::Y)
                            objects_[idx].transform.rotationDeg.y += snappedDeg;
                        else if (rightAxis_ == proto::RightAxis::Z)
                            objects_[idx].transform.rotationDeg.z += snappedDeg;
                    }
                    std::snprintf(badge, sizeof(badge), "ROTATE [%s]: %+.0f deg", axStr, snappedDeg);
                    measurementBadge_ = badge;
                    break;
                }
                case proto::LeftMode::Recolor: {
                    const uint8_t pktMask = hands_[1].isPresent() ? (hands_[1].contactMask & 0x07) : 0;
                    recolorSessionMask_ |= pktMask;
                    activeRecolorMask_ = recolorSessionMask_;
                    if (activeRecolorMask_ != 0) {
                        activeColor_ = recolorColorFromMask(activeRecolorMask_);
                        activeColorName_ = recolorNameFromMask(activeRecolorMask_);
                        for (int idx : targets) {
                            objects_[idx].color = activeColor_;
                        }
                    }
                    std::snprintf(badge, sizeof(badge), "RECOLOR: %s (%u, %u, %u)",
                                  activeColorName_.c_str(), activeColor_.r, activeColor_.g,
                                  activeColor_.b);
                    measurementBadge_ = badge;
                    break;
                }
                case proto::LeftMode::Group:
                case proto::LeftMode::Spawn:
                case proto::LeftMode::Delete:
                default:
                    break;
            }
        }

        // 3. Falling Edge of Pinch Clutch: Bake Transform so Object Stays Frozen in Place
        if (fallingEdge) {
            for (auto& o : objects_) {
                o.anchorTransform = o.transform;
            }
        }

        // 3b. Safe Timed Deletion (1.5s Thumbs-Down Hold + 1.0s Cooldown, group-aware)
        if (leftMode_ == proto::LeftMode::Delete) {
            std::vector<int> deleteTargets = collectTransformTargets();
            if (deleteCooldownTimer_ > 0.0f) {
                deleteHoldTimer_ = 0.0f;
            } else if (!deleteTargets.empty()) {
                deleteHoldTimer_ += dt;
                if (deleteHoldTimer_ >= kDeleteHoldDurationSec) {
                    const int refIdx = (activeObjectIdx_ >= 0 &&
                                        activeObjectIdx_ < static_cast<int>(objects_.size()))
                                           ? activeObjectIdx_
                                           : deleteTargets[0];
                    const int deletedGroupId = objects_[refIdx].groupId;
                    const uint32_t deletedId = objects_[refIdx].id;
                    const std::string deletedLabel = objects_[refIdx].displayLabel();
                    const size_t deletedCount = deleteTargets.size();

                    for (auto it = deleteTargets.rbegin(); it != deleteTargets.rend(); ++it) {
                        objects_.erase(objects_.begin() + *it);
                    }
                    deleteHoldTimer_ = 0.0f;
                    deleteCooldownTimer_ = kDeleteCooldownDurationSec;

                    if (objects_.empty()) {
                        activeObjectIdx_ = -1;
                    } else {
                        activeObjectIdx_ = std::min(
                            std::max(activeObjectIdx_, 0),
                            static_cast<int>(objects_.size()) - 1);
                    }

                    char delMsg[128];
                    if (deletedCount > 1) {
                        if (deletedGroupId >= 0) {
                            std::snprintf(delMsg, sizeof(delMsg),
                                          "DELETED GROUP #%d (%zu OBJECTS) // COOLDOWN 1.0s",
                                          deletedGroupId, deletedCount);
                        } else {
                            std::snprintf(delMsg, sizeof(delMsg),
                                          "DELETED %zu SELECTED OBJECTS // COOLDOWN 1.0s",
                                          deletedCount);
                        }
                    } else {
                        std::snprintf(delMsg, sizeof(delMsg), "DELETED [%s #%u] // COOLDOWN 1.0s",
                                      deletedLabel.c_str(), deletedId);
                    }
                    statusLine_ = delMsg;
                }
            } else {
                deleteHoldTimer_ = 0.0f;
            }
        } else {
            deleteHoldTimer_ = 0.0f;
        }

        // 4. Mode / Hover / Locked Color / Delete HUD Readout
        char badge[160];
        if (leftMode_ == proto::LeftMode::Delete) {
            if (deleteCooldownTimer_ > 0.0f) {
                std::snprintf(badge, sizeof(badge), "DELETE COOLDOWN (%.1fs)", deleteCooldownTimer_);
            } else {
                const size_t targetCount = collectTransformTargets().size();
                if (targetCount > 1) {
                    std::snprintf(badge, sizeof(badge), "DELETING GROUP (%zu)... %.1fs",
                                  targetCount, deleteHoldTimer_);
                } else if (targetCount == 1) {
                    std::snprintf(badge, sizeof(badge), "DELETING OBJECT... %.1fs", deleteHoldTimer_);
                } else {
                    std::snprintf(badge, sizeof(badge), "DELETE [NO OBJECT SELECTED]");
                }
            }
            measurementBadge_ = badge;
        } else if (leftMode_ == proto::LeftMode::Group) {
            std::snprintf(badge, sizeof(badge),
                          "GROUP MODE [%d Selected] // Index=Select, Middle=Group, Ring=Ungroup, Pinky=Clear",
                          countSelectedObjects());
            measurementBadge_ = badge;
        } else if (!isPinching_) {
            if (leftMode_ == proto::LeftMode::Idle) {
                if (activeObjectIdx_ >= 0 && activeObjectIdx_ < static_cast<int>(objects_.size())) {
                    const SceneObject& sel = objects_[activeObjectIdx_];
                    if (sel.groupId >= 0) {
                        std::snprintf(badge, sizeof(badge),
                                      "IDLE [AIM TO SELECT] | TARGET: GRP #%d | AXIS: %s",
                                      sel.groupId, axStr);
                    } else {
                        std::snprintf(badge, sizeof(badge),
                                      "IDLE [AIM TO SELECT] | TARGET: %s #%u | AXIS: %s",
                                      sel.displayLabel().c_str(), sel.id, axStr);
                    }
                } else {
                    std::snprintf(badge, sizeof(badge), "IDLE [AIM TO SELECT] | NO TARGET | AXIS: %s",
                                  axStr);
                }
            } else if (leftMode_ == proto::LeftMode::Recolor) {
                activeRecolorMask_ = 0;
                recolorSessionMask_ = 0;
                std::snprintf(badge, sizeof(badge), "RECOLOR [LOCKED]: %s (%u, %u, %u)",
                              activeColorName_.c_str(), activeColor_.r, activeColor_.g,
                              activeColor_.b);
            } else if (leftMode_ == proto::LeftMode::Spawn) {
                std::snprintf(badge, sizeof(badge),
                              "SPAWN [%s] // Thumb+Index=Cube, Middle=Capsule, Ring=Cyl, Pinky=Ring",
                              primitiveName(activeShape_));
            } else {
                std::snprintf(badge, sizeof(badge), "%s [%s]: +0.0 %s (PINCH <32px TO ENGAGE)",
                              modeName(leftMode_), axStr,
                              leftMode_ == proto::LeftMode::Rotate ? "deg" : "cm");
            }
            measurementBadge_ = badge;
        }
    }

    // --- 3D Scene Rendering (Models, Selection Bounds, Ghost Preview & 3D Cursor) ---
    void draw3DAxisOriginLines() const {
        DrawLine3D({-5.0f, 0.01f, 0.0f}, {5.0f, 0.01f, 0.0f}, Color{220, 60, 70, 180});
        DrawLine3D({0.0f, 0.0f, 0.0f},   {0.0f, 5.0f, 0.0f},  Color{60, 220, 95, 180});
        DrawLine3D({0.0f, 0.01f, -5.0f}, {0.0f, 0.01f, 5.0f}, Color{70, 145, 245, 180});
    }

    // Glowing 3-layer 3D guide line directly through the active/hovered object
    // matching the locked right_axis (Red=X, Green=Y, Blue=Z).
    static void drawGlowingAxisGuide(Vector3 center, proto::RightAxis axis, bool activePinch) {
        Vector3 dir{1.0f, 0.0f, 0.0f};
        Color col{255, 65, 75, 255};  // Red for X
        if (axis == proto::RightAxis::Y) {
            dir = {0.0f, 1.0f, 0.0f};
            col = Color{65, 255, 110, 255};  // Green for Y
        } else if (axis == proto::RightAxis::Z) {
            dir = {0.0f, 0.0f, 1.0f};
            col = Color{65, 165, 255, 255};  // Blue for Z
        }

        constexpr float kHalfLen = 4.5f;
        const Vector3 a = Vector3Subtract(center, Vector3Scale(dir, kHalfLen));
        const Vector3 b = Vector3Add(center, Vector3Scale(dir, kHalfLen));

        // 1. Outer translucent glow cylinder
        DrawCylinderEx(a, b, activePinch ? 0.070f : 0.055f, activePinch ? 0.070f : 0.055f, 10,
                       Fade(col, activePinch ? 0.36f : 0.28f));
        // 2. Inner neon core cylinder
        DrawCylinderEx(a, b, activePinch ? 0.026f : 0.020f, activePinch ? 0.026f : 0.020f, 8, col);
        // 3. Center bright white core line + tick marks every 1.0 world unit (10 cm)
        DrawLine3D(a, b, WHITE);

        Vector3 perp = (axis == proto::RightAxis::Y) ? Vector3{1.0f, 0.0f, 0.0f}
                                                     : Vector3{0.0f, 1.0f, 0.0f};
        for (int t = -4; t <= 4; ++t) {
            if (t == 0) continue;
            const Vector3 tickCenter = Vector3Add(center, Vector3Scale(dir, static_cast<float>(t)));
            const float tickHalf = (t % 2 == 0) ? 0.14f : 0.08f;
            DrawLine3D(Vector3Subtract(tickCenter, Vector3Scale(perp, tickHalf)),
                       Vector3Add(tickCenter, Vector3Scale(perp, tickHalf)), col);
        }
    }

    void draw3DScene(bool drawCursorSphere) {
        const bool lockedMode = isTargetLockedMode(leftMode_);

        // 1. Render all permanent SceneObjects
        for (size_t i = 0; i < objects_.size(); ++i) {
            const SceneObject& obj = objects_[i];
            const bool activeTarget = isObjectInActiveTarget(i);
            const bool idleHoverTarget =
                (leftMode_ == proto::LeftMode::Idle) && isObjectInActiveHover(i);
            const bool isImported = (obj.type == PrimitiveType::ImportedObj &&
                                     obj.importedModel && obj.importedModel->hasModel);
            Model& mdl = isImported ? obj.importedModel->model : primitives_.model(obj.type);
            mdl.transform = isImported
                ? MatrixMultiply(obj.importedModel->normMatrix(), obj.matrix())
                : obj.matrix();

            Color fillCol = obj.color;
            fillCol.a = 210;
            DrawModel(mdl, {0.f, 0.f, 0.f}, 1.0f, fillCol);

            Color wireCol = WHITE;
            if (lockedMode && activeTarget) {
                if (leftMode_ == proto::LeftMode::Delete) wireCol = Color{255, 75, 85, 255};
                else if (leftMode_ == proto::LeftMode::Recolor) wireCol = activeColor_;
                else wireCol = Color{255, 230, 70, 255};
            } else if (obj.isHovered) {
                wireCol = Color{120, 245, 255, 220};
            } else if (obj.isSelected) {
                wireCol = Color{255, 235, 80, 220};
            } else {
                wireCol = Color{
                    static_cast<unsigned char>(std::min(255, obj.color.r + 50)),
                    static_cast<unsigned char>(std::min(255, obj.color.g + 50)),
                    static_cast<unsigned char>(std::min(255, obj.color.b + 50)),
                    255,
                };
            }
            DrawModelWires(mdl, {0.f, 0.f, 0.f}, 1.0f, wireCol);
            mdl.transform = MatrixIdentity();

            if (obj.groupId >= 0) {
                if (lockedMode && activeTarget) {
                    const Color lockCol = (leftMode_ == proto::LeftMode::Delete)
                                              ? Color{255, 70, 80, 255}
                                              : Color{65, 255, 170, 255};
                    DrawBoundingBox(computeWorldBoundingBox(obj, 0.07f), lockCol);
                    DrawBoundingBox(computeWorldBoundingBox(obj, 0.11f), lockCol);
                } else if (idleHoverTarget) {
                    DrawBoundingBox(computeWorldBoundingBox(obj), Color{45, 225, 140, 155});
                } else {
                    DrawBoundingBox(computeWorldBoundingBox(obj),
                                    obj.isSelected ? Color{60, 255, 165, 255}
                                                   : Color{30, 215, 125, 210});
                }
            } else {
                if (lockedMode && activeTarget) {
                    const Color lockCol = (leftMode_ == proto::LeftMode::Delete)
                                              ? Color{255, 70, 80, 255}
                                              : (isPinching_ ? Color{255, 225, 65, 255}
                                                             : Color{65, 245, 255, 255});
                    DrawBoundingBox(computeWorldBoundingBox(obj, 0.07f), lockCol);
                    DrawBoundingBox(computeWorldBoundingBox(obj, 0.11f), lockCol);
                } else if (obj.isSelected) {
                    DrawBoundingBox(computeWorldBoundingBox(obj), Color{50, 235, 255, 255});
                } else if (idleHoverTarget) {
                    DrawBoundingBox(computeWorldBoundingBox(obj), Color{90, 235, 255, 145});
                }
            }

            // Draw glowing 3D guide line through the active primary target object
            if (leftMode_ != proto::LeftMode::Group &&
                leftMode_ != proto::LeftMode::Spawn &&
                static_cast<int>(i) == activeObjectIdx_) {
                drawGlowingAxisGuide(obj.transform.position, rightAxis_, isPinching_);
            }
        }

        // 1b. Group Connecting Lines Between Centers (`groupId != -1`)
        std::vector<int> seenGroups;
        for (const auto& obj : objects_) {
            if (obj.groupId < 0) continue;
            if (std::find(seenGroups.begin(), seenGroups.end(), obj.groupId) != seenGroups.end()) continue;
            seenGroups.push_back(obj.groupId);

            std::vector<Vector3> centers;
            for (const auto& m : objects_) {
                if (m.groupId == obj.groupId) centers.push_back(m.transform.position);
            }
            if (centers.size() >= 2) {
                const Color linkCol{40, 235, 145, 175};
                for (size_t k = 0; k + 1 < centers.size(); ++k) {
                    DrawLine3D(centers[k], centers[k + 1], linkCol);
                }
                if (centers.size() >= 3) {
                    DrawLine3D(centers.back(), centers.front(), linkCol);
                }
                for (const Vector3& c : centers) {
                    DrawSphere(c, 0.045f, Color{50, 245, 155, 210});
                }
            }
        }

        // 2. Ghost Preview in `SPAWN` mode while `is_pinching == false`
        if (leftMode_ == proto::LeftMode::Spawn && !isPinching_) {
            const Vector3 ghostPos{
                snapToStep(cursor3D_.x, kSnapMoveStepUnits),
                snapToStep(cursor3D_.y, kSnapMoveStepUnits),
                snapToStep(cursor3D_.z, kSnapMoveStepUnits),
            };
            Model& ghostMdl = primitives_.model(activeShape_);
            ghostMdl.transform = MatrixTranslate(ghostPos.x, ghostPos.y, ghostPos.z);
            Color ghostFill = activeColor_;
            ghostFill.a = 65;
            DrawModel(ghostMdl, {0.f, 0.f, 0.f}, 1.0f, ghostFill);
            DrawModelWires(ghostMdl, {0.f, 0.f, 0.f}, 1.0f, Color{120, 250, 255, 255});
            ghostMdl.transform = MatrixIdentity();
        }

        // 3. 3D Cursor Sphere & Locked-Axis Gizmo
        if (drawCursorSphere) {
            const Color cCol = isPinching_ ? Color{255, 220, 60, 255} : axisColor(rightAxis_);
            DrawSphere(cursor3D_, isPinching_ ? 0.14f : 0.10f, cCol);
            DrawSphereWires(cursor3D_, 0.18f, 8, 8, WHITE);
        }
    }

    // --- Webcam Texture2D Management & PiP / AR Rendering -------------------
    void initFallbackCameraTexture() {
        const int w = 320, h = 240;
        std::vector<Color> pixels(w * h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const bool grid = (x % 32 == 0) || (y % 32 == 0);
                pixels[y * w + x] = grid ? Color{28, 52, 74, 255} : Color{16, 24, 36, 255};
            }
        }
        Image img{};
        img.data = pixels.data();
        img.width = w;
        img.height = h;
        img.mipmaps = 1;
        img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
        camTex_ = LoadTextureFromImage(img);
    }

    void uploadCameraTexture(const VideoFrame& vf) {
        if (vf.width <= 0 || vf.height <= 0 || vf.pixels.size() < static_cast<size_t>(vf.width * vf.height * 3))
            return;
        if (camTex_.id == 0 || camTex_.width != vf.width || camTex_.height != vf.height ||
            camTex_.format != PIXELFORMAT_UNCOMPRESSED_R8G8B8) {
            if (camTex_.id != 0) UnloadTexture(camTex_);
            Image img{};
            img.data = const_cast<uint8_t*>(vf.pixels.data());
            img.width = vf.width;
            img.height = vf.height;
            img.mipmaps = 1;
            img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8;
            camTex_ = LoadTextureFromImage(img);
        } else {
            UpdateTexture(camTex_, vf.pixels.data());
        }
    }

    void drawCameraTextureFullscreen(int sw, int sh) const {
        if (camTex_.id == 0) return;
        const Rectangle src{0.f, 0.f, static_cast<float>(camTex_.width),
                            static_cast<float>(camTex_.height)};
        const Rectangle dst{0.f, 0.f, static_cast<float>(sw), static_cast<float>(sh)};
        DrawTexturePro(camTex_, src, dst, {0.f, 0.f}, 0.f, Color{220, 240, 255, 255});
    }

    void draw2DHandSkeletonOverlay(int rectX, int rectY, int rectW, int rectH) const {
        for (int s = 0; s < 2; ++s) {
            const proto::HandPacket& h = hands_[s];
            if (!h.isPresent()) continue;

            Vector2 pts[21]{};
            for (int i = 0; i < 21; ++i) {
                pts[i] = {
                    rectX + h.landmarks[i * 3 + 0] * rectW,
                    rectY + h.landmarks[i * 3 + 1] * rectH,
                };
            }

            const Color boneCol = (s == 0) ? Color{40, 230, 255, 240} : Color{80, 255, 165, 240};
            for (const auto& [a, b] : kHandBones) {
                DrawLineEx(pts[a], pts[b], 2.2f, boneCol);
            }
            for (int i = 0; i < 21; ++i) {
                const bool isTip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
                DrawCircleV(pts[i], isTip ? 4.5f : 3.0f, isTip ? YELLOW : WHITE);
            }
            if (s == 1) {
                if (leftMode_ == proto::LeftMode::Recolor) {
                    const uint8_t cm = h.contactMask & 0x07;
                    const int tips[3] = {8, 12, 16};
                    const Color chCols[3] = {
                        Color{255, 60, 70, 255},
                        Color{60, 255, 110, 255},
                        Color{65, 145, 255, 255},
                    };
                    for (int ch = 0; ch < 3; ++ch) {
                        const bool active = (cm & (1 << ch)) != 0;
                        DrawLineEx(pts[4], pts[tips[ch]], active ? 3.5f : 1.4f,
                                   active ? chCols[ch] : Fade(chCols[ch], 0.35f));
                        if (active) DrawCircleV(pts[tips[ch]], 7.5f, chCols[ch]);
                    }
                } else if (leftMode_ == proto::LeftMode::Spawn) {
                    const int shapeTips[4] = {8, 12, 16, 20};
                    const int activeTip = shapeTips[static_cast<int>(activeShape_) % 4];
                    const Color pCol = isPinching_ ? Color{255, 215, 50, 255} : Color{90, 245, 255, 220};
                    DrawLineEx(pts[4], pts[activeTip], isPinching_ ? 3.5f : 2.2f, pCol);
                    DrawCircleV(pts[activeTip], 6.5f, pCol);
                } else if (leftMode_ == proto::LeftMode::Group) {
                    const int groupTips[4] = {8, 12, 16, 20};
                    const int activeTip = groupTips[static_cast<int>(activeGroupAction_) % 4];
                    const Color pCol = isPinching_ ? Color{255, 215, 50, 255} : Color{50, 235, 155, 230};
                    DrawLineEx(pts[4], pts[activeTip], isPinching_ ? 3.5f : 2.2f, pCol);
                    DrawCircleV(pts[activeTip], 6.5f, pCol);
                } else {
                    int activeTip = 8;
                    if (rightAxis_ == proto::RightAxis::Y) activeTip = 12;
                    else if (rightAxis_ == proto::RightAxis::Z) activeTip = 16;
                    const Color pCol = isPinching_ ? Color{255, 215, 50, 255} : Fade(axisColor(rightAxis_), 0.65f);
                    DrawLineEx(pts[4], pts[activeTip], isPinching_ ? 3.5f : 2.0f, pCol);
                    if (isPinching_) {
                        DrawCircleLines(static_cast<int>((pts[4].x + pts[activeTip].x) * 0.5f),
                                        static_cast<int>((pts[4].y + pts[activeTip].y) * 0.5f),
                                        12.f, Color{255, 220, 60, 255});
                    }
                }
            }
        }
    }

    void drawStudioPiP(int pipX, int pipY, int pipW, int pipH) const {
        DrawRectangle(pipX - 3, pipY - 26, pipW + 6, pipH + 29, Color{14, 18, 26, 235});
        DrawRectangleLines(pipX - 3, pipY - 26, pipW + 6, pipH + 29, Color{55, 195, 235, 220});
        DrawText("CAM PiP // 2D HAND SKELETON", pipX + 6, pipY - 20, 14, Color{120, 235, 255, 255});

        if (camTex_.id != 0) {
            const Rectangle src{0.f, 0.f, static_cast<float>(camTex_.width),
                                static_cast<float>(camTex_.height)};
            const Rectangle dst{static_cast<float>(pipX), static_cast<float>(pipY),
                                static_cast<float>(pipW), static_cast<float>(pipH)};
            DrawTexturePro(camTex_, src, dst, {0.f, 0.f}, 0.f, WHITE);
        }
        draw2DHandSkeletonOverlay(pipX, pipY, pipW, pipH);
    }

    // --- Pill-Shaped Top-Center Measurement HUD & Telemetry Panels ----------
    void drawHudOverlays(const Camera3D& cam, int sw, int sh) const {
        // 1. 3D Object Labels, Hover Reticle (IDLE) & Solid Locked Highlight
        const bool lockedMode = isTargetLockedMode(leftMode_);
        for (size_t i = 0; i < objects_.size(); ++i) {
            const auto& obj = objects_[i];
            const bool isSelectedTarget = isObjectInActiveTarget(i);
            const bool isIdleAim =
                (leftMode_ == proto::LeftMode::Idle) && isObjectInActiveHover(i);
            if (!isSelectedTarget && !isIdleAim && !obj.isSelected && !obj.isHovered &&
                obj.groupId < 0) {
                continue;
            }
            const Vector2 sc = GetWorldToScreen(obj.transform.position, cam);
            if (sc.x < 0 || sc.x > sw || sc.y < 0 || sc.y > sh) continue;

            const bool isActiveDeleteTarget =
                (leftMode_ == proto::LeftMode::Delete && isSelectedTarget &&
                 (deleteHoldTimer_ > 0.0f || deleteCooldownTimer_ > 0.0f));
            if (isActiveDeleteTarget) {
                if (deleteCooldownTimer_ > 0.0f) {
                    const float cdFrac =
                        std::clamp(deleteCooldownTimer_ / kDeleteCooldownDurationSec, 0.0f, 1.0f);
                    DrawRing(sc, 32.0f, 40.0f, -90.0f, -90.0f + 360.0f * cdFrac, 48,
                             Color{255, 165, 50, 215});
                } else {
                    const float delFrac =
                        std::clamp(deleteHoldTimer_ / kDeleteHoldDurationSec, 0.0f, 1.0f);
                    DrawRing(sc, 32.0f, 42.0f, 0.0f, 360.0f, 48, Color{45, 10, 15, 195});
                    DrawRing(sc, 32.0f, 42.0f, -90.0f, -90.0f + 360.0f * delFrac, 48,
                             Color{255, 50, 65, 245});
                    DrawRingLines(sc, 31.0f, 43.0f, -90.0f, -90.0f + 360.0f * delFrac, 48,
                                  Color{255, 185, 185, 230});
                }
            } else if (lockedMode && isSelectedTarget) {
                const Color lockCol = (leftMode_ == proto::LeftMode::Delete)
                    ? Color{255, 80, 90, 235}
                    : (isPinching_ ? Color{255, 220, 65, 245}
                                   : (obj.groupId >= 0 ? Color{60, 255, 165, 235}
                                                       : Color{65, 245, 255, 235}));
                drawLockedHighlight2D(sc, lockCol, isPinching_);
            } else if (isIdleAim || (leftMode_ == proto::LeftMode::Group && obj.isHovered)) {
                drawHoverReticle2D(sc, obj.groupId >= 0 ? Color{85, 245, 175, 180}
                                                      : Color{110, 235, 255, 175});
            }

            const bool showLockedTag = lockedMode && isSelectedTarget;
            float dimX = obj.transform.scale.x * kCmPerWorldUnit;
            float dimY = obj.transform.scale.y * kCmPerWorldUnit;
            float dimZ = obj.transform.scale.z * kCmPerWorldUnit;
            if (obj.type == PrimitiveType::ImportedObj && obj.importedModel) {
                const BoundingBox& nb = obj.importedModel->normalizedLocalBox;
                dimX *= std::max(0.01f, nb.max.x - nb.min.x);
                dimY *= std::max(0.01f, nb.max.y - nb.min.y);
                dimZ *= std::max(0.01f, nb.max.z - nb.min.z);
            }
            const std::string labelText = obj.displayLabel();
            char tag[128];
            if (obj.groupId >= 0) {
                std::snprintf(tag, sizeof(tag),
                              showLockedTag ? "%s #%u [GRP %d // LOCKED] (%.0fx%.0fx%.0f cm)"
                                            : "%s #%u [GRP %d] (%.0fx%.0fx%.0f cm)",
                              labelText.c_str(), obj.id, obj.groupId, dimX, dimY, dimZ);
            } else {
                std::snprintf(tag, sizeof(tag),
                              showLockedTag ? "%s #%u [LOCKED] (%.0fx%.0fx%.0f cm)"
                                            : "%s #%u (%.0fx%.0fx%.0f cm)",
                              labelText.c_str(), obj.id, dimX, dimY, dimZ);
            }
            int offsetPx = 38;
            if (isActiveDeleteTarget) offsetPx = 58;
            else if (showLockedTag) offsetPx = 54;
            else if (isIdleAim) offsetPx = 46;
            const int labelY = static_cast<int>(sc.y) - offsetPx;
            DrawText(tag, static_cast<int>(sc.x) - 65, labelY, 14,
                     isActiveDeleteTarget ? RED
                         : (showLockedTag ? Color{255, 230, 85, 255}
                                          : (obj.isSelected ? YELLOW : SKYBLUE)));
        }

        // 2. Top-Center Pill-Shaped Measurement Badge (with RGB Swatch or Delete Progress Bar)
        const int fontSize = 20;
        const bool showSwatch = (leftMode_ == proto::LeftMode::Recolor);
        const bool showDeleteBar = (leftMode_ == proto::LeftMode::Delete &&
                                    deleteCooldownTimer_ <= 0.0f &&
                                    deleteHoldTimer_ > 0.0f);
        const bool showDeleteCooldown = (leftMode_ == proto::LeftMode::Delete &&
                                         deleteCooldownTimer_ > 0.0f);
        const int swatchSpace = showSwatch ? 36 : 0;
        const int textW = MeasureText(measurementBadge_.c_str(), fontSize);
        const int pillW = std::max(380, textW + 48 + swatchSpace);
        const int pillH = showDeleteBar ? 54 : 44;
        const int pillX = (sw - pillW) / 2;
        const Rectangle pillRect{
            static_cast<float>(pillX),
            16.0f,
            static_cast<float>(pillW),
            static_cast<float>(pillH),
        };
        const Color borderCol = showSwatch ? activeColor_
            : (showDeleteBar ? Color{255, 60, 75, 255}
            : (showDeleteCooldown ? Color{255, 165, 50, 255}
            : (leftMode_ == proto::LeftMode::Group
                   ? (isPinching_ ? Color{255, 215, 65, 255} : Color{50, 235, 155, 255})
                   : (isPinching_ ? Color{255, 215, 65, 255} : axisColor(rightAxis_)))));
        DrawRectangleRounded(pillRect, 0.5f, 16, Color{14, 22, 34, 235});
        DrawRectangleRoundedLinesEx(pillRect, 0.5f, 16, 2.5f, borderCol);
        if (showSwatch) {
            const int textStartX = pillX + 24;
            DrawText(measurementBadge_.c_str(), textStartX, 28, fontSize, WHITE);
            const Rectangle swatch{static_cast<float>(textStartX + textW + 14), 24.0f, 26.0f, 26.0f};
            DrawRectangleRounded(swatch, 0.25f, 6, activeColor_);
            DrawRectangleRoundedLinesEx(swatch, 0.25f, 6, 1.5f, WHITE);
        } else if (showDeleteBar) {
            DrawText(measurementBadge_.c_str(), (sw - textW) / 2, 22, fontSize, Color{255, 115, 125, 255});
            const float delFrac = std::clamp(deleteHoldTimer_ / kDeleteHoldDurationSec, 0.0f, 1.0f);
            const float trackW = static_cast<float>(pillW - 44);
            const Rectangle track{static_cast<float>(pillX + 22), 54.0f, trackW, 6.0f};
            const Rectangle fill{track.x, track.y, trackW * delFrac, track.height};
            DrawRectangleRounded(track, 0.5f, 6, Color{60, 18, 22, 230});
            DrawRectangleRounded(fill, 0.5f, 6, Color{255, 55, 68, 255});
        } else {
            DrawText(measurementBadge_.c_str(), (sw - textW) / 2, 28, fontSize, WHITE);
        }

        // 2b. Import / Export Confirmation Toast Banner
        if (toastTimer_ > 0.0f && !toastMessage_.empty()) {
            const float alpha = std::clamp(toastTimer_ / 0.6f, 0.0f, 1.0f);
            const unsigned char aBg = static_cast<unsigned char>(230.0f * alpha);
            const unsigned char aFg = static_cast<unsigned char>(255.0f * alpha);
            const int tFont = 17;
            const int tW = MeasureText(toastMessage_.c_str(), tFont);
            const int bW = std::max(320, tW + 40);
            const int bX = (sw - bW) / 2;
            const int bY = showDeleteBar ? 78 : 70;
            const Rectangle toastRect{
                static_cast<float>(bX), static_cast<float>(bY), static_cast<float>(bW), 34.0f
            };
            const Color fillC = toastIsError_ ? Color{65, 16, 22, aBg} : Color{14, 52, 34, aBg};
            const Color lineC = toastIsError_ ? Color{255, 75, 85, aFg} : Color{60, 245, 145, aFg};
            const Color textC = toastIsError_ ? Color{255, 200, 205, aFg} : Color{210, 255, 230, aFg};
            DrawRectangleRounded(toastRect, 0.5f, 12, fillC);
            DrawRectangleRoundedLinesEx(toastRect, 0.5f, 12, 2.0f, lineC);
            DrawText(toastMessage_.c_str(), (sw - tW) / 2, bY + 9, tFont, textC);
        }

        // 3. Top-Left Mode + Clutch Telemetry Card
        DrawRectangle(18, 18, 330, 152, Color{14, 20, 30, 220});
        DrawRectangleLines(18, 18, 330, 152, Color{55, 195, 235, 200});
        DrawText("VISUAL3D // MODE + CLUTCH ENGINE", 28, 26, 15, Color{110, 235, 255, 255});

        char buf[128];
        std::snprintf(buf, sizeof(buf), "VIEWPORT : %s [TAB]",
                      viewportMode_ == ViewportMode::Studio3D ? "MODE B: STUDIO 3D + PiP"
                                                              : "MODE A: AR CAMERA FULL");
        DrawText(buf, 28, 48, 14, RAYWHITE);

        std::snprintf(buf, sizeof(buf), "L-MODE   : %s (150ms Debounced)", modeName(leftMode_));
        DrawText(buf, 28, 68, 14, Color{90, 245, 255, 255});

        if (leftMode_ == proto::LeftMode::Spawn) {
            std::snprintf(buf, sizeof(buf), "R-HAND   : %s (%s)", primitiveName(activeShape_),
                          spawnFingerName(activeShape_));
            DrawText(buf, 28, 88, 14, Color{90, 245, 255, 255});
        } else if (leftMode_ == proto::LeftMode::Group) {
            std::snprintf(buf, sizeof(buf), "R-HAND   : %s (%s)",
                          groupActionName(activeGroupAction_),
                          groupActionFingerName(activeGroupAction_));
            DrawText(buf, 28, 88, 14, Color{60, 245, 165, 255});
        } else {
            std::snprintf(buf, sizeof(buf), "R-HAND   : %s (%s)", axisName(rightAxis_),
                          axisFingerName(rightAxis_));
            DrawText(buf, 28, 88, 14, axisColor(rightAxis_));
        }

        std::snprintf(buf, sizeof(buf), "CLUTCH   : %s (%.1f px | <32 / >46)",
                      isPinching_ ? "ENGAGED [PINCH]" : "RELEASED [HOVER]", pinchPx_);
        DrawText(buf, 28, 108, 14, isPinching_ ? YELLOW : LIGHTGRAY);

        std::snprintf(buf, sizeof(buf), "FPS / UDP: %d FPS | %.1f PKT/S (%s)",
                      GetFPS(), linkStats_.packetsPerSec,
                      linkStats_.connected ? "ONLINE" : "WAITING");
        DrawText(buf, 28, 128, 14, linkStats_.connected ? GREEN : ORANGE);

        std::snprintf(buf, sizeof(buf), "OBJECTS  : %zu (1.0 unit = 10.0 cm)", objects_.size());
        DrawText(buf, 28, 148, 14, SKYBLUE);

        // 4. Compact Shortcut Legend + Bottom-Left Left-Hand Mode Bar (0..7)
        const char* shortcutLegend =
            "[P] Save Project (.3dmproj)  |  [O] Open Project  |  [I] Import .OBJ  |  [E] Export Group .OBJ";
        const int legFont = 13;
        const int legW = MeasureText(shortcutLegend, legFont) + 20;
        const int legY = sh - 98;
        DrawRectangle(18, legY, legW, 24, Color{14, 20, 30, 215});
        DrawRectangleLines(18, legY, legW, 24, Color{55, 195, 235, 180});
        DrawText(shortcutLegend, 28, legY + 5, legFont, Color{190, 240, 255, 245});

        const char* modeLabels[8] = {
            "0:IDLE", "1:MOVE", "2:RESIZE", "3:ROTATE",
            "4:COLOR", "5:GROUP", "TU:SPAWN", "TD:DEL"
        };
        const int barY = sh - 68;
        for (int m = 0; m < 8; ++m) {
            const int bx = 18 + m * 116;
            const bool active = (static_cast<int>(leftMode_) == m);
            const Color activeFill = (m == 7) ? Color{175, 35, 48, 235} : Color{35, 135, 185, 235};
            const Color activeLine = (m == 7) ? Color{255, 95, 105, 255} : Color{120, 245, 255, 255};
            DrawRectangle(bx, barY, 110, 32, active ? activeFill : Color{16, 24, 36, 200});
            DrawRectangleLines(bx, barY, 110, 32, active ? activeLine : Color{50, 95, 125, 180});
            DrawText(modeLabels[m], bx + 8, barY + 9, 13, active ? WHITE : LIGHTGRAY);
        }
        DrawText(statusLine_.c_str(), 20, sh - 26, 14, Color{150, 220, 245, 230});
    }
};

}  // namespace v3d

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    uint16_t udpPort = 5005;
    int winW = 1360, winH = 820;
    bool autoTracker = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) udpPort = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "--width" && i + 1 < argc) winW = std::atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) winH = std::atoi(argv[++i]);
        else if (arg == "--no-auto-tracker") autoTracker = false;
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(winW, winH, "Visual3D // Raylib Mode + Clutch 3D Studio & AR Modeler");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {0.0f, 3.8f, 8.5f};
    camera.target = {0.0f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    v3d::UDPReceiver udp(udpPort, "127.0.0.1");
    if (udp.start()) {
        std::cout << "[engine] Listening for Mode+Clutch binary UDP on 127.0.0.1:" << udpPort
                  << " (" << sizeof(v3d::proto::TelemetryPacket) << " B)\n";
    } else {
        std::cerr << "[engine] UDP bind warning: " << udp.stats().error << "\n";
    }

    v3d::ProcessManager trackerProc;
    if (autoTracker) {
        if (trackerProc.start(udpPort)) {
            std::cout << "[engine] Auto-started Python hand tracker (" << trackerProc.status()
                      << ")\n";
        } else {
            std::cerr << "[engine] Tracker auto-start warning: " << trackerProc.status() << "\n";
        }
    }

    v3d::ModeClutchApp app;
    app.init();

    while (!WindowShouldClose()) {
        const float dt = std::min(0.05f, GetFrameTime());
        app.update(dt, udp);
        app.render(camera);
    }

    udp.stop();
    app.shutdown();
    CloseWindow();
    trackerProc.stop();
    return 0;
}
