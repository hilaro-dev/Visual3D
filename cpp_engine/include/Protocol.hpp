// Protocol.hpp -- `#pragma pack(push, 1)` binary UDP wire format shared with
// python_tracker/hand_tracker.py (`struct.pack("<4sIddffBBBBBBHfffff...", ...)`).
//
// Zero-Conflict "Mode + Clutch" Spatial Interface Protocol (624 bytes total):
//   TelemetryPacket = Header (56 B) + HandPacket[0: Left] (284 B) + HandPacket[1: Right] (284 B)
//
//   * Left Hand Mode Selector (`left_mode`, 150ms debounced in Python):
//       0 = IDLE        (Safe state, transformations locked)
//       1 = MOVE        (Translate along locked axis, snapped to 0.5 / 1.0 unit = 5 / 10 cm)
//       2 = RESIZE      (Scale along locked axis / uniform, snapped to 0.5 unit = 5 cm)
//       3 = ROTATE      (Rotate around locked axis, snapped to 15.0 deg increments)
//       4 = RECOLOR     (Chromatic RGB color assignment)
//       5 = SPAWN/GROUP (Ghost preview at 3D cursor; pinch spawns or groups selected objects)
//
//   * Right Hand Axis Selector & Hysteresis Pinch Clutch:
//       - `right_axis`  : 1 = X-Axis, 2 = Y-Axis, 3 = Z-Axis
//       - `is_pinching` : 1 when Right Thumb(4)+Index(8) distance < 30px, releases > 45px
//       - `cursor_x/y/z`: 3D cursor position in World Units (`1.0f` world unit = `10 cm`)
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace v3d::proto {

constexpr uint16_t kVersion = 2;
constexpr int kLandmarks = 21;
constexpr size_t kHeaderSize = 56;
constexpr size_t kHandSize = 284;
constexpr size_t kTelemetrySize = kHeaderSize + 2 * kHandSize;  // 624 bytes
constexpr size_t kVideoHeaderSize = 20;

enum class LeftMode : uint8_t {
    Idle = 0,
    Move = 1,
    Resize = 2,
    Rotate = 3,
    Recolor = 4,
    Group = 5,
    Spawn = 6,
    Delete = 7
};

enum class RightAxis : uint8_t {
    None = 0,
    X = 1,
    Y = 2,
    Z = 3
};

enum StateFlags : uint8_t {
    kPinch       = 0x01,
    kFist        = 0x02,
    kOpen        = 0x04,
    kThumbFolded = 0x08
};

enum ContactBits : uint8_t {
    kMiddleRed  = 0x01,
    kRingGreen  = 0x02,
    kPinkyBlue  = 0x04
};

enum HeaderFlags : uint8_t {
    kMirrored = 0x01
};

#pragma pack(push, 1)

struct HandPacket {
    uint8_t present = 0;       // 0 = not detected, 1 = detected
    uint8_t handedness = 0;    // 0 = Physical Left Hand, 1 = Physical Right Hand
    uint8_t fingersMask = 0;   // bit0=Thumb .. bit4=Pinky extended
    uint8_t gesture = 0;       // LeftMode on slot 0, RightAxis on slot 1
    uint8_t contactMask = 0;   // Left Hand RGB bitmask (Middle=R, Ring=G, Pinky=B)
    uint8_t stateFlags = 0;    // StateFlags bitmask
    uint8_t fingerCount = 0;   // Extended finger count
    uint8_t reserved = 0;
    float score = 0.f;
    float pinchDist = 1.f;     // Normalized Thumb(4)-Index(8) distance
    float handScale = 0.f;
    float color[3] = {0.15f, 0.90f, 1.00f};
    float landmarks[kLandmarks * 3]{};  // 21 * (x, y, z) normalized [0, 1] landmarks

    bool isPresent() const { return present != 0; }
    bool pinch() const { return (stateFlags & kPinch) != 0; }
    bool fist() const { return (stateFlags & kFist) != 0; }
    bool open() const { return (stateFlags & kOpen) != 0; }
    bool thumbFolded() const { return (stateFlags & kThumbFolded) != 0; }
};

struct TelemetryPacket {
    char magic[4] = {'V', '3', 'D', 'H'};
    uint32_t seq = 0;
    double captureTs = 0.0;
    double sendTs = 0.0;
    float trackerFps = 0.f;
    float camAspect = 4.f / 3.f;
    uint8_t numHands = 0;
    uint8_t flags = 0;
    uint8_t left_mode = 0;     // LeftMode (0=IDLE, 1=MOVE, 2=RESIZE, 3=ROTATE, 4=RECOLOR, 5=SPAWN/GROUP)
    uint8_t right_axis = 1;    // RightAxis (1=X-Axis, 2=Y-Axis, 3=Z-Axis)
    uint8_t is_pinching = 0;   // Right Thumb+Index Hysteresis Clutch (1=engaged <30px, 0=released >45px)
    uint8_t active_shape = 0;  // 0=Cube, 1=Capsule, 2=Cylinder, 3=Torus
    uint16_t version = kVersion;
    float cursor_x = 0.f;      // 3D cursor X in world units (1.0f = 10 cm)
    float cursor_y = 0.f;      // 3D cursor Y in world units (1.0f = 10 cm)
    float cursor_z = 0.f;      // 3D cursor Z in world units (1.0f = 10 cm)
    float pinch_px = 999.f;    // Right Thumb-Index distance in pixels
    HandPacket hands[2];       // hands[0] = Left Hand, hands[1] = Right Hand
};

struct VideoChunkHeader {
    char magic[4] = {'V', '3', 'D', 'V'};
    uint32_t frameId = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    uint16_t rowStart = 0;
    uint16_t rowCount = 0;
    uint8_t channels = 3;
    uint8_t reserved8 = 0;
    uint16_t reserved16 = 0;
};

#pragma pack(pop)

static_assert(sizeof(HandPacket) == kHandSize, "HandPacket must be 284 bytes");
static_assert(sizeof(TelemetryPacket) == kTelemetrySize, "TelemetryPacket must be 624 bytes");
static_assert(sizeof(VideoChunkHeader) == kVideoHeaderSize, "VideoChunkHeader must be 20 bytes");

inline bool parseTelemetry(const uint8_t* data, size_t size, TelemetryPacket& out) {
    if (size < sizeof(TelemetryPacket)) return false;
    if (std::memcmp(data, "V3DH", 4) != 0) return false;
    std::memcpy(&out, data, sizeof(TelemetryPacket));
    if (out.version != kVersion) return false;
    if (!(out.camAspect > 0.2f && out.camAspect < 5.f)) out.camAspect = 4.f / 3.f;
    if (out.left_mode > 7) out.left_mode = 0;
    if (out.right_axis < 1 || out.right_axis > 3) out.right_axis = 1;
    out.hands[0].contactMask &= 0x07;
    out.hands[1].contactMask &= 0x07;  // Right Hand carries additive R/G/B pinch bitmask in RECOLOR mode
    return true;
}

inline bool parseVideoHeader(const uint8_t* data, size_t size, VideoChunkHeader& hdr) {
    if (size < sizeof(VideoChunkHeader)) return false;
    if (std::memcmp(data, "V3DV", 4) != 0) return false;
    std::memcpy(&hdr, data, sizeof(VideoChunkHeader));
    return (hdr.channels == 3) && hdr.width > 0 && hdr.height > 0 && hdr.width <= 1280 && hdr.height <= 960;
}

}  // namespace v3d::proto
