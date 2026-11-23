#!/usr/bin/env python3
"""
hand_tracker.py -- Visual3D "Mode + Clutch" Spatial Vision Engine (Process 1)
=============================================================================

Zero-Conflict "Mode + Clutch" Gesture State Machine:
  1. Left Hand (Mode Selector with 150ms Debounce Timer):
     Counts extended Left Hand fingers (0..5) and requires a new finger count
     to remain stable for >= 150ms (`MODE_DEBOUNCE_SEC = 0.150`) before switching:
       * 0 Fingers (Closed Fist) : IDLE    (0) -- Safe state, transformations locked
       * 1 Finger                : MOVE    (1) -- Translate along locked axis
       * 2 Fingers               : RESIZE  (2) -- Scale along locked axis / uniform
       * 3 Fingers               : ROTATE  (3) -- Rotate around locked axis (15 deg snap)
       * 4 Fingers (Thumb Tucked): RECOLOR (4) -- Additive RGB finger mix / palette
       * 5 Fingers (Open Palm)   : GROUP   (5) -- Placeholder for multi-object grouping
       * Thumbs-Up  (0+Thumb Up) : SPAWN   (6) -- Ghost preview & single-shot spawn
       * Thumbs-Down(0+Thumb Dn) : DELETE  (7) -- Safe 1.5s hold delete + 1.0s cooldown

  2. Right Hand (Axis Selector & Hysteresis Pinch Clutch):
     * Axis Lock (Hover state, updated while not pinching):
       Counts extended non-thumb Right Hand fingers:
         - 1 Finger  (Index)                = X-Axis (1)
         - 2 Fingers (Index + Middle)       = Y-Axis (2)
         - 3 Fingers (Index + Middle + Ring)= Z-Axis (3)
     * Pinch Clutch (Trigger with Hysteresis):
       Measures Euclidean pixel distance between Right Thumb Tip (4) and
       Right Index Tip (8) on the 640x480 frame:
         - Engage (`is_pinching = 1`) when distance < 30.0 px
           (normalized `< 30 / 640 = 0.046875`)
         - Release (`is_pinching = 0`) when distance > 45.0 px
           (normalized `> 45 / 640 = 0.0703125`)

  3. High-Performance Binary UDP Protocol (`#pragma pack(push, 1)` match):
     * TelemetryPacket (`V3DH`, 624 bytes, `<4sIddffBBBBBBBBfff` + 2 x `HandPacket`):
       Includes `uint8 left_mode`, `uint8 right_axis`, `uint8 is_pinching`,
       3D cursor coordinates (`cursor_x, cursor_y, cursor_z`), and both hands'
       2D/3D landmarks + Left-Hand RGB chromatic mix.
     * Optional low-res RGB camera stream (`V3DV` chunks, `--video rgb` enabled
       by default at 320x240 @ 24 FPS) so the C++ Raylib engine can render the
       live webcam Texture2D in both Mode A (AR Fullscreen) and Mode B (Studio PiP).
"""
from __future__ import annotations

import argparse
import os
import platform
import signal
import socket
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[vision] OpenCV missing. Run: pip install -r requirements.txt")

try:
    import mediapipe as mp
except ImportError:
    sys.exit("[vision] MediaPipe missing. Run: pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# Performance & Gesture Constants
# --------------------------------------------------------------------------- #
DEBUG_VIEW: bool = False
CAPTURE_WIDTH: int = 640
CAPTURE_HEIGHT: int = 480
CAPTURE_FPS: int = 60
MODEL_COMPLEXITY: int = 0

MODE_DEBOUNCE_SEC: float = 0.150   # 150ms debounce timer for Left Hand mode switching
PINCH_ENGAGE_PX: float = 32.0      # Engage Right Thumb+Finger pinch when < 32 px
PINCH_RELEASE_PX: float = 46.0     # Release Right Thumb+Finger pinch when > 46 px
RECOLOR_ENGAGE_PX: float = 38.0    # Engage additive RGB channel when < 38 px
RECOLOR_RELEASE_PX: float = 50.0   # Reset latched RGB channels once all 3 release > 50 px
SPAWN_ENGAGE_PX: float = 32.0      # Single-shot spawn trigger when < 32 px
SPAWN_RELEASE_PX: float = 48.0     # Release spawn clutch when > 48 px
GROUP_ENGAGE_PX: float = 32.0      # Single-shot group action trigger when < 32 px
GROUP_RELEASE_PX: float = 48.0     # Unlock group clutch once all 4 fingers release > 48 px
CHROMA_DISTANCE: float = 0.048     # Normalized proximity for Left-Hand RECOLOR channels

# Left Hand Mode Enum (matches Protocol.hpp :: LeftMode)
MODE_IDLE = 0
MODE_MOVE = 1
MODE_RESIZE = 2
MODE_ROTATE = 3
MODE_RECOLOR = 4
MODE_GROUP = 5
MODE_SPAWN = 6
MODE_DELETE = 7
MODE_NAMES = ["IDLE", "MOVE", "RESIZE", "ROTATE", "RECOLOR", "GROUP", "SPAWN", "DELETE"]

# Right Hand Axis Enum (matches Protocol.hpp :: RightAxis)
AXIS_NONE = 0
AXIS_X = 1
AXIS_Y = 2
AXIS_Z = 3
AXIS_NAMES = ["NONE", "X-AXIS", "Y-AXIS", "Z-AXIS"]
AXIS_FINGER_NAMES = ["None", "Thumb+Index", "Thumb+Middle", "Thumb+Ring"]

# Group Mode Right-Hand Actions (matches Protocol.hpp :: GroupAction)
GROUP_ACTION_SELECT = 0
GROUP_ACTION_GROUP = 1
GROUP_ACTION_UNGROUP = 2
GROUP_ACTION_CLEAR = 3

# --------------------------------------------------------------------------- #
# Fixed-Size Little-Endian Binary UDP Protocol (`#pragma pack(push, 1)` match)
# --------------------------------------------------------------------------- #
PROTOCOL_VERSION = 2
TELEMETRY_MAGIC = b"V3DH"
VIDEO_MAGIC = b"V3DV"

# Header (56 bytes):
#   magic(4s), seq(I), capture_ts(d), send_ts(d), tracker_fps(f), cam_aspect(f),
#   num_hands(B), flags(B), left_mode(B), right_axis(B), is_pinching(B),
#   active_shape(B), version(H), cursor_x(f), cursor_y(f), cursor_z(f),
#   pinch_px(f)
HEADER_FMT = "<4sIddffBBBBBBHffff"
# HandPacket (284 bytes):
#   present(B), handedness(B), fingers_mask(B), gesture(B), contact_mask(B),
#   state_flags(B), finger_count(B), reserved(B), score(f), pinch_dist(f),
#   hand_scale(f), r(f), g(f), b(f), landmarks(63f)
HAND_FMT = "<8B6f63f"
VIDEO_FMT = "<4sIHHHHBBH"

HEADER_SIZE = struct.calcsize(HEADER_FMT)     # 56
HAND_SIZE = struct.calcsize(HAND_FMT)         # 284
PACKET_SIZE = HEADER_SIZE + 2 * HAND_SIZE     # 624
VIDEO_HDR_SIZE = struct.calcsize(VIDEO_FMT)   # 20
MAX_DATAGRAM = 8192
assert HEADER_SIZE == 56 and HAND_SIZE == 284 and PACKET_SIZE == 624

FLAG_MIRRORED = 0x01
FLAG_SPAWN_TRIGGER = 0x02
FLAG_GROUP_TRIGGER = 0x04
S_PINCH, S_FIST, S_OPEN, S_THUMB_FOLDED = 0x01, 0x02, 0x04, 0x08

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
LEFT_CHROMA_TIPS = (MIDDLE_TIP, RING_TIP, PINKY_TIP)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

MASK_COLORS = {
    0b000: (0.15, 0.90, 1.00),
    0b001: (1.00, 0.00, 0.00),  # Red     (255,   0,   0) -- Index + Thumb
    0b010: (0.00, 1.00, 0.00),  # Green   (  0, 255,   0) -- Middle + Thumb
    0b100: (0.00, 0.00, 1.00),  # Blue    (  0,   0, 255) -- Ring + Thumb
    0b011: (1.00, 1.00, 0.00),  # Yellow  (255, 255,   0) -- Index + Middle
    0b110: (0.00, 1.00, 1.00),  # Cyan    (  0, 255, 255) -- Middle + Ring
    0b101: (1.00, 0.00, 1.00),  # Magenta (255,   0, 255) -- Ring + Index
    0b111: (1.00, 1.00, 1.00),  # White   (255, 255, 255) -- Index + Middle + Ring
}

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


# --------------------------------------------------------------------------- #
# Camera Initialization (DirectShow on Windows, 640x480)
# --------------------------------------------------------------------------- #
def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


# --------------------------------------------------------------------------- #
# MediaPipe Hands Wrapper
# --------------------------------------------------------------------------- #
@dataclass
class RawHand:
    landmarks: np.ndarray  # (21, 3) float32 normalized [0.0, 1.0]
    label: str             # "Left" or "Right"
    score: float


def extract_solutions_hands(res) -> List[RawHand]:
    out: List[RawHand] = []
    if res.multi_hand_landmarks and res.multi_handedness:
        for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            arr = np.fromiter(
                (c for lm in lms.landmark for c in (lm.x, lm.y, lm.z)),
                dtype=np.float32,
                count=63,
            ).reshape(21, 3)
            cls = hd.classification[0]
            out.append(RawHand(arr, cls.label, float(cls.score)))
    return out


class TasksHandsFallback:
    def __init__(self, max_hands: int, det_conf: float, track_conf: float, model_path: str) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        if not os.path.isfile(model_path):
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            print(f"[vision] Downloading hand_landmarker.task -> {model_path}")
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=track_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb_frame: np.ndarray, ts_ms: int) -> List[RawHand]:
        ts_ms = max(ts_ms, self._last_ts + 1)
        self._last_ts = ts_ms
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self._lm.detect_for_video(mp_img, ts_ms)
        out: List[RawHand] = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            arr = np.fromiter(
                (c for lm in lms for c in (lm.x, lm.y, lm.z)),
                dtype=np.float32,
                count=63,
            ).reshape(21, 3)
            out.append(RawHand(arr, hd[0].category_name, float(hd[0].score)))
        return out

    def close(self) -> None:
        self._lm.close()


# --------------------------------------------------------------------------- #
# Smoothing & 150ms Time-Based Debouncer
# --------------------------------------------------------------------------- #
class LandmarkEMA:
    """Exponential moving average (alpha = 0.25) on 21 3D landmarks."""

    def __init__(self, alpha: float = 0.25, reset_timeout: float = 0.25) -> None:
        self.alpha = alpha
        self.reset_timeout = reset_timeout
        self._state: Optional[np.ndarray] = None
        self._last_ts = 0.0

    def __call__(self, raw: np.ndarray, now: float) -> np.ndarray:
        if self._state is None or (now - self._last_ts) > self.reset_timeout:
            self._state = raw.astype(np.float32).copy()
        else:
            self._state += self.alpha * (raw - self._state)
        self._last_ts = now
        return self._state

    def reset(self) -> None:
        self._state = None


class TimedDebouncer:
    """Requires a candidate value to remain stable for `hold_sec` (150ms) before switching."""

    def __init__(self, initial: int, hold_sec: float = MODE_DEBOUNCE_SEC) -> None:
        self.value = initial
        self.hold_sec = hold_sec
        self._candidate = initial
        self._candidate_since = 0.0

    def update(self, candidate: int, now: float) -> int:
        if candidate == self.value:
            self._candidate = candidate
            self._candidate_since = now
            return self.value
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now
            return self.value
        if (now - self._candidate_since) >= self.hold_sec:
            self.value = candidate
        return self.value

    def reset(self, val: int, now: float = 0.0) -> None:
        self.value = self._candidate = val
        self._candidate_since = now


# --------------------------------------------------------------------------- #
# "Mode + Clutch" Gesture State Machine
# --------------------------------------------------------------------------- #
@dataclass
class HandFeatures:
    fingers_mask: int = 0
    finger_count: int = 0       # 0..5 total extended fingers on Left; 0..4 non-thumb on Right
    gesture: int = 0
    contact_mask: int = 0
    group_action: int = 0
    pinch: bool = False
    fist: bool = False
    open: bool = False
    thumb_folded: bool = False
    pinch_dist: float = 1.0     # Normalized distance
    pinch_px: float = 999.0     # Pixel distance on 640x480 frame
    hand_scale: float = 0.0
    thumb_tip_dists_px: Tuple[float, float, float, float] = (999.0, 999.0, 999.0, 999.0)  # (Index, Middle, Ring, Pinky) px
    color: Tuple[float, float, float] = (0.15, 0.90, 1.00)


class ModeClutchEngine:
    """
    Implements the complete Mode + Clutch State Machine:
      * Left Hand (Slot 0): 150ms debounced finger counter ->
          0=IDLE, 1=MOVE, 2=RESIZE, 3=ROTATE, 4=RECOLOR, 5=SPAWN/GROUP
      * Right Hand (Slot 1):
          - Direct Finger-to-Thumb Pinch Axis & Clutch:
            Thumb+Index(8) = X-Axis, Thumb+Middle(12) = Y-Axis, Thumb+Ring(16) = Z-Axis
          - Hysteresis: engage < 32px, lock initiating finger, release > 46px
    """

    def __init__(self, frame_w: int = CAPTURE_WIDTH, frame_h: int = CAPTURE_HEIGHT,
                 engage_px: float = PINCH_ENGAGE_PX, release_px: float = PINCH_RELEASE_PX,
                 debounce_sec: float = MODE_DEBOUNCE_SEC) -> None:
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.engage_px = engage_px
        self.release_px = release_px
        self.left_mode_debouncer = TimedDebouncer(MODE_IDLE, debounce_sec)
        self.palm_size_ema = LandmarkEMA(alpha=0.25, reset_timeout=0.25)
        self.left_mode: int = MODE_IDLE
        self.right_axis: int = AXIS_X
        self.is_pinching: bool = False
        self.recolor_latched: List[bool] = [False, False, False]  # [latched_R, latched_G, latched_B]
        self.spawn_shape: int = 0
        self.active_shape: int = 0
        self.spawn_trigger: bool = False
        self.group_action: int = GROUP_ACTION_SELECT
        self.group_trigger: bool = False
        self.smoothed_palm_size: float = 0.18
        self.cursor_3d: Tuple[float, float, float] = (0.0, 0.0, 0.18)
        self.last_color: Tuple[float, float, float] = (0.15, 0.90, 1.00)

    def _analyze_single_hand(self, slot: int, lm: np.ndarray) -> HandFeatures:
        p2 = lm[:, :2]

        def dist2d(a: int, b: int) -> float:
            dx = float(p2[a, 0] - p2[b, 0])
            dy = float(p2[a, 1] - p2[b, 1])
            return (dx * dx + dy * dy) ** 0.5

        def dist_px(a: int, b: int) -> float:
            dx = float(p2[a, 0] - p2[b, 0]) * self.frame_w
            dy = float(p2[a, 1] - p2[b, 1]) * self.frame_h
            return (dx * dx + dy * dy) ** 0.5

        scale = max(dist2d(WRIST, MIDDLE_MCP), 1e-4)

        # 4 non-thumb fingers (Index, Middle, Ring, Pinky) for Left Hand mode selector
        mask = 0
        for i, (_mcp, pip, _dip, tip) in enumerate(FINGERS):
            if dist2d(WRIST, tip) > dist2d(WRIST, pip) * 1.15:
                mask |= 1 << (i + 1)

        # Thumb extension check (distance from Thumb Tip 4 to Pinky MCP 17 vs IP 3 / MCP 2)
        d_tip_pinky = dist2d(THUMB_TIP, PINKY_MCP)
        d_ip_pinky = dist2d(THUMB_IP, PINKY_MCP)
        d_mcp_pinky = dist2d(THUMB_MCP, PINKY_MCP)
        thumb_folded = (d_tip_pinky < d_ip_pinky) or (d_tip_pinky < d_mcp_pinky)
        thumb_ext = (not thumb_folded) and (dist2d(THUMB_TIP, INDEX_MCP) > 0.44 * scale)
        if slot == 0 and thumb_ext:
            mask |= 1

        ext4 = bin(mask >> 1).count("1")
        total5 = ext4 + (1 if thumb_ext else 0)

        f = HandFeatures()
        f.fingers_mask = mask
        f.finger_count = total5 if slot == 0 else ext4
        f.thumb_folded = thumb_folded
        f.fist = (total5 == 0) if slot == 0 else (ext4 == 0 and not thumb_ext)
        f.open = (total5 == 5)
        f.hand_scale = scale

        if slot == 1:
            # Direct Finger-to-Thumb 2D pixel distances on Right Hand:
            #   Index(8), Middle(12), Ring(16), Pinky(20) vs Thumb Tip(4)
            d_idx = dist_px(THUMB_TIP, INDEX_TIP)
            d_mid = dist_px(THUMB_TIP, MIDDLE_TIP)
            d_rng = dist_px(THUMB_TIP, RING_TIP)
            d_pnk = dist_px(THUMB_TIP, PINKY_TIP)
            f.thumb_tip_dists_px = (d_idx, d_mid, d_rng, d_pnk)
            f.pinch_px = min(d_idx, d_mid, d_rng)
            f.pinch_dist = f.pinch_px / max(self.frame_w, 1.0)
        else:
            f.pinch_dist = dist2d(THUMB_TIP, INDEX_TIP)
            f.pinch_px = dist_px(THUMB_TIP, INDEX_TIP)

        # Left Hand Chromatic RGB channels (Middle=R, Ring=G, Pinky=B, or finger chord in RECOLOR mode)
        if slot == 0:
            cmask = 0
            for bit, tip_idx in enumerate(LEFT_CHROMA_TIPS):
                if dist2d(THUMB_TIP, tip_idx) < CHROMA_DISTANCE:
                    cmask |= 1 << bit
            f.contact_mask = cmask
            if cmask != 0:
                self.last_color = MASK_COLORS[cmask]
            f.color = self.last_color
        return f

    def update(self, left_lm: Optional[np.ndarray], right_lm: Optional[np.ndarray],
               now: float) -> Tuple[Optional[HandFeatures], Optional[HandFeatures]]:
        left_f: Optional[HandFeatures] = None
        right_f: Optional[HandFeatures] = None
        self.spawn_trigger = False
        self.group_trigger = False

        # --- 1. Left Hand: 150ms Debounced Mode Selector (0..7) ---
        if left_lm is not None:
            left_f = self._analyze_single_hand(0, left_lm)
            non_thumb = bin(left_f.fingers_mask >> 1).count("1")
            if non_thumb == 0:
                dx = float(left_lm[THUMB_TIP, 0] - left_lm[INDEX_MCP, 0])
                dy = float(left_lm[THUMB_TIP, 1] - left_lm[INDEX_MCP, 1])
                thumb_away = (dx * dx + dy * dy) ** 0.5
                thumb_tip_y = float(left_lm[THUMB_TIP, 1])
                thumb_ip_y = float(left_lm[THUMB_IP, 1])
                wrist_y = float(left_lm[WRIST, 1])
                thumb_extended_away = thumb_away > (0.75 * left_f.hand_scale)
                if thumb_extended_away and (thumb_tip_y < thumb_ip_y - 0.03) and (thumb_tip_y < wrist_y - 0.05):
                    candidate_mode = MODE_SPAWN
                    left_f.fist = False
                elif thumb_extended_away and (thumb_tip_y > thumb_ip_y + 0.03) and (thumb_tip_y > wrist_y + 0.05):
                    candidate_mode = MODE_DELETE
                    left_f.fist = False
                else:
                    candidate_mode = MODE_IDLE
            elif non_thumb == 4 and (left_f.fingers_mask & 1):
                candidate_mode = MODE_GROUP
            elif non_thumb == 4:
                candidate_mode = MODE_RECOLOR
            else:
                candidate_mode = int(np.clip(non_thumb, 1, 3))
            self.left_mode = self.left_mode_debouncer.update(candidate_mode, now)
            left_f.gesture = self.left_mode
        else:
            # Safe fallback when Left Hand leaves frame: lock to IDLE after 150ms
            self.left_mode = self.left_mode_debouncer.update(MODE_IDLE, now)

        # --- 2. Right Hand: Clutch, Peak-Latch RECOLOR, SPAWN, or GROUP ---
        if right_lm is not None:
            right_f = self._analyze_single_hand(1, right_lm)
            dist_index, dist_middle, dist_ring, dist_pinky = right_f.thumb_tip_dists_px
            axis_dists = (
                (AXIS_X, dist_index),
                (AXIS_Y, dist_middle),
                (AXIS_Z, dist_ring),
            )

            if self.left_mode != MODE_RECOLOR:
                self.recolor_latched = [False, False, False]
                right_f.contact_mask = 0

            if self.left_mode == MODE_RECOLOR:
                # Peak-Channel OR-Latch: engage each channel independently (< 38px)
                if dist_index < RECOLOR_ENGAGE_PX:
                    self.recolor_latched[0] = True
                if dist_middle < RECOLOR_ENGAGE_PX:
                    self.recolor_latched[1] = True
                if dist_ring < RECOLOR_ENGAGE_PX:
                    self.recolor_latched[2] = True

                # Session Reset: ONLY clear latched R/G/B once ALL THREE fingers release > 50px
                if (
                    dist_index > RECOLOR_RELEASE_PX
                    and dist_middle > RECOLOR_RELEASE_PX
                    and dist_ring > RECOLOR_RELEASE_PX
                ):
                    self.recolor_latched = [False, False, False]

                recolor_mask = (
                    (0b001 if self.recolor_latched[0] else 0)
                    | (0b010 if self.recolor_latched[1] else 0)
                    | (0b100 if self.recolor_latched[2] else 0)
                )
                right_f.contact_mask = recolor_mask
                self.is_pinching = (recolor_mask != 0)
                if recolor_mask != 0:
                    self.last_color = MASK_COLORS[recolor_mask]
                    right_f.pinch_px = min(
                        d for bit, d in enumerate((dist_index, dist_middle, dist_ring))
                        if (recolor_mask & (1 << bit))
                    )
                else:
                    right_f.pinch_px = min(dist_index, dist_middle, dist_ring)
                right_f.color = self.last_color
                if left_f is not None:
                    left_f.color = self.last_color
            elif self.left_mode == MODE_SPAWN:
                # SPAWN MODE (Left Hand Thumbs-Up): 4-finger Shape Picker + Single-Shot Rising Edge
                # Index=0 (CUBE), Middle=1 (CAPSULE), Ring=2 (CYLINDER), Pinky=3 (RING/TORUS)
                shape_dists = (
                    (0, dist_index),
                    (1, dist_middle),
                    (2, dist_ring),
                    (3, dist_pinky),
                )
                if not self.is_pinching:
                    closest_shape, min_dist = min(shape_dists, key=lambda item: item[1])
                    self.spawn_shape = closest_shape
                    self.active_shape = closest_shape
                    right_f.pinch_px = min_dist
                    if min_dist < SPAWN_ENGAGE_PX:
                        self.is_pinching = True
                        self.spawn_trigger = True  # Single-frame rising-edge event
                else:
                    locked_dist = (dist_index, dist_middle, dist_ring, dist_pinky)[self.active_shape]
                    right_f.pinch_px = locked_dist
                    self.spawn_trigger = False
                    if locked_dist > SPAWN_RELEASE_PX:
                        self.is_pinching = False
                        closest_shape, min_dist = min(shape_dists, key=lambda item: item[1])
                        self.spawn_shape = closest_shape
                        self.active_shape = closest_shape
                        right_f.pinch_px = min_dist
            elif self.left_mode == MODE_GROUP:
                # GROUP MODE (Left Hand 5 fingers): 4-finger rising-edge actions
                # 0=Index (SELECT/DESELECT), 1=Middle (GROUP), 2=Ring (UNGROUP), 3=Pinky (CLEAR)
                group_dists = (
                    (GROUP_ACTION_SELECT, dist_index),
                    (GROUP_ACTION_GROUP, dist_middle),
                    (GROUP_ACTION_UNGROUP, dist_ring),
                    (GROUP_ACTION_CLEAR, dist_pinky),
                )
                closest_action, min_dist = min(group_dists, key=lambda item: item[1])
                if not self.is_pinching:
                    self.group_action = closest_action
                    right_f.pinch_px = min_dist
                    if min_dist < GROUP_ENGAGE_PX:
                        self.is_pinching = True
                        self.group_trigger = True  # Single-frame rising-edge action
                else:
                    locked_dist = (dist_index, dist_middle, dist_ring, dist_pinky)[self.group_action]
                    right_f.pinch_px = locked_dist
                    self.group_trigger = False
                    # Lock until ALL FOUR fingertips release past 48px
                    if (
                        dist_index > GROUP_RELEASE_PX
                        and dist_middle > GROUP_RELEASE_PX
                        and dist_ring > GROUP_RELEASE_PX
                        and dist_pinky > GROUP_RELEASE_PX
                    ):
                        self.is_pinching = False
                        self.group_action = closest_action
                        right_f.pinch_px = min_dist
                right_f.group_action = self.group_action
            else:
                # MOVE / RESIZE / ROTATE / IDLE / DELETE: Single-finger pinch axis & clutch
                closest_axis, min_dist = min(axis_dists, key=lambda item: item[1])
                if not self.is_pinching:
                    self.right_axis = closest_axis
                    right_f.pinch_px = min_dist
                    if min_dist < self.engage_px:
                        self.is_pinching = True
                else:
                    if self.right_axis == AXIS_X:
                        locked_dist = dist_index
                    elif self.right_axis == AXIS_Y:
                        locked_dist = dist_middle
                    else:
                        locked_dist = dist_ring

                    if (
                        closest_axis != self.right_axis
                        and min_dist < self.engage_px
                        and (min_dist + 6.0) < locked_dist
                    ):
                        self.right_axis = closest_axis
                        locked_dist = min_dist

                    right_f.pinch_px = locked_dist
                    if locked_dist > self.release_px:
                        self.is_pinching = False
                        self.right_axis = closest_axis
                        right_f.pinch_px = min_dist

            right_f.pinch = self.is_pinching
            right_f.gesture = self.right_axis

            # 3D Cursor X/Y from midpoint between Right Thumb Tip (4) and active fingertip
            if self.left_mode == MODE_SPAWN:
                active_tip = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)[self.active_shape]
            elif self.left_mode == MODE_GROUP:
                active_tip = INDEX_TIP
            elif self.right_axis == AXIS_X:
                active_tip = INDEX_TIP
            elif self.right_axis == AXIS_Y:
                active_tip = MIDDLE_TIP
            else:
                active_tip = RING_TIP
            cx_norm = 0.5 * float(right_lm[THUMB_TIP, 0] + right_lm[active_tip, 0])
            cy_norm = 0.5 * float(right_lm[THUMB_TIP, 1] + right_lm[active_tip, 1])
            wx = (cx_norm - 0.5) * 10.0          # [-5.0, +5.0] world units (= +/-50 cm)
            wy = (0.5 - cy_norm) * 7.5           # [-3.75, +3.75] world units

            # Apparent 2D palm scale for Z-axis push/pull movement:
            # palm_size = hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
            raw_palm_size = float(
                np.hypot(
                    right_lm[MIDDLE_MCP, 0] - right_lm[WRIST, 0],
                    right_lm[MIDDLE_MCP, 1] - right_lm[WRIST, 1],
                )
            )
            self.smoothed_palm_size = float(
                self.palm_size_ema(np.array([raw_palm_size], dtype=np.float32), now)[0]
            )
            # Send smoothed apparent palm_size as cursor_3d.z
            self.cursor_3d = (float(wx), float(wy), float(self.smoothed_palm_size))
        else:
            self.is_pinching = False
            self.recolor_latched = [False, False, False]
            self.palm_size_ema.reset()

        return left_f, right_f


# --------------------------------------------------------------------------- #
# Hand Slot Assignment (Slot 0 = Physical LEFT, Slot 1 = Physical RIGHT)
# --------------------------------------------------------------------------- #
def assign_hand_slots(hands: List[RawHand], mirrored: bool, swap: bool) -> List[Optional[RawHand]]:
    slots: List[Optional[RawHand]] = [None, None]
    if not hands:
        return slots

    def slot_of(h: RawHand) -> int:
        s = 0 if h.label.lower().startswith("l") else 1
        if not mirrored:
            s = 1 - s
        return 1 - s if swap else s

    if len(hands) >= 2:
        a, b = sorted(hands[:2], key=lambda h: h.score, reverse=True)
        sa, sb = slot_of(a), slot_of(b)
        if sa == sb:
            left_first = a.landmarks[WRIST, 0] < b.landmarks[WRIST, 0]
            if not mirrored:
                left_first = not left_first
            sa, sb = (0, 1) if left_first else (1, 0)
        slots[sa], slots[sb] = a, b
    else:
        slots[slot_of(hands[0])] = hands[0]
    return slots


# --------------------------------------------------------------------------- #
# Binary Packet Serialization & Chunked Video Sender
# --------------------------------------------------------------------------- #
_EMPTY_HAND_BYTES = struct.pack(HAND_FMT, *([0] * 8), *([0.0] * 6), *([0.0] * 63))


def pack_hand_binary(slot: int, lm: np.ndarray, f: HandFeatures, score: float) -> bytes:
    state_flags = (
        (S_PINCH if f.pinch else 0)
        | (S_FIST if f.fist else 0)
        | (S_OPEN if f.open else 0)
        | (S_THUMB_FOLDED if f.thumb_folded else 0)
    )
    return struct.pack(
        HAND_FMT,
        1,
        slot,
        f.fingers_mask,
        f.gesture,
        f.contact_mask,
        state_flags,
        f.finger_count,
        int(f.group_action) & 0x03,
        score,
        f.pinch_dist,
        f.hand_scale,
        f.color[0],
        f.color[1],
        f.color[2],
        *lm.reshape(-1).tolist(),
    )


def send_video_chunks(sock: socket.socket, addr: Tuple[str, int], frame_bgr: np.ndarray,
                      frame_id: int, width: int = 320) -> None:
    h0, w0 = frame_bgr.shape[:2]
    width = max(64, min(width, w0)) & ~3
    height = max(48, int(round(width * h0 / w0)))
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    buf = np.ascontiguousarray(rgb).tobytes()
    row_bytes = width * 3
    rows_per = max(1, (MAX_DATAGRAM - VIDEO_HDR_SIZE) // row_bytes)
    fid = frame_id & 0xFFFFFFFF
    for r in range(0, height, rows_per):
        n = min(rows_per, height - r)
        hdr = struct.pack(VIDEO_FMT, VIDEO_MAGIC, fid, width, height, r, n, 3, 0, 0)
        try:
            sock.sendto(hdr + buf[r * row_bytes:(r + n) * row_bytes], addr)
        except OSError:
            break


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Visual3D Mode + Clutch Hand Tracker")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    ap.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    ap.add_argument("--fps", type=int, default=CAPTURE_FPS)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.70)
    ap.add_argument("--track-conf", type=float, default=0.60)
    ap.add_argument("--model-complexity", type=int, default=MODEL_COMPLEXITY, choices=(0, 1))
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--video", default="rgb", choices=("rgb", "off"),
                    help="stream low-res RGB webcam frames for Raylib AR / Studio PiP")
    ap.add_argument("--video-width", type=int, default=320)
    ap.add_argument("--video-fps", type=float, default=24.0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--swap-hands", action="store_true")
    ap.add_argument("--debug-view", "--preview", dest="debug_view", action="store_true",
                    default=DEBUG_VIEW)
    ap.add_argument("--model", default=os.path.join(here, "models", "hand_landmarker.task"))
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    mirrored = not args.no_mirror

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = actual_w / float(max(1, actual_h))

    # Initialize MediaPipe Hands ONCE outside the main while loop
    use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    hands_detector = None
    tasks_detector = None
    if use_solutions:
        hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.det_conf,
            min_tracking_confidence=args.track_conf,
        )
        backend_name = f"mp.solutions.hands (model_complexity={args.model_complexity})"
    else:
        tasks_detector = TasksHandsFallback(args.max_hands, args.det_conf, args.track_conf, args.model)
        backend_name = "mediapipe.tasks.vision.HandLandmarker (Lite)"

    print(f"[vision] Camera {args.camera}: {actual_w}x{actual_h} @ {args.fps} FPS | {backend_name}")
    print(f"[vision] Mode+Clutch UDP ({PACKET_SIZE}B) -> {args.host}:{args.port} | video={args.video}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
    sock.setblocking(False)
    target_addr = (args.host, args.port)

    smoothers = [LandmarkEMA(args.alpha), LandmarkEMA(args.alpha)]
    engine = ModeClutchEngine(actual_w, actual_h, PINCH_ENGAGE_PX, PINCH_RELEASE_PX, MODE_DEBOUNCE_SEC)

    stop = False

    def _sig_handler(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig_handler)

    t0 = time.time()
    seq = 0
    fps = 0.0
    stat_t = time.perf_counter()
    stat_frames = 0
    last_video_ts = 0.0
    video_period = 1.0 / max(1.0, args.video_fps)
    flags = FLAG_MIRRORED if mirrored else 0

    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                continue
            cap_ts = time.time()

            if mirrored:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            if hands_detector is not None:
                mp_res = hands_detector.process(rgb_frame)
                raw_hands = extract_solutions_hands(mp_res)
            else:
                raw_hands = tasks_detector.process(rgb_frame, int((cap_ts - t0) * 1000.0))

            slots = assign_hand_slots(raw_hands, mirrored, args.swap_hands)

            now = time.time()
            smoothed: List[Optional[np.ndarray]] = [None, None]
            for s in (0, 1):
                if slots[s] is None:
                    smoothers[s].reset()
                else:
                    smoothed[s] = smoothers[s](slots[s].landmarks, now)

            left_f, right_f = engine.update(smoothed[0], smoothed[1], now)
            feats = [left_f, right_f]

            hand_blobs = [_EMPTY_HAND_BYTES, _EMPTY_HAND_BYTES]
            for s in (0, 1):
                if smoothed[s] is not None and feats[s] is not None and slots[s] is not None:
                    hand_blobs[s] = pack_hand_binary(s, smoothed[s], feats[s], slots[s].score)

            num_hands = (1 if smoothed[0] is not None else 0) + (1 if smoothed[1] is not None else 0)
            pinch_px = right_f.pinch_px if right_f is not None else 999.0
            cx, cy, cz = engine.cursor_3d

            pkt_flags = (
                flags
                | (FLAG_SPAWN_TRIGGER if engine.spawn_trigger else 0)
                | (FLAG_GROUP_TRIGGER if engine.group_trigger else 0)
            )
            header = struct.pack(
                HEADER_FMT,
                TELEMETRY_MAGIC,
                seq & 0xFFFFFFFF,
                cap_ts,
                time.time(),
                fps,
                aspect,
                num_hands,
                pkt_flags,
                engine.left_mode,
                engine.right_axis,
                1 if engine.is_pinching else 0,
                engine.active_shape,
                PROTOCOL_VERSION,
                cx,
                cy,
                cz,
                pinch_px,
            )
            try:
                sock.sendto(header + hand_blobs[0] + hand_blobs[1], target_addr)
            except OSError:
                pass

            if args.video != "off" and (now - last_video_ts) >= video_period:
                last_video_ts = now
                send_video_chunks(sock, target_addr, frame, seq, args.video_width)

            seq += 1
            stat_frames += 1
            tnow = time.perf_counter()
            if tnow - stat_t >= 1.0:
                fps = stat_frames / (tnow - stat_t)
                stat_t = tnow
                stat_frames = 0
                if not args.quiet:
                    print(
                        f"[vision] {fps:5.1f} FPS | L_MODE={MODE_NAMES[engine.left_mode]:11s} | "
                        f"R_AXIS={AXIS_NAMES[engine.right_axis]:6s} ({AXIS_FINGER_NAMES[engine.right_axis]}) | PINCH={int(engine.is_pinching)} "
                        f"({pinch_px:5.1f}px) | CURSOR=({cx:+.2f},{cy:+.2f},{cz:+.2f})",
                        flush=True,
                    )
    finally:
        cap.release()
        if hands_detector is not None:
            hands_detector.close()
        if tasks_detector is not None:
            tasks_detector.close()
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
