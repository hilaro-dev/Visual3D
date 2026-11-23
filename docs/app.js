// app.js -- Visual3D Standalone Web Port (Three.js + @mediapipe/tasks-vision)
// Replicates the exact Two-Handed "Mode + Clutch" spatial architecture, gesture
// math, 3D Studio / AR HUD aesthetic, and Wavefront .OBJ & .3dmproj file formats
// from python_tracker/hand_tracker.py and cpp_engine/src/main.cpp.

import * as THREE from "three";
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm";

// ---------------------------------------------------------------------------
// 1. Constants & Enums (Exact match with hand_tracker.py & SceneObject.hpp)
// ---------------------------------------------------------------------------
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;

const MODE_DEBOUNCE_SEC = 0.150;
const PINCH_ENGAGE_PX = 32.0;
const PINCH_RELEASE_PX = 48.0;
const RECOLOR_ENGAGE_PX = 38.0;
const RECOLOR_RELEASE_PX = 50.0;
const SPAWN_ENGAGE_PX = 32.0;
const SPAWN_RELEASE_PX = 48.0;
const GROUP_ENGAGE_PX = 32.0;
const GROUP_RELEASE_PX = 48.0;
const CHROMA_DISTANCE = 0.048;

const CM_PER_WORLD_UNIT = 10.0;
const SNAP_MOVE_STEP_UNITS = 0.5;   // 0.5 world units = 5.0 cm (X/Y)
const SNAP_MOVE_STEP_Z_CM = 1.0;    // 1.0 cm snapped increments along Z-axis
const Z_SENSITIVITY_CM = 320.0;     // Apparent 2D palm scale -> Z displacement (cm)
const SNAP_SCALE_STEP_UNITS = 0.5;  // 0.5 world units = 5.0 cm
const SNAP_ROTATE_STEP_DEG = 15.0;  // 15.0 degree rotation increments

const DELETE_HOLD_DURATION_SEC = 1.50;
const DELETE_COOLDOWN_DURATION_SEC = 1.00;

const LeftMode = Object.freeze({
  Idle: 0,
  Move: 1,
  Resize: 2,
  Rotate: 3,
  Recolor: 4,
  Group: 5,
  Spawn: 6,
  Delete: 7,
});

const RightAxis = Object.freeze({
  None: 0,
  X: 1,
  Y: 2,
  Z: 3,
  XYZ: 4,
});

const GroupAction = Object.freeze({
  SelectToggle: 0,
  Group: 1,
  Ungroup: 2,
  ClearSelection: 3,
});

const PrimitiveType = Object.freeze({
  Cube: 0,
  Capsule: 1,
  Cylinder: 2,
  Torus: 3,
  ImportedObj: 4,
});

const ViewportMode = Object.freeze({
  Studio3D: 0,
  ArCamera: 1,
});

// Landmark indices
const WRIST = 0;
const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

const FINGERS = [
  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
  [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
  [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
];
const LEFT_CHROMA_TIPS = [MIDDLE_TIP, RING_TIP, PINKY_TIP];

const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const MASK_COLORS = {
  0b000: [0.15, 0.90, 1.00],
  0b001: [1.00, 0.00, 0.00], // Red     (255,   0,   0) -- Index + Thumb
  0b010: [0.00, 1.00, 0.00], // Green   (  0, 255,   0) -- Middle + Thumb
  0b100: [0.00, 0.00, 1.00], // Blue    (  0,   0, 255) -- Ring + Thumb
  0b011: [1.00, 1.00, 0.00], // Yellow  (255, 255,   0) -- Index + Middle
  0b110: [0.00, 1.00, 1.00], // Cyan    (  0, 255, 255) -- Middle + Ring
  0b101: [1.00, 0.00, 1.00], // Magenta (255,   0, 255) -- Ring + Index
  0b111: [1.00, 1.00, 1.00], // White   (255, 255, 255) -- Index + Middle + Ring
};

const HAND_LANDMARKER_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const VISION_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

// ---------------------------------------------------------------------------
// 2. Helper Functions (Naming, Colors, Math & Raylib Quaternion Equivalence)
// ---------------------------------------------------------------------------
function modeName(m) {
  switch (m) {
    case LeftMode.Idle: return "IDLE";
    case LeftMode.Move: return "MOVE";
    case LeftMode.Resize: return "RESIZE";
    case LeftMode.Rotate: return "ROTATE";
    case LeftMode.Recolor: return "RECOLOR";
    case LeftMode.Group: return "GROUP";
    case LeftMode.Spawn: return "SPAWN";
    case LeftMode.Delete: return "DELETE";
    default: return "IDLE";
  }
}

function axisName(a) {
  switch (a) {
    case RightAxis.X: return "X-AXIS";
    case RightAxis.Y: return "Y-AXIS";
    case RightAxis.Z: return "Z-AXIS";
    case RightAxis.XYZ: return "XYZ-ALL";
    default: return "X-AXIS";
  }
}

function axisFingerShort(a) {
  switch (a) {
    case RightAxis.X: return "Index";
    case RightAxis.Y: return "Middle";
    case RightAxis.Z: return "Ring";
    case RightAxis.XYZ: return "Pinky";
    default: return "Index";
  }
}

function spawnFingerShort(t) {
  switch (t) {
    case PrimitiveType.Cube: return "Index";
    case PrimitiveType.Capsule: return "Middle";
    case PrimitiveType.Cylinder: return "Ring";
    case PrimitiveType.Torus: return "Pinky";
    default: return "Index";
  }
}

function groupActionName(a) {
  switch (a) {
    case GroupAction.SelectToggle: return "SELECT";
    case GroupAction.Group: return "GROUP";
    case GroupAction.Ungroup: return "UNGROUP";
    case GroupAction.ClearSelection: return "CLEAR";
    default: return "SELECT";
  }
}

function groupActionFingerShort(a) {
  switch (a) {
    case GroupAction.SelectToggle: return "Index";
    case GroupAction.Group: return "Middle";
    case GroupAction.Ungroup: return "Ring";
    case GroupAction.ClearSelection: return "Pinky";
    default: return "Index";
  }
}

function primitiveName(t) {
  switch (t) {
    case PrimitiveType.Cube: return "CUBE";
    case PrimitiveType.Capsule: return "CAPSULE";
    case PrimitiveType.Cylinder: return "CYLINDER";
    case PrimitiveType.Torus: return "RING";
    case PrimitiveType.ImportedObj: return "OBJ";
    default: return "OBJECT";
  }
}

function axisColorRGB(a) {
  switch (a) {
    case RightAxis.X: return { r: 255, g: 75, b: 85 };
    case RightAxis.Y: return { r: 75, g: 245, b: 115 };
    case RightAxis.Z: return { r: 75, g: 165, b: 255 };
    case RightAxis.XYZ: return { r: 210, g: 115, b: 255 };
    default: return { r: 80, g: 225, b: 255 };
  }
}

function recolorColorFromMask(mask) {
  const r = (mask & 0x01) ? 255 : 0;
  const g = (mask & 0x02) ? 255 : 0;
  const b = (mask & 0x04) ? 255 : 0;
  return { r, g, b, a: 255 };
}

function recolorNameFromMask(mask) {
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

function rgbCss(c, alpha = 1.0) {
  return `rgba(${c.r | 0}, ${c.g | 0}, ${c.b | 0}, ${alpha})`;
}

function rgbToHexInt(c) {
  return ((c.r & 0xff) << 16) | ((c.g & 0xff) << 8) | (c.b & 0xff);
}

function snapToStep(value, step) {
  if (step <= 1e-6) return value;
  return Math.round(value / step) * step;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function countBits(n) {
  let c = 0;
  let v = n;
  while (v > 0) {
    c += v & 1;
    v >>= 1;
  }
  return c;
}

function sanitizeObjIdentifier(name) {
  let out = "";
  for (let i = 0; i < name.length; i++) {
    const ch = name[i];
    if (/^[A-Za-z0-9_-]$/.test(ch)) out += ch;
    else out += "_";
  }
  return out.length > 0 ? out : "Object";
}

function extractFilename(path) {
  if (!path) return "";
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

// Exact Raylib QuaternionFromEuler / QuaternionToEuler implementation
// so .3dmproj files are 100% interchangeable between C++ and Web.
function raylibQuaternionFromEuler(pitchRad, yawRad, rollRad) {
  const x0 = Math.cos(pitchRad * 0.5);
  const x1 = Math.sin(pitchRad * 0.5);
  const y0 = Math.cos(yawRad * 0.5);
  const y1 = Math.sin(yawRad * 0.5);
  const z0 = Math.cos(rollRad * 0.5);
  const z1 = Math.sin(rollRad * 0.5);

  return new THREE.Quaternion(
    x1 * y0 * z0 - x0 * y1 * z1,
    x0 * y1 * z0 + x1 * y0 * z1,
    x0 * y0 * z1 - x1 * y1 * z0,
    x0 * y0 * z0 + x1 * y1 * z1
  ).normalize();
}

function raylibQuaternionToEulerDeg(q) {
  const RAD2DEG = 180.0 / Math.PI;
  const x0 = 2.0 * (q.w * q.x + q.y * q.z);
  const x1 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
  const rx = Math.atan2(x0, x1) * RAD2DEG;

  let y0 = 2.0 * (q.w * q.y - q.z * q.x);
  y0 = clamp(y0, -1.0, 1.0);
  const ry = Math.asin(y0) * RAD2DEG;

  const z0 = 2.0 * (q.w * q.z + q.x * q.y);
  const z1 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  const rz = Math.atan2(z0, z1) * RAD2DEG;

  return { x: rx, y: ry, z: rz };
}

// ---------------------------------------------------------------------------
// 3. Smoothing & 150ms Time-Based Debouncer (from hand_tracker.py)
// ---------------------------------------------------------------------------
class LandmarkEMA {
  constructor(alpha = 0.25, resetTimeout = 0.25) {
    this.alpha = alpha;
    this.resetTimeout = resetTimeout;
    this.state = null;
    this.lastTs = 0.0;
  }

  update(raw, nowSec) {
    if (!this.state || (nowSec - this.lastTs) > this.resetTimeout) {
      this.state = raw.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    } else {
      for (let i = 0; i < raw.length; i++) {
        this.state[i].x += this.alpha * (raw[i].x - this.state[i].x);
        this.state[i].y += this.alpha * (raw[i].y - this.state[i].y);
        this.state[i].z += this.alpha * (raw[i].z - this.state[i].z);
      }
    }
    this.lastTs = nowSec;
    return this.state;
  }

  reset() {
    this.state = null;
  }
}

class ScalarEMA {
  constructor(alpha = 0.25, resetTimeout = 0.25) {
    this.alpha = alpha;
    this.resetTimeout = resetTimeout;
    this.state = null;
    this.lastTs = 0.0;
  }

  update(val, nowSec) {
    if (this.state === null || (nowSec - this.lastTs) > this.resetTimeout) {
      this.state = val;
    } else {
      this.state += this.alpha * (val - this.state);
    }
    this.lastTs = nowSec;
    return this.state;
  }

  reset() {
    this.state = null;
  }
}

class TimedDebouncer {
  constructor(initial = LeftMode.Idle, holdSec = MODE_DEBOUNCE_SEC) {
    this.value = initial;
    this.holdSec = holdSec;
    this.candidate = initial;
    this.candidateSince = 0.0;
  }

  update(candidate, nowSec) {
    if (candidate === this.value) {
      this.candidate = candidate;
      this.candidateSince = nowSec;
      return this.value;
    }
    if (candidate !== this.candidate) {
      this.candidate = candidate;
      this.candidateSince = nowSec;
      return this.value;
    }
    if ((nowSec - this.candidateSince) >= this.holdSec) {
      this.value = candidate;
    }
    return this.value;
  }

  reset(val, nowSec = 0.0) {
    this.value = val;
    this.candidate = val;
    this.candidateSince = nowSec;
  }
}

// ---------------------------------------------------------------------------
// 4. "Mode + Clutch" Gesture State Machine (Exact port of ModeClutchEngine)
// ---------------------------------------------------------------------------
class ModeClutchEngine {
  constructor() {
    this.frameW = CAPTURE_WIDTH;
    this.frameH = CAPTURE_HEIGHT;
    this.engagePx = PINCH_ENGAGE_PX;
    this.releasePx = PINCH_RELEASE_PX;
    this.leftModeDebouncer = new TimedDebouncer(LeftMode.Idle, MODE_DEBOUNCE_SEC);
    this.palmSizeEma = new ScalarEMA(0.25, 0.25);

    this.leftMode = LeftMode.Idle;
    this.rightAxis = RightAxis.X;
    this.isPinching = false;
    this.recolorLatched = [false, false, false];
    this.spawnShape = PrimitiveType.Cube;
    this.activeShape = PrimitiveType.Cube;
    this.spawnTrigger = false;
    this.groupAction = GroupAction.SelectToggle;
    this.groupTrigger = false;
    this.smoothedPalmSize = 0.18;
    this.cursor3d = { x: 0.0, y: 0.0, z: 0.18 };
    this.lastColor = [0.15, 0.90, 1.00];
  }

  analyzeSingleHand(slot, lm) {
    const dist2d = (a, b) => {
      const dx = lm[a].x - lm[b].x;
      const dy = lm[a].y - lm[b].y;
      return Math.hypot(dx, dy);
    };

    const distPx = (a, b) => {
      const dx = (lm[a].x - lm[b].x) * this.frameW;
      const dy = (lm[a].y - lm[b].y) * this.frameH;
      return Math.hypot(dx, dy);
    };

    const scale = Math.max(dist2d(WRIST, MIDDLE_MCP), 1e-4);

    // 4 non-thumb fingers (Index, Middle, Ring, Pinky)
    let mask = 0;
    for (let i = 0; i < FINGERS.length; i++) {
      const [, pip, , tip] = FINGERS[i];
      if (dist2d(WRIST, tip) > dist2d(WRIST, pip) * 1.15) {
        mask |= 1 << (i + 1);
      }
    }

    // Thumb extension check
    const dTipPinky = dist2d(THUMB_TIP, PINKY_MCP);
    const dIpPinky = dist2d(THUMB_IP, PINKY_MCP);
    const dMcpPinky = dist2d(THUMB_MCP, PINKY_MCP);
    const thumbFolded = (dTipPinky < dIpPinky) || (dTipPinky < dMcpPinky);
    const thumbExt = (!thumbFolded) && (dist2d(THUMB_TIP, INDEX_MCP) > 0.44 * scale);
    if (slot === 0 && thumbExt) {
      mask |= 1;
    }

    const ext4 = countBits(mask >> 1);
    const total5 = ext4 + (thumbExt ? 1 : 0);

    const f = {
      fingersMask: mask,
      fingerCount: slot === 0 ? total5 : ext4,
      gesture: 0,
      contactMask: 0,
      groupAction: 0,
      pinch: false,
      fist: slot === 0 ? (total5 === 0) : (ext4 === 0 && !thumbExt),
      open: total5 === 5,
      thumbFolded,
      pinchDist: 1.0,
      pinchPx: 999.0,
      handScale: scale,
      thumbTipDistsPx: [999.0, 999.0, 999.0, 999.0],
      color: [...this.lastColor],
    };

    if (slot === 1) {
      const dIdx = distPx(THUMB_TIP, INDEX_TIP);
      const dMid = distPx(THUMB_TIP, MIDDLE_TIP);
      const dRng = distPx(THUMB_TIP, RING_TIP);
      const dPnk = distPx(THUMB_TIP, PINKY_TIP);
      f.thumbTipDistsPx = [dIdx, dMid, dRng, dPnk];
      f.pinchPx = Math.min(dIdx, dMid, dRng, dPnk);
      f.pinchDist = f.pinchPx / Math.max(this.frameW, 1.0);
    } else {
      f.pinchDist = dist2d(THUMB_TIP, INDEX_TIP);
      f.pinchPx = distPx(THUMB_TIP, INDEX_TIP);
    }

    if (slot === 0) {
      let cmask = 0;
      for (let bit = 0; bit < LEFT_CHROMA_TIPS.length; bit++) {
        const tipIdx = LEFT_CHROMA_TIPS[bit];
        if (dist2d(THUMB_TIP, tipIdx) < CHROMA_DISTANCE) {
          cmask |= 1 << bit;
        }
      }
      f.contactMask = cmask;
      if (cmask !== 0) {
        this.lastColor = [...MASK_COLORS[cmask]];
      }
      f.color = [...this.lastColor];
    }

    return f;
  }

  update(leftLm, rightLm, nowSec, manualModeOverride = null) {
    let leftF = null;
    let rightF = null;
    this.spawnTrigger = false;
    this.groupTrigger = false;

    // --- 1. Left Hand: 150ms Debounced Mode Selector (0..7) ---
    if (leftLm) {
      leftF = this.analyzeSingleHand(0, leftLm);
      const nonThumb = countBits(leftF.fingersMask >> 1);
      let candidateMode = LeftMode.Idle;
      if (nonThumb === 0) {
        const dx = leftLm[THUMB_TIP].x - leftLm[INDEX_MCP].x;
        const dy = leftLm[THUMB_TIP].y - leftLm[INDEX_MCP].y;
        const thumbAway = Math.hypot(dx, dy);
        const thumbTipY = leftLm[THUMB_TIP].y;
        const thumbIpY = leftLm[THUMB_IP].y;
        const wristY = leftLm[WRIST].y;
        const thumbExtendedAway = thumbAway > (0.75 * leftF.handScale);
        if (thumbExtendedAway && (thumbTipY < thumbIpY - 0.03) && (thumbTipY < wristY - 0.05)) {
          candidateMode = LeftMode.Spawn;
          leftF.fist = false;
        } else if (thumbExtendedAway && (thumbTipY > thumbIpY + 0.03) && (thumbTipY > wristY + 0.05)) {
          candidateMode = LeftMode.Delete;
          leftF.fist = false;
        } else {
          candidateMode = LeftMode.Idle;
        }
      } else if (nonThumb === 4 && (leftF.fingersMask & 1)) {
        candidateMode = LeftMode.Group;
      } else if (nonThumb === 4) {
        candidateMode = LeftMode.Recolor;
      } else {
        candidateMode = clamp(nonThumb, 1, 3);
      }
      this.leftMode = this.leftModeDebouncer.update(candidateMode, nowSec);
      leftF.gesture = this.leftMode;
    } else if (manualModeOverride !== null) {
      this.leftModeDebouncer.reset(manualModeOverride, nowSec);
      this.leftMode = manualModeOverride;
    } else {
      this.leftMode = this.leftModeDebouncer.update(LeftMode.Idle, nowSec);
    }

    // --- 2. Right Hand: Clutch, Peak-Latch RECOLOR, SPAWN, or GROUP ---
    if (rightLm) {
      rightF = this.analyzeSingleHand(1, rightLm);
      const [distIndex, distMiddle, distRing, distPinky] = rightF.thumbTipDistsPx;
      const axisDists = [
        [RightAxis.X, distIndex],
        [RightAxis.Y, distMiddle],
        [RightAxis.Z, distRing],
        [RightAxis.XYZ, distPinky],
      ];

      if (this.leftMode !== LeftMode.Recolor) {
        this.recolorLatched = [false, false, false];
        rightF.contactMask = 0;
      }

      if (this.leftMode === LeftMode.Recolor) {
        if (distIndex < RECOLOR_ENGAGE_PX) this.recolorLatched[0] = true;
        if (distMiddle < RECOLOR_ENGAGE_PX) this.recolorLatched[1] = true;
        if (distRing < RECOLOR_ENGAGE_PX) this.recolorLatched[2] = true;

        if (
          distIndex > RECOLOR_RELEASE_PX &&
          distMiddle > RECOLOR_RELEASE_PX &&
          distRing > RECOLOR_RELEASE_PX
        ) {
          this.recolorLatched = [false, false, false];
        }

        const recolorMask =
          (this.recolorLatched[0] ? 0b001 : 0) |
          (this.recolorLatched[1] ? 0b010 : 0) |
          (this.recolorLatched[2] ? 0b100 : 0);

        rightF.contactMask = recolorMask;
        this.isPinching = recolorMask !== 0;
        if (recolorMask !== 0) {
          this.lastColor = [...MASK_COLORS[recolorMask]];
          const activeDists = [distIndex, distMiddle, distRing].filter(
            (_, bit) => (recolorMask & (1 << bit)) !== 0
          );
          rightF.pinchPx = Math.min(...activeDists);
        } else {
          rightF.pinchPx = Math.min(distIndex, distMiddle, distRing);
        }
        rightF.color = [...this.lastColor];
        if (leftF) leftF.color = [...this.lastColor];
      } else if (this.leftMode === LeftMode.Spawn) {
        const shapeDists = [
          [PrimitiveType.Cube, distIndex],
          [PrimitiveType.Capsule, distMiddle],
          [PrimitiveType.Cylinder, distRing],
          [PrimitiveType.Torus, distPinky],
        ];
        if (!this.isPinching) {
          let [closestShape, minDist] = shapeDists[0];
          for (const item of shapeDists) {
            if (item[1] < minDist) [closestShape, minDist] = item;
          }
          this.spawnShape = closestShape;
          this.activeShape = closestShape;
          rightF.pinchPx = minDist;
          if (minDist < SPAWN_ENGAGE_PX) {
            this.isPinching = true;
            this.spawnTrigger = true;
          }
        } else {
          const lockedDist = [distIndex, distMiddle, distRing, distPinky][this.activeShape];
          rightF.pinchPx = lockedDist;
          this.spawnTrigger = false;
          if (lockedDist > SPAWN_RELEASE_PX) {
            this.isPinching = false;
            let [closestShape, minDist] = shapeDists[0];
            for (const item of shapeDists) {
              if (item[1] < minDist) [closestShape, minDist] = item;
            }
            this.spawnShape = closestShape;
            this.activeShape = closestShape;
            rightF.pinchPx = minDist;
          }
        }
      } else if (this.leftMode === LeftMode.Group) {
        const groupDists = [
          [GroupAction.SelectToggle, distIndex],
          [GroupAction.Group, distMiddle],
          [GroupAction.Ungroup, distRing],
          [GroupAction.ClearSelection, distPinky],
        ];
        let [closestAction, minDist] = groupDists[0];
        for (const item of groupDists) {
          if (item[1] < minDist) [closestAction, minDist] = item;
        }
        if (!this.isPinching) {
          this.groupAction = closestAction;
          rightF.pinchPx = minDist;
          if (minDist < GROUP_ENGAGE_PX) {
            this.isPinching = true;
            this.groupTrigger = true;
          }
        } else {
          const lockedDist = [distIndex, distMiddle, distRing, distPinky][this.groupAction];
          rightF.pinchPx = lockedDist;
          this.groupTrigger = false;
          if (
            distIndex > GROUP_RELEASE_PX &&
            distMiddle > GROUP_RELEASE_PX &&
            distRing > GROUP_RELEASE_PX &&
            distPinky > GROUP_RELEASE_PX
          ) {
            this.isPinching = false;
            this.groupAction = closestAction;
            rightF.pinchPx = minDist;
          }
        }
        rightF.groupAction = this.groupAction;
      } else {
        let [closestAxis, minDist] = axisDists[0];
        for (const item of axisDists) {
          if (item[1] < minDist) [closestAxis, minDist] = item;
        }
        if (!this.isPinching) {
          this.rightAxis = closestAxis;
          rightF.pinchPx = minDist;
          if (minDist < this.engagePx) {
            this.isPinching = true;
          }
        } else {
          let lockedDist = distPinky;
          if (this.rightAxis === RightAxis.X) lockedDist = distIndex;
          else if (this.rightAxis === RightAxis.Y) lockedDist = distMiddle;
          else if (this.rightAxis === RightAxis.Z) lockedDist = distRing;

          if (
            this.rightAxis !== RightAxis.XYZ &&
            closestAxis !== RightAxis.XYZ &&
            closestAxis !== this.rightAxis &&
            minDist < this.engagePx &&
            (minDist + 6.0) < lockedDist
          ) {
            this.rightAxis = closestAxis;
            lockedDist = minDist;
          }

          rightF.pinchPx = lockedDist;
          if (lockedDist > this.releasePx) {
            this.isPinching = false;
            this.rightAxis = closestAxis;
            rightF.pinchPx = minDist;
          }
        }
      }

      rightF.pinch = this.isPinching;
      rightF.gesture = this.rightAxis;

      // 3D Cursor X/Y from midpoint between Right Thumb Tip (4) and active fingertip
      let activeTip = PINKY_TIP;
      if (this.leftMode === LeftMode.Spawn) {
        activeTip = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP][this.activeShape];
      } else if (this.leftMode === LeftMode.Group) {
        activeTip = INDEX_TIP;
      } else if (this.rightAxis === RightAxis.X) {
        activeTip = INDEX_TIP;
      } else if (this.rightAxis === RightAxis.Y) {
        activeTip = MIDDLE_TIP;
      } else if (this.rightAxis === RightAxis.Z) {
        activeTip = RING_TIP;
      }

      const cxNorm = 0.5 * (rightLm[THUMB_TIP].x + rightLm[activeTip].x);
      const cyNorm = 0.5 * (rightLm[THUMB_TIP].y + rightLm[activeTip].y);
      const wx = (cxNorm - 0.5) * 10.0; // [-5.0, +5.0] world units (= +/-50 cm)
      const wy = (0.5 - cyNorm) * 7.5;  // [-3.75, +3.75] world units

      const rawPalmSize = Math.hypot(
        rightLm[MIDDLE_MCP].x - rightLm[WRIST].x,
        rightLm[MIDDLE_MCP].y - rightLm[WRIST].y
      );
      this.smoothedPalmSize = this.palmSizeEma.update(rawPalmSize, nowSec);
      this.cursor3d = { x: wx, y: wy, z: this.smoothedPalmSize };
    } else {
      this.isPinching = false;
      this.recolorLatched = [false, false, false];
      this.palmSizeEma.reset();
    }

    return [leftF, rightF];
  }
}

// Hand slot assignment (Slot 0 = Physical Left Hand, Slot 1 = Physical Right Hand)
function assignHandSlots(rawHands, mirrored = true, swap = false) {
  const slots = [null, null];
  if (!rawHands || rawHands.length === 0) return slots;

  const slotOf = (h) => {
    let s = h.label.toLowerCase().startsWith("l") ? 0 : 1;
    if (!mirrored) s = 1 - s;
    return swap ? 1 - s : s;
  };

  if (rawHands.length >= 2) {
    const sorted = [...rawHands].sort((a, b) => b.score - a.score).slice(0, 2);
    const [a, b] = sorted;
    let sa = slotOf(a);
    let sb = slotOf(b);
    if (sa === sb) {
      let leftFirst = a.landmarks[WRIST].x < b.landmarks[WRIST].x;
      if (!mirrored) leftFirst = !leftFirst;
      sa = leftFirst ? 0 : 1;
      sb = leftFirst ? 1 : 0;
    }
    slots[sa] = a;
    slots[sb] = b;
  } else {
    slots[slotOf(rawHands[0])] = rawHands[0];
  }
  return slots;
}

// ---------------------------------------------------------------------------
// 5. Procedural Primitive Library & Wavefront .OBJ Parser / Exporter
// ---------------------------------------------------------------------------
function buildCapsuleBufferGeometry(radius = 0.45, cylHalfH = 0.65, slices = 20, rings = 8) {
  // Exact port of PrimitiveLibrary::buildCapsuleMesh from SceneObject.hpp
  const prof = [];
  for (let i = 0; i <= rings; i++) {
    const t = i / rings;
    const phi = -0.5 * Math.PI + t * (0.5 * Math.PI);
    prof.push({
      y: -cylHalfH + radius * Math.sin(phi),
      r: radius * Math.cos(phi),
      ny: Math.sin(phi),
    });
  }
  for (let i = 0; i <= rings; i++) {
    const t = i / rings;
    const phi = t * (0.5 * Math.PI);
    prof.push({
      y: cylHalfH + radius * Math.sin(phi),
      r: radius * Math.cos(phi),
      ny: Math.sin(phi),
    });
  }

  const numR = prof.length;
  const stride = slices + 1;
  const vCount = numR * stride;
  const triCount = (numR - 1) * slices * 2;

  const positions = new Float32Array(vCount * 3);
  const normals = new Float32Array(vCount * 3);
  const indices = new Uint32Array(triCount * 3);

  let vIdx = 0;
  for (let r = 0; r < numR; r++) {
    const ny = prof[r].ny;
    const nr = Math.sqrt(Math.max(0.0, 1.0 - ny * ny));
    for (let s = 0; s <= slices; s++) {
      const u = s / slices;
      const th = u * 2.0 * Math.PI;
      const cs = Math.cos(th);
      const sn = Math.sin(th);
      positions[vIdx * 3 + 0] = prof[r].r * cs;
      positions[vIdx * 3 + 1] = prof[r].y;
      positions[vIdx * 3 + 2] = prof[r].r * sn;
      normals[vIdx * 3 + 0] = nr * cs;
      normals[vIdx * 3 + 1] = ny;
      normals[vIdx * 3 + 2] = nr * sn;
      vIdx++;
    }
  }

  let iIdx = 0;
  for (let r = 0; r < numR - 1; r++) {
    for (let s = 0; s < slices; s++) {
      const i0 = r * stride + s;
      const i1 = i0 + 1;
      const i2 = i0 + stride;
      const i3 = i2 + 1;
      indices[iIdx++] = i0;
      indices[iIdx++] = i2;
      indices[iIdx++] = i1;
      indices[iIdx++] = i1;
      indices[iIdx++] = i2;
      indices[iIdx++] = i3;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  geom.computeBoundingBox();
  return geom;
}

function extractCpuMeshFromGeometry(geom) {
  const posAttr = geom.getAttribute("position");
  const normAttr = geom.getAttribute("normal");
  const idxAttr = geom.getIndex();

  const vertices = [];
  const normals = [];
  for (let i = 0; i < posAttr.count; i++) {
    vertices.push(new THREE.Vector3(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)));
    if (normAttr) {
      normals.push(new THREE.Vector3(normAttr.getX(i), normAttr.getY(i), normAttr.getZ(i)).normalize());
    } else {
      normals.push(new THREE.Vector3(0, 1, 0));
    }
  }

  const indices = [];
  if (idxAttr) {
    for (let i = 0; i < idxAttr.count; i++) {
      indices.push(idxAttr.getX(i));
    }
  } else {
    const usable = Math.floor(posAttr.count / 3) * 3;
    for (let i = 0; i < usable; i++) {
      indices.push(i);
    }
  }

  return { vertices, normals, indices };
}

class PrimitiveLibrary {
  constructor() {
    this.geometries = {};
    this.cpuMeshes = {};
    this.localBoxes = {};
  }

  init() {
    // 0: Cube (1.0 x 1.0 x 1.0)
    const cubeGeom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
    cubeGeom.computeBoundingBox();

    // 1: Capsule (radius=0.45, cylHalfH=0.65, slices=20, rings=8)
    const capsuleGeom = buildCapsuleBufferGeometry(0.45, 0.65, 20, 8);

    // 2: Cylinder (radius=0.5, height=1.1, slices=24)
    const cylinderGeom = new THREE.CylinderGeometry(0.5, 0.5, 1.1, 24, 1);
    cylinderGeom.computeBoundingBox();

    // 3: Ring / Torus (Raylib GenMeshTorus(0.32f, 0.95f, 20, 28) -> major=0.475, minor=0.16)
    const torusGeom = new THREE.TorusGeometry(0.475, 0.16, 20, 28);
    torusGeom.computeBoundingBox();

    const list = [
      [PrimitiveType.Cube, cubeGeom],
      [PrimitiveType.Capsule, capsuleGeom],
      [PrimitiveType.Cylinder, cylinderGeom],
      [PrimitiveType.Torus, torusGeom],
    ];

    for (const [type, geom] of list) {
      this.geometries[type] = geom;
      this.cpuMeshes[type] = extractCpuMeshFromGeometry(geom);
      this.localBoxes[type] = geom.boundingBox.clone();
    }
  }

  geometry(type) {
    return this.geometries[type];
  }

  cpuMesh(type) {
    return this.cpuMeshes[type];
  }

  localBox(type) {
    return this.localBoxes[type];
  }
}

// Parse & Normalize Wavefront .OBJ text (matches LoadNormalizedOBJAsset in SceneObject.hpp)
function parseAndNormalizeOBJ(objText, filename, targetExtent = 2.0) {
  const rawPositions = [];
  const rawNormals = [];
  const outVertices = [];
  const outNormals = [];
  const outIndices = [];
  let hasAnyNormals = false;

  const lines = objText.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    const cmd = parts[0];

    if (cmd === "v" && parts.length >= 4) {
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        rawPositions.push(new THREE.Vector3(x, y, z));
      }
    } else if (cmd === "vn" && parts.length >= 4) {
      const nx = parseFloat(parts[1]);
      const ny = parseFloat(parts[2]);
      const nz = parseFloat(parts[3]);
      if (Number.isFinite(nx) && Number.isFinite(ny) && Number.isFinite(nz)) {
        rawNormals.push(new THREE.Vector3(nx, ny, nz));
      }
    } else if (cmd === "f" && parts.length >= 4) {
      const faceVerts = [];
      for (let i = 1; i < parts.length; i++) {
        const tok = parts[i];
        if (!tok) continue;
        const sub = tok.split("/");
        let vIdx = parseInt(sub[0], 10);
        if (!Number.isFinite(vIdx) || vIdx === 0) continue;
        vIdx = vIdx > 0 ? vIdx - 1 : rawPositions.length + vIdx;
        if (vIdx < 0 || vIdx >= rawPositions.length) continue;

        let nIdx = -1;
        if (sub.length >= 3 && sub[2].length > 0) {
          const parsedN = parseInt(sub[2], 10);
          if (Number.isFinite(parsedN) && parsedN !== 0) {
            nIdx = parsedN > 0 ? parsedN - 1 : rawNormals.length + parsedN;
          }
        }
        faceVerts.push({ vIdx, nIdx });
      }

      // Fan triangulation for polygons with >= 3 vertices
      for (let i = 1; i + 1 < faceVerts.length; i++) {
        const tri = [faceVerts[0], faceVerts[i], faceVerts[i + 1]];
        const baseIdx = outVertices.length;
        for (const tv of tri) {
          outVertices.push(rawPositions[tv.vIdx].clone());
          if (tv.nIdx >= 0 && tv.nIdx < rawNormals.length) {
            const n = rawNormals[tv.nIdx].clone();
            if (n.lengthSq() > 1e-12) n.normalize();
            else n.set(0, 1, 0);
            outNormals.push(n);
            hasAnyNormals = true;
          } else {
            outNormals.push(null);
          }
        }
        outIndices.push(baseIdx, baseIdx + 1, baseIdx + 2);
      }
    }
  }

  if (outVertices.length === 0 || outIndices.length < 3) {
    return null;
  }

  // Raw bounding box
  const rawMin = outVertices[0].clone();
  const rawMax = outVertices[0].clone();
  for (let i = 1; i < outVertices.length; i++) {
    rawMin.min(outVertices[i]);
    rawMax.max(outVertices[i]);
  }

  // ComputeNormalizationParams (exact match with SceneObject.hpp lines 162-189)
  const rawCenter = new THREE.Vector3(
    0.5 * (rawMin.x + rawMax.x),
    0.5 * (rawMin.y + rawMax.y),
    0.5 * (rawMin.z + rawMax.z)
  );
  const extX = Math.max(0.0, rawMax.x - rawMin.x);
  const extY = Math.max(0.0, rawMax.y - rawMin.y);
  const extZ = Math.max(0.0, rawMax.z - rawMin.z);
  const maxExtent = Math.max(extX, extY, extZ, 1e-5);

  const pivotOffset = rawCenter.clone().negate();
  const normScale = targetExtent / maxExtent;
  const normalizedLocalBox = new THREE.Box3(
    new THREE.Vector3(
      (rawMin.x + pivotOffset.x) * normScale,
      (rawMin.y + pivotOffset.y) * normScale,
      (rawMin.z + pivotOffset.z) * normScale
    ),
    new THREE.Vector3(
      (rawMax.x + pivotOffset.x) * normScale,
      (rawMax.y + pivotOffset.y) * normScale,
      (rawMax.z + pivotOffset.z) * normScale
    )
  );

  // Transform vertices into normalized local space and synthesize missing flat normals
  for (let i = 0; i < outVertices.length; i++) {
    outVertices[i].add(pivotOffset).multiplyScalar(normScale);
  }

  for (let i = 0; i + 2 < outIndices.length; i += 3) {
    const i0 = outIndices[i];
    const i1 = outIndices[i + 1];
    const i2 = outIndices[i + 2];
    if (!hasAnyNormals || !outNormals[i0] || !outNormals[i1] || !outNormals[i2]) {
      const e1 = new THREE.Vector3().subVectors(outVertices[i1], outVertices[i0]);
      const e2 = new THREE.Vector3().subVectors(outVertices[i2], outVertices[i0]);
      const fn = new THREE.Vector3().crossVectors(e1, e2);
      if (fn.lengthSq() > 1e-12) fn.normalize();
      else fn.set(0, 1, 0);
      outNormals[i0] = outNormals[i0] || fn.clone();
      outNormals[i1] = outNormals[i1] || fn.clone();
      outNormals[i2] = outNormals[i2] || fn.clone();
    }
  }

  const posArr = new Float32Array(outVertices.length * 3);
  const normArr = new Float32Array(outVertices.length * 3);
  for (let i = 0; i < outVertices.length; i++) {
    posArr[i * 3 + 0] = outVertices[i].x;
    posArr[i * 3 + 1] = outVertices[i].y;
    posArr[i * 3 + 2] = outVertices[i].z;
    const n = outNormals[i] || new THREE.Vector3(0, 1, 0);
    normArr[i * 3 + 0] = n.x;
    normArr[i * 3 + 1] = n.y;
    normArr[i * 3 + 2] = n.z;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normArr, 3));
  geometry.setIndex(outIndices);
  geometry.computeBoundingBox();

  return {
    displayName: extractFilename(filename),
    sourcePath: filename,
    pivotOffset,
    normScale,
    normalizedLocalBox,
    geometry,
    cpuMesh: {
      vertices: outVertices,
      normals: outNormals,
      indices: outIndices,
    },
  };
}

// Compose world Matrix4 from ObjectTransform (T * R * S)
function composeTransformMatrix(transform) {
  const matScale = new THREE.Matrix4().makeScale(
    transform.scale.x,
    transform.scale.y,
    transform.scale.z
  );
  const matTrans = new THREE.Matrix4().makeTranslation(
    transform.position.x,
    transform.position.y,
    transform.position.z
  );
  return new THREE.Matrix4()
    .multiplyMatrices(matTrans, transform.rotMatrix)
    .multiply(matScale);
}

function cloneTransform(t) {
  return {
    position: { x: t.position.x, y: t.position.y, z: t.position.z },
    scale: { x: t.scale.x, y: t.scale.y, z: t.scale.z },
    rotationDeg: { x: t.rotationDeg.x, y: t.rotationDeg.y, z: t.rotationDeg.z },
    rotMatrix: t.rotMatrix.clone(),
  };
}

// Wavefront .OBJ Exporter (Exact match with ExportObjectsToOBJ in SceneObject.hpp)
function serializeObjectsToOBJ(objects, primitives) {
  const lines = [];
  lines.push("# Visual3D Wavefront .OBJ Export");
  lines.push("# Scale: 1.0 world unit = 10.0 cm");
  lines.push(`# Objects exported: ${objects.length}`);
  lines.push("");

  let vertexOffset = 1; // Wavefront .obj indices are 1-based
  for (const obj of objects) {
    let base = null;
    if (obj.type === PrimitiveType.ImportedObj) {
      if (obj.importedModel && obj.importedModel.cpuMesh.vertices.length > 0) {
        base = obj.importedModel.cpuMesh;
      }
    } else {
      base = primitives.cpuMesh(obj.type);
    }
    if (!base || base.vertices.length === 0) continue;

    const label =
      obj.type === PrimitiveType.ImportedObj && obj.importedModel
        ? obj.importedModel.displayName
        : primitiveName(obj.type);
    let objHeader = `o ${sanitizeObjIdentifier(label)}_${obj.id}`;
    if (obj.groupId >= 0) objHeader += `_Group${obj.groupId}`;
    lines.push(objHeader);
    if (obj.groupId >= 0) {
      lines.push(`g Group_${obj.groupId}`);
    }

    const worldMat = composeTransformMatrix(obj.transform);
    const matRot = obj.transform.rotMatrix;
    const invSx = 1.0 / (Math.abs(obj.transform.scale.x) > 1e-5 ? obj.transform.scale.x : 1.0);
    const invSy = 1.0 / (Math.abs(obj.transform.scale.y) > 1e-5 ? obj.transform.scale.y : 1.0);
    const invSz = 1.0 / (Math.abs(obj.transform.scale.z) > 1e-5 ? obj.transform.scale.z : 1.0);

    for (const v of base.vertices) {
      const tv = v.clone().applyMatrix4(worldMat);
      lines.push(`v ${tv.x.toFixed(6)} ${tv.y.toFixed(6)} ${tv.z.toFixed(6)}`);
    }

    for (const n of base.normals) {
      const scaledN = new THREE.Vector3(n.x * invSx, n.y * invSy, n.z * invSz);
      const tn = scaledN.applyMatrix4(matRot);
      if (tn.lengthSq() > 1e-12) tn.normalize();
      else tn.set(0, 1, 0);
      lines.push(`vn ${tn.x.toFixed(6)} ${tn.y.toFixed(6)} ${tn.z.toFixed(6)}`);
    }

    for (let i = 0; i + 2 < base.indices.length; i += 3) {
      const i0 = vertexOffset + base.indices[i];
      const i1 = vertexOffset + base.indices[i + 1];
      const i2 = vertexOffset + base.indices[i + 2];
      lines.push(`f ${i0}//${i0} ${i1}//${i1} ${i2}//${i2}`);
    }
    lines.push("");
    vertexOffset += base.vertices.length;
  }

  return lines.join("\n");
}

// .3dmproj Serializer & Parser (Exact match with SaveProjectToFile / LoadProjectFromFile)
function projectTypeToken(t) {
  switch (t) {
    case PrimitiveType.Cube: return "CUBE";
    case PrimitiveType.Capsule: return "CAPSULE";
    case PrimitiveType.Cylinder: return "CYLINDER";
    case PrimitiveType.Torus: return "RING";
    case PrimitiveType.ImportedObj: return "IMPORTED_OBJ";
    default: return "CUBE";
  }
}

function parseProjectTypeToken(tok) {
  if (tok === "CUBE") return PrimitiveType.Cube;
  if (tok === "CAPSULE") return PrimitiveType.Capsule;
  if (tok === "CYLINDER") return PrimitiveType.Cylinder;
  if (tok === "RING") return PrimitiveType.Torus;
  if (tok === "IMPORTED_OBJ") return PrimitiveType.ImportedObj;
  return null;
}

function escapeProjectPath(raw) {
  let out = '"';
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === "\\" || c === '"') out += "\\";
    out += c;
  }
  out += '"';
  return out;
}

function parseQuotedProjectPath(rest) {
  const openPos = rest.indexOf('"');
  if (openPos === -1) return null;
  let out = "";
  let closed = false;
  for (let i = openPos + 1; i < rest.length; i++) {
    const c = rest[i];
    if (c === "\\" && i + 1 < rest.length) {
      const nxt = rest[i + 1];
      if (nxt === "\\" || nxt === '"') {
        out += nxt;
        i++;
      } else {
        out += c;
      }
    } else if (c === '"') {
      closed = true;
      break;
    } else {
      out += c;
    }
  }
  return closed ? out : null;
}

function serializeProjectTo3dmproj(objects, nextGroupId) {
  const safeNextGroupId = Math.max(1, nextGroupId | 0);
  const lines = [`NEXT_GROUP_ID ${safeNextGroupId}`];

  for (const obj of objects) {
    const q = new THREE.Quaternion().setFromRotationMatrix(obj.transform.rotMatrix);
    const eulerDeg = raylibQuaternionToEulerDeg(q);
    const srcPath =
      obj.type === PrimitiveType.ImportedObj && obj.importedModel
        ? obj.importedModel.sourcePath
        : "";

    lines.push(
      [
        projectTypeToken(obj.type),
        obj.transform.position.x.toFixed(6),
        obj.transform.position.y.toFixed(6),
        obj.transform.position.z.toFixed(6),
        eulerDeg.x.toFixed(6),
        eulerDeg.y.toFixed(6),
        eulerDeg.z.toFixed(6),
        obj.transform.scale.x.toFixed(6),
        obj.transform.scale.y.toFixed(6),
        obj.transform.scale.z.toFixed(6),
        obj.color.r | 0,
        obj.color.g | 0,
        obj.color.b | 0,
        obj.color.a | 0,
        obj.groupId | 0,
        obj.isSelected ? 1 : 0,
        escapeProjectPath(srcPath),
      ].join(" ")
    );
  }
  return lines.join("\n") + "\n";
}

function parseProjectFrom3dmproj(text, importedAssetCache) {
  const lines = text.split(/\r?\n/);
  const parsed = [];
  let headerNextGroupId = -1;
  let maxGroupId = 0;
  let nextLocalId = 1;
  let parsedHeader = false;
  let missingObjCount = 0;

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    if (!parsedHeader) {
      const headerParts = trimmed.split(/\s+/);
      if (headerParts[0] === "NEXT_GROUP_ID") {
        if (headerParts.length < 2) return null;
        headerNextGroupId = parseInt(headerParts[1], 10);
        if (!Number.isFinite(headerNextGroupId) || headerNextGroupId < 1) return null;
      } else {
        headerNextGroupId = parseInt(headerParts[0], 10);
        if (!Number.isFinite(headerNextGroupId) || headerNextGroupId < 1) return null;
      }
      parsedHeader = true;
      continue;
    }

    // Parse 16 whitespace-separated fields followed by quoted path
    const tokenRegex = /^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$/;
    const match = trimmed.match(tokenRegex);
    if (!match) return null;

    const typeTok = match[1];
    const pType = parseProjectTypeToken(typeTok);
    if (pType === null) return null;

    const nums = match.slice(2, 17).map(Number);
    if (nums.some((n) => !Number.isFinite(n))) return null;

    const [px, py, pz, rx, ry, rz, sx, sy, sz, cr, cg, cb, ca, groupId, isSel] = nums;
    const remainder = match[17] || "";
    const srcPath = parseQuotedProjectPath(remainder);
    if (srcPath === null && pType === PrimitiveType.ImportedObj) {
      return null;
    }

    const DEG2RAD = Math.PI / 180.0;
    const q = raylibQuaternionFromEuler(rx * DEG2RAD, ry * DEG2RAD, rz * DEG2RAD);
    const rotMatrix = new THREE.Matrix4().makeRotationFromQuaternion(q);

    const transform = {
      position: { x: px, y: py, z: pz },
      rotationDeg: { x: rx, y: ry, z: rz },
      rotMatrix,
      scale: {
        x: Math.max(0.05, sx),
        y: Math.max(0.05, sy),
        z: Math.max(0.05, sz),
      },
    };

    let importedModel = null;
    if (pType === PrimitiveType.ImportedObj) {
      const baseName = extractFilename(srcPath || "");
      importedModel =
        importedAssetCache.get(srcPath) ||
        importedAssetCache.get(baseName) ||
        null;
      if (!importedModel) {
        missingObjCount++;
        continue;
      }
    }

    const obj = {
      id: nextLocalId++,
      type: pType,
      transform,
      anchorTransform: cloneTransform(transform),
      color: {
        r: clamp(cr | 0, 0, 255),
        g: clamp(cg | 0, 0, 255),
        b: clamp(cb | 0, 0, 255),
        a: clamp(ca | 0, 0, 255),
      },
      groupId: groupId >= 0 ? (groupId | 0) : -1,
      isSelected: isSel !== 0,
      isHovered: false,
      importedModel,
    };

    if (obj.groupId >= 0) {
      maxGroupId = Math.max(maxGroupId, obj.groupId);
    }
    parsed.push(obj);
  }

  if (!parsedHeader) return null;

  return {
    objects: parsed,
    nextGroupId: Math.max(1, headerNextGroupId, maxGroupId + 1),
    missingObjCount,
  };
}

function triggerBlobDownload(filename, textContent) {
  const blob = new Blob([textContent], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------
// 6. Main Visual3D Web Application
// ---------------------------------------------------------------------------
class Visual3DWebApp {
  constructor() {
    // DOM Elements
    this.arBgCanvas = document.getElementById("ar-bg-canvas");
    this.arBgCtx = this.arBgCanvas.getContext("2d");
    this.threeCanvas = document.getElementById("three-canvas");
    this.hud2dCanvas = document.getElementById("hud-2d-canvas");
    this.hud2dCtx = this.hud2dCanvas.getContext("2d");
    this.pipContainer = document.getElementById("pip-container");
    this.pipCanvas = document.getElementById("pip-canvas");
    this.pipCtx = this.pipCanvas.getContext("2d");
    this.videoEl = document.getElementById("webcam-video");

    // Offscreen 640x480 mirrored capture canvas (feeds MediaPipe identically to cv2.flip(frame, 1))
    this.captureCanvas = document.createElement("canvas");
    this.captureCanvas.width = CAPTURE_WIDTH;
    this.captureCanvas.height = CAPTURE_HEIGHT;
    this.captureCtx = this.captureCanvas.getContext("2d", { willReadFrequently: false });

    // HUD Elements
    this.telemetryCard = document.getElementById("telemetry-card");
    this.telViewport = document.getElementById("tel-viewport");
    this.telLMode = document.getElementById("tel-lmode");
    this.telRAxis = document.getElementById("tel-raxis");
    this.telClutch = document.getElementById("tel-clutch");
    this.telEngine = document.getElementById("tel-engine");
    this.telScene = document.getElementById("tel-scene");

    this.modeBanner = document.getElementById("mode-banner");
    this.modeBannerText = document.getElementById("mode-banner-text");
    this.modeBannerSwatch = document.getElementById("mode-banner-swatch");
    this.modeDeleteTrack = document.getElementById("mode-delete-track");
    this.modeDeleteFill = document.getElementById("mode-delete-fill");

    this.toastBanner = document.getElementById("toast-banner");
    this.bottomHud = document.getElementById("bottom-hud");
    this.dockButtons = Array.from(document.querySelectorAll(".dock-btn"));
    this.manualDrawerWrapper = document.getElementById("manual-drawer-wrapper");
    this.manualTab = document.getElementById("manual-tab");
    this.dropOverlay = document.getElementById("drop-overlay");
    this.fileInputObj = document.getElementById("file-input-obj");
    this.fileInputProj = document.getElementById("file-input-proj");
    this.camToggleBtn = document.getElementById("cam-toggle-btn");

    // Vision & Mode+Clutch Engine
    this.handLandmarker = null;
    this.visionReady = false;
    this.cameraReady = false;
    this.visionStatus = "LOADING MEDIAPIPE...";
    this.lastVideoTime = -1;
    this.smoothers = [new LandmarkEMA(0.25, 0.25), new LandmarkEMA(0.25, 0.25)];
    this.engine = new ModeClutchEngine();
    this.hands = [
      { present: false, landmarks: null, features: null, score: 0 },
      { present: false, landmarks: null, features: null, score: 0 },
    ];
    this.manualModeOverride = null; // Dock click mode fallback when Left Hand is not in frame

    // Scene & State Machine Variables (exact match with ModeClutchApp in main.cpp)
    this.primitives = new PrimitiveLibrary();
    this.importedAssetCache = new Map();
    this.objects = [];
    this.nextId = 1;
    this.nextGroupId = 1;
    this.activeObjectIdx = -1;

    this.viewportMode = ViewportMode.Studio3D;
    this.leftMode = LeftMode.Idle;
    this.rightAxis = RightAxis.X;
    this.lastTrackerRightAxis = RightAxis.X;
    this.hasKeyboardAxisOverride = false;
    this.activeShape = PrimitiveType.Cube;
    this.activeGroupAction = GroupAction.SelectToggle;
    this.activeColor = { r: 38, g: 230, b: 255, a: 255 };
    this.activeRecolorMask = 0;
    this.recolorSessionMask = 0;
    this.activeColorName = "CYAN";

    this.isPinching = false;
    this.wasPinching = false;
    this.spawnTriggerFromPkt = false;
    this.groupTriggerFromPkt = false;
    this.pinchPx = 999.0;
    this.deleteHoldTimer = 0.0;
    this.deleteCooldownTimer = 0.0;
    this.currentPalmSize = 0.18;
    this.anchorPalmSize = 0.18;
    this.cursor3D = new THREE.Vector3(0.0, 0.5, 0.0);
    this.anchorHandPos = new THREE.Vector3(0.0, 0.5, 0.0);

    this.measurementBadge = "IDLE [SAFE] // 1:MOVE 2:SIZE 3:ROT 4:CLR 5:GRP TU:SPAWN TD:DEL";
    this.cleanHud = false;
    this.manualOpen = false;
    this.toastMessage = "";
    this.toastTimer = 0.0;
    this.toastIsError = false;

    // FPS tracking
    this.fps = 60;
    this.trackerFps = 0;
    this.frameCounter = 0;
    this.trackerFrameCounter = 0;
    this.fpsTimer = performance.now();
    this.lastFrameTime = performance.now();
  }

  async init() {
    this.primitives.init();
    this.initThreeJS();
    this.resetDefaultScene();
    this.bindEvents();
    this.onResize();

    // Start render loop right away so 3D studio is interactive immediately
    requestAnimationFrame((ts) => this.loop(ts));

    // Initialize Webcam + MediaPipe HandLandmarker asynchronously
    this.initWebcamAndVision();
  }

  initThreeJS() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.threeCanvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);

    this.scene = new THREE.Scene();

    // Exact camera from main.cpp lines 2287-2292
    this.camera = new THREE.PerspectiveCamera(
      45.0,
      window.innerWidth / Math.max(1, window.innerHeight),
      0.1,
      100.0
    );
    this.camera.position.set(0.0, 3.8, 8.5);
    this.camera.up.set(0.0, 1.0, 0.0);
    this.camera.lookAt(0.0, 0.5, 0.0);

    // Studio lighting to complement unlit/shaded CAD materials
    const ambient = new THREE.AmbientLight(0xffffff, 0.82);
    this.scene.add(ambient);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.65);
    dirLight1.position.set(5, 10, 7);
    this.scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0x90d8ff, 0.28);
    dirLight2.position.set(-6, -4, -5);
    this.scene.add(dirLight2);

    // Studio Grid & Origin Axis Lines (Mode B only)
    this.studioEnvGroup = new THREE.Group();
    // DrawGrid(20, 0.5f) -> 20 slices * 0.5f spacing = 10.0 total size, 20 divisions
    const gridHelper = new THREE.GridHelper(10.0, 20, 0x556070, 0x363c48);
    this.studioEnvGroup.add(gridHelper);

    // draw3DAxisOriginLines()
    const makeOriginLine = (p1, p2, hexColor) => {
      const g = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const m = new THREE.LineBasicMaterial({
        color: hexColor,
        transparent: true,
        opacity: 0.72,
      });
      return new THREE.Line(g, m);
    };
    this.studioEnvGroup.add(
      makeOriginLine(new THREE.Vector3(-5, 0.012, 0), new THREE.Vector3(5, 0.012, 0), 0xdc3c46)
    );
    this.studioEnvGroup.add(
      makeOriginLine(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 5, 0), 0x3cdc5f)
    );
    this.studioEnvGroup.add(
      makeOriginLine(new THREE.Vector3(0, 0.012, -5), new THREE.Vector3(0, 0.012, 5), 0x4691f5)
    );
    this.scene.add(this.studioEnvGroup);

    // Dynamic Scene Groups (re-populated or updated per frame)
    this.objectsGroup = new THREE.Group();
    this.scene.add(this.objectsGroup);

    this.overlays3DGroup = new THREE.Group();
    this.scene.add(this.overlays3DGroup);

    // Pre-build 3D Cursor Sphere + Wireframe Sphere
    this.cursorGroup = new THREE.Group();
    const cursorInnerGeom = new THREE.SphereGeometry(1.0, 16, 16);
    this.cursorInnerMat = new THREE.MeshBasicMaterial({ color: 0xff4b55 });
    this.cursorInnerMesh = new THREE.Mesh(cursorInnerGeom, this.cursorInnerMat);
    this.cursorGroup.add(this.cursorInnerMesh);

    const cursorWireGeom = new THREE.SphereGeometry(0.18, 8, 8);
    const cursorWireMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.85,
    });
    this.cursorWireMesh = new THREE.Mesh(cursorWireGeom, cursorWireMat);
    this.cursorGroup.add(this.cursorWireMesh);
    this.scene.add(this.cursorGroup);

    // Pre-build Ghost Preview Group for SPAWN mode
    this.ghostGroup = new THREE.Group();
    this.ghostFillMat = new THREE.MeshLambertMaterial({
      color: 0x26e6ff,
      transparent: true,
      opacity: 65 / 255,
      depthWrite: false,
    });
    this.ghostWireMat = new THREE.MeshBasicMaterial({
      color: 0x78faff,
      wireframe: true,
    });
    this.ghostFillMesh = new THREE.Mesh(this.primitives.geometry(PrimitiveType.Cube), this.ghostFillMat);
    this.ghostWireMesh = new THREE.Mesh(this.primitives.geometry(PrimitiveType.Cube), this.ghostWireMat);
    this.ghostGroup.add(this.ghostFillMesh);
    this.ghostGroup.add(this.ghostWireMesh);
    this.ghostGroup.visible = false;
    this.scene.add(this.ghostGroup);

    // Pre-build Glowing Axis Guide Builders (X, Y, Z)
    this.axisGuidesGroup = new THREE.Group();
    this.scene.add(this.axisGuidesGroup);
  }

  async initWebcamAndVision() {
    try {
      this.visionStatus = "STARTING WEBCAM...";
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          frameRate: { ideal: 60 },
          facingMode: "user",
        },
        audio: false,
      });
      this.videoEl.srcObject = stream;
      await this.videoEl.play();
      this.cameraReady = true;
    } catch (err) {
      console.warn("[vision] Webcam access unavailable:", err);
      this.cameraReady = false;
      this.visionStatus = "NO WEBCAM (CLICK CAM)";
    }

    try {
      this.visionStatus = "LOADING HAND MODEL...";
      const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_LANDMARKER_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.70,
        minHandPresenceConfidence: 0.50,
        minTrackingConfidence: 0.60,
      });
      this.visionReady = true;
      this.visionStatus = this.cameraReady ? "TRACKING ACTIVE" : "NO WEBCAM";
    } catch (err) {
      console.error("[vision] MediaPipe HandLandmarker initialization error:", err);
      this.visionReady = false;
      this.visionStatus = "MEDIAPIPE ERROR";
    }
  }

  resetDefaultScene() {
    this.clearThreeObjects();
    this.objects = [];
    this.nextId = 1;
    this.nextGroupId = 1;

    this.spawnObject(PrimitiveType.Cube, { x: -2.0, y: 0.5, z: 0.0 }, { r: 45, g: 225, b: 255, a: 255 });
    this.spawnObject(PrimitiveType.Capsule, { x: -0.5, y: 0.6, z: 0.0 }, { r: 80, g: 245, b: 145, a: 255 });
    this.spawnObject(PrimitiveType.Cylinder, { x: 1.0, y: 0.55, z: 0.0 }, { r: 255, g: 95, b: 210, a: 255 });
    this.spawnObject(PrimitiveType.Torus, { x: 2.5, y: 0.6, z: 0.0 }, { r: 255, g: 215, b: 65, a: 255 });

    this.activeObjectIdx = this.objects.length > 0 ? 0 : -1;
    this.deleteHoldTimer = 0.0;
    this.deleteCooldownTimer = 0.0;
    this.measurementBadge = "IDLE [SAFE] // 1:MOVE 2:SIZE 3:ROT 4:CLR 5:GRP TU:SPAWN TD:DEL";
  }

  clearThreeObjects() {
    while (this.objectsGroup.children.length > 0) {
      const child = this.objectsGroup.children[0];
      this.objectsGroup.remove(child);
    }
  }

  createThreeNodeForObject(obj) {
    const geom =
      obj.type === PrimitiveType.ImportedObj && obj.importedModel
        ? obj.importedModel.geometry
        : this.primitives.geometry(obj.type);

    const group = new THREE.Group();
    group.matrixAutoUpdate = false;

    const fillMat = new THREE.MeshLambertMaterial({
      color: rgbToHexInt(obj.color),
      transparent: true,
      opacity: 210 / 255,
    });
    const fillMesh = new THREE.Mesh(geom, fillMat);

    const wireMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 1.0,
    });
    const wireMesh = new THREE.Mesh(geom, wireMat);

    group.add(fillMesh);
    group.add(wireMesh);
    this.objectsGroup.add(group);

    obj.threeNode = { group, fillMesh, fillMat, wireMesh, wireMat };
  }

  spawnObject(type, pos, col, importedModel = null) {
    const transform = {
      position: {
        x: snapToStep(pos.x, SNAP_MOVE_STEP_UNITS),
        y: snapToStep(pos.y, SNAP_MOVE_STEP_UNITS),
        z: snapToStep(pos.z, SNAP_MOVE_STEP_UNITS),
      },
      scale: { x: 1.0, y: 1.0, z: 1.0 },
      rotationDeg: { x: 0.0, y: 0.0, z: 0.0 },
      rotMatrix: new THREE.Matrix4().identity(),
    };
    const obj = {
      id: this.nextId++,
      type,
      transform,
      anchorTransform: cloneTransform(transform),
      color: { r: col.r, g: col.g, b: col.b, a: col.a ?? 255 },
      groupId: -1,
      isSelected: false,
      isHovered: false,
      importedModel,
      threeNode: null,
    };
    this.createThreeNodeForObject(obj);
    this.objects.push(obj);
    return obj;
  }

  displayLabel(obj) {
    if (
      obj.type === PrimitiveType.ImportedObj &&
      obj.importedModel &&
      obj.importedModel.displayName
    ) {
      return obj.importedModel.displayName;
    }
    return primitiveName(obj.type);
  }

  showToast(msg, isError = false, duration = 4.0) {
    this.toastMessage = msg;
    this.toastIsError = isError;
    this.toastTimer = duration;
  }

  // -------------------------------------------------------------------------
  // 7. Grouping, Target Resolution & World Bounding Box Helpers (main.cpp)
  // -------------------------------------------------------------------------
  findClosestObjectToCursor(pos, maxDist = 1.35) {
    let best = -1;
    let bestD = maxDist;
    for (let i = 0; i < this.objects.length; i++) {
      const p = this.objects[i].transform.position;
      const dx = p.x - pos.x;
      const dy = p.y - pos.y;
      const dz = p.z - pos.z;
      const d = Math.hypot(dx, dy) + 0.25 * Math.abs(dz);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  isTargetLockedMode(m) {
    return (
      m === LeftMode.Move ||
      m === LeftMode.Resize ||
      m === LeftMode.Rotate ||
      m === LeftMode.Recolor ||
      m === LeftMode.Delete
    );
  }

  collectGroupMembers(idx) {
    const indices = [];
    if (idx < 0 || idx >= this.objects.length) return indices;
    const gid = this.objects[idx].groupId;
    if (gid >= 0) {
      for (let i = 0; i < this.objects.length; i++) {
        if (this.objects[i].groupId === gid) indices.push(i);
      }
    } else {
      indices.push(idx);
    }
    return indices;
  }

  collectTransformTargets() {
    const set = new Set();
    const addWithGroup = (baseIdx) => {
      for (const m of this.collectGroupMembers(baseIdx)) set.add(m);
    };
    if (this.activeObjectIdx >= 0 && this.activeObjectIdx < this.objects.length) {
      addWithGroup(this.activeObjectIdx);
      if (this.objects[this.activeObjectIdx].isSelected) {
        for (let i = 0; i < this.objects.length; i++) {
          if (this.objects[i].isSelected) addWithGroup(i);
        }
      }
    } else {
      for (let i = 0; i < this.objects.length; i++) {
        if (this.objects[i].isSelected) addWithGroup(i);
      }
    }
    return Array.from(set).sort((a, b) => a - b);
  }

  collectGroupExportTargets() {
    let isGroup = false;
    if (
      this.activeObjectIdx >= 0 &&
      this.activeObjectIdx < this.objects.length &&
      this.objects[this.activeObjectIdx].groupId >= 0
    ) {
      return { indices: this.collectGroupMembers(this.activeObjectIdx), isGroup: true };
    }
    const selCount = this.countSelectedObjects();
    if (selCount >= 2) {
      isGroup = true;
      const set = new Set();
      for (let i = 0; i < this.objects.length; i++) {
        if (this.objects[i].isSelected) {
          for (const m of this.collectGroupMembers(i)) set.add(m);
        }
      }
      return { indices: Array.from(set).sort((a, b) => a - b), isGroup };
    }
    if (this.activeObjectIdx >= 0 && this.activeObjectIdx < this.objects.length) {
      return { indices: [this.activeObjectIdx], isGroup: false };
    }
    if (selCount === 1) {
      for (let i = 0; i < this.objects.length; i++) {
        if (this.objects[i].isSelected) {
          const members = this.collectGroupMembers(i);
          return { indices: members, isGroup: members.length > 1 };
        }
      }
    }
    return { indices: [], isGroup: false };
  }

  isObjectInActiveTarget(idx) {
    return this.collectTransformTargets().includes(idx);
  }

  isObjectInActiveHover(idx) {
    if (this.activeObjectIdx < 0 || this.activeObjectIdx >= this.objects.length) {
      return false;
    }
    return this.collectGroupMembers(this.activeObjectIdx).includes(idx);
  }

  countSelectedObjects() {
    let c = 0;
    for (const o of this.objects) {
      if (o.isSelected) c++;
    }
    return c;
  }

  computeWorldBoundingBox(obj, pad = 0.08) {
    let localBox = new THREE.Box3(
      new THREE.Vector3(-0.5, -0.5, -0.5),
      new THREE.Vector3(0.5, 0.5, 0.5)
    );
    if (obj.type === PrimitiveType.ImportedObj && obj.importedModel) {
      localBox = obj.importedModel.normalizedLocalBox;
    } else if (this.primitives.localBox(obj.type)) {
      localBox = this.primitives.localBox(obj.type);
    }

    const mat = composeTransformMatrix(obj.transform);
    const corners = [
      new THREE.Vector3(localBox.min.x, localBox.min.y, localBox.min.z),
      new THREE.Vector3(localBox.max.x, localBox.min.y, localBox.min.z),
      new THREE.Vector3(localBox.min.x, localBox.max.y, localBox.min.z),
      new THREE.Vector3(localBox.max.x, localBox.max.y, localBox.min.z),
      new THREE.Vector3(localBox.min.x, localBox.min.y, localBox.max.z),
      new THREE.Vector3(localBox.max.x, localBox.min.y, localBox.max.z),
      new THREE.Vector3(localBox.min.x, localBox.max.y, localBox.max.z),
      new THREE.Vector3(localBox.max.x, localBox.max.y, localBox.max.z),
    ];

    const worldBox = new THREE.Box3();
    for (const c of corners) {
      c.applyMatrix4(mat);
      worldBox.expandByPoint(c);
    }
    worldBox.min.subScalar(pad);
    worldBox.max.addScalar(pad);
    return worldBox;
  }

  groupSelectedObjects() {
    const count = this.countSelectedObjects();
    if (count < 2 && this.objects.length >= 2) {
      const gid = this.nextGroupId++;
      for (const o of this.objects) {
        o.groupId = gid;
        o.isSelected = true;
      }
      this.showToast(`GROUP #${gid} CREATED (${this.objects.length} objects)`, false, 2.5);
      return;
    }
    if (count < 2) {
      this.showToast("GROUP NEEDS >= 2 OBJECTS", true, 2.5);
      return;
    }
    const gid = this.nextGroupId++;
    for (const o of this.objects) {
      if (o.isSelected) o.groupId = gid;
    }
    this.showToast(`ASSIGNED GROUP #${gid} (${count} objects)`, false, 2.5);
  }

  executeGroupModeAction(action, hovIdx) {
    switch (action) {
      case GroupAction.SelectToggle: {
        const targetIdx =
          hovIdx >= 0 ? hovIdx : this.findClosestObjectToCursor(this.cursor3D);
        if (targetIdx < 0 || targetIdx >= this.objects.length) {
          return;
        }
        const nextSel = !this.objects[targetIdx].isSelected;
        const toggleSet = this.collectGroupMembers(targetIdx);
        for (const idx of toggleSet) {
          this.objects[idx].isSelected = nextSel;
        }
        this.activeObjectIdx = targetIdx;
        break;
      }
      case GroupAction.Group: {
        const selCount = this.countSelectedObjects();
        if (selCount < 2) {
          this.showToast("GROUP NEEDS >= 2 SELECTED OBJECTS (INDEX PINCH)", true, 2.5);
          return;
        }
        const gid = this.nextGroupId++;
        for (let i = 0; i < this.objects.length; i++) {
          if (this.objects[i].isSelected) {
            this.objects[i].groupId = gid;
            if (
              this.activeObjectIdx < 0 ||
              this.activeObjectIdx >= this.objects.length ||
              !this.objects[this.activeObjectIdx].isSelected
            ) {
              this.activeObjectIdx = i;
            }
          }
        }
        this.showToast(`GROUPED ${selCount} OBJECTS -> GROUP #${gid}`, false, 2.5);
        break;
      }
      case GroupAction.Ungroup: {
        const gids = [];
        for (const o of this.objects) {
          if (o.isSelected && o.groupId >= 0 && !gids.includes(o.groupId)) {
            gids.push(o.groupId);
          }
        }
        let ungrouped = 0;
        if (gids.length > 0) {
          for (const o of this.objects) {
            const inTargetGid = o.groupId >= 0 && gids.includes(o.groupId);
            if ((o.isSelected || inTargetGid) && o.groupId >= 0) {
              o.groupId = -1;
              ungrouped++;
            }
          }
        } else {
          const targetIdx =
            hovIdx >= 0 ? hovIdx : this.findClosestObjectToCursor(this.cursor3D);
          if (
            targetIdx >= 0 &&
            targetIdx < this.objects.length &&
            this.objects[targetIdx].groupId >= 0
          ) {
            const hovGid = this.objects[targetIdx].groupId;
            for (const o of this.objects) {
              if (o.groupId === hovGid) {
                o.groupId = -1;
                ungrouped++;
              }
            }
          }
        }
        if (ungrouped > 0) {
          this.showToast(`UNGROUPED ${ungrouped} OBJECTS`, false, 2.5);
        } else {
          this.showToast("NO GROUPED OBJECTS TO UNGROUP", true, 2.2);
        }
        break;
      }
      case GroupAction.ClearSelection: {
        for (const o of this.objects) o.isSelected = false;
        this.showToast("CLEARED SELECTION", false, 2.0);
        break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 8. Core "Mode + Clutch" Anchor & Snapped Delta State Machine (main.cpp)
  // -------------------------------------------------------------------------
  runModeClutchStateMachine(dt) {
    if (this.deleteCooldownTimer > 0.0) {
      this.deleteCooldownTimer = Math.max(0.0, this.deleteCooldownTimer - dt);
    }

    for (const o of this.objects) o.isHovered = false;
    let hovIdx = -1;
    if (this.leftMode === LeftMode.Idle && this.hands[1].present) {
      hovIdx = this.findClosestObjectToCursor(this.cursor3D);
      if (hovIdx >= 0) {
        this.activeObjectIdx = hovIdx;
        for (const idx of this.collectGroupMembers(this.activeObjectIdx)) {
          this.objects[idx].isHovered = true;
        }
      }
    } else if (this.leftMode === LeftMode.Group && this.hands[1].present) {
      hovIdx = this.findClosestObjectToCursor(this.cursor3D);
      if (hovIdx >= 0) {
        for (const idx of this.collectGroupMembers(hovIdx)) {
          this.objects[idx].isHovered = true;
        }
      }
    }

    const wasNotPinching = !this.wasPinching;
    const risingEdge = this.isPinching && wasNotPinching;
    const fallingEdge = !this.isPinching && this.wasPinching;
    const spawnTrigger =
      this.leftMode === LeftMode.Spawn &&
      (risingEdge || (this.spawnTriggerFromPkt && wasNotPinching));
    const groupTrigger =
      this.leftMode === LeftMode.Group &&
      (risingEdge || (this.groupTriggerFromPkt && wasNotPinching));
    this.wasPinching = this.isPinching;

    if (this.leftMode !== LeftMode.Recolor || !this.isPinching) {
      this.recolorSessionMask = 0;
    }

    const axStr = axisName(this.rightAxis);

    // 1. Single-Shot Rising-Edge Spawn/Group OR Anchor Snapshot on Pinch Clutch
    if (spawnTrigger) {
      const created = this.spawnObject(this.activeShape, this.cursor3D, this.activeColor);
      this.activeObjectIdx = this.objects.length - 1;
      this.spawnTriggerFromPkt = false;
      const cxCm = (created.transform.position.x * CM_PER_WORLD_UNIT).toFixed(0);
      const cyCm = (created.transform.position.y * CM_PER_WORLD_UNIT).toFixed(0);
      const czCm = (created.transform.position.z * CM_PER_WORLD_UNIT).toFixed(0);
      this.measurementBadge = `SPAWNED [${primitiveName(this.activeShape)}] @ (${cxCm}, ${cyCm}, ${czCm}) cm // RELEASE (>48px) FOR NEXT`;
    } else if (groupTrigger) {
      this.executeGroupModeAction(this.activeGroupAction, hovIdx);
      this.groupTriggerFromPkt = false;
    } else if (risingEdge) {
      this.anchorHandPos.copy(this.cursor3D);
      this.anchorPalmSize = this.currentPalmSize;
      for (const o of this.objects) {
        o.anchorTransform = cloneTransform(o.transform);
      }
    }

    // 2. While isPinching is Held: Compute Axis-Locked Snapped Delta from Anchor
    const targets = this.collectTransformTargets();
    if (this.isPinching && targets.length > 0) {
      const rawDelta = new THREE.Vector3().subVectors(this.cursor3D, this.anchorHandPos);
      const rawZDeltaCm = -(this.currentPalmSize - this.anchorPalmSize) * Z_SENSITIVITY_CM;
      let axisRawDelta = 0.0;
      if (this.rightAxis === RightAxis.X) axisRawDelta = rawDelta.x;
      else if (this.rightAxis === RightAxis.Y) axisRawDelta = rawDelta.y;
      else if (this.rightAxis === RightAxis.Z) axisRawDelta = rawZDeltaCm / CM_PER_WORLD_UNIT;

      const fmtSigned = (val, decimals = 0) => {
        const s = val >= 0 ? "+" : "";
        return `${s}${val.toFixed(decimals)}`;
      };

      switch (this.leftMode) {
        case LeftMode.Idle: {
          this.measurementBadge = "IDLE [LOCKED] // PINCH HELD";
          break;
        }
        case LeftMode.Move: {
          if (this.rightAxis === RightAxis.XYZ) {
            const snappedXCm = snapToStep(rawDelta.x * CM_PER_WORLD_UNIT, 1.0);
            const snappedYCm = snapToStep(rawDelta.y * CM_PER_WORLD_UNIT, 1.0);
            const snappedZCm = snapToStep(rawZDeltaCm, 1.0);
            for (const idx of targets) {
              this.objects[idx].transform = cloneTransform(this.objects[idx].anchorTransform);
              this.objects[idx].transform.position.x += snappedXCm / CM_PER_WORLD_UNIT;
              this.objects[idx].transform.position.y += snappedYCm / CM_PER_WORLD_UNIT;
              this.objects[idx].transform.position.z += snappedZCm / CM_PER_WORLD_UNIT;
            }
            this.measurementBadge = `MOVE [XYZ-ALL]: (${fmtSigned(snappedXCm, 0)}, ${fmtSigned(snappedYCm, 0)}, ${fmtSigned(snappedZCm, 0)}) cm`;
          } else {
            const snappedCm =
              this.rightAxis === RightAxis.Z
                ? snapToStep(rawZDeltaCm, SNAP_MOVE_STEP_Z_CM)
                : snapToStep(axisRawDelta, SNAP_MOVE_STEP_UNITS) * CM_PER_WORLD_UNIT;
            const snappedUnits = snappedCm / CM_PER_WORLD_UNIT;
            for (const idx of targets) {
              this.objects[idx].transform = cloneTransform(this.objects[idx].anchorTransform);
              if (this.rightAxis === RightAxis.X)
                this.objects[idx].transform.position.x += snappedUnits;
              else if (this.rightAxis === RightAxis.Y)
                this.objects[idx].transform.position.y += snappedUnits;
              else if (this.rightAxis === RightAxis.Z)
                this.objects[idx].transform.position.z += snappedUnits;
            }
            this.measurementBadge = `MOVE [${axStr}]: ${fmtSigned(snappedCm, 1)} cm`;
          }
          break;
        }
        case LeftMode.Resize: {
          if (this.rightAxis === RightAxis.XYZ) {
            const depthUnits = rawZDeltaCm / CM_PER_WORLD_UNIT;
            const uniformDriver =
              Math.abs(rawDelta.x) >= Math.abs(depthUnits) ? rawDelta.x : depthUnits;
            const snappedStep = snapToStep(uniformDriver, 0.25);
            const factor = Math.max(0.25, 1.0 + snappedStep);
            for (const idx of targets) {
              const anc = this.objects[idx].anchorTransform;
              this.objects[idx].transform = cloneTransform(anc);
              this.objects[idx].transform.scale.x = Math.max(0.25, anc.scale.x * factor);
              this.objects[idx].transform.scale.y = Math.max(0.25, anc.scale.y * factor);
              this.objects[idx].transform.scale.z = Math.max(0.25, anc.scale.z * factor);
            }
            this.measurementBadge = `RESIZE [XYZ-ALL]: ${factor.toFixed(2)}x (${fmtSigned((factor - 1.0) * 100.0, 0)}%)`;
          } else {
            const snappedUnits = snapToStep(axisRawDelta, SNAP_SCALE_STEP_UNITS);
            const snappedCm = snappedUnits * CM_PER_WORLD_UNIT;
            for (const idx of targets) {
              const anc = this.objects[idx].anchorTransform;
              this.objects[idx].transform = cloneTransform(anc);
              if (this.rightAxis === RightAxis.X) {
                this.objects[idx].transform.scale.x = Math.max(0.5, anc.scale.x + snappedUnits);
              } else if (this.rightAxis === RightAxis.Y) {
                this.objects[idx].transform.scale.y = Math.max(0.5, anc.scale.y + snappedUnits);
              } else if (this.rightAxis === RightAxis.Z) {
                this.objects[idx].transform.scale.z = Math.max(0.5, anc.scale.z + snappedUnits);
              }
            }
            this.measurementBadge = `RESIZE [${axStr}]: ${fmtSigned(snappedCm, 1)} cm`;
          }
          break;
        }
        case LeftMode.Rotate: {
          const DEG2RAD = Math.PI / 180.0;
          if (this.rightAxis === RightAxis.XYZ) {
            const snappedPitchDeg = snapToStep(-rawDelta.y * 45.0, SNAP_ROTATE_STEP_DEG);
            const snappedYawDeg = snapToStep(rawDelta.x * 45.0, SNAP_ROTATE_STEP_DEG);
            const snappedRollDeg = snapToStep(
              (rawZDeltaCm / CM_PER_WORLD_UNIT) * 45.0,
              SNAP_ROTATE_STEP_DEG
            );
            // Matches Raylib MatrixMultiply(MatrixMultiply(RotX, RotY), RotZ) -> RotZ * RotY * RotX in column-vector math
            const rotX = new THREE.Matrix4().makeRotationX(snappedPitchDeg * DEG2RAD);
            const rotY = new THREE.Matrix4().makeRotationY(snappedYawDeg * DEG2RAD);
            const rotZ = new THREE.Matrix4().makeRotationZ(snappedRollDeg * DEG2RAD);
            const deltaRot = new THREE.Matrix4().multiplyMatrices(rotZ, rotY).multiply(rotX);

            for (const idx of targets) {
              const anc = this.objects[idx].anchorTransform;
              this.objects[idx].transform = cloneTransform(anc);
              this.objects[idx].transform.rotMatrix = new THREE.Matrix4().multiplyMatrices(
                deltaRot,
                anc.rotMatrix
              );
              this.objects[idx].transform.rotationDeg.x += snappedPitchDeg;
              this.objects[idx].transform.rotationDeg.y += snappedYawDeg;
              this.objects[idx].transform.rotationDeg.z += snappedRollDeg;
            }
            this.measurementBadge = `ROTATE [XYZ-ALL]: (${fmtSigned(snappedPitchDeg, 0)}, ${fmtSigned(snappedYawDeg, 0)}, ${fmtSigned(snappedRollDeg, 0)}) deg`;
          } else {
            let rotDriver = axisRawDelta;
            if (this.rightAxis === RightAxis.X) {
              rotDriver = Math.abs(rawDelta.y) > Math.abs(rawDelta.x) ? -rawDelta.y : rawDelta.x;
            } else if (this.rightAxis === RightAxis.Y) {
              rotDriver = Math.abs(rawDelta.x) > Math.abs(rawDelta.y) ? rawDelta.x : rawDelta.y;
            } else if (this.rightAxis === RightAxis.Z) {
              rotDriver = Math.abs(rawDelta.x) > Math.abs(axisRawDelta) ? -rawDelta.x : axisRawDelta;
            }
            const rawDeg = rotDriver * 45.0;
            const snappedDeg = snapToStep(rawDeg, SNAP_ROTATE_STEP_DEG);
            const axisVec = new THREE.Vector3(
              this.rightAxis === RightAxis.X ? 1 : 0,
              this.rightAxis === RightAxis.Y ? 1 : 0,
              this.rightAxis === RightAxis.Z ? 1 : 0
            );
            const deltaRot = new THREE.Matrix4().makeRotationAxis(axisVec, snappedDeg * DEG2RAD);
            for (const idx of targets) {
              const anc = this.objects[idx].anchorTransform;
              this.objects[idx].transform = cloneTransform(anc);
              this.objects[idx].transform.rotMatrix = new THREE.Matrix4().multiplyMatrices(
                deltaRot,
                anc.rotMatrix
              );
              if (this.rightAxis === RightAxis.X)
                this.objects[idx].transform.rotationDeg.x += snappedDeg;
              else if (this.rightAxis === RightAxis.Y)
                this.objects[idx].transform.rotationDeg.y += snappedDeg;
              else if (this.rightAxis === RightAxis.Z)
                this.objects[idx].transform.rotationDeg.z += snappedDeg;
            }
            this.measurementBadge = `ROTATE [${axStr}]: ${fmtSigned(snappedDeg, 0)} deg`;
          }
          break;
        }
        case LeftMode.Recolor: {
          const pktMask =
            this.hands[1].present && this.hands[1].features
              ? this.hands[1].features.contactMask & 0x07
              : 0;
          this.recolorSessionMask |= pktMask;
          this.activeRecolorMask = this.recolorSessionMask;
          if (this.activeRecolorMask !== 0) {
            this.activeColor = recolorColorFromMask(this.activeRecolorMask);
            this.activeColorName = recolorNameFromMask(this.activeRecolorMask);
            for (const idx of targets) {
              this.objects[idx].color = { ...this.activeColor };
            }
          }
          this.measurementBadge = `RECOLOR: ${this.activeColorName} (${this.activeColor.r}, ${this.activeColor.g}, ${this.activeColor.b})`;
          break;
        }
        default:
          break;
      }
    }

    // 3. Falling Edge of Pinch Clutch: Bake Transform so Object Stays Frozen in Place
    if (fallingEdge) {
      for (const o of this.objects) {
        o.anchorTransform = cloneTransform(o.transform);
      }
    }

    // 3b. Safe Timed Deletion (1.5s Thumbs-Down Hold + 1.0s Cooldown, group-aware)
    if (this.leftMode === LeftMode.Delete) {
      const deleteTargets = this.collectTransformTargets();
      if (this.deleteCooldownTimer > 0.0) {
        this.deleteHoldTimer = 0.0;
      } else if (deleteTargets.length > 0) {
        this.deleteHoldTimer += dt;
        if (this.deleteHoldTimer >= DELETE_HOLD_DURATION_SEC) {
          const refIdx =
            this.activeObjectIdx >= 0 && this.activeObjectIdx < this.objects.length
              ? this.activeObjectIdx
              : deleteTargets[0];
          const deletedGroupId = this.objects[refIdx].groupId;
          const deletedId = this.objects[refIdx].id;
          const deletedLabel = this.displayLabel(this.objects[refIdx]);
          const deletedCount = deleteTargets.length;

          for (let i = deleteTargets.length - 1; i >= 0; i--) {
            const delIdx = deleteTargets[i];
            const removed = this.objects[delIdx];
            if (removed.threeNode) {
              this.objectsGroup.remove(removed.threeNode.group);
            }
            this.objects.splice(delIdx, 1);
          }
          this.deleteHoldTimer = 0.0;
          this.deleteCooldownTimer = DELETE_COOLDOWN_DURATION_SEC;

          if (this.objects.length === 0) {
            this.activeObjectIdx = -1;
          } else {
            this.activeObjectIdx = clamp(this.activeObjectIdx, 0, this.objects.length - 1);
          }

          let delMsg = "";
          if (deletedCount > 1) {
            if (deletedGroupId >= 0) {
              delMsg = `DELETED GROUP #${deletedGroupId} (${deletedCount} OBJECTS) // COOLDOWN 1.0s`;
            } else {
              delMsg = `DELETED ${deletedCount} SELECTED OBJECTS // COOLDOWN 1.0s`;
            }
          } else {
            delMsg = `DELETED [${deletedLabel} #${deletedId}] // COOLDOWN 1.0s`;
          }
          this.showToast(delMsg, false, 2.5);
        }
      } else {
        this.deleteHoldTimer = 0.0;
      }
    } else {
      this.deleteHoldTimer = 0.0;
    }

    // 4. Mode / Hover / Locked Color / Delete HUD Readout
    if (this.leftMode === LeftMode.Delete) {
      if (this.deleteCooldownTimer > 0.0) {
        this.measurementBadge = `DELETE COOLDOWN (${this.deleteCooldownTimer.toFixed(1)}s)`;
      } else {
        const targetCount = this.collectTransformTargets().length;
        if (targetCount > 1) {
          this.measurementBadge = `DELETING GROUP (${targetCount})... ${this.deleteHoldTimer.toFixed(1)}s`;
        } else if (targetCount === 1) {
          this.measurementBadge = `DELETING OBJECT... ${this.deleteHoldTimer.toFixed(1)}s`;
        } else {
          this.measurementBadge = "DELETE [NO OBJECT SELECTED]";
        }
      }
    } else if (this.leftMode === LeftMode.Group) {
      this.measurementBadge = `GROUP MODE [${this.countSelectedObjects()} Selected] // Index=Select, Middle=Group, Ring=Ungroup, Pinky=Clear`;
    } else if (!this.isPinching) {
      if (this.leftMode === LeftMode.Idle) {
        if (this.activeObjectIdx >= 0 && this.activeObjectIdx < this.objects.length) {
          const sel = this.objects[this.activeObjectIdx];
          if (sel.groupId >= 0) {
            this.measurementBadge = `IDLE [AIM TO SELECT] | TARGET: GRP #${sel.groupId} | AXIS: ${axStr}`;
          } else {
            this.measurementBadge = `IDLE [AIM TO SELECT] | TARGET: ${this.displayLabel(sel)} #${sel.id} | AXIS: ${axStr}`;
          }
        } else {
          this.measurementBadge = `IDLE [AIM TO SELECT] | NO TARGET | AXIS: ${axStr}`;
        }
      } else if (this.leftMode === LeftMode.Recolor) {
        this.activeRecolorMask = 0;
        this.recolorSessionMask = 0;
        this.measurementBadge = `RECOLOR [LOCKED]: ${this.activeColorName} (${this.activeColor.r}, ${this.activeColor.g}, ${this.activeColor.b})`;
      } else if (this.leftMode === LeftMode.Spawn) {
        this.measurementBadge = `SPAWN [${primitiveName(this.activeShape)}] // Thumb+Index=Cube, Middle=Capsule, Ring=Cyl, Pinky=Ring`;
      } else {
        const unit = this.leftMode === LeftMode.Rotate ? "deg" : "cm";
        this.measurementBadge = `${modeName(this.leftMode)} [${axStr}]: +0.0 ${unit} (PINCH <32px TO ENGAGE)`;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 9. Vision Frame Processing & 3D / 2D Rendering
  // -------------------------------------------------------------------------
  processVisionFrame(nowSec, nowMs) {
    if (!this.cameraReady || this.videoEl.readyState < 2) {
      this.hands[0].present = false;
      this.hands[1].present = false;
      return;
    }

    // Draw mirrored 640x480 webcam frame onto captureCanvas (matches cv2.flip(frame, 1))
    this.captureCtx.save();
    this.captureCtx.translate(CAPTURE_WIDTH, 0);
    this.captureCtx.scale(-1, 1);
    this.captureCtx.drawImage(this.videoEl, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    this.captureCtx.restore();

    if (!this.visionReady || !this.handLandmarker) return;
    if (this.videoEl.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = this.videoEl.currentTime;
    this.trackerFrameCounter++;

    const res = this.handLandmarker.detectForVideo(this.captureCanvas, Math.floor(nowMs));
    const rawHands = [];
    if (res && res.landmarks && res.handedness) {
      for (let i = 0; i < res.landmarks.length; i++) {
        const lms = res.landmarks[i];
        const hd = res.handedness[i] && res.handedness[i][0];
        if (!lms || !hd) continue;
        rawHands.push({
          landmarks: lms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
          label: hd.categoryName || hd.displayName || "Right",
          score: hd.score || 0.9,
        });
      }
    }

    const slots = assignHandSlots(rawHands, true, false);
    const smoothed = [null, null];
    for (const s of [0, 1]) {
      if (!slots[s]) {
        this.smoothers[s].reset();
      } else {
        smoothed[s] = this.smoothers[s].update(slots[s].landmarks, nowSec);
      }
    }

    // If Left Hand is detected in frame, clear manual dock mode override
    if (smoothed[0]) {
      this.manualModeOverride = null;
    }

    const [leftF, rightF] = this.engine.update(
      smoothed[0],
      smoothed[1],
      nowSec,
      this.manualModeOverride
    );

    this.hands[0] = {
      present: smoothed[0] !== null,
      landmarks: smoothed[0],
      features: leftF,
      score: slots[0] ? slots[0].score : 0,
    };
    this.hands[1] = {
      present: smoothed[1] !== null,
      landmarks: smoothed[1],
      features: rightF,
      score: slots[1] ? slots[1].score : 0,
    };

    // Sync telemetry into app state (exact match with main.cpp lines 444-490)
    this.leftMode = this.engine.leftMode;
    if (this.engine.rightAxis !== this.lastTrackerRightAxis) {
      this.lastTrackerRightAxis = this.engine.rightAxis;
      this.hasKeyboardAxisOverride = false;
    }
    if (!this.hasKeyboardAxisOverride) {
      this.rightAxis = this.engine.rightAxis;
    }
    this.isPinching = this.engine.isPinching;
    if (this.engine.spawnTrigger) this.spawnTriggerFromPkt = true;
    if (this.engine.groupTrigger) this.groupTriggerFromPkt = true;
    this.activeGroupAction =
      this.hands[1].present && rightF ? rightF.groupAction : GroupAction.SelectToggle;
    this.pinchPx = rightF ? rightF.pinchPx : 999.0;
    if (this.leftMode === LeftMode.Spawn) {
      this.activeShape = this.engine.activeShape;
    }

    if (this.hands[1].present) {
      const dt = 1 / 60;
      const alpha = Math.min(1.0, dt * 18.0);
      this.cursor3D.x += (this.engine.cursor3d.x - this.cursor3D.x) * alpha;
      this.cursor3D.y += (this.engine.cursor3d.y - this.cursor3D.y) * alpha;
      this.currentPalmSize += (this.engine.cursor3d.z - this.currentPalmSize) * alpha;
      this.cursor3D.z = clamp(
        -(this.currentPalmSize - 0.18) * (Z_SENSITIVITY_CM / CM_PER_WORLD_UNIT),
        -4.5,
        4.5
      );
    }

    if (this.leftMode !== LeftMode.Recolor && this.hands[0].present && leftF) {
      this.activeColor = {
        r: clamp(Math.round(leftF.color[0] * 255), 0, 255),
        g: clamp(Math.round(leftF.color[1] * 255), 0, 255),
        b: clamp(Math.round(leftF.color[2] * 255), 0, 255),
        a: 255,
      };
    }
  }

  buildGlowingAxisGuideThree(center, axis, activePinch) {
    if (axis === RightAxis.XYZ) {
      this.buildGlowingAxisGuideThree(center, RightAxis.X, activePinch);
      this.buildGlowingAxisGuideThree(center, RightAxis.Y, activePinch);
      this.buildGlowingAxisGuideThree(center, RightAxis.Z, activePinch);
      return;
    }

    let dir = new THREE.Vector3(1, 0, 0);
    let hexCol = 0xff414b;
    if (axis === RightAxis.Y) {
      dir.set(0, 1, 0);
      hexCol = 0x41ff6e;
    } else if (axis === RightAxis.Z) {
      dir.set(0, 0, 1);
      hexCol = 0x41a5ff;
    }

    const halfLen = 4.5;
    const len = halfLen * 2.0;

    // 1. Outer translucent glow cylinder + 2. Inner neon core cylinder
    const outerRad = activePinch ? 0.070 : 0.055;
    const innerRad = activePinch ? 0.026 : 0.020;

    const cylGroup = new THREE.Group();
    cylGroup.position.copy(center);
    if (axis === RightAxis.X) cylGroup.rotation.z = -Math.PI * 0.5;
    else if (axis === RightAxis.Z) cylGroup.rotation.x = Math.PI * 0.5;

    const outerGeom = new THREE.CylinderGeometry(outerRad, outerRad, len, 10);
    const outerMat = new THREE.MeshBasicMaterial({
      color: hexCol,
      transparent: true,
      opacity: activePinch ? 0.36 : 0.28,
      depthWrite: false,
    });
    cylGroup.add(new THREE.Mesh(outerGeom, outerMat));

    const innerGeom = new THREE.CylinderGeometry(innerRad, innerRad, len, 8);
    const innerMat = new THREE.MeshBasicMaterial({ color: hexCol });
    cylGroup.add(new THREE.Mesh(innerGeom, innerMat));
    this.axisGuidesGroup.add(cylGroup);

    // 3. Center bright white core line + tick marks every 1.0 world unit (10 cm)
    const a = center.clone().addScaledVector(dir, -halfLen);
    const b = center.clone().addScaledVector(dir, halfLen);
    const centerLineGeom = new THREE.BufferGeometry().setFromPoints([a, b]);
    const centerLineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
    this.axisGuidesGroup.add(new THREE.Line(centerLineGeom, centerLineMat));

    const perp =
      axis === RightAxis.Y ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    const tickPoints = [];
    for (let t = -4; t <= 4; t++) {
      if (t === 0) continue;
      const tickCenter = center.clone().addScaledVector(dir, t);
      const tickHalf = t % 2 === 0 ? 0.14 : 0.08;
      tickPoints.push(
        tickCenter.clone().addScaledVector(perp, -tickHalf),
        tickCenter.clone().addScaledVector(perp, tickHalf)
      );
    }
    const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPoints);
    const tickMat = new THREE.LineBasicMaterial({ color: hexCol });
    this.axisGuidesGroup.add(new THREE.LineSegments(tickGeom, tickMat));
  }

  addThreeBoundingBoxHelper(box, hexColor, opacity = 1.0) {
    const helper = new THREE.Box3Helper(box, hexColor);
    if (opacity < 1.0) {
      helper.material.transparent = true;
      helper.material.opacity = opacity;
    }
    this.overlays3DGroup.add(helper);
  }

  updateThreeScene() {
    const isStudio = this.viewportMode === ViewportMode.Studio3D;
    this.studioEnvGroup.visible = isStudio;
    if (isStudio) {
      this.renderer.setClearColor(0x181a20, 1.0); // Color{24, 26, 32, 255}
    } else {
      this.renderer.setClearColor(0x000000, 0.0); // Transparent over fullscreen AR canvas
    }

    // Clear per-frame dynamic 3D overlays (bounding boxes, group links, axis guides)
    while (this.overlays3DGroup.children.length > 0) {
      const c = this.overlays3DGroup.children[0];
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
      this.overlays3DGroup.remove(c);
    }
    while (this.axisGuidesGroup.children.length > 0) {
      const c = this.axisGuidesGroup.children[0];
      this.axisGuidesGroup.remove(c);
    }

    const lockedMode = this.isTargetLockedMode(this.leftMode);

    // 1. Update all permanent SceneObjects
    for (let i = 0; i < this.objects.length; i++) {
      const obj = this.objects[i];
      if (!obj.threeNode) this.createThreeNodeForObject(obj);

      const activeTarget = this.isObjectInActiveTarget(i);
      const idleHoverTarget =
        this.leftMode === LeftMode.Idle && this.isObjectInActiveHover(i);

      const worldMat = composeTransformMatrix(obj.transform);
      obj.threeNode.group.matrix.copy(worldMat);
      obj.threeNode.fillMat.color.setHex(rgbToHexInt(obj.color));

      let wireCol = { r: 255, g: 255, b: 255, a: 255 };
      if (lockedMode && activeTarget) {
        if (this.leftMode === LeftMode.Delete) wireCol = { r: 255, g: 75, b: 85, a: 255 };
        else if (this.leftMode === LeftMode.Recolor) wireCol = { ...this.activeColor };
        else wireCol = { r: 255, g: 230, b: 70, a: 255 };
      } else if (obj.isHovered) {
        wireCol = { r: 120, g: 245, b: 255, a: 220 };
      } else if (obj.isSelected) {
        wireCol = { r: 255, g: 235, b: 80, a: 220 };
      } else {
        wireCol = {
          r: Math.min(255, obj.color.r + 50),
          g: Math.min(255, obj.color.g + 50),
          b: Math.min(255, obj.color.b + 50),
          a: 255,
        };
      }
      obj.threeNode.wireMat.color.setHex(rgbToHexInt(wireCol));
      obj.threeNode.wireMat.opacity = (wireCol.a ?? 255) / 255;

      // Selection & Group 3D Bounding Boxes (exact match with main.cpp lines 1422-1449)
      if (obj.groupId >= 0) {
        if (lockedMode && activeTarget) {
          const lockHex = this.leftMode === LeftMode.Delete ? 0xff4650 : 0x41ffaa;
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj, 0.07), lockHex);
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj, 0.11), lockHex);
        } else if (idleHoverTarget) {
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj), 0x2de18c, 155 / 255);
        } else {
          this.addThreeBoundingBoxHelper(
            this.computeWorldBoundingBox(obj),
            obj.isSelected ? 0x3cffa5 : 0x1ed77d,
            obj.isSelected ? 1.0 : 210 / 255
          );
        }
      } else {
        if (lockedMode && activeTarget) {
          const lockHex =
            this.leftMode === LeftMode.Delete
              ? 0xff4650
              : this.isPinching
                ? 0xffe141
                : 0x41f5ff;
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj, 0.07), lockHex);
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj, 0.11), lockHex);
        } else if (obj.isSelected) {
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj), 0x32ebff);
        } else if (idleHoverTarget) {
          this.addThreeBoundingBoxHelper(this.computeWorldBoundingBox(obj), 0x5aebff, 145 / 255);
        }
      }

      // Glowing 3D axis guide line through the active primary target object
      if (
        this.leftMode !== LeftMode.Group &&
        this.leftMode !== LeftMode.Spawn &&
        i === this.activeObjectIdx
      ) {
        this.buildGlowingAxisGuideThree(
          new THREE.Vector3(
            obj.transform.position.x,
            obj.transform.position.y,
            obj.transform.position.z
          ),
          this.rightAxis,
          this.isPinching
        );
      }
    }

    // 1b. Group Connecting Lines Between Centers (groupId != -1)
    const seenGroups = new Set();
    for (const obj of this.objects) {
      if (obj.groupId < 0 || seenGroups.has(obj.groupId)) continue;
      seenGroups.add(obj.groupId);

      const centers = [];
      for (const m of this.objects) {
        if (m.groupId === obj.groupId) {
          centers.push(
            new THREE.Vector3(
              m.transform.position.x,
              m.transform.position.y,
              m.transform.position.z
            )
          );
        }
      }
      if (centers.length >= 2) {
        const linePts = [...centers];
        if (centers.length >= 3) linePts.push(centers[0]);
        const lineGeom = new THREE.BufferGeometry().setFromPoints(linePts);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x28eb91,
          transparent: true,
          opacity: 175 / 255,
        });
        this.overlays3DGroup.add(new THREE.Line(lineGeom, lineMat));

        for (const c of centers) {
          const spGeom = new THREE.SphereGeometry(0.045, 10, 10);
          const spMat = new THREE.MeshBasicMaterial({
            color: 0x32f59b,
            transparent: true,
            opacity: 210 / 255,
          });
          const sp = new THREE.Mesh(spGeom, spMat);
          sp.position.copy(c);
          this.overlays3DGroup.add(sp);
        }
      }
    }

    // 2. Ghost Preview in SPAWN mode while isPinching == false
    if (this.leftMode === LeftMode.Spawn && !this.isPinching) {
      this.ghostGroup.visible = true;
      const ghostGeom = this.primitives.geometry(this.activeShape);
      this.ghostFillMesh.geometry = ghostGeom;
      this.ghostWireMesh.geometry = ghostGeom;
      this.ghostFillMat.color.setHex(rgbToHexInt(this.activeColor));
      this.ghostGroup.position.set(
        snapToStep(this.cursor3D.x, SNAP_MOVE_STEP_UNITS),
        snapToStep(this.cursor3D.y, SNAP_MOVE_STEP_UNITS),
        snapToStep(this.cursor3D.z, SNAP_MOVE_STEP_UNITS)
      );
    } else {
      this.ghostGroup.visible = false;
    }

    // 3. 3D Cursor Sphere (Studio 3D mode)
    this.cursorGroup.visible = isStudio;
    if (isStudio) {
      this.cursorGroup.position.copy(this.cursor3D);
      const rad = this.isPinching ? 0.14 : 0.10;
      this.cursorInnerMesh.scale.set(rad, rad, rad);
      const cCol = this.isPinching ? { r: 255, g: 220, b: 60 } : axisColorRGB(this.rightAxis);
      this.cursorInnerMat.color.setHex(rgbToHexInt(cCol));
    }
  }

  drawFallbackGridToCanvas(ctx, w, h) {
    ctx.fillStyle = "rgb(16, 24, 36)";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgb(28, 52, 74)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 32) {
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(w, y + 0.5);
      ctx.stroke();
    }
  }

  // Exact port of draw2DHandSkeletonOverlay from main.cpp lines 1553-1616
  draw2DHandSkeletonOverlay(ctx, rectW, rectH) {
    for (let s = 0; s < 2; s++) {
      const h = this.hands[s];
      if (!h.present || !h.landmarks) continue;

      const pts = h.landmarks.map((lm) => ({
        x: lm.x * rectW,
        y: lm.y * rectH,
      }));

      ctx.strokeStyle =
        s === 0 ? "rgba(40, 230, 255, 0.94)" : "rgba(80, 255, 165, 0.94)";
      ctx.lineWidth = 2.2;
      for (const [a, b] of HAND_BONES) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
        ctx.stroke();
      }

      for (let i = 0; i < 21; i++) {
        const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        ctx.fillStyle = isTip ? "rgb(253, 249, 0)" : "#ffffff";
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, isTip ? 4.5 : 3.0, 0, Math.PI * 2);
        ctx.fill();
      }

      if (s === 1) {
        if (this.leftMode === LeftMode.Recolor) {
          const cm = h.features ? h.features.contactMask & 0x07 : 0;
          const tips = [8, 12, 16];
          const chCols = [
            { r: 255, g: 60, b: 70 },
            { r: 60, g: 255, b: 110 },
            { r: 65, g: 145, b: 255 },
          ];
          for (let ch = 0; ch < 3; ch++) {
            const active = (cm & (1 << ch)) !== 0;
            ctx.strokeStyle = rgbCss(chCols[ch], active ? 1.0 : 0.35);
            ctx.lineWidth = active ? 3.5 : 1.4;
            ctx.beginPath();
            ctx.moveTo(pts[4].x, pts[4].y);
            ctx.lineTo(pts[tips[ch]].x, pts[tips[ch]].y);
            ctx.stroke();
            if (active) {
              ctx.fillStyle = rgbCss(chCols[ch], 1.0);
              ctx.beginPath();
              ctx.arc(pts[tips[ch]].x, pts[tips[ch]].y, 7.5, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        } else if (this.leftMode === LeftMode.Spawn) {
          const shapeTips = [8, 12, 16, 20];
          const activeTip = shapeTips[this.activeShape % 4];
          const pCol = this.isPinching
            ? "rgba(255, 215, 50, 1.0)"
            : "rgba(90, 245, 255, 0.86)";
          ctx.strokeStyle = pCol;
          ctx.lineWidth = this.isPinching ? 3.5 : 2.2;
          ctx.beginPath();
          ctx.moveTo(pts[4].x, pts[4].y);
          ctx.lineTo(pts[activeTip].x, pts[activeTip].y);
          ctx.stroke();
          ctx.fillStyle = pCol;
          ctx.beginPath();
          ctx.arc(pts[activeTip].x, pts[activeTip].y, 6.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (this.leftMode === LeftMode.Group) {
          const groupTips = [8, 12, 16, 20];
          const activeTip = groupTips[this.activeGroupAction % 4];
          const pCol = this.isPinching
            ? "rgba(255, 215, 50, 1.0)"
            : "rgba(50, 235, 155, 0.90)";
          ctx.strokeStyle = pCol;
          ctx.lineWidth = this.isPinching ? 3.5 : 2.2;
          ctx.beginPath();
          ctx.moveTo(pts[4].x, pts[4].y);
          ctx.lineTo(pts[activeTip].x, pts[activeTip].y);
          ctx.stroke();
          ctx.fillStyle = pCol;
          ctx.beginPath();
          ctx.arc(pts[activeTip].x, pts[activeTip].y, 6.5, 0, Math.PI * 2);
          ctx.fill();
        } else {
          let activeTip = 8;
          if (this.rightAxis === RightAxis.Y) activeTip = 12;
          else if (this.rightAxis === RightAxis.Z) activeTip = 16;
          else if (this.rightAxis === RightAxis.XYZ) activeTip = 20;
          const axC = axisColorRGB(this.rightAxis);
          const pCol = this.isPinching
            ? "rgba(255, 215, 50, 1.0)"
            : rgbCss(axC, 0.65);
          ctx.strokeStyle = pCol;
          ctx.lineWidth = this.isPinching ? 3.5 : 2.0;
          ctx.beginPath();
          ctx.moveTo(pts[4].x, pts[4].y);
          ctx.lineTo(pts[activeTip].x, pts[activeTip].y);
          ctx.stroke();
          if (this.isPinching) {
            const mx = (pts[4].x + pts[activeTip].x) * 0.5;
            const my = (pts[4].y + pts[activeTip].y) * 0.5;
            ctx.strokeStyle = "rgba(255, 220, 60, 1.0)";
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.arc(mx, my, 12.0, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
    }
  }

  renderWebcamPiPOrAR() {
    const sw = window.innerWidth;
    const sh = window.innerHeight;

    if (this.viewportMode === ViewportMode.ArCamera) {
      this.arBgCanvas.style.display = "block";
      this.pipContainer.classList.add("hidden");
      if (this.cameraReady && this.videoEl.readyState >= 2) {
        this.arBgCtx.drawImage(this.captureCanvas, 0, 0, sw, sh);
      } else {
        this.drawFallbackGridToCanvas(this.arBgCtx, sw, sh);
      }
      this.draw2DHandSkeletonOverlay(this.arBgCtx, sw, sh);
    } else {
      this.arBgCanvas.style.display = "none";
      this.pipContainer.classList.remove("hidden");
      const pipW = 320;
      const pipH = 240;
      if (this.cameraReady && this.videoEl.readyState >= 2) {
        this.pipCtx.drawImage(this.captureCanvas, 0, 0, pipW, pipH);
      } else {
        this.drawFallbackGridToCanvas(this.pipCtx, pipW, pipH);
      }
      this.draw2DHandSkeletonOverlay(this.pipCtx, pipW, pipH);
    }
  }

  // Project 3D position to 2D screen coordinates (matches Raylib GetWorldToScreen)
  worldToScreen(pos3, sw, sh) {
    const v = new THREE.Vector3(pos3.x, pos3.y, pos3.z);
    v.project(this.camera);
    if (v.z > 1.0) return null; // Behind camera
    return {
      x: (v.x * 0.5 + 0.5) * sw,
      y: (-v.y * 0.5 + 0.5) * sh,
    };
  }

  drawHoverReticle2D(ctx, center, cssColor) {
    const arcSpans = [
      [15, 75],
      [105, 165],
      [195, 255],
      [285, 345],
    ];
    ctx.strokeStyle = cssColor;
    ctx.lineWidth = 3.0;
    for (const [startDeg, endDeg] of arcSpans) {
      ctx.beginPath();
      ctx.arc(
        center.x,
        center.y,
        26.5,
        (startDeg * Math.PI) / 180,
        (endDeg * Math.PI) / 180
      );
      ctx.stroke();
    }

    const tickRad = 30.0;
    const tickOuter = 36.0;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(center.x - tickOuter, center.y);
    ctx.lineTo(center.x - tickRad, center.y);
    ctx.moveTo(center.x + tickRad, center.y);
    ctx.lineTo(center.x + tickOuter, center.y);
    ctx.moveTo(center.x, center.y - tickOuter);
    ctx.lineTo(center.x, center.y - tickRad);
    ctx.moveTo(center.x, center.y + tickRad);
    ctx.lineTo(center.x, center.y + tickOuter);
    ctx.stroke();
  }

  drawLockedHighlight2D(ctx, center, cssColor, activePinch) {
    ctx.strokeStyle = cssColor;
    ctx.lineWidth = activePinch ? 5.5 : 4.0;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 29.0, 0, Math.PI * 2);
    ctx.stroke();

    const offset = 35.0;
    const arm = 11.0;
    ctx.lineWidth = activePinch ? 3.2 : 2.6;
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        const cx = center.x + sx * offset;
        const cy = center.y + sy * offset;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - sx * arm, cy);
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, cy - sy * arm);
        ctx.stroke();
      }
    }
  }

  drawRoundedPill2D(ctx, x, y, w, h, radius, fillStyle, strokeStyle, lineWidth = 1.2) {
    const r = Math.min(radius, w * 0.5, h * 0.5);
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(x, y, w, h, r);
    } else {
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }
    ctx.fillStyle = fillStyle;
    ctx.fill();
    if (strokeStyle) {
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = strokeStyle;
      ctx.stroke();
    }
  }

  // Exact port of drawHudOverlays from main.cpp lines 1634-1965
  render2DHudOverlays() {
    const sw = window.innerWidth;
    const sh = window.innerHeight;
    const ctx = this.hud2dCtx;
    ctx.clearRect(0, 0, sw, sh);

    const lockedMode = this.isTargetLockedMode(this.leftMode);
    let primaryLabelIdx = this.activeObjectIdx;
    if (this.leftMode === LeftMode.Group && this.hands[1].present) {
      const hovered = this.findClosestObjectToCursor(this.cursor3D);
      if (hovered >= 0) primaryLabelIdx = hovered;
    }

    // 1. Decluttered Context-Aware 3D Reticles & Floating Object Pills
    ctx.font = '700 13px "JetBrains Mono", "Cascadia Mono", Consolas, monospace';
    ctx.textBaseline = "middle";

    for (let i = 0; i < this.objects.length; i++) {
      const obj = this.objects[i];
      const isSelectedTarget = this.isObjectInActiveTarget(i);
      const isIdleAim =
        this.leftMode === LeftMode.Idle && this.isObjectInActiveHover(i);
      const isPrimaryLabel = i === primaryLabelIdx;
      const isGroupBadge = this.leftMode === LeftMode.Group && obj.groupId >= 0;

      if (
        !isSelectedTarget &&
        !isIdleAim &&
        !isPrimaryLabel &&
        !isGroupBadge &&
        !obj.isSelected &&
        !obj.isHovered
      ) {
        continue;
      }

      const sc = this.worldToScreen(obj.transform.position, sw, sh);
      if (!sc || sc.x < 0 || sc.x > sw || sc.y < 0 || sc.y > sh) continue;

      const isActiveDeleteTarget =
        this.leftMode === LeftMode.Delete &&
        isSelectedTarget &&
        (this.deleteHoldTimer > 0.0 || this.deleteCooldownTimer > 0.0);

      if (isActiveDeleteTarget) {
        if (this.deleteCooldownTimer > 0.0) {
          const cdFrac = clamp(
            this.deleteCooldownTimer / DELETE_COOLDOWN_DURATION_SEC,
            0.0,
            1.0
          );
          ctx.strokeStyle = "rgba(255, 165, 50, 0.85)";
          ctx.lineWidth = 8.0;
          ctx.beginPath();
          ctx.arc(
            sc.x,
            sc.y,
            36.0,
            -Math.PI * 0.5,
            -Math.PI * 0.5 + Math.PI * 2 * cdFrac
          );
          ctx.stroke();
        } else {
          const delFrac = clamp(
            this.deleteHoldTimer / DELETE_HOLD_DURATION_SEC,
            0.0,
            1.0
          );
          ctx.strokeStyle = "rgba(45, 10, 15, 0.78)";
          ctx.lineWidth = 10.0;
          ctx.beginPath();
          ctx.arc(sc.x, sc.y, 37.0, 0, Math.PI * 2);
          ctx.stroke();

          ctx.strokeStyle = "rgba(255, 50, 65, 0.96)";
          ctx.lineWidth = 10.0;
          ctx.beginPath();
          ctx.arc(
            sc.x,
            sc.y,
            37.0,
            -Math.PI * 0.5,
            -Math.PI * 0.5 + Math.PI * 2 * delFrac
          );
          ctx.stroke();
        }
      } else if (lockedMode && isSelectedTarget) {
        const lockCol =
          this.leftMode === LeftMode.Delete
            ? "rgba(255, 80, 90, 0.92)"
            : this.isPinching
              ? "rgba(255, 220, 65, 0.96)"
              : obj.groupId >= 0
                ? "rgba(60, 255, 165, 0.92)"
                : "rgba(65, 245, 255, 0.92)";
        this.drawLockedHighlight2D(ctx, sc, lockCol, this.isPinching);
      } else if (isIdleAim || (this.leftMode === LeftMode.Group && obj.isHovered)) {
        this.drawHoverReticle2D(
          ctx,
          sc,
          obj.groupId >= 0
            ? "rgba(85, 245, 175, 0.72)"
            : "rgba(110, 235, 255, 0.70)"
        );
      }

      if (isPrimaryLabel) {
        const showLockedTag = lockedMode && isSelectedTarget;
        let dimX = obj.transform.scale.x * CM_PER_WORLD_UNIT;
        let dimY = obj.transform.scale.y * CM_PER_WORLD_UNIT;
        let dimZ = obj.transform.scale.z * CM_PER_WORLD_UNIT;
        if (obj.type === PrimitiveType.ImportedObj && obj.importedModel) {
          const nb = obj.importedModel.normalizedLocalBox;
          dimX *= Math.max(0.01, nb.max.x - nb.min.x);
          dimY *= Math.max(0.01, nb.max.y - nb.min.y);
          dimZ *= Math.max(0.01, nb.max.z - nb.min.z);
        }
        const labelText = this.displayLabel(obj);
        const dimsStr = `${ dimX.toFixed(0)}x${dimY.toFixed(0)}x${dimZ.toFixed(0)} cm`;
        let tag = "";
        if (obj.groupId >= 0) {
          tag = showLockedTag
            ? `${labelText} #${obj.id} [GRP ${obj.groupId} // LOCKED]  ${dimsStr}`
            : `${labelText} #${obj.id} [GRP ${obj.groupId}]  ${dimsStr}`;
        } else {
          tag = showLockedTag
            ? `${labelText} #${obj.id} [LOCKED]  ${dimsStr}`
            : `${labelText} #${obj.id}  ${dimsStr}`;
        }

        const textCol = isActiveDeleteTarget
          ? { r: 255, g: 110, b: 120 }
          : showLockedTag
            ? { r: 255, g: 230, b: 95 }
            : obj.groupId >= 0
              ? { r: 100, g: 250, b: 180 }
              : { r: 140, g: 235, b: 255 };

        ctx.font = '700 13px "JetBrains Mono", "Cascadia Mono", Consolas, monospace';
        const tagW = Math.ceil(ctx.measureText(tag).width);
        const lPillW = tagW + 16;
        const lPillH = 22;
        const bottomGap = isActiveDeleteTarget
          ? 48
          : showLockedTag || isIdleAim
            ? 44
            : 30;
        const lx = clamp(Math.round(sc.x - lPillW / 2), 8, Math.max(8, sw - lPillW - 8));
        const ly = Math.max(8, Math.round(sc.y - bottomGap - lPillH));

        this.drawRoundedPill2D(
          ctx,
          lx,
          ly,
          lPillW,
          lPillH,
          10,
          "rgba(12, 16, 22, 0.85)",
          rgbCss(textCol, 0.65),
          1.2
        );
        ctx.fillStyle = rgbCss(textCol, 1.0);
        ctx.fillText(tag, lx + 8, ly + lPillH / 2);
      } else if (isGroupBadge) {
        const grpTag = `[GRP ${obj.groupId}]`;
        ctx.font = '700 12px "JetBrains Mono", "Cascadia Mono", Consolas, monospace';
        const gW = Math.ceil(ctx.measureText(grpTag).width);
        const gPillW = gW + 12;
        const gPillH = 18;
        const gx = clamp(Math.round(sc.x - gPillW / 2), 8, Math.max(8, sw - gPillW - 8));
        const gy = Math.max(8, Math.round(sc.y - 26 - gPillH));

        this.drawRoundedPill2D(
          ctx,
          gx,
          gy,
          gPillW,
          gPillH,
          8,
          "rgba(12, 22, 18, 0.80)",
          "rgba(55, 225, 145, 0.70)",
          1.0
        );
        ctx.fillStyle = "rgb(95, 250, 175)";
        ctx.fillText(grpTag, gx + 6, gy + gPillH / 2);
      }
    }

    // 2. Update DOM Telemetry Card & Bottom HUD Visibility
    if (this.cleanHud) {
      this.telemetryCard.classList.add("hidden");
      this.bottomHud.classList.add("hidden");
    } else {
      this.telemetryCard.classList.remove("hidden");
      this.bottomHud.classList.remove("hidden");

      this.telViewport.textContent = `VIEWPORT : ${this.viewportMode === ViewportMode.Studio3D ? "STUDIO 3D" : "AR OVERLAY"} [TAB]`;
      this.telLMode.textContent = `L-MODE   : ${modeName(this.leftMode)}`;

      if (this.leftMode === LeftMode.Spawn) {
        this.telRAxis.textContent = `R-PICK   : ${primitiveName(this.activeShape)} (${spawnFingerShort(this.activeShape)})`;
        this.telRAxis.className = "tel-row tel-cyan";
      } else if (this.leftMode === LeftMode.Group) {
        this.telRAxis.textContent = `R-ACTION : ${groupActionName(this.activeGroupAction)} (${groupActionFingerShort(this.activeGroupAction)})`;
        this.telRAxis.className = "tel-row tel-green";
      } else {
        this.telRAxis.textContent = `R-AXIS   : ${axisName(this.rightAxis)} (${axisFingerShort(this.rightAxis)})`;
        const axisClass =
          this.rightAxis === RightAxis.Y
            ? "tel-axis-y"
            : this.rightAxis === RightAxis.Z
              ? "tel-axis-z"
              : this.rightAxis === RightAxis.XYZ
                ? "tel-axis-xyz"
                : "tel-axis-x";
        this.telRAxis.className = `tel-row ${axisClass}`;
      }

      if (!this.hands[1].present || this.pinchPx >= 900.0) {
        this.telClutch.textContent = "CLUTCH   : NO HAND";
        this.telClutch.className = "tel-row tel-gray";
      } else {
        this.telClutch.textContent = `CLUTCH   : ${this.isPinching ? "PINCH" : "HOVER"} (${this.pinchPx.toFixed(0)} px)`;
        this.telClutch.className = `tel-row ${this.isPinching ? "tel-yellow" : "tel-gray"}`;
      }

      if (this.visionReady && this.cameraReady) {
        this.telEngine.textContent = `ENGINE   : ${this.fps} FPS | ${this.trackerFps} TRK/S`;
        this.telEngine.className = "tel-row tel-green";
      } else {
        this.telEngine.textContent = `ENGINE   : ${this.fps} FPS | ${this.visionStatus}`;
        this.telEngine.className = "tel-row tel-orange";
      }

      this.telScene.textContent = `SCENE    : ${this.objects.length} ${this.objects.length === 1 ? "OBJECT" : "OBJECTS"} (1 unit = 10 cm)`;

      // Highlight active dock button
      for (const btn of this.dockButtons) {
        const m = parseInt(btn.dataset.mode, 10);
        btn.classList.toggle("active", m === this.leftMode);
      }
    }

    // 3. Auto-Sizing Top-Center Mode Banner
    const showSwatch = this.leftMode === LeftMode.Recolor;
    const showDeleteBar =
      this.leftMode === LeftMode.Delete &&
      this.deleteCooldownTimer <= 0.0 &&
      this.deleteHoldTimer > 0.0;
    const showDeleteCooldown =
      this.leftMode === LeftMode.Delete && this.deleteCooldownTimer > 0.0;

    this.modeBannerText.textContent = this.measurementBadge;
    this.modeBannerText.style.color = showDeleteBar
      ? "rgb(255, 115, 125)"
      : "#ffffff";

    if (showSwatch) {
      this.modeBannerSwatch.classList.remove("hidden");
      this.modeBannerSwatch.style.backgroundColor = rgbCss(this.activeColor, 1.0);
    } else {
      this.modeBannerSwatch.classList.add("hidden");
    }

    if (showDeleteBar) {
      this.modeDeleteTrack.classList.remove("hidden");
      const delFrac = clamp(this.deleteHoldTimer / DELETE_HOLD_DURATION_SEC, 0.0, 1.0);
      this.modeDeleteFill.style.width = `${(delFrac * 100).toFixed(1)}%`;
    } else {
      this.modeDeleteTrack.classList.add("hidden");
    }

    let borderCol = rgbCss(axisColorRGB(this.rightAxis), 1.0);
    if (showSwatch) borderCol = rgbCss(this.activeColor, 1.0);
    else if (showDeleteBar) borderCol = "rgb(255, 60, 75)";
    else if (showDeleteCooldown) borderCol = "rgb(255, 165, 50)";
    else if (this.leftMode === LeftMode.Group) {
      borderCol = this.isPinching ? "rgb(255, 215, 65)" : "rgb(50, 235, 155)";
    } else if (this.isPinching) {
      borderCol = "rgb(255, 215, 65)";
    }
    this.modeBanner.style.borderColor = borderCol;
    if (!this.cleanHud) {
      const telRight = this.telemetryCard.offsetLeft + this.telemetryCard.offsetWidth;
      const pillW = this.modeBanner.offsetWidth;
      const centeredLeft = (sw - pillW) * 0.5;
      if (centeredLeft < telRight + 12) {
        const clampedLeft = Math.max(16, Math.min(sw - pillW - 16, telRight + 12));
        this.modeBanner.style.left = `${clampedLeft}px`;
        this.modeBanner.style.transform = "none";
      } else {
        this.modeBanner.style.left = "50%";
        this.modeBanner.style.transform = "translateX(-50%)";
      }
    } else {
      this.modeBanner.style.left = "50%";
      this.modeBanner.style.transform = "translateX(-50%)";
    }

    // 3b. Toast Notification Banner
    if (this.toastTimer > 0.0 && this.toastMessage) {
      this.toastBanner.classList.remove("hidden");
      this.toastBanner.textContent = this.toastMessage;
      this.toastBanner.classList.toggle("toast-error", this.toastIsError);
      const alpha = clamp(this.toastTimer / 0.6, 0.0, 1.0);
      this.toastBanner.style.opacity = alpha.toFixed(2);
    } else {
      this.toastBanner.classList.add("hidden");
    }
  }

  // -------------------------------------------------------------------------
  // 10. File I/O Actions (.OBJ Import/Export & .3dmproj Save/Load)
  // -------------------------------------------------------------------------
  async handleImportedOBJText(text, filename) {
    const asset = parseAndNormalizeOBJ(text, filename, 2.0);
    if (!asset) {
      this.showToast(`IMPORT FAILED: ${extractFilename(filename)}`, true);
      return false;
    }
    this.importedAssetCache.set(asset.sourcePath, asset);
    this.importedAssetCache.set(asset.displayName, asset);

    this.spawnObject(
      PrimitiveType.ImportedObj,
      { x: 0.0, y: 0.0, z: 0.0 },
      { r: 215, g: 225, b: 240, a: 255 },
      asset
    );
    this.activeObjectIdx = this.objects.length - 1;
    this.showToast(`IMPORTED: ${asset.displayName}`, false);
    return true;
  }

  async handleLoadedProjectText(text, filename) {
    const res = parseProjectFrom3dmproj(text, this.importedAssetCache);
    if (!res) {
      this.showToast(`LOAD FAILED: ${extractFilename(filename)}`, true);
      return false;
    }
    this.clearThreeObjects();
    this.objects = res.objects;
    for (const obj of this.objects) {
      this.createThreeNodeForObject(obj);
    }
    this.nextGroupId = res.nextGroupId;
    this.nextId = this.objects.length + 1;
    this.deleteHoldTimer = 0.0;
    this.deleteCooldownTimer = 0.0;

    this.activeObjectIdx = -1;
    for (let i = 0; i < this.objects.length; i++) {
      if (this.objects[i].isSelected) {
        this.activeObjectIdx = i;
        break;
      }
    }
    if (this.activeObjectIdx < 0 && this.objects.length > 0) {
      this.activeObjectIdx = 0;
    }

    let msg = `PROJECT LOADED -> ${extractFilename(filename)} (${this.objects.length} objects)`;
    if (res.missingObjCount > 0) {
      msg += ` [${res.missingObjCount} missing .OBJ skipped]`;
    }
    this.showToast(msg, false);
    return true;
  }

  saveProjectDownload() {
    const content = serializeProjectTo3dmproj(this.objects, this.nextGroupId);
    triggerBlobDownload("untitled.3dmproj", content);
    this.showToast("PROJECT SAVED -> untitled.3dmproj", false);
  }

  exportGroupOBJDownload() {
    const { indices, isGroup } = this.collectGroupExportTargets();
    if (indices.length === 0) {
      this.showToast("NOTHING TO EXPORT: SELECT AN OBJECT OR GROUP FIRST", true);
      return;
    }
    const subset = indices.map((idx) => this.objects[idx]);
    const objContent = serializeObjectsToOBJ(subset, this.primitives);
    const filename = "group_export.obj";
    triggerBlobDownload(filename, objContent);
    if (isGroup || subset.length > 1) {
      this.showToast(`EXPORTED GROUP (${subset.length} objects) -> ${filename}`, false);
    } else {
      this.showToast(`EXPORTED OBJECT (1 object) -> ${filename}`, false);
    }
  }

  exportSceneOBJDownload() {
    if (this.objects.length === 0) {
      this.showToast("SCENE IS EMPTY: NOTHING TO EXPORT", true);
      return;
    }
    const objContent = serializeObjectsToOBJ(this.objects, this.primitives);
    const filename = "scene_export.obj";
    triggerBlobDownload(filename, objContent);
    this.showToast(`EXPORTED SCENE (${this.objects.length} objects) -> ${filename}`, false);
  }

  toggleManualDrawer() {
    this.manualOpen = !this.manualOpen;
    this.manualDrawerWrapper.classList.toggle("drawer-open", this.manualOpen);
    this.manualDrawerWrapper.classList.toggle("drawer-closed", !this.manualOpen);
  }

  // -------------------------------------------------------------------------
  // 11. Keyboard Shortcuts, Drag-and-Drop & UI Event Listeners
  // -------------------------------------------------------------------------
  handleKeyCode(code, ctrlKey = false, event = null) {
    if (code === "Tab") {
      if (event) event.preventDefault();
      this.viewportMode =
        this.viewportMode === ViewportMode.Studio3D
          ? ViewportMode.ArCamera
          : ViewportMode.Studio3D;
    } else if (code === "F1") {
      if (event) event.preventDefault();
      this.toggleManualDrawer();
    } else if (code === "Digit1") {
      this.activeShape = PrimitiveType.Cube;
    } else if (code === "Digit2") {
      this.activeShape = PrimitiveType.Capsule;
    } else if (code === "Digit3") {
      this.activeShape = PrimitiveType.Cylinder;
    } else if (code === "Digit4") {
      this.activeShape = PrimitiveType.Torus;
    } else if (code === "KeyX") {
      this.rightAxis = RightAxis.X;
      this.hasKeyboardAxisOverride = true;
    } else if (code === "KeyY") {
      this.rightAxis = RightAxis.Y;
      this.hasKeyboardAxisOverride = true;
    } else if (code === "KeyZ") {
      this.rightAxis = RightAxis.Z;
      this.hasKeyboardAxisOverride = true;
    } else if (code === "KeyG") {
      this.groupSelectedObjects();
    } else if (code === "KeyU") {
      for (const o of this.objects) o.groupId = -1;
      this.showToast("UNGROUPED ALL OBJECTS", false, 2.2);
    } else if (code === "KeyH") {
      this.cleanHud = !this.cleanHud;
      this.showToast(
        this.cleanHud ? "CLEAN HUD: ON  [PRESS H TO RESTORE]" : "CLEAN HUD: OFF",
        false,
        2.2
      );
    } else if (code === "Space") {
      if (event) event.preventDefault();
      this.spawnObject(this.activeShape, this.cursor3D, this.activeColor);
      this.activeObjectIdx = this.objects.length - 1;
    } else if (code === "KeyP" || (ctrlKey && code === "KeyS")) {
      if (event) event.preventDefault();
      this.saveProjectDownload();
    } else if (code === "KeyO") {
      if (event) event.preventDefault();
      this.fileInputProj.value = "";
      this.fileInputProj.click();
    } else if (code === "KeyI") {
      if (event) event.preventDefault();
      this.fileInputObj.value = "";
      this.fileInputObj.click();
    } else if (code === "KeyE") {
      if (event) event.preventDefault();
      this.exportGroupOBJDownload();
    } else if (!ctrlKey && code === "KeyS") {
      if (event) event.preventDefault();
      this.exportSceneOBJDownload();
    } else if (code === "KeyC") {
      this.clearThreeObjects();
      this.objects = [];
      this.activeObjectIdx = -1;
      this.deleteHoldTimer = 0.0;
      this.deleteCooldownTimer = 0.0;
      this.showToast("SCENE CLEARED", false, 2.2);
    } else if (code === "KeyR") {
      this.resetDefaultScene();
      this.showToast("SCENE RESET TO DEFAULTS", false, 2.2);
    }
  }

  bindEvents() {
    window.addEventListener("resize", () => this.onResize());

    window.addEventListener("keydown", (e) => {
      if (e.repeat) return;
      this.handleKeyCode(e.code, e.ctrlKey || e.metaKey, e);
    });

    this.manualTab.addEventListener("click", () => this.toggleManualDrawer());

    for (const btn of document.querySelectorAll(".strip-btn")) {
      btn.addEventListener("click", () => {
        const key = btn.dataset.key;
        if (key) this.handleKeyCode(key, false, null);
      });
    }

    for (const btn of this.dockButtons) {
      btn.addEventListener("click", () => {
        const m = parseInt(btn.dataset.mode, 10);
        if (Number.isFinite(m)) {
          this.manualModeOverride = m;
          this.leftMode = m;
        }
      });
    }

    this.camToggleBtn.addEventListener("click", () => {
      this.initWebcamAndVision();
    });

    // File Picker Listeners
    this.fileInputObj.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const text = await file.text();
      await this.handleImportedOBJText(text, file.name);
    });

    this.fileInputProj.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const text = await file.text();
      await this.handleLoadedProjectText(text, file.name);
    });

    // Drag-and-Drop .obj and .3dmproj onto window
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropOverlay.classList.remove("hidden");
    });

    window.addEventListener("dragleave", (e) => {
      if (e.relatedTarget === null) {
        this.dropOverlay.classList.add("hidden");
      }
    });

    window.addEventListener("drop", async (e) => {
      e.preventDefault();
      this.dropOverlay.classList.add("hidden");
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || files.length === 0) return;

      // First load any .obj files so they are cached if a .3dmproj is dropped alongside them
      const fileList = Array.from(files);
      const objFiles = fileList.filter((f) => f.name.toLowerCase().endsWith(".obj"));
      const projFiles = fileList.filter((f) => f.name.toLowerCase().endsWith(".3dmproj"));
      const otherFiles = fileList.filter(
        (f) =>
          !f.name.toLowerCase().endsWith(".obj") &&
          !f.name.toLowerCase().endsWith(".3dmproj")
      );

      for (const f of objFiles) {
        const text = await f.text();
        await this.handleImportedOBJText(text, f.name);
      }
      for (const f of projFiles) {
        const text = await f.text();
        await this.handleLoadedProjectText(text, f.name);
      }
      if (otherFiles.length > 0) {
        this.showToast("DRAG & DROP: ONLY .3DMPROJ AND .OBJ FILES SUPPORTED", true);
      }
    });
  }

  onResize() {
    const sw = window.innerWidth;
    const sh = window.innerHeight;
    this.arBgCanvas.width = sw;
    this.arBgCanvas.height = sh;
    this.hud2dCanvas.width = sw;
    this.hud2dCanvas.height = sh;
    this.camera.aspect = sw / Math.max(1, sh);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(sw, sh);
  }

  // -------------------------------------------------------------------------
  // 12. Main 60 FPS Loop
  // -------------------------------------------------------------------------
  loop(nowMs) {
    const dt = Math.min(0.05, Math.max(0.0, (nowMs - this.lastFrameTime) / 1000.0));
    this.lastFrameTime = nowMs;
    const nowSec = nowMs / 1000.0;

    this.frameCounter++;
    if (nowMs - this.fpsTimer >= 1000.0) {
      const elapsedSec = (nowMs - this.fpsTimer) / 1000.0;
      this.fps = Math.round(this.frameCounter / elapsedSec);
      this.trackerFps = Math.round(this.trackerFrameCounter / elapsedSec);
      this.frameCounter = 0;
      this.trackerFrameCounter = 0;
      this.fpsTimer = nowMs;
    }

    if (this.toastTimer > 0.0) {
      this.toastTimer = Math.max(0.0, this.toastTimer - dt);
    }

    // 1. Vision & Gesture Detection
    this.processVisionFrame(nowSec, nowMs);

    // 2. Mode + Clutch State Machine
    this.runModeClutchStateMachine(dt);

    // 3. Render Webcam PiP / Fullscreen AR Overlay + 2D Hand Skeleton
    this.renderWebcamPiPOrAR();

    // 4. Update & Render Three.js 3D Scene
    this.updateThreeScene();
    this.renderer.render(this.scene, this.camera);

    // 5. Render 2D Screen-Space HUD Overlays
    this.render2DHudOverlays();

    requestAnimationFrame((ts) => this.loop(ts));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new Visual3DWebApp();
  app.init();
});
