import * as THREE from 'three';
import { createWasmPhysicsKernel, WasmPhysicsKernel } from './wasmKernel';

/**
 * Pure 3D Hydrodynamic, Gyroscope Sloshing & Exact Continuous Torus/OBB Collision Engine
 * for 1976 Tomy Waterful Ring-Toss.
 *
 * Complete Audit Resolution Summary:
 * 1. P0 Gyroscope & Sensor Bridge (`deviceorientation` + `devicemotion` `rotationRate` + `requestPermission` + landscape swap):
 *    - Built-in `setGyroscopeAngularVelocity` + finite-difference angular rate tracking on `tiltRoll`/`tiltPitch`
 *      + full `sensorBridge.ts` integration.
 * 2. P0 Tilt/Rotation -> Whole-Tank Water Inertial Sloshing Flow (`waterAngularVel` & `-ω × r` in `sampleWaterVelocityField`):
 *    - Rotating or tilting the console excites `state.waterAngularVel` (counter-rotating inertial slosh in the tank frame
 *      that decays with wall viscosity), adding `waterAngularVel × pos` velocity and `2 * waterAngularVel` vorticity
 *      throughout the entire 3D water volume!
 * 3. P1 `gz = -Math.sin(clampedPitch)` Sign Fix:
 *    - Matches exact right-handed `Rz(roll) * Rx(pitch)` transpose applied to world gravity `(0, -1, 0)`:
 *      `gx = sin(roll) * cos(pitch)`, `gy = -cos(roll) * cos(pitch)`, `gz = -sin(pitch)`.
 * 4. P1 Isolated Per-Instance WASM Memory & Live `uniformFloatView` (Offset 512+):
 *    - Uses `createWasmPhysicsKernel()` per `SimulationState` (zero cross-instance state clobbering)
 *      and writes `subDt`, `damping`, `gravityDir`, and `waterAngularVel` into `uniformFloatView`.
 * 5. P1/P2 Zero-GC Pooled Contact Manifold, Tight `0.0025` Deduplication Threshold, Bidirectional 16-Seed Alternating Projection,
 *    Explicit `invMassA/B`, 4-Point Distributed Wall Contact (`tLen <= 1e-6`), and Unified Position-Correction Clamping.
 * 6. P2 3D Geometric Plane-Piercing `evaluateHookedRings`:
 *    - Computes where the 3D peg line pierces the tilted ring plane `(P - C) · n_ring = 0`, rejecting edge-on rings
 *      (`|n_ring · u_peg| < 0.25`) and verifying true 3D hole clearance `||P_pierce - C|| < holeRadius - pegRadiusAtY`.
 */

export interface TankBounds {
  halfWidthBottom: number;
  halfWidthTop: number;
  halfHeight: number;
  halfDepth: number;
}

export interface PegSpec {
  id: string;
  baseX: number;
  baseY: number;
  baseZ: number;
  height: number;
  baseRadius: number;
  tipRadius: number;
  holderCupRadius: number;
  holderCupY: number;
}

export interface NozzleSpec {
  id: 'left' | 'right';
  x: number;
  y: number;
  z: number;
  dirX: number;
  dirY: number;
  dirZ: number;
}

export interface OBB3D {
  center: THREE.Vector3;
  axes: [THREE.Vector3, THREE.Vector3, THREE.Vector3];
  halfExtents: THREE.Vector3;
}

export interface RingState {
  id: number;
  colorHex: number;
  colorName: string;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  quaternion: THREE.Quaternion;
  angularVelocity: THREE.Vector3;
  majorRadius: number;
  minorRadius: number;
  mass: number;
  hookedPegId: string | null;
}

export interface BubbleParticle {
  active: boolean;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  radius: number;
  life: number;
  maxLife: number;
  wobblePhase: number;
}

export interface FlowTracerParticle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  baseSize: number;
  phase: number;
}

export const WAVE_NX = 48;
export const WAVE_NZ = 16;

export type PumpForcePreset = 'light' | 'medium' | 'heavy';

export interface PumpState {
  depressed: number;
  targetDepth: number;
  velocity: number;
  jetStrength: number;
  suctionStrength: number;
  globalCirculation: number;
  pressCount: number;
  targetHeld: boolean;
  lastForcePreset: PumpForcePreset;
}

export interface PhysicsConfig {
  ringDensity: number;
  waterDensity: number;
  normalDragCoeff: number;
  tangentDragCoeff: number;
  angularDragCoeff: number;
  jetPowerMultiplier: number;
  turbulenceStrength: number;
  wasmSubsteps: number;
}

export interface SimulationState {
  tank: TankBounds;
  pegs: PegSpec[];
  nozzles: NozzleSpec[];
  rings: RingState[];
  bubbles: BubbleParticle[];
  flowParticles: FlowTracerParticle[];
  surfaceWaveHeight: Float32Array;
  surfaceWaveVel: Float32Array;
  leftPump: PumpState;
  rightPump: PumpState;
  gravityDir: THREE.Vector3;
  gravityMagnitude: number;
  tiltPitch: number;
  tiltRoll: number;
  prevTiltPitch: number;
  prevTiltRoll: number;
  // Gyroscope angular velocity of the console (rad/s) from DeviceMotion.rotationRate or tilt derivative
  deviceAngularVel: THREE.Vector3;
  // Relative angular velocity of the sloshing water body inside the tank frame (rad/s)
  waterAngularVel: THREE.Vector3;
  config: PhysicsConfig;
  timeElapsed: number;
  score: number;
  wasmKernel: WasmPhysicsKernel;
  totalWasmStepsExecuted: number;
  lastKineticEnergy: number;
  effectiveHz: number;
}

export const DEFAULT_TANK_BOUNDS: TankBounds = {
  halfWidthBottom: 1.58,
  halfWidthTop: 1.74,
  halfHeight: 2.15,
  halfDepth: 0.33, // Calibrated 3D chamber thickness (0.66 total Z depth vs 0.54 ring diameter) so rings thread needles reliably
};

export const DEFAULT_PEGS: PegSpec[] = [
  {
    id: 'peg-left',
    baseX: -0.46,
    baseY: -0.55,
    baseZ: 0.0,
    height: 1.35,
    baseRadius: 0.054,
    tipRadius: 0.026,
    holderCupRadius: 0.175,
    holderCupY: 0.14,
  },
  {
    id: 'peg-right',
    baseX: 0.46,
    baseY: -0.55,
    baseZ: 0.0,
    height: 1.35,
    baseRadius: 0.054,
    tipRadius: 0.026,
    holderCupRadius: 0.175,
    holderCupY: 0.14,
  },
];

export const DEFAULT_NOZZLES: NozzleSpec[] = [
  {
    id: 'left',
    x: -1.16,
    y: -2.11,
    z: 0.0,
    dirX: 0.04,
    dirY: 0.9992,
    dirZ: 0.0,
  },
  {
    id: 'right',
    x: 1.16,
    y: -2.11,
    z: 0.0,
    dirX: -0.04,
    dirY: 0.9992,
    dirZ: 0.0,
  },
];

export const DEFAULT_PHYSICS_CONFIG: PhysicsConfig = {
  ringDensity: 1.28,
  waterDensity: 1.0,
  normalDragCoeff: 2.05,
  tangentDragCoeff: 0.78,
  angularDragCoeff: 0.65,
  jetPowerMultiplier: 1.0,
  turbulenceStrength: 0.85,
  wasmSubsteps: 24,
};

export const PUMP_FORCE_VALUES: Record<PumpForcePreset, number> = {
  light: 0.22,
  medium: 0.64,
  heavy: 1.05,
};

const RING_PALETTE = [
  { hex: 0xff3b30, name: '珊瑚红' },
  { hex: 0xffcc00, name: '柠檬黄' },
  { hex: 0x34c759, name: '翡翠绿' },
  { hex: 0x00b0ff, name: '海洋蓝' },
  { hex: 0xff6d00, name: '活力橙' },
  { hex: 0xf50057, name: '荧光粉' },
];

// Pre-allocated scratch vectors for zero-GC simulation loop
const _ringNormal = new THREE.Vector3();
const _ringNormalB = new THREE.Vector3();
const _ringAxisU = new THREE.Vector3();
const _ringAxisV = new THREE.Vector3();
const _ringAxisUB = new THREE.Vector3();
const _ringAxisVB = new THREE.Vector3();
const _relVel = new THREE.Vector3();
const _velNormal = new THREE.Vector3();
const _velTangent = new THREE.Vector3();
const _dragForce = new THREE.Vector3();
const _torque = new THREE.Vector3();
const _fluidVel = new THREE.Vector3();
const _fluidVorticity = new THREE.Vector3();
const _sampleVel = new THREE.Vector3();
const _crossTmp = new THREE.Vector3();
const _diff = new THREE.Vector3();
const _dq = new THREE.Quaternion();
const _contactPt = new THREE.Vector3();
const _segA0 = new THREE.Vector3();
const _segA1 = new THREE.Vector3();
const _segB0 = new THREE.Vector3();
const _segB1 = new THREE.Vector3();
const _closestA = new THREE.Vector3();
const _closestB = new THREE.Vector3();
const _posDeltaAccum = new THREE.Vector3();

export const NUM_TORUS_SAMPLES = 16;
export const TORUS_SAMPLE_ANGLES = Array.from(
  { length: NUM_TORUS_SAMPLES },
  (_, i) => (i * Math.PI * 2) / NUM_TORUS_SAMPLES
);

const TORUS_COS = TORUS_SAMPLE_ANGLES.map((a) => Math.cos(a));
const TORUS_SIN = TORUS_SAMPLE_ANGLES.map((a) => Math.sin(a));

// 8 evenly-spaced seed angles for fast bidirectional Ring-A <-> Ring-B alternating projection
const NUM_BI_SEEDS = 8;
const BI_SEED_COS = Array.from({ length: NUM_BI_SEEDS }, (_, i) =>
  Math.cos((i * Math.PI * 2) / NUM_BI_SEEDS)
);
const BI_SEED_SIN = Array.from({ length: NUM_BI_SEEDS }, (_, i) =>
  Math.sin((i * Math.PI * 2) / NUM_BI_SEEDS)
);

export interface TorusContactManifold {
  pointOnA: THREE.Vector3;
  pointOnB: THREE.Vector3;
  normalAToB: THREE.Vector3;
  penetration: number;
}

// Pre-allocated Zero-GC Contact Manifold Pool (eliminates `.clone()` inside the simulation loop!)
const MAX_MANIFOLD_CONTACTS = 4;
const _manifoldPool: TorusContactManifold[] = Array.from(
  { length: MAX_MANIFOLD_CONTACTS },
  () => ({
    pointOnA: new THREE.Vector3(),
    pointOnB: new THREE.Vector3(),
    normalAToB: new THREE.Vector3(),
    penetration: 0,
  })
);
const _activeManifoldSlice: TorusContactManifold[] = [];

/**
 * Exact 3D Segment-to-Segment Closest Points Solver (`closestPointsSegmentSegment3D`)
 */
export function closestPointsSegmentSegment3D(
  p1: THREE.Vector3,
  q1: THREE.Vector3,
  p2: THREE.Vector3,
  q2: THREE.Vector3,
  outP: THREE.Vector3,
  outQ: THREE.Vector3
): { s: number; t: number; distSq: number } {
  const d1x = q1.x - p1.x;
  const d1y = q1.y - p1.y;
  const d1z = q1.z - p1.z;

  const d2x = q2.x - p2.x;
  const d2y = q2.y - p2.y;
  const d2z = q2.z - p2.z;

  const rx = p1.x - p2.x;
  const ry = p1.y - p2.y;
  const rz = p1.z - p2.z;

  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;

  let s = 0;
  let t = 0;

  if (a <= 1e-10 && e <= 1e-10) {
    outP.copy(p1);
    outQ.copy(p2);
    return { s: 0, t: 0, distSq: rx * rx + ry * ry + rz * rz };
  }
  if (a <= 1e-10) {
    s = 0;
    t = THREE.MathUtils.clamp(f / e, 0, 1);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= 1e-10) {
      t = 0;
      s = THREE.MathUtils.clamp(-c / a, 0, 1);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;

      if (denom !== 0) {
        s = THREE.MathUtils.clamp((b * f - c * e) / denom, 0, 1);
      } else {
        s = 0;
      }

      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = THREE.MathUtils.clamp(-c / a, 0, 1);
      } else if (t > 1) {
        t = 1;
        s = THREE.MathUtils.clamp((b - c) / a, 0, 1);
      }
    }
  }

  outP.set(p1.x + d1x * s, p1.y + d1y * s, p1.z + d1z * s);
  outQ.set(p2.x + d2x * t, p2.y + d2y * t, p2.z + d2z * t);
  const dx = outP.x - outQ.x;
  const dy = outP.y - outQ.y;
  const dz = outP.z - outQ.z;
  return { s, t, distSq: dx * dx + dy * dy + dz * dz };
}

/**
 * Exact 3D Separating Axis Theorem (SAT) Test between two 3D Oriented Bounding Boxes (OBBs).
 */
export function testOBBvsOBB_SAT(
  a: OBB3D,
  b: OBB3D,
  outNormal: THREE.Vector3
): { intersects: boolean; penetration: number } {
  const tx = b.center.x - a.center.x;
  const ty = b.center.y - a.center.y;
  const tz = b.center.z - a.center.z;

  let minPen = Infinity;
  let bestAxisX = 0;
  let bestAxisY = 1;
  let bestAxisZ = 0;

  const testAxis = (ax: number, ay: number, az: number): boolean => {
    const lenSq = ax * ax + ay * ay + az * az;
    if (lenSq < 1e-8) return true;
    const invLen = 1 / Math.sqrt(lenSq);
    const ux = ax * invLen;
    const uy = ay * invLen;
    const uz = az * invLen;

    const rA =
      Math.abs(ux * a.axes[0].x + uy * a.axes[0].y + uz * a.axes[0].z) * a.halfExtents.x +
      Math.abs(ux * a.axes[1].x + uy * a.axes[1].y + uz * a.axes[1].z) * a.halfExtents.y +
      Math.abs(ux * a.axes[2].x + uy * a.axes[2].y + uz * a.axes[2].z) * a.halfExtents.z;

    const rB =
      Math.abs(ux * b.axes[0].x + uy * b.axes[0].y + uz * b.axes[0].z) * b.halfExtents.x +
      Math.abs(ux * b.axes[1].x + uy * b.axes[1].y + uz * b.axes[1].z) * b.halfExtents.y +
      Math.abs(ux * b.axes[2].x + uy * b.axes[2].y + uz * b.axes[2].z) * b.halfExtents.z;

    const distProj = tx * ux + ty * uy + tz * uz;
    const pen = rA + rB - Math.abs(distProj);
    if (pen <= 0) return false;

    if (pen < minPen) {
      minPen = pen;
      const sign = distProj >= 0 ? 1 : -1;
      bestAxisX = ux * sign;
      bestAxisY = uy * sign;
      bestAxisZ = uz * sign;
    }
    return true;
  };

  for (let i = 0; i < 3; i++) {
    if (!testAxis(a.axes[i].x, a.axes[i].y, a.axes[i].z)) {
      return { intersects: false, penetration: 0 };
    }
    if (!testAxis(b.axes[i].x, b.axes[i].y, b.axes[i].z)) {
      return { intersects: false, penetration: 0 };
    }
  }

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const cx = a.axes[i].y * b.axes[j].z - a.axes[i].z * b.axes[j].y;
      const cy = a.axes[i].z * b.axes[j].x - a.axes[i].x * b.axes[j].z;
      const cz = a.axes[i].x * b.axes[j].y - a.axes[i].y * b.axes[j].x;
      if (!testAxis(cx, cy, cz)) {
        return { intersects: false, penetration: 0 };
      }
    }
  }

  outNormal.set(bestAxisX, bestAxisY, bestAxisZ);
  return { intersects: true, penetration: minPen };
}

/**
 * Exact Analytical Closest Point on a 3D Circle (center C, unit normal n, radius R)
 * to an arbitrary 3D query point Q.
 */
export function closestPointOnCircle3D(
  circleCenter: THREE.Vector3,
  circleNormal: THREE.Vector3,
  circleAxisU: THREE.Vector3,
  majorRadius: number,
  queryPoint: THREE.Vector3,
  outClosest: THREE.Vector3
): void {
  const dx = queryPoint.x - circleCenter.x;
  const dy = queryPoint.y - circleCenter.y;
  const dz = queryPoint.z - circleCenter.z;

  const dotN = dx * circleNormal.x + dy * circleNormal.y + dz * circleNormal.z;
  const px = dx - dotN * circleNormal.x;
  const py = dy - dotN * circleNormal.y;
  const pz = dz - dotN * circleNormal.z;

  const lenSq = px * px + py * py + pz * pz;
  if (lenSq > 1e-10) {
    const scale = majorRadius / Math.sqrt(lenSq);
    outClosest.set(
      circleCenter.x + px * scale,
      circleCenter.y + py * scale,
      circleCenter.z + pz * scale
    );
  } else {
    outClosest.set(
      circleCenter.x + circleAxisU.x * majorRadius,
      circleCenter.y + circleAxisU.y * majorRadius,
      circleCenter.z + circleAxisU.z * majorRadius
    );
  }
}

/**
 * Zero-GC Bidirectional Exact 3D Shortest Distance & Contact Manifold between two 3D Tori:
 * - Seeds alternating projection from BOTH Ring A (8 seeds) AND Ring B (8 seeds) so global minima
 *   are guaranteed to be found whether contact is near A's rim or B's rim.
 * - Uses pre-allocated `_manifoldPool` (zero `new THREE.Vector3()` or `.clone()` allocations!).
 * - Uses tight deduplication threshold `0.0025` (distance < 0.05m, smaller than tube radius 0.058m)
 *   so distinct contact points are never falsely merged.
 */
export function computeExactTorusToTorusContacts(
  a: RingState,
  b: RingState
): TorusContactManifold[] {
  _activeManifoldSlice.length = 0;

  _diff.subVectors(b.position, a.position);
  const centerDistSq = _diff.lengthSq();
  const maxReach = a.majorRadius + a.minorRadius + b.majorRadius + b.minorRadius;
  if (centerDistSq > maxReach * maxReach) return _activeManifoldSlice;

  _ringAxisU.set(1, 0, 0).applyQuaternion(a.quaternion);
  _ringAxisV.set(0, 1, 0).applyQuaternion(a.quaternion);
  _ringNormal.set(0, 0, 1).applyQuaternion(a.quaternion).normalize();

  _ringAxisUB.set(1, 0, 0).applyQuaternion(b.quaternion);
  _ringAxisVB.set(0, 1, 0).applyQuaternion(b.quaternion);
  _ringNormalB.set(0, 0, 1).applyQuaternion(b.quaternion).normalize();

  const minTubeSum = a.minorRadius + b.minorRadius;
  const minTubeSumSq = minTubeSum * minTubeSum;
  let contactCount = 0;

  const evaluateSeedPair = () => {
    const dx = _closestB.x - _closestA.x;
    const dy = _closestB.y - _closestA.y;
    const dz = _closestB.z - _closestA.z;
    const distSq = dx * dx + dy * dy + dz * dz;

    if (distSq < minTubeSumSq && distSq > 1e-10) {
      const dist = Math.sqrt(distSq);
      const pen = minTubeSum - dist;

      // Tight deduplication threshold: 0.0025 (~0.05 units < tube radius 0.058)
      for (let k = 0; k < contactCount; k++) {
        const existing = _manifoldPool[k];
        if (existing.pointOnA.distanceToSquared(_closestA) < 0.0025) {
          if (pen > existing.penetration) {
            existing.pointOnA.copy(_closestA);
            existing.pointOnB.copy(_closestB);
            existing.normalAToB.set(dx / dist, dy / dist, dz / dist);
            existing.penetration = pen;
          }
          return;
        }
      }

      if (contactCount < MAX_MANIFOLD_CONTACTS) {
        const slot = _manifoldPool[contactCount++];
        slot.pointOnA.copy(_closestA);
        slot.pointOnB.copy(_closestB);
        slot.normalAToB.set(dx / dist, dy / dist, dz / dist);
        slot.penetration = pen;
        _activeManifoldSlice.push(slot);
      }
    }
  };

  // Pass 1: 8 seeds on Ring A -> project A->B->A->B
  for (let i = 0; i < NUM_BI_SEEDS; i++) {
    const c = BI_SEED_COS[i] * a.majorRadius;
    const s = BI_SEED_SIN[i] * a.majorRadius;
    _closestA.set(
      a.position.x + _ringAxisU.x * c + _ringAxisV.x * s,
      a.position.y + _ringAxisU.y * c + _ringAxisV.y * s,
      a.position.z + _ringAxisU.z * c + _ringAxisV.z * s
    );

    for (let iter = 0; iter < 3; iter++) {
      closestPointOnCircle3D(b.position, _ringNormalB, _ringAxisUB, b.majorRadius, _closestA, _closestB);
      closestPointOnCircle3D(a.position, _ringNormal, _ringAxisU, a.majorRadius, _closestB, _closestA);
    }
    closestPointOnCircle3D(b.position, _ringNormalB, _ringAxisUB, b.majorRadius, _closestA, _closestB);
    evaluateSeedPair();
  }

  // Pass 2: 8 seeds on Ring B -> project B->A->B->A (guarantees global symmetry & catches B-side crossings!)
  for (let i = 0; i < NUM_BI_SEEDS; i++) {
    const c = BI_SEED_COS[i] * b.majorRadius;
    const s = BI_SEED_SIN[i] * b.majorRadius;
    _closestB.set(
      b.position.x + _ringAxisUB.x * c + _ringAxisVB.x * s,
      b.position.y + _ringAxisUB.y * c + _ringAxisVB.y * s,
      b.position.z + _ringAxisUB.z * c + _ringAxisVB.z * s
    );

    for (let iter = 0; iter < 3; iter++) {
      closestPointOnCircle3D(a.position, _ringNormal, _ringAxisU, a.majorRadius, _closestB, _closestA);
      closestPointOnCircle3D(b.position, _ringNormalB, _ringAxisUB, b.majorRadius, _closestA, _closestB);
    }
    closestPointOnCircle3D(a.position, _ringNormal, _ringAxisU, a.majorRadius, _closestB, _closestA);
    evaluateSeedPair();
  }

  return _activeManifoldSlice;
}

export function createInitialSimulationState(ringCount = 12): SimulationState {
  const rings: RingState[] = [];
  // Compact center hole (holeRadius = 0.185 - 0.066 = 0.119) & substantial plastic tube (minorRadius = 0.066)
  const majorRadius = 0.185;
  const minorRadius = 0.066;

  for (let i = 0; i < ringCount; i++) {
    const paletteItem = RING_PALETTE[i % RING_PALETTE.length];
    const col = i % 6;
    const row = Math.floor(i / 6);
    const x = -1.1 + col * 0.44 + (row % 2 === 0 ? 0.04 : -0.04);
    const y = -1.82 + row * 0.24 + (col % 2) * 0.05;
    const z = ((i % 3) - 1) * 0.06;

    const q = new THREE.Quaternion();
    const euler = new THREE.Euler(
      Math.PI * 0.46 + ((i % 5) - 2) * 0.08,
      ((i % 7) - 3) * 0.22,
      ((i % 3) - 1) * 0.12
    );
    q.setFromEuler(euler);

    rings.push({
      id: i + 1,
      colorHex: paletteItem.hex,
      colorName: paletteItem.name,
      position: new THREE.Vector3(x, y, z),
      velocity: new THREE.Vector3(0, 0, 0),
      quaternion: q,
      angularVelocity: new THREE.Vector3(0, 0, 0),
      majorRadius,
      minorRadius,
      mass: 1.0,
      hookedPegId: null,
    });
  }

  // Minimal, clean bubble pool (32 max) so the water stays clear and uncluttered
  const bubbles: BubbleParticle[] = Array.from({ length: 32 }, (_, idx) => ({
    active: false,
    position: new THREE.Vector3(0, -2.0, 0),
    velocity: new THREE.Vector3(0, 0, 0),
    radius: 0.012 + (idx % 5) * 0.0045,
    life: 0,
    maxLife: 0.9,
    wobblePhase: idx * 0.71,
  }));

  const halton = (index: number, base: number): number => {
    let result = 0;
    let f = 1 / base;
    let i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  };

  // 64 subtle Lagrangian fluid tracer particles distributed in 3D via (2, 3, 5) Halton sequence
  const flowParticles: FlowTracerParticle[] = Array.from({ length: 64 }, (_, idx) => {
    const u = halton(idx + 1, 2) * 2 - 1;
    const v = (halton(idx + 1, 3) * 2 - 1) * 0.9;
    const w = (halton(idx + 1, 5) * 2 - 1) * 0.82;
    const y = v * DEFAULT_TANK_BOUNDS.halfHeight;
    const yNorm = (y + DEFAULT_TANK_BOUNDS.halfHeight) / (DEFAULT_TANK_BOUNDS.halfHeight * 2);
    const halfW = THREE.MathUtils.lerp(
      DEFAULT_TANK_BOUNDS.halfWidthBottom,
      DEFAULT_TANK_BOUNDS.halfWidthTop,
      yNorm
    );
    return {
      position: new THREE.Vector3(
        u * halfW * 0.9,
        y,
        w * DEFAULT_TANK_BOUNDS.halfDepth
      ),
      velocity: new THREE.Vector3(0, 0, 0),
      baseSize: 0.0065 + (idx % 4) * 0.0015,
      phase: idx * 1.13,
    };
  });

  const surfaceWaveHeight = new Float32Array(WAVE_NX * WAVE_NZ);
  const surfaceWaveVel = new Float32Array(WAVE_NX * WAVE_NZ);

  // Create an isolated WASM kernel instance with its own dedicated WebAssembly.Memory
  const wasmKernel = createWasmPhysicsKernel();

  return {
    tank: { ...DEFAULT_TANK_BOUNDS },
    pegs: DEFAULT_PEGS.map((p) => ({ ...p })),
    nozzles: DEFAULT_NOZZLES.map((n) => ({ ...n })),
    rings,
    bubbles,
    flowParticles,
    surfaceWaveHeight,
    surfaceWaveVel,
    leftPump: {
      depressed: 0,
      targetDepth: 0,
      velocity: 0,
      jetStrength: 0,
      suctionStrength: 0,
      globalCirculation: 0,
      pressCount: 0,
      targetHeld: false,
      lastForcePreset: 'heavy',
    },
    rightPump: {
      depressed: 0,
      targetDepth: 0,
      velocity: 0,
      jetStrength: 0,
      suctionStrength: 0,
      globalCirculation: 0,
      pressCount: 0,
      targetHeld: false,
      lastForcePreset: 'heavy',
    },
    gravityDir: new THREE.Vector3(0, -1, 0),
    gravityMagnitude: 9.81,
    tiltPitch: 0,
    tiltRoll: 0,
    prevTiltPitch: 0,
    prevTiltRoll: 0,
    deviceAngularVel: new THREE.Vector3(0, 0, 0),
    waterAngularVel: new THREE.Vector3(0, 0, 0),
    config: { ...DEFAULT_PHYSICS_CONFIG },
    timeElapsed: 0,
    score: 0,
    wasmKernel,
    totalWasmStepsExecuted: 0,
    lastKineticEnergy: 0,
    effectiveHz: 24 * 60,
  };
}

/**
 * Finite-Stroke Bellows Pump Control (`setPumpInput`):
 * - Pressing a button compresses the rubber bellows to a finite depth (`targetDepth <= 1.0`).
 * - Water is ONLY ejected while `targetDepth` is increasing during the compression stroke (`0 ~ 260ms`).
 * - Holding the button down after the bellows bottoms out (`targetDepth === 1.0`) injects ZERO additional water!
 */
export function setPumpInput(
  state: SimulationState,
  side: 'left' | 'right',
  isHeld: boolean,
  forcePresetOrLevel: PumpForcePreset | number = 'heavy'
): void {
  const pump = side === 'left' ? state.leftPump : state.rightPump;

  let normalizedForce = 1.0;
  if (typeof forcePresetOrLevel === 'string') {
    pump.lastForcePreset = forcePresetOrLevel;
    normalizedForce = PUMP_FORCE_VALUES[forcePresetOrLevel];
  } else {
    normalizedForce = THREE.MathUtils.clamp(forcePresetOrLevel, 0.1, 1.2);
    if (normalizedForce < 0.55) pump.lastForcePreset = 'light';
    else if (normalizedForce < 0.88) pump.lastForcePreset = 'medium';
    else pump.lastForcePreset = 'heavy';
  }

  if (isHeld) {
    const clampedDepth = THREE.MathUtils.clamp(normalizedForce, 0.08, 1.15);
    if (!pump.targetHeld) {
      pump.pressCount += 1;
      // Progressive soft-start impulse: ultra-light taps (clampedDepth ~ 0.18..0.24) produce a delicate gentle puff,
      // while deeper presses scale up smoothly to full power.
      const softKickScale = 0.72 + 0.24 * Math.min(1.0, clampedDepth);
      pump.jetStrength = Math.min(1.75, pump.jetStrength + softKickScale * clampedDepth);
      pump.suctionStrength *= 0.15;
      pump.globalCirculation = Math.min(
        1.9,
        pump.globalCirculation + 0.62 * clampedDepth
      );
      pump.targetDepth = clampedDepth;
    } else {
      const deltaForce = Math.max(0, clampedDepth - pump.targetDepth);
      if (deltaForce > 0) {
        pump.jetStrength = Math.min(1.75, pump.jetStrength + 1.14 * deltaForce);
        pump.globalCirculation = Math.min(
          1.9,
          pump.globalCirculation + 0.76 * deltaForce
        );
        pump.targetDepth = clampedDepth;
      }
    }
    pump.targetHeld = true;
  } else {
    if (pump.targetHeld || pump.depressed > 0.06) {
      const releaseVolume = Math.max(0.18, pump.depressed);
      pump.suctionStrength = Math.min(1.25, pump.suctionStrength + releaseVolume * 0.82);
      // If released early during a quick light tap, immediately curtail residual upward jet
      if (pump.depressed < 0.35) {
        pump.jetStrength *= 0.55;
      }
    }
    pump.targetHeld = false;
    pump.targetDepth = 0;
  }
}

/**
 * Feeds real-time gyroscope angular velocity (`devicemotion` `rotationRate` in rad/s)
 * into the simulation, exciting inertial water sloshing (`waterAngularVel`) in the 3D tank!
 */
export function setGyroscopeAngularVelocity(
  state: SimulationState,
  omegaPitchX: number,
  omegaRollZ: number,
  omegaYawY = 0
): void {
  state.deviceAngularVel.set(omegaPitchX, omegaYawY, omegaRollZ);
  // In the tank's reference frame, when the casing rotates by +omega, the fluid initially lags behind (-omega)
  state.waterAngularVel.x = THREE.MathUtils.clamp(
    state.waterAngularVel.x - omegaPitchX * 0.55,
    -6.0,
    6.0
  );
  state.waterAngularVel.y = THREE.MathUtils.clamp(
    state.waterAngularVel.y - omegaYawY * 0.45,
    -6.0,
    6.0
  );
  state.waterAngularVel.z = THREE.MathUtils.clamp(
    state.waterAngularVel.z - omegaRollZ * 0.65,
    -6.0,
    6.0
  );
}

/**
 * Updates gravity direction from mobile `deviceorientation` or manual tilt pad,
 * using the exact right-handed `Rz(roll) * Rx(pitch)` transpose formula (`gz = -Math.sin(clampedPitch)`),
 * and excites whole-tank water sloshing (`waterAngularVel`) proportional to the tilt change!
 */
export function setGravityFromTilt(
  state: SimulationState,
  rollRad: number,
  pitchRad: number
): void {
  // Wrap rollRad cleanly to [-pi, pi] (no artificial ±0.85*pi clamp!) so holding the phone
  // with its head pointing straight down (rollRad = ±pi) produces exact upward screen gravity (gx = 0, gy = +1).
  const clampedRoll =
    THREE.MathUtils.euclideanModulo(rollRad + Math.PI, Math.PI * 2) - Math.PI;
  const clampedPitch = THREE.MathUtils.clamp(pitchRad, -Math.PI, Math.PI);

  let deltaRoll = clampedRoll - state.tiltRoll;
  if (deltaRoll > Math.PI) deltaRoll -= Math.PI * 2;
  if (deltaRoll < -Math.PI) deltaRoll += Math.PI * 2;
  const deltaPitch = clampedPitch - state.tiltPitch;

  state.tiltRoll = clampedRoll;
  state.tiltPitch = clampedPitch;

  // Rectangular water tank coupling (`方形的水体本身是会随着转动被方形的限制框带动的`):
  // The 4 flat walls of the rectangular frame positively carry the bulk water along with the box;
  // only a mild irrotational core eddy lags in the interior, damped rapidly by the flat walls.
  if (Math.abs(deltaRoll) > 1e-5 || Math.abs(deltaPitch) > 1e-5) {
    state.waterAngularVel.z = THREE.MathUtils.clamp(
      state.waterAngularVel.z - deltaRoll * 0.55,
      -3.5,
      3.5
    );
    state.waterAngularVel.x = THREE.MathUtils.clamp(
      state.waterAngularVel.x - deltaPitch * 0.45,
      -3.5,
      3.5
    );
  }

  // Exact right-handed R_local<-world = (Rz(roll) * Rx(pitch))^T applied to world gravity (0, -1, 0):
  // gx = sin(roll) * cos(pitch)
  // gy = -cos(roll) * cos(pitch)
  // gz = -sin(pitch)
  const gx = Math.sin(clampedRoll) * Math.cos(clampedPitch);
  const gy = -Math.cos(clampedRoll) * Math.cos(clampedPitch);
  const gz = -Math.sin(clampedPitch);
  state.gravityDir.set(gx, gy, gz).normalize();
}

/**
 * Computes the 3D fluid velocity field and vorticity inside the closed 3D water tank at point (x, y, z).
 * Includes:
 * 1. Ejection Jet Cone + Bernoulli Entrainment from the Left/Right Jet Nozzles (`jetStrength`)
 * 2. Separate Return-Vent Suction Flow (`suctionStrength` drawn into the bottom-center & outer-corner intake vents, NEVER into the jet nozzles!)
 * 3. Whole-Tank 3D Toroidal Recirculation Cell (`globalCirculation`)
 * 4. Gyroscope / Tilt-Induced Inertial Water Sloshing (`waterAngularVel × pos`)
 */
export function sampleWaterVelocityField(
  state: SimulationState,
  pos: THREE.Vector3,
  outVel: THREE.Vector3,
  outVorticity: THREE.Vector3
): void {
  outVel.set(0, 0, 0);
  outVorticity.set(0, 0, 0);

  const powerMult = state.config.jetPowerMultiplier;
  const turbMult = state.config.turbulenceStrength;
  const t = state.timeElapsed;

  const pumps = [
    {
      nozzle: state.nozzles[0],
      jet: state.leftPump.jetStrength * powerMult,
      suction: state.leftPump.suctionStrength * powerMult,
      circ: state.leftPump.globalCirculation * powerMult,
      sign: -1,
    },
    {
      nozzle: state.nozzles[1],
      jet: state.rightPump.jetStrength * powerMult,
      suction: state.rightPump.suctionStrength * powerMult,
      circ: state.rightPump.globalCirculation * powerMult,
      sign: 1,
    },
  ];

  const heightNorm = THREE.MathUtils.clamp(
    (pos.y + state.tank.halfHeight) / (state.tank.halfHeight * 2),
    0,
    1
  );

  for (let i = 0; i < pumps.length; i++) {
    const { nozzle, jet, suction, circ, sign } = pumps[i];
    if (jet <= 0.001 && suction <= 0.001 && circ <= 0.001) continue;

    const dx = pos.x - nozzle.x;
    const dy = Math.max(0.06, pos.y - nozzle.y);
    const dz = pos.z - nozzle.z;
    const dist3DSq = dx * dx + dy * dy + dz * dz;

    // --- A. EJECTION PHASE (`喷水口只向上喷水` — Analytical 3D Gaussian Turbulent Plume) ---
    if (jet > 0.001) {
      const jetCenterX = nozzle.x + (nozzle.dirX / nozzle.dirY) * dy;
      const jetCenterZ = nozzle.z + (nozzle.dirZ / nozzle.dirY) * dy;
      const rx = pos.x - jetCenterX;
      const rz = pos.z - jetCenterZ;

      const spreadRadius = 0.48 + dy * 0.16;
      const radialDistSq = rx * rx + rz * rz * 1.15;
      const bell = Math.exp(-0.5 * (radialDistSq / (spreadRadius * spreadRadius)));

      const verticalDecay = Math.max(0.34, 1.0 - heightNorm * 0.36);
      const jetSpeed = jet * 12.6 * bell * verticalDecay;

      const hDiff = Math.max(0, heightNorm - 0.48);
      const upperArchInward =
        heightNorm > 0.48
          ? -sign * 2.35 * jet * hDiff * bell
          : -rx * 1.45 * jet * bell;

      const hCube = heightNorm * heightNorm * heightNorm;
      outVel.x += nozzle.dirX * jetSpeed + upperArchInward;
      outVel.y += nozzle.dirY * jetSpeed * (1.0 - 0.42 * hCube);
      outVel.z += nozzle.dirZ * jetSpeed;

      if (heightNorm < 0.48 && Math.abs(rx) > 0.12) {
        const entrainBell = Math.exp(-dist3DSq * 0.28);
        const entrainStrength = jet * 1.25 * (1.0 - heightNorm * 1.8) * entrainBell;
        outVel.x += (-rx / (Math.abs(rx) + 0.18)) * entrainStrength;
        outVel.z += (-rz / (Math.abs(rz) + 0.18)) * entrainStrength * 0.55;
      }

      const invSpreadSq = 1.0 / (spreadRadius * spreadRadius);
      outVorticity.z += -2 * rx * invSpreadSq * bell * jet * 3.8 * turbMult;
      outVorticity.x += 2 * rz * invSpreadSq * bell * jet * 2.8 * turbMult;
    }

    // --- B. SEPARATE RETURN-VENT SUCTION (`独立回水口吸水：从底部正中央与外侧回水阀吸水，绝不从喷水口吸水！`) ---
    if (suction > 0.001) {
      const centerIntakeY = -state.tank.halfHeight + 0.04;
      const cdx = pos.x;
      const cdy = Math.max(0.05, pos.y - centerIntakeY);
      const cdz = pos.z;
      const cDistSq = cdx * cdx + cdy * cdy + cdz * cdz;
      const cDist = Math.sqrt(cDistSq);

      const outerIntakeX = sign * (state.tank.halfWidthBottom - 0.06);
      const outerIntakeY = -state.tank.halfHeight + 0.22;
      const odx = pos.x - outerIntakeX;
      const ody = Math.max(0.05, pos.y - outerIntakeY);
      const odz = pos.z;
      const oDistSq = odx * odx + ody * ody + odz * odz;
      const oDist = Math.sqrt(oDistSq);

      const centerFalloff = 1.0 / (0.32 + cDistSq * 0.45);
      const outerFalloff = 1.0 / (0.38 + oDistSq * 0.55);

      const centerSinkSpeed = suction * 2.8 * centerFalloff;
      const outerSinkSpeed = suction * 2.1 * outerFalloff;

      outVel.x +=
        (-cdx / (cDist + 0.06)) * centerSinkSpeed +
        (-odx / (oDist + 0.06)) * outerSinkSpeed;
      outVel.y +=
        (-cdy / (cDist + 0.06)) * centerSinkSpeed * 1.2 +
        (-ody / (oDist + 0.06)) * outerSinkSpeed * 1.1 -
        suction * 0.55 * (1.0 - heightNorm * 0.35);
      outVel.z +=
        (-cdz / (cDist + 0.06)) * centerSinkSpeed +
        (-odz / (oDist + 0.06)) * outerSinkSpeed;

      outVorticity.z += sign * suction * 1.6 * centerFalloff * turbMult;
      outVorticity.x -= suction * 1.1 * centerFalloff * (cdz / (cDist + 0.1));
    }

    // --- C. WHOLE-TANK 3D GLOBAL RECIRCULATION EDDY ---
    const netCirc = circ + jet * 0.55 + suction * 0.25;
    if (Math.abs(netCirc) > 0.001) {
      const vortexCenterX = sign * 0.52;
      const vortexCenterY = -0.05;
      const vx = pos.x - vortexCenterX;
      const vy = pos.y - vortexCenterY;
      const vortexDistSq = vx * vx * 0.6 + vy * vy * 0.5;
      const cellStrength = (netCirc * 1.85) / (1.0 + vortexDistSq * 0.35);

      outVel.x += -sign * vy * cellStrength * 0.85;
      outVel.y += sign * vx * cellStrength * 0.95;

      const zRollStrength = (netCirc * 0.65) / (1.0 + dist3DSq * 0.22);
      outVel.z += -pos.y * 0.25 * zRollStrength * sign;
      outVel.y += pos.z * 0.32 * zRollStrength * sign;

      outVorticity.z += -sign * cellStrength * 1.1;
      outVorticity.x += sign * zRollStrength * 0.8;
    }
  }

  // --- D. RECTANGULAR TANK BOUNDARY-CONFINED SLOSHING (`方形限制框带动水体`) ---
  // Near the flat rectangular walls ((x/W)^2 -> 1), the rectangular frame carries the water directly with it
  // so relative sloshing vanishes at the walls and only a gentle Stokes core eddy exists near the center.
  const wx = state.waterAngularVel.x;
  const wy = state.waterAngularVel.y;
  const wz = state.waterAngularVel.z;
  if (wx * wx + wy * wy + wz * wz > 1e-6) {
    const sloshVx = wy * pos.z - wz * pos.y;
    const sloshVy = wz * pos.x - wx * pos.z;
    const sloshVz = wx * pos.y - wy * pos.x;

    const xRatioSq = (pos.x * pos.x) / (state.tank.halfWidthTop * state.tank.halfWidthTop);
    const yRatioSq = (pos.y * pos.y) / (state.tank.halfHeight * state.tank.halfHeight);
    const wallConfinement = Math.max(0.15, (1.0 - 0.65 * xRatioSq) * (1.0 - 0.35 * yRatioSq));
    const gain = 0.62 * wallConfinement;

    outVel.x += sloshVx * gain;
    outVel.y += sloshVy * gain;
    outVel.z += sloshVz * gain;

    outVorticity.x += 1.2 * wx * wallConfinement;
    outVorticity.y += 1.2 * wy * wallConfinement;
    outVorticity.z += 1.2 * wz * wallConfinement;
  }
}

function injectSurfaceWaveImpulse(
  state: SimulationState,
  worldX: number,
  worldZ: number,
  impulseStrength: number,
  radiusGrid = 2
): void {
  const xNorm = THREE.MathUtils.clamp(
    (worldX + state.tank.halfWidthTop) / (state.tank.halfWidthTop * 2),
    0,
    0.999
  );
  const zNorm = THREE.MathUtils.clamp(
    (worldZ + state.tank.halfDepth) / (state.tank.halfDepth * 2),
    0,
    0.999
  );
  const cx = Math.floor(xNorm * WAVE_NX);
  const cz = Math.floor(zNorm * WAVE_NZ);

  for (let dz = -radiusGrid; dz <= radiusGrid; dz++) {
    const iz = cz + dz;
    if (iz < 0 || iz >= WAVE_NZ) continue;
    for (let dx = -radiusGrid; dx <= radiusGrid; dx++) {
      const ix = cx + dx;
      if (ix < 0 || ix >= WAVE_NX) continue;
      const d2 = dx * dx + dz * dz;
      const weight = Math.exp(-d2 / Math.max(1, radiusGrid * radiusGrid * 0.65));
      state.surfaceWaveVel[iz * WAVE_NX + ix] += impulseStrength * weight;
    }
  }
}

function updateSurfaceWavePDE(state: SimulationState, dt: number): void {
  const h = state.surfaceWaveHeight;
  const v = state.surfaceWaveVel;
  const topY = state.tank.halfHeight - 0.14;

  // 1. Sample upward/downward fluid velocity right below the free surface to drive upwelling boils & suction troughs
  for (let iz = 1; iz < WAVE_NZ - 1; iz += 2) {
    const zWorld = ((iz / (WAVE_NZ - 1)) * 2 - 1) * state.tank.halfDepth;
    for (let ix = 1; ix < WAVE_NX - 1; ix += 2) {
      const xWorld = ((ix / (WAVE_NX - 1)) * 2 - 1) * state.tank.halfWidthTop;
      _segA0.set(xWorld, topY - 0.25, zWorld);
      sampleWaterVelocityField(state, _segA0, _fluidVel, _fluidVorticity);
      // Upwelling vertical velocity + gyroscope tilt hydrostatic slope forcing
      const tiltForcing = -state.gravityDir.x * xWorld * 0.45 - state.gravityDir.z * zWorld * 0.35;
      const forcing = _fluidVel.y * 0.32 + tiltForcing;
      v[iz * WAVE_NX + ix] += forcing * dt;
    }
  }

  // 2. 2D Shallow-Water Wave PDE step (c^2 Laplacian(h) - damping * v - restoring * h)
  const waveSpeedSq = 18.0;
  const restoring = 14.5;
  const damping = 3.2;

  for (let iz = 1; iz < WAVE_NZ - 1; iz++) {
    const row = iz * WAVE_NX;
    for (let ix = 1; ix < WAVE_NX - 1; ix++) {
      const idx = row + ix;
      const centerH = h[idx];
      const laplacian =
        h[idx - 1] +
        h[idx + 1] +
        h[idx - WAVE_NX] +
        h[idx + WAVE_NX] -
        4.0 * centerH;
      const accel = waveSpeedSq * laplacian - restoring * centerH - damping * v[idx];
      v[idx] = THREE.MathUtils.clamp(v[idx] + accel * dt, -2.5, 2.5);
    }
  }

  // 3. Integrate wave height & apply reflective Neumann boundary condition on walls
  for (let iz = 0; iz < WAVE_NZ; iz++) {
    const row = iz * WAVE_NX;
    for (let ix = 0; ix < WAVE_NX; ix++) {
      const idx = row + ix;
      h[idx] = THREE.MathUtils.clamp(h[idx] + v[idx] * dt, -0.22, 0.22);
    }
  }
  for (let ix = 0; ix < WAVE_NX; ix++) {
    h[ix] = h[WAVE_NX + ix] * 0.9;
    h[(WAVE_NZ - 1) * WAVE_NX + ix] = h[(WAVE_NZ - 2) * WAVE_NX + ix] * 0.9;
  }
  for (let iz = 0; iz < WAVE_NZ; iz++) {
    h[iz * WAVE_NX] = h[iz * WAVE_NX + 1] * 0.9;
    h[iz * WAVE_NX + (WAVE_NX - 1)] = h[iz * WAVE_NX + (WAVE_NX - 2)] * 0.9;
  }
}

function updateFlowParticles(state: SimulationState, dt: number): void {
  const bh = state.tank.halfHeight;
  const bd = state.tank.halfDepth;

  for (let i = 0; i < state.flowParticles.length; i++) {
    const fp = state.flowParticles[i];
    sampleWaterVelocityField(state, fp.position, _fluidVel, _fluidVorticity);

    // Couple to nearby moving rings (hydrodynamic wake entrainment)
    for (let rIdx = 0; rIdx < state.rings.length; rIdx++) {
      const ring = state.rings[rIdx];
      const dx = fp.position.x - ring.position.x;
      const dy = fp.position.y - ring.position.y;
      const dz = fp.position.z - ring.position.z;
      const dSq = dx * dx + dy * dy + dz * dz;
      if (dSq < 0.22) {
        const wakeWeight = (1.0 - dSq / 0.22) * 0.55;
        _fluidVel.addScaledVector(ring.velocity, wakeWeight);
      }
    }

    // Sub-grid curl-like micro-turbulence + fast Stokes relaxation to fluid velocity
    const swirlX = Math.sin(state.timeElapsed * 4.2 + fp.phase + fp.position.y * 2.5) * 0.14;
    const swirlY = Math.cos(state.timeElapsed * 3.8 + fp.phase + fp.position.x * 2.5) * 0.14;
    const swirlZ = Math.sin(state.timeElapsed * 4.6 + fp.phase * 1.3) * 0.1;

    const relax = Math.min(1.0, dt * 11.0);
    fp.velocity.x += (_fluidVel.x + swirlX - fp.velocity.x) * relax;
    fp.velocity.y += (_fluidVel.y + swirlY - fp.velocity.y) * relax;
    fp.velocity.z += (_fluidVel.z + swirlZ - fp.velocity.z) * relax;

    fp.position.addScaledVector(fp.velocity, dt);

    // Volume-conserving recirculation: if a particle hits the top free-surface boundary,
    // deflect it outward/downward along the return current or re-inject at the active nozzle orifice
    if (fp.position.y > bh - 0.18) {
      if ((i + Math.floor(state.timeElapsed * 60)) % 5 === 0) {
        const useLeft = state.leftPump.jetStrength >= state.rightPump.jetStrength;
        const noz = useLeft ? state.nozzles[0] : state.nozzles[1];
        const activeStr = useLeft ? state.leftPump.jetStrength : state.rightPump.jetStrength;
        if (activeStr > 0.05) {
          const angle = state.timeElapsed * 37.0 + i * 2.39996;
          const r = ((i % 7) / 7) * 0.24;
          fp.position.set(
            noz.x + Math.cos(angle) * r,
            noz.y + 0.08,
            noz.z + Math.sin(angle) * r
          );
          fp.velocity.set(noz.dirX * activeStr * 6.0, activeStr * 7.0, noz.dirZ * activeStr * 6.0);
          continue;
        }
      }
      fp.position.y = bh - 0.18;
      fp.velocity.y = -Math.abs(fp.velocity.y) * 0.65 - 0.35;
      fp.velocity.x += (fp.position.x >= 0 ? -0.6 : 0.6);
    } else if (fp.position.y < -bh + 0.06) {
      fp.position.y = -bh + 0.06;
      fp.velocity.y = Math.abs(fp.velocity.y) * 0.5;
    }

    const yNorm = THREE.MathUtils.clamp((fp.position.y + bh) / (bh * 2), 0, 1);
    const maxW =
      THREE.MathUtils.lerp(state.tank.halfWidthBottom, state.tank.halfWidthTop, yNorm) - 0.06;
    if (fp.position.x < -maxW) {
      fp.position.x = -maxW;
      fp.velocity.x = Math.abs(fp.velocity.x) * 0.5;
    } else if (fp.position.x > maxW) {
      fp.position.x = maxW;
      fp.velocity.x = -Math.abs(fp.velocity.x) * 0.5;
    }

    if (fp.position.z < -bd + 0.05) {
      fp.position.z = -bd + 0.05;
      fp.velocity.z = Math.abs(fp.velocity.z) * 0.5;
    } else if (fp.position.z > bd - 0.05) {
      fp.position.z = bd - 0.05;
      fp.velocity.z = -Math.abs(fp.velocity.z) * 0.5;
    }
  }
}

let _bubbleSpawnCursor = 0;

function updateBubbles(state: SimulationState, dt: number): void {
  const pumps = [
    {
      nozzle: state.nozzles[0],
      strength: state.leftPump.jetStrength,
      vel: state.leftPump.velocity,
    },
    {
      nozzle: state.nozzles[1],
      strength: state.rightPump.jetStrength,
      vel: state.rightPump.velocity,
    },
  ];

  // Emit only a few delicate cavitation bubbles during active pump compression stroke
  for (let p = 0; p < pumps.length; p++) {
    const { nozzle, strength, vel } = pumps[p];
    if (strength > 0.14 && vel > 0.05) {
      const countToSpawn = Math.min(2, Math.ceil(strength * dt * 32));
      for (let k = 0; k < countToSpawn; k++) {
        let freeBubble: BubbleParticle | null = null;
        for (let tries = 0; tries < state.bubbles.length; tries++) {
          const candidate = state.bubbles[(_bubbleSpawnCursor + tries) % state.bubbles.length];
          if (!candidate.active) {
            freeBubble = candidate;
            _bubbleSpawnCursor = (_bubbleSpawnCursor + tries + 1) % state.bubbles.length;
            break;
          }
        }
        if (!freeBubble) break;

        const seed = state.timeElapsed * 173.3 + _bubbleSpawnCursor * 19.7 + k * 7.1 + p * 53.0;
        const radial = (Math.sin(seed) * 0.5 + 0.5) * 0.12;
        const angle = seed * 2.39996;
        const rx = Math.cos(angle) * radial;
        const rz = Math.sin(angle) * radial * 0.6;

        freeBubble.active = true;
        freeBubble.life = 0;
        freeBubble.maxLife = 0.65 + ((_bubbleSpawnCursor % 5) * 0.12);
        freeBubble.radius = 0.011 + ((_bubbleSpawnCursor % 5) * 0.0038);
        freeBubble.position.set(
          nozzle.x + rx,
          nozzle.y + 0.05 + (k % 2) * 0.03,
          nozzle.z + rz
        );
        const speedSpread = 0.82 + (k % 3) * 0.1;
        freeBubble.velocity.set(
          nozzle.dirX * (3.2 + strength * 4.5) * speedSpread + rx * 2.2,
          (3.2 + strength * 5.2) * speedSpread,
          nozzle.dirZ * (3.2 + strength * 4.5) * speedSpread + rz * 2.2
        );
      }
    }
  }

  const baseSurfaceY = state.tank.halfHeight - 0.14;
  for (let i = 0; i < state.bubbles.length; i++) {
    const b = state.bubbles[i];
    if (!b.active) continue;
    b.life += dt;

    // Sample local dynamic wave height at bubble's (x, z)
    const xNormSurf = THREE.MathUtils.clamp(
      (b.position.x + state.tank.halfWidthTop) / (state.tank.halfWidthTop * 2),
      0,
      0.999
    );
    const zNormSurf = THREE.MathUtils.clamp(
      (b.position.z + state.tank.halfDepth) / (state.tank.halfDepth * 2),
      0,
      0.999
    );
    const surfIdx =
      Math.floor(zNormSurf * WAVE_NZ) * WAVE_NX + Math.floor(xNormSurf * WAVE_NX);
    const dynamicSurfaceY = baseSurfaceY + (state.surfaceWaveHeight[surfIdx] || 0);

    if (b.position.y >= dynamicSurfaceY) {
      // Bubble hits the dynamic free surface -> excites a circular ripple in the 2D Shallow-Water Wave Grid!
      injectSurfaceWaveImpulse(state, b.position.x, b.position.z, b.radius * 9.5, 2);
      b.active = false;
      continue;
    }
    if (b.life >= b.maxLife || b.position.y < -state.tank.halfHeight + 0.03) {
      b.active = false;
      continue;
    }

    sampleWaterVelocityField(state, b.position, _fluidVel, _fluidVorticity);

    // Maxey-Riley Multiphase Bubble Equation:
    // 1. Radius-dependent Schiller-Naumann / Stokes relaxation rate
    const invTau = 1.0 / (0.065 + b.radius * 1.8);
    const dragRelax = Math.min(1, dt * invTau);

    // Relative slip velocity (u_fluid - v_bubble)
    const slipX = _fluidVel.x - b.velocity.x;
    const slipY = _fluidVel.y - b.velocity.y;
    const slipZ = _fluidVel.z - b.velocity.z;

    // 2. Auton-Saffman shear lift force: F_lift = C_L * (u_fluid - v_bubble) x omega_fluid
    const cLift = 0.42;
    const liftX = cLift * (slipY * _fluidVorticity.z - slipZ * _fluidVorticity.y);
    const liftY = cLift * (slipZ * _fluidVorticity.x - slipX * _fluidVorticity.z);
    const liftZ = cLift * (slipX * _fluidVorticity.y - slipY * _fluidVorticity.x);

    b.velocity.x += slipX * dragRelax + liftX * dt;
    b.velocity.y += slipY * dragRelax + liftY * dt;
    b.velocity.z += slipZ * dragRelax + liftZ * dt;

    // 3. Archimedes net buoyancy + path-instability helical oscillation (wake vortex shedding at Re > 200)
    const buoyancyAccel = 4.6 + b.radius * 95.0;
    const helicalX = Math.sin(state.timeElapsed * 15.0 + b.wobblePhase) * 1.25;
    const helicalZ = Math.cos(state.timeElapsed * 13.5 + b.wobblePhase) * 1.15;

    b.velocity.x += (-state.gravityDir.x * buoyancyAccel + helicalX) * dt;
    b.velocity.y += (-state.gravityDir.y * buoyancyAccel) * dt;
    b.velocity.z += (-state.gravityDir.z * buoyancyAccel + helicalZ) * dt;

    b.position.addScaledVector(b.velocity, dt);

    // 4. Two-Way Bubble-to-Ring Collision & Micro-Buoyancy Transfer
    for (let rIdx = 0; rIdx < state.rings.length; rIdx++) {
      const ring = state.rings[rIdx];
      const maxR = ring.majorRadius + ring.minorRadius + b.radius;
      const dxR = b.position.x - ring.position.x;
      const dyR = b.position.y - ring.position.y;
      const dzR = b.position.z - ring.position.z;
      if (dxR * dxR + dyR * dyR + dzR * dzR > maxR * maxR) continue;

      _ringAxisU.set(1, 0, 0).applyQuaternion(ring.quaternion);
      _ringNormal.set(0, 0, 1).applyQuaternion(ring.quaternion).normalize();
      closestPointOnCircle3D(
        ring.position,
        _ringNormal,
        _ringAxisU,
        ring.majorRadius,
        b.position,
        _contactPt
      );

      const nx = b.position.x - _contactPt.x;
      const ny = b.position.y - _contactPt.y;
      const nz = b.position.z - _contactPt.z;
      const dSq = nx * nx + ny * ny + nz * nz;
      const minD = ring.minorRadius + b.radius;
      if (dSq < minD * minD && dSq > 1e-8) {
        const d = Math.sqrt(dSq);
        const unx = nx / d;
        const uny = ny / d;
        const unz = nz / d;
        const pen = minD - d;

        // Push bubble outside the ring tube surface
        b.position.x += unx * pen;
        b.position.y += uny * pen;
        b.position.z += unz * pen;

        // Transfer micro-buoyancy & impact impulse to the plastic ring!
        const vnRel =
          (b.velocity.x - ring.velocity.x) * unx +
          (b.velocity.y - ring.velocity.y) * uny +
          (b.velocity.z - ring.velocity.z) * unz;
        if (vnRel < 0) {
          b.velocity.x -= 1.35 * vnRel * unx;
          b.velocity.y -= 1.35 * vnRel * uny;
          b.velocity.z -= 1.35 * vnRel * unz;

          const bubbleImpulse = -vnRel * (b.radius * 0.36);
          ring.velocity.x -= unx * bubbleImpulse;
          ring.velocity.y -= uny * bubbleImpulse;
          ring.velocity.z -= unz * bubbleImpulse;

          const armX = _contactPt.x - ring.position.x;
          const armY = _contactPt.y - ring.position.y;
          const armZ = _contactPt.z - ring.position.z;
          ring.angularVelocity.x -= (armY * unz - armZ * uny) * bubbleImpulse * 1.6;
          ring.angularVelocity.y -= (armZ * unx - armX * unz) * bubbleImpulse * 1.6;
          ring.angularVelocity.z -= (armX * uny - armY * unx) * bubbleImpulse * 1.6;
        }
      }
    }

    const yNorm = THREE.MathUtils.clamp(
      (b.position.y + state.tank.halfHeight) / (state.tank.halfHeight * 2),
      0,
      1
    );
    const maxW =
      THREE.MathUtils.lerp(state.tank.halfWidthBottom, state.tank.halfWidthTop, yNorm) - 0.05;
    b.position.x = THREE.MathUtils.clamp(b.position.x, -maxW, maxW);
    b.position.z = THREE.MathUtils.clamp(
      b.position.z,
      -state.tank.halfDepth + 0.05,
      state.tank.halfDepth - 0.05
    );
  }
}

/**
 * True 3D Geometric Plane-Piercing Hooked-Ring Evaluator (`evaluateHookedRings`):
 * Checks not only the ring center's Y bounds, but computes the exact 3D intersection
 * `P_pierce` where the vertical peg shaft line `(peg.baseX, y, peg.baseZ)` pierces the
 * tilted ring plane `(P - C) · n_ring = 0`:
 * 1. Rejects edge-on / vertical rings (`|n_ring.y| < 0.28`) which cannot physically encircle a vertical peg.
 * 2. Verifies `P_pierce.y` lies along the peg shaft between the catcher cup and peg tip.
 * 3. Verifies the 3D in-plane distance `||P_pierce - ring.position||` is strictly less than
 *    `holeRadius - pegRadiusAtY` (so the peg shaft genuinely passes through the torus aperture!).
 */
/**
 * Incremental Potential Contact (IPC, Li et al. 2020) C^2 Smooth Logarithmic Barrier Gradient:
 * b(d) = -(d - dHat)^2 * ln(d / dHat) for 0 < d < dHat, and 0 for d >= dHat.
 * Since db/dd = -2 * (d - dHat) * ln(d / dHat) - (d - dHat)^2 / d < 0,
 * returns the positive repulsive barrier force magnitude -db/dd >= 0:
 *   -db/dd = 2 * (d - dHat) * ln(d / dHat) + (d - dHat)^2 / d
 * Regularized at the d -> 0+ singularity via C^infinity smoothing d_reg = sqrt(max(0, d)^2 + epsD^2).
 */
export function computeIPCBarrierGradient(d: number, dHat: number, stiffness = 1.0): number {
  if (d >= dHat) return 0.0;
  const posD = Math.max(0.0, d);
  const clampedD = Math.sqrt(posD * posD + 0.002 * 0.002);
  if (clampedD >= dHat) return 0.0;
  const diff = clampedD - dHat; // negative
  const logRatio = Math.log(clampedD / dHat); // negative
  const grad = 2.0 * diff * logRatio + (diff * diff) / clampedD; // strictly positive
  return Math.min(stiffness * grad, 32.0);
}

/**
 * Incremental Potential Contact (IPC, Li et al. 2020) C^2 Logarithmic Barrier Hessian (Second Derivative):
 *   d^2 b / dd^2 = -2 * ln(d / dHat) - 4 * (d - dHat) / d + (d - dHat)^2 / d^2 > 0
 * Used for second-order Newton / Levenberg-Marquardt curvature damping at contact singularities.
 */
export function computeIPCBarrierHessian(d: number, dHat: number, stiffness = 1.0): number {
  if (d >= dHat) return 0.0;
  const posD = Math.max(0.0, d);
  const clampedD = Math.sqrt(posD * posD + 0.002 * 0.002);
  if (clampedD >= dHat) return 0.0;
  const diff = clampedD - dHat; // negative
  const logRatio = Math.log(clampedD / dHat); // negative
  const hessian =
    -2.0 * logRatio - (4.0 * diff) / clampedD + (diff * diff) / (clampedD * clampedD);
  return Math.max(0.0, stiffness * hessian);
}

/**
 * Exact 3D Gauss Linking Integral Lk(ring, peg) between the closed 16-segment torus centerline circle
 * and the multi-segment vertical needle shaft:
 *   Lk = (1 / 4pi) \oint_{C_1} \oint_{C_2} ((r_1 - r_2) . (dr_1 x dr_2)) / ||r_1 - r_2||^3
 * Measures the true topological winding number of the ring around the needle shaft.
 */
export function computeGaussLinkingNumberRingPeg(ring: RingState, peg: PegSpec): number {
  const R = ring.majorRadius;
  const pegTopY = peg.baseY + peg.height;
  _ringAxisU.set(1, 0, 0).applyQuaternion(ring.quaternion);
  _ringAxisV.set(0, 1, 0).applyQuaternion(ring.quaternion);

  const byStart = peg.baseY - 0.18;
  const byEnd = pegTopY + 0.14;
  const pegSegments = 4;
  const segLenY = (byEnd - byStart) / pegSegments;

  let integralSum = 0.0;

  for (let s = 0; s < NUM_TORUS_SAMPLES; s++) {
    const sNext = (s + 1) % NUM_TORUS_SAMPLES;
    const ax0 = ring.position.x + (_ringAxisU.x * TORUS_COS[s] + _ringAxisV.x * TORUS_SIN[s]) * R;
    const ay0 = ring.position.y + (_ringAxisU.y * TORUS_COS[s] + _ringAxisV.y * TORUS_SIN[s]) * R;
    const az0 = ring.position.z + (_ringAxisU.z * TORUS_COS[s] + _ringAxisV.z * TORUS_SIN[s]) * R;

    const ax1 =
      ring.position.x + (_ringAxisU.x * TORUS_COS[sNext] + _ringAxisV.x * TORUS_SIN[sNext]) * R;
    const ay1 =
      ring.position.y + (_ringAxisU.y * TORUS_COS[sNext] + _ringAxisV.y * TORUS_SIN[sNext]) * R;
    const az1 =
      ring.position.z + (_ringAxisU.z * TORUS_COS[sNext] + _ringAxisV.z * TORUS_SIN[sNext]) * R;

    const dAx = ax1 - ax0;
    const dAy = ay1 - ay0;
    const dAz = az1 - az0;
    const midAx = 0.5 * (ax0 + ax1);
    const midAy = 0.5 * (ay0 + ay1);
    const midAz = 0.5 * (az0 + az1);

    // Since dBx = 0 and dBz = 0 for the vertical peg shaft, (dA x dB) = (dAz * dBy, 0, -dAx * dBy)
    const cx = dAz * segLenY;
    const cz = -dAx * segLenY;

    for (let pSeg = 0; pSeg < pegSegments; pSeg++) {
      const midBy = byStart + (pSeg + 0.5) * segLenY;
      const rx = midAx - peg.baseX;
      const ry = midAy - midBy;
      const rz = midAz - peg.baseZ;
      const distSq = Math.max(0.0004, rx * rx + ry * ry + rz * rz);
      const dist = Math.sqrt(distSq);
      integralSum += (cx * rx + cz * rz) / (distSq * dist);
    }
  }

  return Math.abs(integralSum) / (4.0 * Math.PI);
}

export function evaluateHookedRings(state: SimulationState): number {
  let hookedCount = 0;
  for (let i = 0; i < state.rings.length; i++) {
    const ring = state.rings[i];
    const holeRadius = ring.majorRadius - ring.minorRadius;

    _ringNormal.set(0, 0, 1).applyQuaternion(ring.quaternion).normalize();
    const absNy = Math.abs(_ringNormal.y);

    // 1. If already topologically hooked on a peg, verify it is still on the peg shaft strictly above the holder tray (`cupY`)
    if (ring.hookedPegId !== null) {
      const currentPeg = state.pegs.find((p) => p.id === ring.hookedPegId);
      if (currentPeg) {
        const pegTopY = currentPeg.baseY + currentPeg.height;
        const cupY = currentPeg.baseY + currentPeg.holderCupY;
        const horizDist = Math.hypot(
          ring.position.x - currentPeg.baseX,
          ring.position.z - currentPeg.baseZ
        );
        const linkingNumber = computeGaussLinkingNumberRingPeg(ring, currentPeg);
        if (
          ring.position.y <= pegTopY + ring.minorRadius * 1.45 &&
          ring.position.y >= cupY - ring.minorRadius * 0.35 &&
          horizDist <= holeRadius + currentPeg.baseRadius + 0.04 &&
          absNy >= 0.28 &&
          linkingNumber >= 0.18
        ) {
          hookedCount++;
          continue;
        }
      }
      ring.hookedPegId = null;
    }

    // 2. Otherwise check if the ring plane is pierced by a peg shaft strictly above the holder tray (`cupY`)
    // AND topologically linked via Gauss Linking Integral
    if (absNy < 0.28) continue;

    for (let p = 0; p < state.pegs.length; p++) {
      const peg = state.pegs[p];
      const pegTopY = peg.baseY + peg.height;
      const cupY = peg.baseY + peg.holderCupY;

      // Never mark a ring below the holder tray (`cupY`) as hooked!
      if (ring.position.y < cupY) continue;

      const dx = peg.baseX - ring.position.x;
      const dz = peg.baseZ - ring.position.z;

      const pierceY =
        ring.position.y - (dx * _ringNormal.x + dz * _ringNormal.z) / _ringNormal.y;

      if (pierceY >= cupY && pierceY <= pegTopY - 0.01) {
        const dy = pierceY - ring.position.y;
        const distInRingPlane = Math.hypot(dx, dy, dz);
        const hFrac = THREE.MathUtils.clamp((pierceY - peg.baseY) / peg.height, 0, 1);
        const pegRadiusAtY = THREE.MathUtils.lerp(peg.baseRadius, peg.tipRadius, hFrac);

        if (
          distInRingPlane < holeRadius - pegRadiusAtY * 0.35 &&
          computeGaussLinkingNumberRingPeg(ring, peg) >= 0.18
        ) {
          ring.hookedPegId = peg.id;
          hookedCount++;
          break;
        }
      }
    }
  }
  state.score = hookedCount;
  return hookedCount;
}

function syncRingsAndUniformsToWasmMemory(state: SimulationState): void {
  const buf = state.wasmKernel.ringFloatView;
  for (let i = 0; i < state.rings.length; i++) {
    const r = state.rings[i];
    const base = i * 16;
    buf[base + 0] = r.position.x;
    buf[base + 1] = r.position.y;
    buf[base + 2] = r.position.z;
    buf[base + 3] = r.velocity.x;
    buf[base + 4] = r.velocity.y;
    buf[base + 5] = r.velocity.z;
    buf[base + 6] = r.quaternion.x;
    buf[base + 7] = r.quaternion.y;
    buf[base + 8] = r.quaternion.z;
    buf[base + 9] = r.quaternion.w;
    buf[base + 10] = r.angularVelocity.x;
    buf[base + 11] = r.angularVelocity.y;
    buf[base + 12] = r.angularVelocity.z;
    buf[base + 13] = r.majorRadius;
    buf[base + 14] = r.minorRadius;
    buf[base + 15] = r.hookedPegId ? 1.0 : 0.0;
  }

  const rhoRing = state.config.ringDensity;
  const rhoWater = state.config.waterDensity;
  const netBuoyancyFactor = (rhoRing - rhoWater) / (rhoRing + 0.55 * rhoWater);
  const netGravScale = state.gravityMagnitude * netBuoyancyFactor * 2.15;
  const powerMult = state.config.jetPowerMultiplier;
  const leftNoz = state.nozzles[0];
  const rightNoz = state.nozzles[1];

  const uBuf = state.wasmKernel.uniformFloatView;
  uBuf[4] = state.gravityDir.x * netGravScale;
  uBuf[5] = state.gravityDir.y * netGravScale;
  uBuf[6] = state.gravityDir.z * netGravScale;
  uBuf[7] = state.waterAngularVel.x;
  uBuf[8] = state.waterAngularVel.y;
  uBuf[9] = state.waterAngularVel.z;
  uBuf[10] = state.gravityDir.x;
  uBuf[11] = state.gravityDir.y;
  uBuf[12] = state.gravityDir.z;
  uBuf[13] = state.config.normalDragCoeff;
  uBuf[14] = state.config.tangentDragCoeff;
  uBuf[15] = state.config.angularDragCoeff;
  uBuf[16] = leftNoz.x;
  uBuf[17] = leftNoz.y;
  uBuf[18] = leftNoz.z;
  uBuf[19] = state.leftPump.jetStrength * powerMult;
  uBuf[20] =
    (state.leftPump.globalCirculation +
      state.leftPump.jetStrength * 0.55 +
      state.leftPump.suctionStrength * 0.25) *
    powerMult;
  uBuf[21] = rightNoz.x;
  uBuf[22] = rightNoz.y;
  uBuf[23] = rightNoz.z;
  uBuf[24] = state.rightPump.jetStrength * powerMult;
  uBuf[25] =
    (state.rightPump.globalCirculation +
      state.rightPump.jetStrength * 0.55 +
      state.rightPump.suctionStrength * 0.25) *
    powerMult;
  uBuf[26] = (state.leftPump.suctionStrength + state.rightPump.suctionStrength) * powerMult;
  uBuf[27] = state.tank.halfHeight;
  uBuf[28] = Math.sin(state.timeElapsed * 5.2) * 0.12 * state.config.turbulenceStrength;
  uBuf[29] = Math.cos(state.timeElapsed * 4.7) * 0.08 * state.config.turbulenceStrength;
  uBuf[30] = 1.0; // Enable full WASM 3D hydrodynamic + Morison + quaternion kernel
}

export function stepWaterSimulation(state: SimulationState, rawDt: number): void {
  const dt = Math.min(rawDt, 0.04);
  if (dt <= 0) return;

  state.timeElapsed += dt;

  // Track finite-difference tilt rate (with shortest-angle wrap across ±pi) and damp rectangular-cavity water sloshing (`waterAngularVel`)
  state.prevTiltRoll = state.tiltRoll;
  state.prevTiltPitch = state.tiltPitch;

  // Rectangular frame walls rapidly arrest residual relative water rotation (`方形限制框阻尼`)
  state.waterAngularVel.x = THREE.MathUtils.damp(state.waterAngularVel.x, 0, 6.8, dt);
  state.waterAngularVel.y = THREE.MathUtils.damp(state.waterAngularVel.y, 0, 6.8, dt);
  state.waterAngularVel.z = THREE.MathUtils.damp(state.waterAngularVel.z, 0, 6.8, dt);
  state.deviceAngularVel.multiplyScalar(1.0 / (1.0 + 6.0 * dt));

  const pumps = [state.leftPump, state.rightPump];
  for (let i = 0; i < pumps.length; i++) {
    const pump = pumps[i];
    const target = pump.targetHeld ? pump.targetDepth : 0.0;
    const prevDepressed = pump.depressed;
    const rate = pump.targetHeld ? 15.0 : 11.5;
    pump.depressed = THREE.MathUtils.damp(pump.depressed, target, rate, dt);
    pump.velocity = (pump.depressed - prevDepressed) / Math.max(dt, 0.0001);

    // Finite-stroke bellows rule: ONLY eject water while the bellows is actively compressing inward!
    // Once the button bottoms out (velocity -> 0), water spraying stops completely even if held down.
    if (pump.velocity > 0.05 && pump.depressed < pump.targetDepth * 0.98) {
      pump.jetStrength = Math.min(1.75, pump.jetStrength + pump.velocity * dt * 0.68);
      pump.globalCirculation = Math.min(1.9, pump.globalCirculation + pump.velocity * dt * 0.45);
    } else if (pump.velocity < -0.05) {
      pump.suctionStrength = Math.min(
        1.25,
        pump.suctionStrength + Math.abs(pump.velocity) * dt * 0.78
      );
    }

    // Always decay jetStrength toward 0.0 so holding a button down NEVER sprays continuously!
    pump.jetStrength = THREE.MathUtils.damp(pump.jetStrength, 0.0, 4.6, dt);
    pump.suctionStrength = THREE.MathUtils.damp(pump.suctionStrength, 0.0, 4.4, dt);
    pump.globalCirculation = THREE.MathUtils.damp(pump.globalCirculation, 0.0, 2.2, dt);
  }

  updateBubbles(state, dt);
  updateFlowParticles(state, dt);
  updateSurfaceWavePDE(state, dt);

  const subSteps = Math.max(4, Math.min(64, Math.round(state.config.wasmSubsteps || 24)));
  const subDt = dt / subSteps;
  state.effectiveHz = Math.round(1 / Math.max(subDt, 1e-5));

  const wasmBuf = state.wasmKernel.ringFloatView;

  if (state.totalWasmStepsExecuted === 0) {
    evaluateHookedRings(state);
  }

  // Sync ring states and all 31 hydrodynamic uniforms to WASM linear memory ONCE before the substep loop
  syncRingsAndUniformsToWasmMemory(state);

  for (let step = 0; step < subSteps; step++) {
    // Execute 3D fluid velocity sampling, Morison drag, hydrodynamic torques,
    // quaternion integration, and position integration inside WebAssembly!
    state.lastKineticEnergy = state.wasmKernel.wasmIntegrateStep(
      state.rings.length,
      subDt,
      1.0
    );
    state.totalWasmStepsExecuted += 1;

    for (let i = 0; i < state.rings.length; i++) {
      const ring = state.rings[i];
      const base = i * 16;
      const prevX = ring.position.x;
      const prevY = ring.position.y;
      const prevZ = ring.position.z;

      ring.position.set(wasmBuf[base + 0], wasmBuf[base + 1], wasmBuf[base + 2]);
      ring.velocity.set(wasmBuf[base + 3], wasmBuf[base + 4], wasmBuf[base + 5]);
      ring.quaternion.set(
        wasmBuf[base + 6],
        wasmBuf[base + 7],
        wasmBuf[base + 8],
        wasmBuf[base + 9]
      );
      ring.angularVelocity.set(wasmBuf[base + 10], wasmBuf[base + 11], wasmBuf[base + 12]);

      resolveRingTankAndPegCollisionsExact(state, ring, subDt);

      // Strict per-substep anti-teleportation displacement guard ("不要因为受了力瞬移"):
      const stepDx = ring.position.x - prevX;
      const stepDy = ring.position.y - prevY;
      const stepDz = ring.position.z - prevZ;
      const stepDistSq = stepDx * stepDx + stepDy * stepDy + stepDz * stepDz;
      const maxSubstepMove = 6.0 * subDt + 0.008;
      if (stepDistSq > maxSubstepMove * maxSubstepMove) {
        const clampScale = maxSubstepMove / Math.sqrt(stepDistSq);
        ring.position.set(
          prevX + stepDx * clampScale,
          prevY + stepDy * clampScale,
          prevZ + stepDz * clampScale
        );
      }

      // Write collision-resolved state back to WASM linear memory for the next substep
      wasmBuf[base + 0] = ring.position.x;
      wasmBuf[base + 1] = ring.position.y;
      wasmBuf[base + 2] = ring.position.z;
      wasmBuf[base + 3] = ring.velocity.x;
      wasmBuf[base + 4] = ring.velocity.y;
      wasmBuf[base + 5] = ring.velocity.z;
      wasmBuf[base + 10] = ring.angularVelocity.x;
      wasmBuf[base + 11] = ring.angularVelocity.y;
      wasmBuf[base + 12] = ring.angularVelocity.z;
    }

    // Decimate Ring-to-Ring manifold solver to every 2nd substep when running ultra-high substeps (>= 16)
    if (step % 2 === 0 || subSteps < 8) {
      resolveRingToRingCollisionsExact3D(state.rings);
      for (let i = 0; i < state.rings.length; i++) {
        const r = state.rings[i];
        const base = i * 16;
        wasmBuf[base + 0] = r.position.x;
        wasmBuf[base + 1] = r.position.y;
        wasmBuf[base + 2] = r.position.z;
        wasmBuf[base + 3] = r.velocity.x;
        wasmBuf[base + 4] = r.velocity.y;
        wasmBuf[base + 5] = r.velocity.z;
        wasmBuf[base + 10] = r.angularVelocity.x;
        wasmBuf[base + 11] = r.angularVelocity.y;
        wasmBuf[base + 12] = r.angularVelocity.z;
      }
    }
  }

  evaluateHookedRings(state);
}

/**
 * Singularity-Regularized IPC Barrier Force + C^1 Mollified Tangential Friction Deceleration ("算出奇点时考虑一下可以有摩擦力（减速）"):
 * Applies both the smooth C^2 normal repulsion and physical Coulomb tangential friction + rotational torque,
 * strictly scaled by `subDt` so barrier singularities never cause velocity spikes or position teleportation.
 */
function applyIPCBarrierWithFriction(
  ring: RingState,
  armX: number,
  armY: number,
  armZ: number,
  nx: number,
  ny: number,
  nz: number,
  gap: number,
  dHat: number,
  stiffness: number,
  frictionCoeff: number,
  invMass: number,
  invInertia: number,
  subDt: number
): void {
  const fN = computeIPCBarrierGradient(gap, dHat, stiffness);
  if (fN <= 0) return;

  // Second-order IPC Hessian curvature damping (Newton/Levenberg-Marquardt step regularization)
  // prevents stiff logarithmic barrier forces from overshooting at contact singularities:
  const hN = computeIPCBarrierHessian(gap, dHat, stiffness);
  const newtonDampedForce = fN / (1.0 + 0.012 * hN * subDt);
  const jN = Math.min(newtonDampedForce * subDt, 0.065);

  const vcX = ring.velocity.x + (ring.angularVelocity.y * armZ - ring.angularVelocity.z * armY);
  const vcY = ring.velocity.y + (ring.angularVelocity.z * armX - ring.angularVelocity.x * armZ);
  const vcZ = ring.velocity.z + (ring.angularVelocity.x * armY - ring.angularVelocity.y * armX);

  const vn = vcX * nx + vcY * ny + vcZ * nz;
  // Add viscous normal damping when approaching the barrier singularity (vn < 0)
  const normalDampImpulse = vn < 0 ? Math.min(-vn * 0.28, 0.045) : 0.0;
  const totalJN = jN + normalDampImpulse;

  ring.velocity.x += nx * totalJN * invMass;
  ring.velocity.y += ny * totalJN * invMass;
  ring.velocity.z += nz * totalJN * invMass;

  const rxnX = armY * nz - armZ * ny;
  const rxnY = armZ * nx - armX * nz;
  const rxnZ = armX * ny - armY * nx;
  ring.angularVelocity.x += rxnX * totalJN * invInertia;
  ring.angularVelocity.y += rxnY * totalJN * invInertia;
  ring.angularVelocity.z += rxnZ * totalJN * invInertia;

  // IPC C^1 smooth mollified tangential friction (decelerates tangential slip & induces realistic rolling torque)
  const vtX = vcX - vn * nx;
  const vtY = vcY - vn * ny;
  const vtZ = vcZ - vn * nz;
  const vtLen = Math.hypot(vtX, vtY, vtZ);
  if (vtLen > 1e-4 && frictionCoeff > 0) {
    const tx = vtX / vtLen;
    const ty = vtY / vtLen;
    const tz = vtZ / vtLen;

    const rxtX = armY * tz - armZ * ty;
    const rxtY = armZ * tx - armX * tz;
    const rxtZ = armX * ty - armY * tx;
    const effMassT = invMass + (rxtX * rxtX + rxtY * rxtY + rxtZ * rxtZ) * invInertia;

    const epsV = 0.015;
    const mollifier = vtLen / Math.sqrt(vtLen * vtLen + epsV * epsV);
    const maxNoReverseJt = (vtLen / Math.max(1e-4, effMassT)) * 0.42;
    const jt = Math.min(frictionCoeff * totalJN * mollifier, maxNoReverseJt);

    ring.velocity.x -= tx * jt * invMass;
    ring.velocity.y -= ty * jt * invMass;
    ring.velocity.z -= tz * jt * invMass;

    ring.angularVelocity.x -= rxtX * jt * invInertia;
    ring.angularVelocity.y -= rxtY * jt * invInertia;
    ring.angularVelocity.z -= rxtZ * jt * invInertia;
  }
}

/**
 * Exact Analytical 3D Torus vs. Half-Space Plane Collision Helper:
 * When `tLen <= 1e-6` (the ring is parallel to the wall plane), distributes the contact
 * across 4 symmetric points (`±ringAxisU * R`, `±ringAxisV * R`) so parallel wall impacts
 * never apply a spurious asymmetric torque!
 */
function resolveTorusAgainstHalfSpacePlane(
  ring: RingState,
  ringNormal: THREE.Vector3,
  ringAxisU: THREE.Vector3,
  ringAxisV: THREE.Vector3,
  planeNx: number,
  planeNy: number,
  planeNz: number,
  planeOffset: number,
  restitution: number,
  friction: number,
  invMass: number,
  invInertia: number,
  posDeltaAccum: THREE.Vector3
): void {
  const R = ring.majorRadius;
  const r = ring.minorRadius;

  const dotNn = planeNx * ringNormal.x + planeNy * ringNormal.y + planeNz * ringNormal.z;
  const tx = -planeNx + dotNn * ringNormal.x;
  const ty = -planeNy + dotNn * ringNormal.y;
  const tz = -planeNz + dotNn * ringNormal.z;
  const tLen = Math.hypot(tx, ty, tz);

  const centerSignedDist =
    ring.position.x * planeNx +
    ring.position.y * planeNy +
    ring.position.z * planeNz -
    planeOffset;

  const minSurfaceDist = centerSignedDist - R * tLen - r;
  if (minSurfaceDist >= 0) return;

  const penetration = -minSurfaceDist;

  if (tLen > 1e-6) {
    const scale = R / tLen;
    applyContactImpulse(
      ring,
      tx * scale,
      ty * scale,
      tz * scale,
      planeNx,
      planeNy,
      planeNz,
      penetration,
      restitution,
      friction,
      invMass,
      invInertia,
      0.28,
      posDeltaAccum
    );
  } else {
    // Ring plane is strictly parallel to wall: apply 4-point symmetric distributed contact!
    const quarterPen = penetration * 0.25;
    const arms = [
      [ringAxisU.x * R, ringAxisU.y * R, ringAxisU.z * R],
      [-ringAxisU.x * R, -ringAxisU.y * R, -ringAxisU.z * R],
      [ringAxisV.x * R, ringAxisV.y * R, ringAxisV.z * R],
      [-ringAxisV.x * R, -ringAxisV.y * R, -ringAxisV.z * R],
    ];
    for (let k = 0; k < 4; k++) {
      applyContactImpulse(
        ring,
        arms[k][0],
        arms[k][1],
        arms[k][2],
        planeNx,
        planeNy,
        planeNz,
        quarterPen,
        restitution,
        friction,
        invMass,
        invInertia,
        0.28,
        posDeltaAccum
      );
    }
  }
}

export function resolveRingTankAndPegCollisionsExact(
  state: SimulationState,
  ring: RingState,
  subDt = 1 / 1440
): void {
  const { tank, pegs } = state;
  const R = ring.majorRadius;
  const r = ring.minorRadius;

  const invMass = 1.0 / (ring.mass || 1.0);
  const invInertia = invMass / (0.65 * R * R);

  _ringAxisU.set(1, 0, 0).applyQuaternion(ring.quaternion);
  _ringAxisV.set(0, 1, 0).applyQuaternion(ring.quaternion);
  _ringNormal.set(0, 0, 1).applyQuaternion(ring.quaternion).normalize();
  _posDeltaAccum.set(0, 0, 0);

  // --- A. Exact Analytical 3D Circle vs. All 6 Trapezoidal Tank Planes ---
  const dxSlope = tank.halfWidthTop - tank.halfWidthBottom;
  const dySlope = tank.halfHeight * 2;
  const slopeLen = Math.hypot(dxSlope, dySlope);
  const leftNx = dySlope / slopeLen;
  const leftNy = dxSlope / slopeLen;
  const leftOffset = -tank.halfWidthBottom * leftNx + -tank.halfHeight * leftNy;

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    leftNx,
    leftNy,
    0,
    leftOffset,
    0.35,
    0.18,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    -leftNx,
    leftNy,
    0,
    leftOffset,
    0.35,
    0.18,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    0,
    1,
    0,
    -tank.halfHeight,
    0.28,
    0.25,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    0,
    -1,
    0,
    -tank.halfHeight,
    0.28,
    0.18,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    0,
    0,
    1,
    -tank.halfDepth,
    0.32,
    0.15,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  resolveTorusAgainstHalfSpacePlane(
    ring,
    _ringNormal,
    _ringAxisU,
    _ringAxisV,
    0,
    0,
    -1,
    -tank.halfDepth,
    0.32,
    0.15,
    invMass,
    invInertia,
    _posDeltaAccum
  );

  // --- B. Smooth Analytical IPC Barrier + Singularity Friction + Gauss Linking Contact vs. 3D Tapered Pegs ---
  const holeRadius = R - r;
  const ipcDHat = 0.035;

  for (let p = 0; p < pegs.length; p++) {
    const peg = pegs[p];
    const pegTopY = peg.baseY + peg.height;
    const cupY = peg.baseY + peg.holderCupY;

    const dxCenter = ring.position.x - peg.baseX;
    const dzCenter = ring.position.z - peg.baseZ;
    const horizDist = Math.hypot(dxCenter, dzCenter);
    const maxRadialReach = R + r + Math.max(peg.baseRadius, peg.holderCupRadius) + 0.14;
    if (
      horizDist > maxRadialReach ||
      ring.position.y + R + r < peg.baseY - 0.12 ||
      ring.position.y - R - r > pegTopY + peg.tipRadius + 0.14
    ) {
      if (ring.hookedPegId === peg.id && ring.position.y > pegTopY + r * 0.45) {
        ring.hookedPegId = null;
      }
      continue;
    }

    // Check if the ring is threading downward over the conical needle tip right now
    // (Do not re-capture if gravity points upward / phone head is down or ring is flying upward off the tip!)
    if (
      ring.hookedPegId === null &&
      Math.abs(_ringNormal.y) > 0.28 &&
      ring.velocity.y <= 0.15 &&
      state.gravityDir.y <= 0.25
    ) {
      const pierceY =
        ring.position.y -
        ((peg.baseX - ring.position.x) * _ringNormal.x +
          (peg.baseZ - ring.position.z) * _ringNormal.z) /
          _ringNormal.y;

      const tipCaptureBand =
        (pierceY >= pegTopY - r * 1.8 && pierceY <= pegTopY + r * 1.4) ||
        (ring.position.y >= pegTopY - r * 1.8 && ring.position.y <= pegTopY + r * 1.4);
      if (
        tipCaptureBand &&
        horizDist < holeRadius - peg.tipRadius * 0.15 &&
        computeGaussLinkingNumberRingPeg(ring, peg) >= 0.18
      ) {
        ring.hookedPegId = peg.id;
      }
    }

    // --- CASE 1: RING IS TOPOLOGICALLY LINKED ON THIS NEEDLE (`ring.hookedPegId === peg.id`) ---
    // Pure 3D Multi-Point Rigid-Body Contact + IPC Barrier + Coulomb Friction:
    // Evaluates all 8 points around the 3D torus circle against the tapered conical needle shaft [cupY, pegTopY]
    // and the 3D rounded holder tray disk (cupDiskY). Every linear and angular velocity change comes strictly
    // from physical contact forces & impulses (Delta omega = I^-1 * (r x J)) at true 3D lever arms!
    if (ring.hookedPegId === peg.id) {
      if (
        ring.position.y > pegTopY + r * 0.65 &&
        (ring.velocity.y > 0 || state.gravityDir.y > 0.1)
      ) {
        ring.hookedPegId = null;
        continue;
      }

      const coneSlope = (peg.baseRadius - peg.tipRadius) / Math.max(0.1, peg.height);
      const invConeNorm = 1.0 / Math.sqrt(1.0 + coneSlope * coneSlope);
      const cupRimRadius = 0.024;
      const cupDiskY = cupY - 0.012;
      const diskCoreR = Math.max(0.04, peg.holderCupRadius - cupRimRadius);

      for (let k = 0; k < NUM_BI_SEEDS; k++) {
        const c = BI_SEED_COS[k] * R;
        const sn = BI_SEED_SIN[k] * R;
        const armX = _ringAxisU.x * c + _ringAxisV.x * sn;
        const armY = _ringAxisU.y * c + _ringAxisV.y * sn;
        const armZ = _ringAxisU.z * c + _ringAxisV.z * sn;

        const ptX = ring.position.x + armX;
        const ptY = ring.position.y + armY;
        const ptZ = ring.position.z + armZ;

        // 1A. 3D Inner-Rim Point vs. Tapered Conical Needle Shaft [cupY, pegTopY]:
        const armHoriz = Math.hypot(armX, armZ);
        if (armHoriz > 1e-5) {
          const ex = armX / armHoriz;
          const ez = armZ / armHoriz;

          const clampedY = THREE.MathUtils.clamp(ptY, cupY, pegTopY);
          const hFrac = THREE.MathUtils.clamp((clampedY - peg.baseY) / peg.height, 0, 1);
          const pegRadiusAtY = THREE.MathUtils.lerp(peg.baseRadius, peg.tipRadius, hFrac);

          // Signed radial coordinate of the torus centerline point along outward direction (ex, ez) from needle axis:
          const signedRadial = (ptX - peg.baseX) * ex + (ptZ - peg.baseZ) * ez;
          // Also account for the axial tube thickness (r * n_ring) when tilted relative to the vertical shaft:
          const tiltRimShift = Math.abs(_ringNormal.x * ex + _ringNormal.z * ez) * r * 0.45;
          const minInnerRadius = r + pegRadiusAtY + tiltRimShift;
          const gapShaft = signedRadial - minInnerRadius;

          // Contact point on the inner tube surface of the torus (lever arm from ring center of mass):
          const armInnerX = armX - ex * r;
          const armInnerY = armY;
          const armInnerZ = armZ - ez * r;

          // True 3D outward normal of the tapered conical needle surface at azimuth (ex, ez):
          const nxShaft = ex * invConeNorm;
          const nyShaft = coneSlope * invConeNorm;
          const nzShaft = ez * invConeNorm;

          if (gapShaft < ipcDHat) {
            applyIPCBarrierWithFriction(
              ring,
              armInnerX,
              armInnerY,
              armInnerZ,
              nxShaft,
              nyShaft,
              nzShaft,
              Math.max(0.0, gapShaft),
              ipcDHat,
              0.28,
              0.14,
              invMass,
              invInertia,
              subDt
            );
          }

          if (gapShaft < 0) {
            applyContactImpulse(
              ring,
              armInnerX,
              armInnerY,
              armInnerZ,
              nxShaft,
              nyShaft,
              nzShaft,
              -gapShaft,
              0.12,
              0.18,
              invMass,
              invInertia,
              0.22,
              _posDeltaAccum
            );
          }
        }

        // 1B. 3D Torus Point vs. Rounded Holder Tray Disk (`cupDiskY`) at the bottom of the needle:
        const dxD = ptX - peg.baseX;
        const dzD = ptZ - peg.baseZ;
        const radD = Math.hypot(dxD, dzD);
        const qx = radD > diskCoreR && radD > 1e-6 ? peg.baseX + (dxD / radD) * diskCoreR : ptX;
        const qz = radD > diskCoreR && radD > 1e-6 ? peg.baseZ + (dzD / radD) * diskCoreR : ptZ;

        const dxCup = ptX - qx;
        const dyCup = ptY - cupDiskY;
        const dzCup = ptZ - qz;
        const distCup = Math.hypot(dxCup, dyCup, dzCup);
        const minCupDist = r + cupRimRadius;
        const gapCup = ptY >= cupDiskY ? distCup - minCupDist : -(minCupDist + distCup);

        if (gapCup < ipcDHat) {
          const rawNx = dxCup;
          const rawNy = Math.max(0.01, dyCup);
          const rawNz = dzCup;
          const rawLen = Math.hypot(rawNx, rawNy, rawNz) || 1.0;
          const nxCup = rawNx / rawLen;
          const nyCup = rawNy / rawLen;
          const nzCup = rawNz / rawLen;

          // Contact point on the lower surface of the torus tube at point k:
          const armCupX = armX - nxCup * r;
          const armCupY = armY - nyCup * r;
          const armCupZ = armZ - nzCup * r;

          applyIPCBarrierWithFriction(
            ring,
            armCupX,
            armCupY,
            armCupZ,
            nxCup,
            nyCup,
            nzCup,
            Math.max(0.0, gapCup),
            ipcDHat,
            0.32,
            0.22,
            invMass,
            invInertia,
            subDt
          );

          if (gapCup < 0) {
            applyContactImpulse(
              ring,
              armCupX,
              armCupY,
              armCupZ,
              nxCup,
              nyCup,
              nzCup,
              -gapCup,
              0.12,
              0.24,
              invMass,
              invInertia,
              0.24,
              _posDeltaAccum
            );
          }
        }
      }
      continue;
    }

    // --- CASE 2: RING IS UNHOOKED (`ring.hookedPegId !== peg.id`) ---
    // 2A. Exact Analytical 3D Circle vs. Needle Shaft Segment [cupY, pegTopY] (strictly above the holder tray!)
    _segB0.set(peg.baseX, cupY, peg.baseZ);
    _segB1.set(peg.baseX, pegTopY, peg.baseZ);

    let bestDistSq = Infinity;
    let bestPegY = THREE.MathUtils.clamp(ring.position.y, cupY, pegTopY);
    for (let k = 0; k < NUM_BI_SEEDS; k++) {
      const c = BI_SEED_COS[k] * R;
      const sn = BI_SEED_SIN[k] * R;
      const py = THREE.MathUtils.clamp(
        ring.position.y + _ringAxisU.y * c + _ringAxisV.y * sn,
        cupY,
        pegTopY
      );
      _closestB.set(peg.baseX, py, peg.baseZ);
      closestPointOnCircle3D(ring.position, _ringNormal, _ringAxisU, R, _closestB, _contactPt);
      const cy = THREE.MathUtils.clamp(_contactPt.y, cupY, pegTopY);
      const d2 =
        (_contactPt.x - peg.baseX) * (_contactPt.x - peg.baseX) +
        (_contactPt.y - cy) * (_contactPt.y - cy) +
        (_contactPt.z - peg.baseZ) * (_contactPt.z - peg.baseZ);
      if (d2 < bestDistSq) {
        bestDistSq = d2;
        bestPegY = cy;
      }
    }

    // Check proximity to the Torus Medial-Axis Singularity (where the needle line passes near the ring's normal axis):
    const qDx = peg.baseX - ring.position.x;
    const qDy = bestPegY - ring.position.y;
    const qDz = peg.baseZ - ring.position.z;
    const qDotN = qDx * _ringNormal.x + qDy * _ringNormal.y + qDz * _ringNormal.z;
    const projX = qDx - qDotN * _ringNormal.x;
    const projY = qDy - qDotN * _ringNormal.y;
    const projZ = qDz - qDotN * _ringNormal.z;
    const inPlaneAxisDist = Math.hypot(projX, projY, projZ);

    // Torus Medial-Axis Singularity Frictional Deceleration ("算出奇点时考虑一下可以有摩擦力（减速）"):
    if (inPlaneAxisDist < R * 0.55 && Math.abs(ring.position.y - pegTopY) < R + r * 1.8) {
      const singProximity = 1.0 - inPlaneAxisDist / (R * 0.55);
      const singFrictionDamp = Math.exp(-3.8 * singProximity * subDt);
      ring.velocity.x *= singFrictionDamp;
      ring.velocity.z *= singFrictionDamp;
      ring.angularVelocity.multiplyScalar(Math.exp(-5.8 * singProximity * subDt));
    }

    // Helper to resolve a candidate point on the 3D circle against the needle shaft/tip [cupY, pegTopY]
    const resolveCirclePointAgainstNeedle = (
      ptX: number,
      ptY: number,
      ptZ: number,
      weight: number
    ) => {
      // Only resolve against the needle shaft if the point is above the underside of the holder tray
      if (ptY < cupY - r * 0.45) return;
      const clampedY = THREE.MathUtils.clamp(ptY, cupY, pegTopY);
      const hFrac = THREE.MathUtils.clamp((clampedY - peg.baseY) / peg.height, 0, 1);
      const pegRadiusAtY = THREE.MathUtils.lerp(peg.baseRadius, peg.tipRadius, hFrac);

      const dx = ptX - peg.baseX;
      const dy = ptY - clampedY;
      const dz = ptZ - peg.baseZ;
      const dist = Math.hypot(dx, dy, dz);
      const minDist = r + pegRadiusAtY;

      // Detect if an unhooked ring's leading rim is attempting to cross the needle shaft between cupY and pegTopY
      const isShaftCrossover =
        ptY >= cupY &&
        clampedY < pegTopY - r * 0.65 &&
        dx * dxCenter + dz * dzCenter < 0;

      const surfaceGap = isShaftCrossover ? -(minDist + dist) : dist - minDist;
      if (surfaceGap >= ipcDHat) return;

      // Smoothly regularize the contact normal across the dist -> 0 tube-centerline singularity
      const fallbackNx = horizDist > 1e-5 ? dxCenter / horizDist : 1.0;
      const fallbackNy = clampedY >= pegTopY - r * 0.65 ? 0.35 : 0.0;
      const fallbackNz = horizDist > 1e-5 ? dzCenter / horizDist : 0.0;
      const signFlip = isShaftCrossover ? -1.0 : 1.0;
      const regEps = 0.0035;
      const rawNx = dx * signFlip + fallbackNx * regEps;
      const rawNy = dy * signFlip + fallbackNy * regEps;
      const rawNz = dz * signFlip + fallbackNz * regEps;
      const rawLen = Math.hypot(rawNx, rawNy, rawNz) || 1.0;
      const nx = rawNx / rawLen;
      const ny = rawNy / rawLen;
      const nz = rawNz / rawLen;

      const armX = ptX - ring.position.x;
      const armY = ptY - ring.position.y;
      const armZ = ptZ - ring.position.z;

      applyIPCBarrierWithFriction(
        ring,
        armX,
        armY,
        armZ,
        nx,
        ny,
        nz,
        Math.max(0.0, surfaceGap),
        ipcDHat,
        0.52 * weight,
        0.18,
        invMass,
        invInertia,
        subDt
      );

      if (surfaceGap < 0) {
        const pen = -surfaceGap;
        applyContactImpulse(
          ring,
          armX,
          armY,
          armZ,
          nx,
          ny,
          nz,
          pen,
          0.24,
          0.18,
          invMass,
          invInertia,
          0.26 * weight,
          _posDeltaAccum
        );
      }
    };

    // Refine primary analytical closest point on circle to the needle shaft
    _closestB.set(peg.baseX, bestPegY, peg.baseZ);
    closestPointOnCircle3D(ring.position, _ringNormal, _ringAxisU, R, _closestB, _contactPt);

    if (inPlaneAxisDist < R * 0.45) {
      const primaryW = Math.max(0.25, inPlaneAxisDist / (R * 0.45));
      const symW = (1.0 - primaryW) * 0.25;
      resolveCirclePointAgainstNeedle(_contactPt.x, _contactPt.y, _contactPt.z, primaryW);
      for (let k = 0; k < 8; k += 2) {
        const c = BI_SEED_COS[k] * R;
        const sn = BI_SEED_SIN[k] * R;
        resolveCirclePointAgainstNeedle(
          ring.position.x + _ringAxisU.x * c + _ringAxisV.x * sn,
          ring.position.y + _ringAxisU.y * c + _ringAxisV.y * sn,
          ring.position.z + _ringAxisU.z * c + _ringAxisV.z * sn,
          symW
        );
      }
    } else {
      resolveCirclePointAgainstNeedle(_contactPt.x, _contactPt.y, _contactPt.z, 1.0);
    }

    // 2B. Exact 3D Circle-to-Disk (Rounded Holder Tray `托盘`) Shortest Distance & Contact Solver:
    // Models the holder tray at (peg.baseX, cupDiskY, peg.baseZ) as a true 3D rounded circular disk
    // (core disk radius = holderCupRadius - cupRimRadius, rounded rim radius = cupRimRadius = 0.024).
    // Eliminates the old axis-aligned vertical-only box check that popped rings vertically on side contact!
    const cupRimRadius = 0.024;
    const cupDiskY = cupY - 0.012;
    const diskCoreR = Math.max(0.04, peg.holderCupRadius - cupRimRadius);

    if (Math.abs(ring.position.y - cupDiskY) <= R + r + cupRimRadius + 0.08) {
      let bestDiskDistSq = Infinity;
      let bestRingPtX = _contactPt.x;
      let bestRingPtY = _contactPt.y;
      let bestRingPtZ = _contactPt.z;
      let bestDiskPtX = peg.baseX;
      let bestDiskPtY = cupDiskY;
      let bestDiskPtZ = peg.baseZ;

      for (let k = 0; k < NUM_BI_SEEDS; k++) {
        const c = BI_SEED_COS[k] * R;
        const sn = BI_SEED_SIN[k] * R;
        let rx = ring.position.x + _ringAxisU.x * c + _ringAxisV.x * sn;
        let ry = ring.position.y + _ringAxisU.y * c + _ringAxisV.y * sn;
        let rz = ring.position.z + _ringAxisU.z * c + _ringAxisV.z * sn;

        // 2 iterations of alternating projection between 3D ring circle and 3D horizontal tray disk
        let qx = peg.baseX;
        let qz = peg.baseZ;
        for (let iter = 0; iter < 2; iter++) {
          const dxD = rx - peg.baseX;
          const dzD = rz - peg.baseZ;
          const radD = Math.hypot(dxD, dzD);
          if (radD > diskCoreR && radD > 1e-6) {
            qx = peg.baseX + (dxD / radD) * diskCoreR;
            qz = peg.baseZ + (dzD / radD) * diskCoreR;
          } else {
            qx = rx;
            qz = rz;
          }
          _closestB.set(qx, cupDiskY, qz);
          closestPointOnCircle3D(ring.position, _ringNormal, _ringAxisU, R, _closestB, _closestA);
          rx = _closestA.x;
          ry = _closestA.y;
          rz = _closestA.z;
        }

        const dxFinal = rx - peg.baseX;
        const dzFinal = rz - peg.baseZ;
        const radFinal = Math.hypot(dxFinal, dzFinal);
        if (radFinal > diskCoreR && radFinal > 1e-6) {
          qx = peg.baseX + (dxFinal / radFinal) * diskCoreR;
          qz = peg.baseZ + (dzFinal / radFinal) * diskCoreR;
        } else {
          qx = rx;
          qz = rz;
        }

        const d2 =
          (rx - qx) * (rx - qx) +
          (ry - cupDiskY) * (ry - cupDiskY) +
          (rz - qz) * (rz - qz);
        if (d2 < bestDiskDistSq) {
          bestDiskDistSq = d2;
          bestRingPtX = rx;
          bestRingPtY = ry;
          bestRingPtZ = rz;
          bestDiskPtX = qx;
          bestDiskPtY = cupDiskY;
          bestDiskPtZ = qz;
        }
      }

      const dxCup = bestRingPtX - bestDiskPtX;
      const dyCup = bestRingPtY - bestDiskPtY;
      const dzCup = bestRingPtZ - bestDiskPtZ;
      const distCup = Math.hypot(dxCup, dyCup, dzCup);
      const minCupDist = r + cupRimRadius;
      const cupSurfaceGap = distCup - minCupDist;

      if (cupSurfaceGap < ipcDHat) {
        const regNx = dxCup + (horizDist > 1e-5 ? (dxCenter / horizDist) * 0.002 : 0.0);
        const regNy = dyCup + (ring.position.y >= cupDiskY ? 0.002 : -0.002);
        const regNz = dzCup + (horizDist > 1e-5 ? (dzCenter / horizDist) * 0.002 : 0.0);
        const regLen = Math.hypot(regNx, regNy, regNz) || 1.0;
        const nxCup = regNx / regLen;
        const nyCup = regNy / regLen;
        const nzCup = regNz / regLen;

        const armX = bestRingPtX - ring.position.x;
        const armY = bestRingPtY - ring.position.y;
        const armZ = bestRingPtZ - ring.position.z;

        applyIPCBarrierWithFriction(
          ring,
          armX,
          armY,
          armZ,
          nxCup,
          nyCup,
          nzCup,
          Math.max(0.0, cupSurfaceGap),
          ipcDHat,
          0.48,
          0.18,
          invMass,
          invInertia,
          subDt
        );

        if (cupSurfaceGap < 0) {
          applyContactImpulse(
            ring,
            armX,
            armY,
            armZ,
            nxCup,
            nyCup,
            nzCup,
            -cupSurfaceGap,
            0.2,
            0.18,
            invMass,
            invInertia,
            0.25,
            _posDeltaAccum
          );
        }
      }
    }
  }

  // Strictly bounded per-substep position correction prevents any multi-contact or singularity teleportation!
  const maxStepCorrection = 0.0075;
  if (_posDeltaAccum.lengthSq() > maxStepCorrection * maxStepCorrection) {
    _posDeltaAccum.setLength(maxStepCorrection);
  }
  ring.position.add(_posDeltaAccum);
}

function applyContactImpulse(
  ring: RingState,
  armX: number,
  armY: number,
  armZ: number,
  nx: number,
  ny: number,
  nz: number,
  penetration: number,
  restitution: number,
  friction: number,
  invMass: number,
  invInertia: number,
  posCorrectionScale: number,
  posDeltaAccum: THREE.Vector3
): void {
  const posCorrection = Math.min(0.0045, penetration * posCorrectionScale);
  posDeltaAccum.x += nx * posCorrection;
  posDeltaAccum.y += ny * posCorrection;
  posDeltaAccum.z += nz * posCorrection;

  const wx = ring.angularVelocity.x;
  const wy = ring.angularVelocity.y;
  const wz = ring.angularVelocity.z;
  _sampleVel.set(
    ring.velocity.x + (wy * armZ - wz * armY),
    ring.velocity.y + (wz * armX - wx * armZ),
    ring.velocity.z + (wx * armY - wy * armX)
  );

  const rxnX = armY * nz - armZ * ny;
  const rxnY = armZ * nx - armX * nz;
  const rxnZ = armX * ny - armY * nx;
  const angTerm = (rxnX * rxnX + rxnY * rxnY + rxnZ * rxnZ) * invInertia;
  const denom = invMass + angTerm;

  const vn = _sampleVel.x * nx + _sampleVel.y * ny + _sampleVel.z * nz;
  let normalImpulseForFriction = Math.min(0.04, penetration * 0.45) / denom;

  if (vn < 0) {
    const j = (-(1 + restitution) * vn) / denom;
    ring.velocity.x += nx * j * invMass;
    ring.velocity.y += ny * j * invMass;
    ring.velocity.z += nz * j * invMass;

    ring.angularVelocity.x += rxnX * j * invInertia;
    ring.angularVelocity.y += rxnY * j * invInertia;
    ring.angularVelocity.z += rxnZ * j * invInertia;
    normalImpulseForFriction = Math.max(normalImpulseForFriction, j);
  }

  // IPC C1 smooth mollified tangential friction with full linear deceleration AND rotational torque
  const vtX = _sampleVel.x - vn * nx;
  const vtY = _sampleVel.y - vn * ny;
  const vtZ = _sampleVel.z - vn * nz;
  const vtLen = Math.hypot(vtX, vtY, vtZ);
  if (vtLen > 0.0005 && friction > 0) {
    const tx = vtX / vtLen;
    const ty = vtY / vtLen;
    const tz = vtZ / vtLen;

    const rxtX = armY * tz - armZ * ty;
    const rxtY = armZ * tx - armX * tz;
    const rxtZ = armX * ty - armY * tx;
    const effMassT = invMass + (rxtX * rxtX + rxtY * rxtY + rxtZ * rxtZ) * invInertia;

    const mollifier = vtLen / Math.sqrt(vtLen * vtLen + 0.015 * 0.015);
    const jt = Math.min(
      friction * normalImpulseForFriction * mollifier,
      (vtLen / Math.max(1e-4, effMassT)) * 0.38
    );

    ring.velocity.x -= tx * jt * invMass;
    ring.velocity.y -= ty * jt * invMass;
    ring.velocity.z -= tz * jt * invMass;

    ring.angularVelocity.x -= rxtX * jt * invInertia;
    ring.angularVelocity.y -= rxtY * jt * invInertia;
    ring.angularVelocity.z -= rxtZ * jt * invInertia;
  }
}

/**
 * Exact 3D Torus-to-Torus Collision Resolution (`resolveRingToRingCollisionsExact3D`):
 * - Uses 8-Point 3D Torus-to-Torus Stack Contact Solver when two rings are hooked on the SAME needle (`a.hookedPegId === b.hookedPegId`),
 *   computing true 3D lever arms `r_A, r_B` and rotational contact torques `r x J` so stacked tilted rings physically rock and settle onto each other!
 * - Protects hooked rings from being pushed radially through the needle shaft by unhooked rings!
 */
export function resolveRingToRingCollisionsExact3D(rings: RingState[]): void {
  const n = rings.length;
  for (let i = 0; i < n; i++) {
    const a = rings[i];
    const invMassA = 1.0 / (a.mass || 1.0);
    const invInertiaA = invMassA / (0.65 * a.majorRadius * a.majorRadius);

    for (let j = i + 1; j < n; j++) {
      const b = rings[j];
      const invMassB = 1.0 / (b.mass || 1.0);
      const invInertiaB = invMassB / (0.65 * b.majorRadius * b.majorRadius);
      const totalInvMass = invMassA + invMassB;

      // --- SPECIAL CASE: Both rings are hooked on the SAME needle (`3D multi-point coaxial stack`) ---
      if (a.hookedPegId !== null && a.hookedPegId === b.hookedPegId) {
        const minStackDist = a.minorRadius + b.minorRadius + 0.003;
        if (Math.abs(b.position.y - a.position.y) > a.majorRadius + b.majorRadius + minStackDist) {
          continue;
        }

        _ringAxisU.set(1, 0, 0).applyQuaternion(a.quaternion);
        _ringAxisV.set(0, 1, 0).applyQuaternion(a.quaternion);
        _ringNormalB.set(0, 0, 1).applyQuaternion(b.quaternion).normalize();
        _ringAxisUB.set(1, 0, 0).applyQuaternion(b.quaternion);

        const signY = b.position.y >= a.position.y ? 1.0 : -1.0;

        for (let k = 0; k < NUM_BI_SEEDS; k++) {
          const c = BI_SEED_COS[k] * a.majorRadius;
          const sn = BI_SEED_SIN[k] * a.majorRadius;
          const armAx = _ringAxisU.x * c + _ringAxisV.x * sn;
          const armAy = _ringAxisU.y * c + _ringAxisV.y * sn;
          const armAz = _ringAxisU.z * c + _ringAxisV.z * sn;

          _closestA.set(a.position.x + armAx, a.position.y + armAy, a.position.z + armAz);
          closestPointOnCircle3D(
            b.position,
            _ringNormalB,
            _ringAxisUB,
            b.majorRadius,
            _closestA,
            _closestB
          );

          const dyPt = (_closestB.y - _closestA.y) * signY;
          if (dyPt >= minStackDist) continue;

          const penY = minStackDist - dyPt;
          const stepStackCorr = Math.min(0.0018, penY * 0.14);
          a.position.y -= signY * stepStackCorr;
          b.position.y += signY * stepStackCorr;

          const armBx = _closestB.x - b.position.x;
          const armBy = _closestB.y - b.position.y;
          const armBz = _closestB.z - b.position.z;

          const vAx = a.velocity.x + (a.angularVelocity.y * armAz - a.angularVelocity.z * armAy);
          const vAy = a.velocity.y + (a.angularVelocity.z * armAx - a.angularVelocity.x * armAz);
          const vAz = a.velocity.z + (a.angularVelocity.x * armAy - a.angularVelocity.y * armAx);

          const vBx = b.velocity.x + (b.angularVelocity.y * armBz - b.angularVelocity.z * armBy);
          const vBy = b.velocity.y + (b.angularVelocity.z * armBx - b.angularVelocity.x * armBz);
          const vBz = b.velocity.z + (b.angularVelocity.x * armBy - b.angularVelocity.y * armBx);

          const relVy = (vBy - vAy) * signY;
          // Normal is (0, signY, 0), so r x n = (-armZ * signY, 0, armX * signY)
          const rxnAx = -armAz * signY;
          const rxnAz = armAx * signY;
          const rxnBx = -armBz * signY;
          const rxnBz = armBx * signY;

          const angTerm =
            (rxnAx * rxnAx + rxnAz * rxnAz) * invInertiaA +
            (rxnBx * rxnBx + rxnBz * rxnBz) * invInertiaB;
          const denom = totalInvMass + angTerm;

          const penImpulse = (Math.min(0.018, penY * 0.25) / denom) * 0.25;
          const velImpulse = relVy < 0 ? (-(1.0 + 0.12) * relVy) / denom * 0.25 : 0.0;
          const jStack = penImpulse + velImpulse;

          a.velocity.y -= signY * jStack * invMassA;
          b.velocity.y += signY * jStack * invMassB;

          a.angularVelocity.x -= rxnAx * jStack * invInertiaA;
          a.angularVelocity.z -= rxnAz * jStack * invInertiaA;
          b.angularVelocity.x += rxnBx * jStack * invInertiaB;
          b.angularVelocity.z += rxnBz * jStack * invInertiaB;

          // Physical Coulomb tangential friction between stacked rings at their 3D contact arms
          const rvx = vBx - vAx;
          const rvz = vBz - vAz;
          const vtLen = Math.hypot(rvx, rvz);
          if (vtLen > 1e-4) {
            const tx = rvx / vtLen;
            const tz = rvz / vtLen;
            const jt = Math.min(0.18 * jStack, (vtLen / totalInvMass) * 0.12);
            a.velocity.x += tx * jt * invMassA;
            a.velocity.z += tz * jt * invMassA;
            b.velocity.x -= tx * jt * invMassB;
            b.velocity.z -= tz * jt * invMassB;

            const rxtAy = armAz * tx - armAx * tz;
            const rxtBy = armBz * tx - armBx * tz;
            a.angularVelocity.y += rxtAy * jt * invInertiaA;
            b.angularVelocity.y -= rxtBy * jt * invInertiaB;
          }
        }
        continue;
      }

      const contacts = computeExactTorusToTorusContacts(a, b);
      if (contacts.length === 0) continue;

      const share = 1.0 / contacts.length;
      const aHooked = a.hookedPegId !== null;
      const bHooked = b.hookedPegId !== null;

      for (let c = 0; c < contacts.length; c++) {
        const contact = contacts[c];
        const nx = contact.normalAToB.x;
        const ny = contact.normalAToB.y;
        const nz = contact.normalAToB.z;
        const corr = Math.min(0.0045, contact.penetration * 0.28 * share) / totalInvMass;

        // If one ring is hooked on a needle and the other is free, do not push the hooked ring radially off the needle!
        const weightAxz = aHooked && !bHooked ? 0.0 : !aHooked && bHooked ? 2.0 : 1.0;
        const weightBxz = bHooked && !aHooked ? 0.0 : !bHooked && aHooked ? 2.0 : 1.0;

        a.position.x -= nx * corr * invMassA * weightAxz;
        a.position.y -= ny * corr * invMassA;
        a.position.z -= nz * corr * invMassA * weightAxz;
        b.position.x += nx * corr * invMassB * weightBxz;
        b.position.y += ny * corr * invMassB;
        b.position.z += nz * corr * invMassB * weightBxz;

        const armAx = contact.pointOnA.x - a.position.x;
        const armAy = contact.pointOnA.y - a.position.y;
        const armAz = contact.pointOnA.z - a.position.z;

        const armBx = contact.pointOnB.x - b.position.x;
        const armBy = contact.pointOnB.y - b.position.y;
        const armBz = contact.pointOnB.z - b.position.z;

        const vAx = a.velocity.x + (a.angularVelocity.y * armAz - a.angularVelocity.z * armAy);
        const vAy = a.velocity.y + (a.angularVelocity.z * armAx - a.angularVelocity.x * armAz);
        const vAz = a.velocity.z + (a.angularVelocity.x * armAy - a.angularVelocity.y * armAx);

        const vBx = b.velocity.x + (b.angularVelocity.y * armBz - b.angularVelocity.z * armBy);
        const vBy = b.velocity.y + (b.angularVelocity.z * armBx - b.angularVelocity.x * armBz);
        const vBz = b.velocity.z + (b.angularVelocity.x * armBy - b.angularVelocity.y * armBx);

        const rvx = vBx - vAx;
        const rvy = vBy - vAy;
        const rvz = vBz - vAz;
        const relVn = rvx * nx + rvy * ny + rvz * nz;
        if (relVn < 0) {
          const rxnAx = armAy * nz - armAz * ny;
          const rxnAy = armAz * nx - armAx * nz;
          const rxnAz = armAx * ny - armAy * nx;

          const rxnBx = armBy * nz - armBz * ny;
          const rxnBy = armBz * nx - armBx * nz;
          const rxnBz = armBx * ny - armBy * nx;

          const angTerm =
            (rxnAx * rxnAx + rxnAy * rxnAy + rxnAz * rxnAz) * invInertiaA +
            (rxnBx * rxnBx + rxnBy * rxnBy + rxnBz * rxnBz) * invInertiaB;

          const restitution = 0.22;
          const jImp = ((-(1 + restitution) * relVn) / (totalInvMass + angTerm)) * share;

          a.velocity.x -= nx * jImp * invMassA * weightAxz;
          a.velocity.y -= ny * jImp * invMassA;
          a.velocity.z -= nz * jImp * invMassA * weightAxz;
          b.velocity.x += nx * jImp * invMassB * weightBxz;
          b.velocity.y += ny * jImp * invMassB;
          b.velocity.z += nz * jImp * invMassB * weightBxz;

          a.angularVelocity.x -= rxnAx * jImp * invInertiaA;
          a.angularVelocity.y -= rxnAy * jImp * invInertiaA;
          a.angularVelocity.z -= rxnAz * jImp * invInertiaA;

          b.angularVelocity.x += rxnBx * jImp * invInertiaB;
          b.angularVelocity.y += rxnBy * jImp * invInertiaB;
          b.angularVelocity.z += rxnBz * jImp * invInertiaB;

          // Ring-to-Ring IPC C1 mollified tangential friction
          const vtX = rvx - relVn * nx;
          const vtY = rvy - relVn * ny;
          const vtZ = rvz - relVn * nz;
          const vtLen = Math.hypot(vtX, vtY, vtZ);
          if (vtLen > 1e-4) {
            const tx = vtX / vtLen;
            const ty = vtY / vtLen;
            const tz = vtZ / vtLen;
            const mollifier = vtLen / Math.sqrt(vtLen * vtLen + 0.015 * 0.015);
            const jt = Math.min(0.16 * jImp * mollifier, (vtLen / totalInvMass) * 0.3 * share);
            a.velocity.x += tx * jt * invMassA * weightAxz;
            a.velocity.y += ty * jt * invMassA;
            a.velocity.z += tz * jt * invMassA * weightAxz;
            b.velocity.x -= tx * jt * invMassB * weightBxz;
            b.velocity.y -= ty * jt * invMassB;
            b.velocity.z -= tz * jt * invMassB * weightBxz;
          }
        }
      }
    }
  }
}

export const WATERFUL_STORAGE_KEY = 'tomy_waterful_ring_toss_state_v1';

export interface SerializedRingSnapshot {
  id: number;
  position: [number, number, number];
  velocity: [number, number, number];
  quaternion: [number, number, number, number];
  angularVelocity: [number, number, number];
  hookedPegId: string | null;
}

export interface SerializedSimulationSnapshot {
  version: 1;
  score: number;
  elapsedTime: number;
  tiltRoll: number;
  tiltPitch: number;
  rings: SerializedRingSnapshot[];
}

/**
 * Persists the current 3D ring-toss game state (all ring 3D positions, velocities,
 * orientations, hooked needle states, score, and device tilt) to `localStorage`.
 */
export function saveSimulationStateToStorage(
  state: SimulationState,
  storageKey: string = WATERFUL_STORAGE_KEY
): boolean {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return false;
    const payload: SerializedSimulationSnapshot = {
      version: 1,
      score: state.score,
      elapsedTime: state.timeElapsed,
      tiltRoll: state.tiltRoll,
      tiltPitch: state.tiltPitch,
      rings: state.rings.map((r) => ({
        id: r.id,
        position: [r.position.x, r.position.y, r.position.z],
        velocity: [r.velocity.x, r.velocity.y, r.velocity.z],
        quaternion: [r.quaternion.x, r.quaternion.y, r.quaternion.z, r.quaternion.w],
        angularVelocity: [r.angularVelocity.x, r.angularVelocity.y, r.angularVelocity.z],
        hookedPegId: r.hookedPegId,
      })),
    };
    window.localStorage.setItem(storageKey, JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

/**
 * Restores a previously saved 3D ring-toss game state from `localStorage` into `state`.
 */
export function loadSimulationStateFromStorage(
  state: SimulationState,
  storageKey: string = WATERFUL_STORAGE_KEY
): boolean {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return false;
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return false;
    const parsed = JSON.parse(raw) as Partial<SerializedSimulationSnapshot>;
    if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.rings)) return false;
    if (parsed.rings.length !== state.rings.length) return false;

    const maxX = state.tank.halfWidthTop - 0.12;
    const maxY = state.tank.halfHeight - 0.12;
    const maxZ = state.tank.halfDepth - 0.08;

    for (let i = 0; i < state.rings.length; i++) {
      const snap = parsed.rings[i];
      const ring = state.rings[i];
      if (!snap || !Array.isArray(snap.position) || !Array.isArray(snap.quaternion)) {
        return false;
      }
      const [px, py, pz] = snap.position;
      const [vx, vy, vz] = Array.isArray(snap.velocity) ? snap.velocity : [0, 0, 0];
      const [qx, qy, qz, qw] = snap.quaternion;
      const [wx, wy, wz] = Array.isArray(snap.angularVelocity)
        ? snap.angularVelocity
        : [0, 0, 0];

      if (![px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz].every(Number.isFinite)) {
        return false;
      }

      ring.position.set(
        THREE.MathUtils.clamp(px, -maxX, maxX),
        THREE.MathUtils.clamp(py, -maxY, maxY),
        THREE.MathUtils.clamp(pz, -maxZ, maxZ)
      );
      ring.velocity.set(
        THREE.MathUtils.clamp(vx, -6, 6),
        THREE.MathUtils.clamp(vy, -6, 6),
        THREE.MathUtils.clamp(vz, -6, 6)
      );
      ring.quaternion.set(qx, qy, qz, qw).normalize();
      ring.angularVelocity.set(
        THREE.MathUtils.clamp(wx, -12, 12),
        THREE.MathUtils.clamp(wy, -12, 12),
        THREE.MathUtils.clamp(wz, -12, 12)
      );
      ring.hookedPegId =
        typeof snap.hookedPegId === 'string' &&
        state.pegs.some((p) => p.id === snap.hookedPegId)
          ? snap.hookedPegId
          : null;
    }

    if (Number.isFinite(parsed.elapsedTime)) {
      state.timeElapsed = Math.max(0, Number(parsed.elapsedTime));
    }
    if (Number.isFinite(parsed.tiltRoll) && Number.isFinite(parsed.tiltPitch)) {
      setGravityFromTilt(state, Number(parsed.tiltRoll), Number(parsed.tiltPitch));
    }

    syncRingsAndUniformsToWasmMemory(state);
    evaluateHookedRings(state);
    return true;
  } catch {
    return false;
  }
}

