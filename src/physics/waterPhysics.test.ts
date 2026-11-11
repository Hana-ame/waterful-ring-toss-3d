import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  createInitialSimulationState,
  setPumpInput,
  setGravityFromTilt,
  setGyroscopeAngularVelocity,
  stepWaterSimulation,
  evaluateHookedRings,
  sampleWaterVelocityField,
  computeExactTorusToTorusContacts,
  computeIPCBarrierGradient,
  computeIPCBarrierHessian,
  computeGaussLinkingNumberRingPeg,
  saveSimulationStateToStorage,
  loadSimulationStateFromStorage,
  resolveRingTankAndPegCollisionsExact,
  resolveRingToRingCollisionsExact3D,
  testOBBvsOBB_SAT,
  OBB3D,
} from './waterPhysics';
import { mapDeviceOrientationToTiltRadians } from './sensorBridge';
import { waterAudio } from '../audio/waterAudio';

test('P0 & P1: Gyroscope rotationRate, tilt sloshing, and phone head-down (手机头向下 beta < 0) gravity direction', () => {
  const state = createInitialSimulationState(4);

  // 1. Verify exact right-handed gz = -Math.sin(pitch) sign
  const pitchRad = Math.PI / 6; // +30 deg
  setGravityFromTilt(state, 0, pitchRad);
  assert.ok(
    Math.abs(state.gravityDir.z - -Math.sin(pitchRad)) < 1e-5,
    `Expected gz = -sin(pitch) = ${-Math.sin(pitchRad)}, got ${state.gravityDir.z}`
  );

  // 2. Verify gyroscope angular velocity excites inertial water sloshing (`waterAngularVel × r`) in `sampleWaterVelocityField`
  // even when both water pumps are completely idle!
  setGyroscopeAngularVelocity(state, 0, 4.0, 0); // Roll angular velocity = +4 rad/s
  assert.ok(Math.abs(state.waterAngularVel.z) > 1.0, 'waterAngularVel.z excited by gyroscope');

  const sloshVelTop = new THREE.Vector3();
  const sloshVelBottom = new THREE.Vector3();
  const vort = new THREE.Vector3();
  sampleWaterVelocityField(state, new THREE.Vector3(0, 1.5, 0), sloshVelTop, vort);
  sampleWaterVelocityField(state, new THREE.Vector3(0, -1.5, 0), sloshVelBottom, vort);

  assert.ok(
    Math.abs(sloshVelTop.x) > 0.5 && sloshVelTop.x * sloshVelBottom.x < 0,
    `Expected opposite inertial sloshing velocities at top (${sloshVelTop.x}) vs bottom (${sloshVelBottom.x})`
  );

  // 3. Verify normal upright portrait & landscape orientation axis remapping in sensorBridge
  const portrait = mapDeviceOrientationToTiltRadians(60, 25, 0);
  const landscape90 = mapDeviceOrientationToTiltRadians(25, 60, 90);
  assert.ok(portrait.rollRad > 0.3, 'Portrait gamma maps to positive roll');
  assert.ok(landscape90.rollRad < -0.3, 'Landscape 90° beta maps to negative roll');

  // 4. Verify phone head pointing down (`手机头向下`: beta = -60° and beta = -90°, gamma = 0° in portrait)
  // produces upward screen gravity (`state.gravityDir.y > 0.85`) with ZERO sideways drift (`|gx| < 1e-5`)!
  const headDown60 = mapDeviceOrientationToTiltRadians(-60, 0, 0);
  setGravityFromTilt(state, headDown60.rollRad, headDown60.pitchRad);
  assert.ok(
    state.gravityDir.y > 0.99 && Math.abs(state.gravityDir.x) < 1e-5,
    `When phone head points down at handheld angle (beta=-60, gamma=0), gravity must be (0, +1, 0), got (${state.gravityDir.x}, ${state.gravityDir.y}, ${state.gravityDir.z})`
  );

  const headDown90 = mapDeviceOrientationToTiltRadians(-90, 0, 0);
  setGravityFromTilt(state, headDown90.rollRad, headDown90.pitchRad);
  assert.ok(
    state.gravityDir.y > 0.85 && Math.abs(state.gravityDir.x) < 1e-5,
    `When phone is vertical head-down (beta=-90, gamma=0), gravity must point toward +Y with zero sideways drift, got (${state.gravityDir.x}, ${state.gravityDir.y}, ${state.gravityDir.z})`
  );

  // Also verify slight tilt while phone head points down (beta = -80°, gamma = 20°) keeps gy > 0.8 and pulls toward +X
  const headDownTilted = mapDeviceOrientationToTiltRadians(-80, 20, 0);
  setGravityFromTilt(state, headDownTilted.rollRad, headDownTilted.pitchRad);
  assert.ok(
    state.gravityDir.y > 0.85 && state.gravityDir.x > 0.12,
    `Expected head-down +20° gamma to keep gy > 0.85 and gx > 0.12, got (${state.gravityDir.x}, ${state.gravityDir.y})`
  );
});

test('P1 & P2: Isolated per-instance WebAssembly memory and live uniformFloatView at offset 2048', () => {
  const stateA = createInitialSimulationState(4);
  const stateB = createInitialSimulationState(4);

  assert.notEqual(
    stateA.wasmKernel.memory,
    stateB.wasmKernel.memory,
    'Each SimulationState owns an isolated WebAssembly.Memory instance (no singleton clobbering)'
  );

  stateA.config.wasmSubsteps = 32;
  stepWaterSimulation(stateA, 1 / 60);

  assert.equal(stateA.totalWasmStepsExecuted, 32);
  assert.equal(
    stateA.wasmKernel.uniformFloatView[2],
    32,
    'WASM binary actively increments stepCounter in uniformFloatView[2] at byte offset 2056'
  );
  assert.equal(
    stateB.wasmKernel.uniformFloatView[2],
    0,
    'Second WASM instance memory is completely untouched by stateA'
  );
});

test('exact analytical Circle-Plane & Capsule-Segment collision has zero inter-vertex gaps on sloped corners and peg tips', () => {
  const state = createInitialSimulationState(1);
  state.gravityMagnitude = 0;
  const ring = state.rings[0];
  const leftPeg = state.pegs[0];

  const testAngle = Math.PI / 16; // Strictly midway between sample 0 and sample 1
  ring.quaternion.identity();
  const R = ring.majorRadius;
  const r = ring.minorRadius;
  const pegY = leftPeg.baseY + leftPeg.height * 0.8;
  const pegRad = THREE.MathUtils.lerp(leftPeg.baseRadius, leftPeg.tipRadius, 0.8);

  const targetDist = R + r + pegRad - 0.03;
  ring.position.set(
    leftPeg.baseX - Math.cos(testAngle) * targetDist,
    pegY - Math.sin(testAngle) * targetDist,
    leftPeg.baseZ
  );
  ring.velocity.set(Math.cos(testAngle) * 1.5, Math.sin(testAngle) * 1.5, 0);

  stepWaterSimulation(state, 1 / 60);

  const afterDist = Math.hypot(ring.position.x - leftPeg.baseX, ring.position.y - pegY);
  assert.ok(
    afterDist > targetDist + 0.015,
    `Expected continuous analytical circle/capsule collision to push ring out at midpoint angle pi/16 (before=${targetDist}, after=${afterDist})`
  );
});

test('Zero-GC bidirectional 3D Torus-to-Torus shortest distance solver resolves side-by-side and perpendicular contacts', () => {
  const state = createInitialSimulationState(2);
  const a = state.rings[0];
  const b = state.rings[1];
  const R = a.majorRadius;
  const r = a.minorRadius;

  a.quaternion.identity();
  b.quaternion.identity();
  a.position.set(0, 0, 0);
  const sideCenterSep = 2 * R + 2 * r - 0.03;
  b.position.set(sideCenterSep, 0, 0);

  const sideContacts = computeExactTorusToTorusContacts(a, b);
  assert.ok(sideContacts.length >= 1, 'Detected side-by-side coplanar rim contact');
  assert.ok(
    Math.abs(sideContacts[0].penetration - 0.03) < 0.005,
    `Expected exact penetration ~0.03, got ${sideContacts[0].penetration}`
  );
  assert.ok(sideContacts[0].normalAToB.x > 0.98, 'Contact normal points along +X between rims');

  b.position.set(2 * R + 2 * r + 0.02, 0, 0);
  const nonTouchingContacts = computeExactTorusToTorusContacts(a, b);
  assert.equal(nonTouchingContacts.length, 0, 'Non-touching side-by-side rings have 0 false contacts');

  b.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  b.position.set(2 * R + 2 * r - 0.025, 0, 0);
  const tJunctionContacts = computeExactTorusToTorusContacts(a, b);
  assert.ok(tJunctionContacts.length >= 1, 'Detected 90-degree perpendicular 3D torus contact');
  assert.ok(Math.abs(tJunctionContacts[0].penetration - 0.025) < 0.005);

  const obbA: OBB3D = {
    center: new THREE.Vector3(0, 0, 0),
    axes: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)],
    halfExtents: new THREE.Vector3(0.5, 0.2, 0.2),
  };
  const obbB: OBB3D = {
    center: new THREE.Vector3(0.85, 0, 0),
    axes: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)],
    halfExtents: new THREE.Vector3(0.45, 0.2, 0.2),
  };
  const satNormal = new THREE.Vector3();
  const satRes = testOBBvsOBB_SAT(obbA, obbB, satNormal);
  assert.equal(satRes.intersects, true);
  assert.ok(Math.abs(satRes.penetration - 0.1) < 1e-5);
});

test('P2: evaluateHookedRings checks true 3D peg-line plane piercing and rejects vertical edge-on rings', () => {
  const state = createInitialSimulationState(2);
  const leftPeg = state.pegs[0];

  // Ring 0 is centered on the peg, but oriented vertically (hole normal along +Z, so |n_y| = 0)!
  // A vertical ring resting against or overlapping the peg shaft is NOT hooked around the peg!
  state.rings[0].position.set(leftPeg.baseX, leftPeg.baseY + 0.6, leftPeg.baseZ);
  state.rings[0].quaternion.identity(); // hole axis is +Z, so n_y = 0
  assert.equal(
    evaluateHookedRings(state),
    0,
    'Vertical edge-on ring (|n_y| = 0) must NOT be counted as hooked even if its center is near the peg'
  );

  // Now orient Ring 0 horizontally (hole axis along +Y, |n_y| = 1) so the peg pierces its aperture
  state.rings[0].quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  assert.equal(
    evaluateHookedRings(state),
    1,
    'Horizontal ring encircling the peg shaft is counted as hooked'
  );
});

test('both water ejection (喷水 from nozzle) and bellows release suction (吸水 from separate return vents, never nozzle) work properly', () => {
  const state = createInitialSimulationState(6);
  const leftNozzle = state.nozzles[0];

  setPumpInput(state, 'left', true, 'heavy');
  for (let i = 0; i < 8; i++) {
    stepWaterSimulation(state, 1 / 60);
  }

  const velAboveNozzle = new THREE.Vector3();
  const velOppositeSide = new THREE.Vector3();
  const vort = new THREE.Vector3();

  sampleWaterVelocityField(
    state,
    new THREE.Vector3(leftNozzle.x, leftNozzle.y + 0.8, leftNozzle.z),
    velAboveNozzle,
    vort
  );
  sampleWaterVelocityField(
    state,
    new THREE.Vector3(0.4, 0.0, 0.1),
    velOppositeSide,
    vort
  );

  assert.ok(velAboveNozzle.y > 2.5, `Expected strong upward jet above nozzle, got vy=${velAboveNozzle.y}`);
  assert.ok(
    velOppositeSide.length() > 0.05,
    'Whole-tank circulation affects water velocity even in the middle/opposite side of the tank'
  );

  // Release button: suction must draw water into the separate bottom-center return intake vent (x = 0.0),
  // and NEVER suck downward into the jet nozzle itself!
  state.leftPump.jetStrength = 0;
  state.leftPump.globalCirculation = 0;
  setPumpInput(state, 'left', false);
  stepWaterSimulation(state, 1 / 60);

  assert.ok(state.leftPump.suctionStrength > 0.2, 'Releasing pump activates bellows suction strength');
  const velAtCenterReturnVent = new THREE.Vector3();
  sampleWaterVelocityField(
    state,
    new THREE.Vector3(0.0, -state.tank.halfHeight + 0.42, 0.0),
    velAtCenterReturnVent,
    vort
  );
  assert.ok(
    velAtCenterReturnVent.y < -0.15,
    `Expected downward intake suction flow at separate bottom-center return vent (x=0), got vy=${velAtCenterReturnVent.y}`
  );
});

test('mobile multi-force pump presets (light, medium, heavy) produce distinct water jet heights', () => {
  const heights: Record<'light' | 'medium' | 'heavy', number> = {
    light: 0,
    medium: 0,
    heavy: 0,
  };

  for (const preset of ['light', 'medium', 'heavy'] as const) {
    const state = createInitialSimulationState(1);
    const leftNozzle = state.nozzles[0];
    state.rings[0].position.set(leftNozzle.x, leftNozzle.y + 0.35, leftNozzle.z);
    state.rings[0].velocity.set(0, 0, 0);

    setPumpInput(state, 'left', true, preset);
    for (let i = 0; i < 20; i++) {
      stepWaterSimulation(state, 1 / 60);
    }
    heights[preset] = state.rings[0].position.y;
  }

  assert.ok(
    heights.light < heights.medium && heights.medium < heights.heavy,
    `Expected light (${heights.light}) < medium (${heights.medium}) < heavy (${heights.heavy})`
  );
});

test('3D torus-to-peg collision catches ring on peg cup and increments score, and tilting upside down slides it off', () => {
  const state = createInitialSimulationState(2);
  const leftPeg = state.pegs[0];

  state.rings[0].position.set(leftPeg.baseX, leftPeg.baseY + leftPeg.height + 0.15, leftPeg.baseZ);
  state.rings[0].quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  state.rings[0].velocity.set(0, -0.5, 0);

  state.rings[1].position.set(1.2, -1.8, 0);

  for (let i = 0; i < 90; i++) {
    stepWaterSimulation(state, 1 / 60);
  }

  const scoreHooked = evaluateHookedRings(state);
  assert.equal(scoreHooked, 1, 'Ring should be hooked on left peg');
  assert.equal(state.rings[0].hookedPegId, 'peg-left');
  assert.ok(
    state.rings[0].position.y >= leftPeg.baseY + leftPeg.holderCupY - 0.12,
    'Hooked ring rests on peg holder cup flange rather than falling to floor'
  );

  setGravityFromTilt(state, (150 * Math.PI) / 180, 0);
  for (let i = 0; i < 120; i++) {
    stepWaterSimulation(state, 1 / 60);
  }

  assert.equal(state.score, 0, 'Tilting device upside down slides the ring off the peg');
});

test('Touch-duration force ramping, finite-stroke jet cutoff, anti-tunneling needle shaft collision, and SWE waves', () => {
  // 1. Compare short touch (3 frames) vs full stroke hold (18 frames ramping 0.54 -> 1.12)
  const shortTapState = createInitialSimulationState(1);
  const leftNoz = shortTapState.nozzles[0];
  shortTapState.rings[0].position.set(leftNoz.x, leftNoz.y + 0.35, leftNoz.z);
  shortTapState.rings[0].velocity.set(0, 0, 0);
  setPumpInput(shortTapState, 'left', true, 0.54);
  for (let f = 0; f < 3; f++) {
    stepWaterSimulation(shortTapState, 1 / 60);
  }
  setPumpInput(shortTapState, 'left', false);
  for (let f = 0; f < 21; f++) {
    stepWaterSimulation(shortTapState, 1 / 60);
  }

  const longHoldState = createInitialSimulationState(1);
  longHoldState.rings[0].position.set(leftNoz.x, leftNoz.y + 0.35, leftNoz.z);
  longHoldState.rings[0].velocity.set(0, 0, 0);
  for (let f = 0; f < 24; f++) {
    const holdSec = f / 60;
    const force = Math.min(1.12, 0.54 + (holdSec / 0.24) * 0.58);
    setPumpInput(longHoldState, 'left', true, force);
    stepWaterSimulation(longHoldState, 1 / 60);
  }

  assert.ok(
    longHoldState.rings[0].position.y > shortTapState.rings[0].position.y + 0.35,
    `Expected longer touch hold (${longHoldState.rings[0].position.y}) to propel ring significantly higher than short tap (${shortTapState.rings[0].position.y})`
  );
  assert.ok(
    longHoldState.rings[0].position.y > 0.2,
    `Expected full stroke to launch ring high into upper chamber (> 0.2), got y=${longHoldState.rings[0].position.y}`
  );

  // 2. Verify finite bellows stroke: continuing to hold the button down stops spraying water!
  for (let f = 0; f < 90; f++) {
    setPumpInput(longHoldState, 'left', true, 1.12);
    stepWaterSimulation(longHoldState, 1 / 60);
  }
  assert.ok(
    longHoldState.leftPump.jetStrength < 0.05,
    `Expected held button to stop spraying once bellows bottoms out, got jetStrength=${longHoldState.leftPump.jetStrength}`
  );

  // 3. Verify unhooked ring cannot penetrate sideways through the middle of the needle shaft
  const needleTestState = createInitialSimulationState(1);
  needleTestState.gravityMagnitude = 0;
  const leftPeg = needleTestState.pegs[0];
  const testRing = needleTestState.rings[0];
  testRing.hookedPegId = null;
  testRing.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)); // horizontal ring
  testRing.position.set(leftPeg.baseX - 0.32, leftPeg.baseY + leftPeg.height * 0.5, leftPeg.baseZ);
  testRing.velocity.set(3.2, 0, 0); // flying fast straight toward the middle of the needle shaft
  for (let f = 0; f < 20; f++) {
    stepWaterSimulation(needleTestState, 1 / 60);
  }
  assert.ok(
    testRing.position.x < leftPeg.baseX,
    `Unhooked ring hitting needle shaft from the left must bounce off and NEVER penetrate to +X side (got x=${testRing.position.x}, pegX=${leftPeg.baseX})`
  );

  // 4. Verify minimal bubble pool and SWE waves
  let maxWaveAbs = 0;
  for (let i = 0; i < longHoldState.surfaceWaveHeight.length; i++) {
    maxWaveAbs = Math.max(maxWaveAbs, Math.abs(longHoldState.surfaceWaveHeight[i]));
  }
  assert.ok(maxWaveAbs > 1e-4, `Expected 2D Shallow-Water Wave PDE free surface to ripple, maxWaveAbs=${maxWaveAbs}`);
});

test('Incremental Potential Contact (IPC C2 log-barrier) + Gauss Linking Integral: tilted and stacked rings entering needle NEVER jam in collision box', () => {
  // 1. Verify C^2 smooth IPC logarithmic barrier gradient & Hessian properties
  assert.equal(computeIPCBarrierGradient(0.04, 0.035), 0.0, 'Zero barrier force outside dHat');
  assert.equal(computeIPCBarrierHessian(0.04, 0.035), 0.0, 'Zero barrier Hessian outside dHat');
  const gFar = computeIPCBarrierGradient(0.025, 0.035);
  const gNear = computeIPCBarrierGradient(0.005, 0.035);
  const hFar = computeIPCBarrierHessian(0.025, 0.035);
  const hNear = computeIPCBarrierHessian(0.005, 0.035);
  assert.ok(gNear > gFar && gFar > 0, `Expected IPC barrier force to grow smoothly as d -> 0+ (${gNear} > ${gFar})`);
  assert.ok(hNear > hFar && hFar > 0, `Expected IPC barrier Hessian to grow smoothly as d -> 0+ (${hNear} > ${hFar})`);

  // 2. Test tilted rings (15 deg, 35 deg, 48 deg tilt) entering the needle tip:
  // None of them may get stuck/jammed mid-shaft ("环进入之后不可以卡碰撞箱") — all must slide all the way down to the holder cup!
  for (const tiltDeg of [15, 35, 48]) {
    const state = createInitialSimulationState(1);
    const peg = state.pegs[0];
    const cupY = peg.baseY + peg.holderCupY;
    const pegTopY = peg.baseY + peg.height;
    const ring = state.rings[0];

    // Position the tilted ring right at the needle tip, slightly off-center
    ring.position.set(peg.baseX + 0.03, pegTopY + 0.04, peg.baseZ - 0.02);
    ring.quaternion.setFromEuler(
      new THREE.Euler(Math.PI / 2 + THREE.MathUtils.degToRad(tiltDeg), 0, THREE.MathUtils.degToRad(tiltDeg * 0.6))
    );
    ring.velocity.set(0, -0.4, 0);

    for (let f = 0; f < 95; f++) {
      stepWaterSimulation(state, 1 / 60);
    }

    assert.equal(ring.hookedPegId, peg.id, `Tilted ring (${tiltDeg} deg) should be hooked on peg`);
    const linking = computeGaussLinkingNumberRingPeg(ring, peg);
    assert.ok(linking > 0.25, `Expected positive Gauss linking number around peg, got ${linking}`);

    // Verify ring slid all the way down to the bottom holder cup and did NOT jam mid-shaft!
    const expectedRestY = cupY + ring.minorRadius + 0.012;
    assert.ok(
      Math.abs(ring.position.y - expectedRestY) < 0.06,
      `Tilted ring (${tiltDeg} deg) jammed on needle shaft at y=${ring.position.y} instead of reaching cup y=${expectedRestY}`
    );
  }

  // 3. Test 4 rings entering the SAME needle consecutively (coaxial stacking without jamming!)
  const stackState = createInitialSimulationState(4);
  const leftPeg = stackState.pegs[0];
  const cupY = leftPeg.baseY + leftPeg.holderCupY;
  const pegTopY = leftPeg.baseY + leftPeg.height;

  for (let idx = 0; idx < 4; idx++) {
    const r = stackState.rings[idx];
    r.position.set(leftPeg.baseX + (idx % 2 === 0 ? 0.02 : -0.02), pegTopY - 0.05 - idx * 0.22, leftPeg.baseZ);
    r.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 + 0.35 * (idx % 2 === 0 ? 1 : -1), 0, 0.25));
    r.velocity.set(0, -0.5, 0);
  }

  for (let f = 0; f < 120; f++) {
    stepWaterSimulation(stackState, 1 / 60);
  }

  assert.equal(stackState.score, 4, 'All 4 rings should be hooked on the left peg');
  const sortedY = stackState.rings.map((r) => r.position.y).sort((a, b) => a - b);
  const bottomExpectedY = cupY + stackState.rings[0].minorRadius + 0.012;
  assert.ok(
    Math.abs(sortedY[0] - bottomExpectedY) < 0.06,
    `Bottom ring in 4-ring stack should rest on holder cup (${bottomExpectedY}), got ${sortedY[0]}`
  );
  for (let idx = 1; idx < 4; idx++) {
    const gap = sortedY[idx] - sortedY[idx - 1];
    assert.ok(
      gap > 0.11 && gap < 0.18,
      `Expected clean coaxial ring stack spacing (~0.136) between ring ${idx - 1} and ${idx}, got gap=${gap}`
    );
  }
});

test('Singularity friction deceleration (奇点摩擦力减速), zero position teleportation (不瞬移), silent ring insertion, and localStorage persistence', () => {
  // 1. Verify regularized IPC barrier remains bounded even at d = 0 (contact singularity)
  const gZero = computeIPCBarrierGradient(0.0, 0.035, 1.0);
  assert.ok(
    Number.isFinite(gZero) && gZero > 0 && gZero <= 32.0,
    `Expected finite regularized IPC gradient at d=0 singularity, got ${gZero}`
  );

  // 2. Verify a ring passing across the torus medial-axis singularity at the needle tip
  // decelerates smoothly via singularity friction and NEVER teleports (per-frame ||dx|| < 0.12)
  const singState = createInitialSimulationState(1);
  singState.gravityMagnitude = 0;
  const peg = singState.pegs[0];
  const pegTopY = peg.baseY + peg.height;
  const ring = singState.rings[0];

  // Place ring edge-on / tilted right at the needle tip singularity (inPlaneAxisDist ~ 0)
  ring.hookedPegId = null;
  ring.quaternion.identity(); // vertical ring so it cannot hook, testing unhooked singularity contact
  ring.position.set(peg.baseX + 0.01, pegTopY - 0.04, peg.baseZ);
  ring.velocity.set(1.2, -0.3, 0.4);
  ring.angularVelocity.set(4.0, -3.0, 2.5);

  const initialHorizSpeed = Math.hypot(ring.velocity.x, ring.velocity.z);
  const initialAngSpeed = ring.angularVelocity.length();
  const prevPos = ring.position.clone();

  for (let f = 0; f < 15; f++) {
    stepWaterSimulation(singState, 1 / 60);
    const frameJump = ring.position.distanceTo(prevPos);
    assert.ok(
      frameJump < 0.12,
      `Ring must NEVER teleport under singularity forces (frame ${f} jumped by ${frameJump}m)`
    );
    prevPos.copy(ring.position);
  }

  const finalHorizSpeed = Math.hypot(ring.velocity.x, ring.velocity.z);
  const finalAngSpeed = ring.angularVelocity.length();
  assert.ok(
    finalHorizSpeed < initialHorizSpeed * 0.75 && finalAngSpeed < initialAngSpeed * 0.75,
    `Expected singularity friction to decelerate linear (${finalHorizSpeed} < ${initialHorizSpeed}) and angular (${finalAngSpeed} < ${initialAngSpeed}) speeds`
  );

  // 3. Verify localStorage game state save & restore round-trip
  const storeMap = new Map<string, string>();
  const prevWindow = (globalThis as unknown as { window?: unknown }).window;
  (globalThis as unknown as { window: unknown }).window = {
    localStorage: {
      getItem: (k: string) => storeMap.get(k) ?? null,
      setItem: (k: string, v: string) => {
        storeMap.set(k, v);
      },
    },
  };

  try {
    const stateToSave = createInitialSimulationState(3);
    stateToSave.rings[0].position.set(-0.55, -0.42, 0.02);
    stateToSave.rings[0].velocity.set(0.15, -0.25, 0.05);
    stateToSave.rings[0].hookedPegId = 'peg-left';
    setGravityFromTilt(stateToSave, 0.22, -0.14);
    evaluateHookedRings(stateToSave);

    const savedOk = saveSimulationStateToStorage(stateToSave, 'test_waterful_key');
    assert.equal(savedOk, true, 'saveSimulationStateToStorage should succeed');

    const restoredState = createInitialSimulationState(3);
    const loadedOk = loadSimulationStateFromStorage(restoredState, 'test_waterful_key');
    assert.equal(loadedOk, true, 'loadSimulationStateFromStorage should succeed');
    assert.equal(restoredState.rings[0].hookedPegId, 'peg-left');
    assert.equal(restoredState.score, 1);
    assert.ok(Math.abs(restoredState.rings[0].position.x - -0.55) < 1e-5);
    assert.ok(Math.abs(restoredState.tiltRoll - 0.22) < 1e-5);
  } finally {
    (globalThis as unknown as { window?: unknown }).window = prevWindow;
  }

  // 4. Verify playRingHooked is a silent no-op ("插入的时候不要有任何音效")
  assert.doesNotThrow(() => {
    waterAudio.playRingHooked(1);
    waterAudio.playVictoryFanfare();
  });
});

test('Fluttering + Tumbling (飘摆 + 翻滚) falling hydrodynamics, exact 3D holder tray disk collision (托盘碰撞箱), and ultra-gentle tap force (最最轻击)', () => {
  // 1. Verify falling ring exhibits true Tumbling (翻滚: flips over around horizontal axis while descending) + Fluttering lateral drift
  const fallState = createInitialSimulationState(1);
  // Move pegs out of the way so we test pure free-water settling hydrodynamics from top to bottom
  fallState.pegs = [];
  const ring = fallState.rings[0];
  ring.position.set(0.0, 1.65, 0.0);
  // Start tilted at 28 degrees from horizontal with a small initial glide velocity
  ring.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 - 0.48, 0, 0.32));
  ring.velocity.set(0.35, -0.2, 0.0);
  ring.angularVelocity.set(0.0, 0.0, 1.2);

  let totalHorizontalSpinRad = 0;
  let minNy = 1.0;
  let maxNy = -1.0;
  let maxLateralDrift = 0;
  const normalVec = new THREE.Vector3();

  for (let f = 0; f < 130; f++) {
    const dt = 1 / 60;
    stepWaterSimulation(fallState, dt);
    const omegaHoriz = Math.hypot(ring.angularVelocity.x, ring.angularVelocity.z);
    totalHorizontalSpinRad += omegaHoriz * dt;

    normalVec.set(0, 0, 1).applyQuaternion(ring.quaternion).normalize();
    minNy = Math.min(minNy, normalVec.y);
    maxNy = Math.max(maxNy, normalVec.y);
    maxLateralDrift = Math.max(maxLateralDrift, Math.hypot(ring.position.x, ring.position.z));
  }

  assert.ok(
    totalHorizontalSpinRad > Math.PI * 1.15 && minNy * maxNy < 0,
    `Expected falling ring to tumble/flip over its horizontal axis (totalHorizontalSpinRad=${totalHorizontalSpinRad.toFixed(2)} rad, minNy=${minNy.toFixed(2)}, maxNy=${maxNy.toFixed(2)})`
  );
  assert.ok(
    maxLateralDrift > 0.18,
    `Expected falling ring to flutter/drift laterally while tumbling down (maxLateralDrift=${maxLateralDrift.toFixed(3)})`
  );

  // 2. Verify exact 3D Circle-to-Disk Holder Tray (`两根针的托盘`) side contact deflects horizontally without vertical box pop
  const trayState = createInitialSimulationState(1);
  trayState.gravityMagnitude = 0;
  const leftPeg = trayState.pegs[0];
  const cupDiskY = leftPeg.baseY + leftPeg.holderCupY - 0.012;
  const trayRing = trayState.rings[0];
  trayRing.hookedPegId = null;
  trayRing.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)); // horizontal ring approaching tray rim from the left
  const startX =
    leftPeg.baseX - (leftPeg.holderCupRadius + trayRing.majorRadius + trayRing.minorRadius + 0.04);
  trayRing.position.set(startX, cupDiskY, leftPeg.baseZ);
  trayRing.velocity.set(1.8, 0, 0);

  for (let f = 0; f < 15; f++) {
    stepWaterSimulation(trayState, 1 / 60);
  }

  assert.equal(
    trayRing.hookedPegId,
    null,
    'Unhooked ring hitting the side/underside of the holder tray must NEVER be falsely marked as hooked'
  );
  assert.ok(
    trayRing.velocity.x < 0.2 && Math.abs(trayRing.velocity.y) < 0.35,
    `Ring hitting side rim of holder tray should be deflected radially without a huge vertical box pop (vx=${trayRing.velocity.x}, vy=${trayRing.velocity.y})`
  );

  // 3. Verify ultra-gentle minimum tap (`最最轻击` force = 0.18 for 2 frames) produces a subtle, gentle lift (< 0.25m)
  const gentleTapState = createInitialSimulationState(1);
  const leftNoz = gentleTapState.nozzles[0];
  const tapRing = gentleTapState.rings[0];
  const initialY = leftNoz.y + 0.28;
  tapRing.position.set(leftNoz.x, initialY, leftNoz.z);
  tapRing.velocity.set(0, 0, 0);

  setPumpInput(gentleTapState, 'left', true, 0.18);
  stepWaterSimulation(gentleTapState, 1 / 60);
  stepWaterSimulation(gentleTapState, 1 / 60);
  setPumpInput(gentleTapState, 'left', false);

  let peakY = initialY;
  for (let f = 0; f < 25; f++) {
    stepWaterSimulation(gentleTapState, 1 / 60);
    peakY = Math.max(peakY, tapRing.position.y);
  }
  const gentleLift = peakY - initialY;
  assert.ok(
    gentleLift > 0.005 && gentleLift < 0.26,
    `Expected ultra-light tap (0.18) to produce a delicate, gentle lift in (0.005, 0.26), got ${gentleLift}`
  );

  // 4. Verify ZERO artificial coaxial leveling torque on a hooked ring in mid-air within aperture clearance,
  // and real 3D contact-driven leveling (r x J) when resting on the tapered shaft & holder tray
  const hookedState = createInitialSimulationState(1);
  hookedState.gravityMagnitude = 0;
  hookedState.config.turbulenceStrength = 0;
  const hPeg = hookedState.pegs[0];
  const hRing = hookedState.rings[0];
  hRing.hookedPegId = hPeg.id;
  // Place ring centered on the thin upper needle shaft, tilted by 20 degrees (well within hole clearance R*cos(20)-r = 0.115 > r_peg + ipcDHat)
  hRing.position.set(hPeg.baseX, hPeg.baseY + hPeg.height * 0.78, hPeg.baseZ);
  hRing.velocity.set(0, 0, 0);
  hRing.angularVelocity.set(0, 0, 0);
  hRing.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 - 0.35, 0, 0));

  stepWaterSimulation(hookedState, 1 / 60);
  assert.ok(
    Math.hypot(hRing.angularVelocity.x, hRing.angularVelocity.z) < 1e-4,
    `Hooked ring in mid-air within hole clearance must NOT experience any artificial leveling torque (got omega=${Math.hypot(hRing.angularVelocity.x, hRing.angularVelocity.z)})`
  );

  // Now turn on gravity so the tilted hooked ring slides down the tapered shaft onto the holder tray and levels via real 3D contact (r x J)
  hookedState.gravityMagnitude = 9.81;
  for (let f = 0; f < 90; f++) {
    stepWaterSimulation(hookedState, 1 / 60);
  }
  normalVec.set(0, 0, 1).applyQuaternion(hRing.quaternion).normalize();
  assert.ok(
    Math.abs(normalVec.y) > 0.94,
    `Hooked ring resting on holder tray should level naturally via 3D contact torques (r x J), got |n.y|=${Math.abs(normalVec.y)}`
  );
});

test('Needle Tip (尖尖头) Exhaustive Continuity & Anti-Teleportation Stress Test across offsets and tilts', () => {
  // 1. Verify ZERO mid-air repulsion when ring tube is 1.2cm (0.012m > ipcDHat=0.008m) above the needle tip
  const airState = createInitialSimulationState(1);
  airState.gravityMagnitude = 0;
  airState.config.turbulenceStrength = 0;
  const peg = airState.pegs[0];
  const pegTopY = peg.baseY + peg.height;
  const r0 = airState.rings[0];
  r0.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  // Place ring tube at x = peg.baseX + R, y = pegTopY + r + peg.tipRadius + 0.012 (1.2cm clear of tip)
  r0.position.set(peg.baseX + r0.majorRadius, pegTopY + r0.minorRadius + peg.tipRadius + 0.012, peg.baseZ);
  r0.velocity.set(0, 0, 0);
  r0.angularVelocity.set(0, 0, 0);
  stepWaterSimulation(airState, 1 / 60);
  assert.ok(
    r0.velocity.length() < 1e-4 && r0.angularVelocity.length() < 1e-4,
    `Ring 1.2cm outside needle tip surface must NOT hit any invisible mid-air force bubble (vel=${r0.velocity.length()})`
  );

  // 2. Sweep across critical radial offsets (straddling tip, hitting tip head-on, grazing outside/inside) and tilt angles
  const offsets = [0.0, 0.05, 0.11, 0.145, 0.165, 0.185, 0.21];
  const tilts = [0, 22, 48, 68, 82];

  for (const offset of offsets) {
    for (const tiltDeg of tilts) {
      const tipState = createInitialSimulationState(1);
      const tPeg = tipState.pegs[0];
      const tTopY = tPeg.baseY + tPeg.height;
      const ring = tipState.rings[0];

      const tiltRad = THREE.MathUtils.degToRad(tiltDeg);
      ring.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 - tiltRad, 0, 0.2));
      ring.position.set(tPeg.baseX + offset, tTopY + 0.09, tPeg.baseZ + 0.01);
      ring.velocity.set(-0.25, -0.85, 0.1);
      ring.angularVelocity.set(0.8, 0.2, -0.6);

      const prevPos = ring.position.clone();
      for (let f = 0; f < 45; f++) {
        stepWaterSimulation(tipState, 1 / 60);
        const frameDelta = ring.position.distanceTo(prevPos);
        const speed = ring.velocity.length();
        assert.ok(
          frameDelta < 0.055,
          `Needle tip teleportation detected at offset=${offset}, tilt=${tiltDeg}deg, frame=${f}: frameDelta=${frameDelta.toFixed(4)}m`
        );
        assert.ok(
          speed < 3.2,
          `Non-physical velocity spike at needle tip (offset=${offset}, tilt=${tiltDeg}deg, frame=${f}): speed=${speed.toFixed(3)}m/s`
        );
        prevPos.copy(ring.position);
      }
    }
  }

  // 3. Verify 100% Unified Non-Piecewise Physics ("不要分段物理，统一"):
  // Calling `resolveRingTankAndPegCollisionsExact` and `resolveRingToRingCollisionsExact3D`
  // must produce bit-for-bit identical positions, velocities, and angular velocities regardless of `hookedPegId`.
  const stateA = createInitialSimulationState(2);
  const stateB = createInitialSimulationState(2);
  const pegA = stateA.pegs[0];
  for (const [idx, s] of [stateA, stateB].entries()) {
    s.rings[0].position.set(pegA.baseX + 0.03, pegA.baseY + pegA.holderCupY + 0.08, pegA.baseZ);
    s.rings[0].velocity.set(-0.3, -0.4, 0.12);
    s.rings[0].quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 - 0.35, 0.1, 0.2));
    s.rings[0].angularVelocity.set(1.1, -0.4, 0.7);
    s.rings[0].hookedPegId = idx === 0 ? pegA.id : null;

    s.rings[1].position.set(pegA.baseX + 0.02, pegA.baseY + pegA.holderCupY + 0.19, pegA.baseZ);
    s.rings[1].velocity.set(0.2, -0.5, -0.1);
    s.rings[1].quaternion.setFromEuler(new THREE.Euler(Math.PI / 2 + 0.25, -0.1, -0.15));
    s.rings[1].angularVelocity.set(-0.8, 0.3, -0.5);
    s.rings[1].hookedPegId = idx === 0 ? pegA.id : null;
  }

  resolveRingToRingCollisionsExact3D(stateA.rings);
  resolveRingToRingCollisionsExact3D(stateB.rings);
  resolveRingTankAndPegCollisionsExact(stateA, stateA.rings[0], 1 / 1440);
  resolveRingTankAndPegCollisionsExact(stateB, stateB.rings[0], 1 / 1440);

  assert.equal(
    stateA.rings[0].position.distanceTo(stateB.rings[0].position),
    0,
    'Unified collision solver must produce identical position regardless of hookedPegId'
  );
  assert.equal(
    stateA.rings[0].velocity.distanceTo(stateB.rings[0].velocity),
    0,
    'Unified collision solver must produce identical velocity regardless of hookedPegId'
  );
  assert.equal(
    stateA.rings[0].angularVelocity.distanceTo(stateB.rings[0].angularVelocity),
    0,
    'Unified collision solver must produce identical angularVelocity regardless of hookedPegId'
  );
});
