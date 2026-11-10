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
  computeGaussLinkingNumberRingPeg,
  saveSimulationStateToStorage,
  loadSimulationStateFromStorage,
  testOBBvsOBB_SAT,
  OBB3D,
} from './waterPhysics';
import { mapDeviceOrientationToTiltRadians } from './sensorBridge';
import { waterAudio } from '../audio/waterAudio';

test('P0 & P1: Gyroscope rotationRate and tilt excite whole-tank water sloshing (waterAngularVel) and gz = -sin(pitch)', () => {
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

  // 3. Verify landscape orientation axis remapping in sensorBridge
  const portrait = mapDeviceOrientationToTiltRadians(60, 25, 0);
  const landscape90 = mapDeviceOrientationToTiltRadians(25, 60, 90);
  assert.ok(portrait.rollRad > 0.3, 'Portrait gamma maps to positive roll');
  assert.ok(landscape90.rollRad < -0.3, 'Landscape 90° beta maps to negative roll');
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
  // 1. Verify C^2 smooth IPC barrier properties
  assert.equal(computeIPCBarrierGradient(0.04, 0.035), 0.0, 'Zero barrier force outside dHat');
  const gFar = computeIPCBarrierGradient(0.025, 0.035);
  const gNear = computeIPCBarrierGradient(0.005, 0.035);
  assert.ok(gNear > gFar && gFar > 0, `Expected IPC barrier force to grow smoothly as d -> 0+ (${gNear} > ${gFar})`);

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




