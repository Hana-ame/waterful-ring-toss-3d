import * as THREE from 'three';
import {
  SimulationState,
  setGravityFromTilt,
  setGyroscopeAngularVelocity,
} from './waterPhysics';

/**
 * Hardware Mobile Sensor Bridge (`sensorBridge.ts`)
 * Connects:
 * 1. `deviceorientation` (beta, gamma, alpha) using the exact W3C 3D gravity vector projection
 *    onto the screen plane across Portrait (0°), Landscape-Left (90°), Landscape-Right (-90° / 270°),
 *    and Upside-Down (180°), accurately handling inverted / head-down phone orientations (`手机头向下`).
 * 2. `devicemotion` (`rotationRate.alpha`, `rotationRate.beta`, `rotationRate.gamma`) for real-time gyroscope angular velocity
 *    so shaking or tilting the phone drives whole-tank water inertial sloshing (`waterAngularVel`).
 * 3. iOS 13+ `DeviceOrientationEvent.requestPermission()` & `DeviceMotionEvent.requestPermission()` handshake.
 */

export interface SensorBridgeController {
  requestPermissionAndStart: () => Promise<boolean>;
  stop: () => void;
  isActive: () => boolean;
}

function getScreenOrientationAngle(): number {
  if (typeof window === 'undefined') return 0;
  if (window.screen?.orientation && typeof window.screen.orientation.angle === 'number') {
    return window.screen.orientation.angle;
  }
  const legacyAngle = (window as unknown as { orientation?: number }).orientation;
  return typeof legacyAngle === 'number' ? legacyAngle : 0;
}

/**
 * Maps raw W3C DeviceOrientation (beta, gamma) in degrees into console-local (rollRad, pitchRad)
 * using the exact 3D gravity vector projection onto the device screen frame:
 * - In W3C DeviceOrientation:
 *   - `beta` in [-180°, 180°] is front-to-back tilt around +X_device.
 *     When `beta > 0°`, the phone's top/head (+Y_device) points UP above the horizon (`g_y_dev = -sin(beta) < 0`).
 *     When `beta < 0°` (`手机头向下`), the phone's top/head (+Y_device) points DOWN below the horizon (`g_y_dev = -sin(beta) > 0`)!
 *   - `gamma` in [-90°, 90°] is side-to-side tilt around +Y_device.
 *     When `|beta| > 90°` (screen facing downward toward the ground), W3C `gamma` inverts sign; multiplying by
 *     the regularized `cos(beta)` sign keeps lateral gravity `g_x_dev` strictly continuous across vertical!
 */
export function mapDeviceOrientationToTiltRadians(
  betaDeg: number,
  gammaDeg: number,
  screenAngleDeg = 0
): { rollRad: number; pitchRad: number } {
  const betaRad = THREE.MathUtils.degToRad(betaDeg);
  const gammaRad = THREE.MathUtils.degToRad(gammaDeg);

  const sinB = Math.sin(betaRad);
  const cosB = Math.cos(betaRad);
  const sinG = Math.sin(gammaRad);
  const cosG = Math.cos(gammaRad);

  // Regularized cos(beta) factor preserves continuous lateral sign when pitching past ±90°
  // without attenuating lateral roll sensitivity near upright handheld angles (beta ~ 60°..85°)
  const cosBSign = cosB / Math.sqrt(cosB * cosB + 0.12 * sinB * sinB + 1e-8);

  // 3D gravity components in hardware device frame (+X right, +Y top/head `手机头`, +Z out of screen)
  const gxDev = sinG * cosBSign;
  const gyDev = -sinB;

  let gxScreen = gxDev;
  let gyScreen = gyDev;
  let pitchDeg = (betaDeg >= 0 ? betaDeg - 60 : -betaDeg - 60) * (cosG >= 0 ? 1 : -1);

  if (screenAngleDeg === 90) {
    // Landscape primary: phone rotated 90° CCW (+Y_device is on the LEFT of the screen)
    gxScreen = gyDev;
    gyScreen = -gxDev;
    pitchDeg = gammaDeg - 60;
  } else if (screenAngleDeg === -90 || screenAngleDeg === 270) {
    // Landscape secondary: phone rotated 90° CW (+Y_device is on the RIGHT of the screen)
    gxScreen = -gyDev;
    gyScreen = gxDev;
    pitchDeg = -(gammaDeg + 60);
  } else if (screenAngleDeg === 180) {
    // Upside-down portrait
    gxScreen = -gxDev;
    gyScreen = -gyDev;
    pitchDeg = -(betaDeg - 60);
  }

  // Exact full-circle [-π, π] in-screen gravity angle measured from -Y_screen:
  // - When gyScreen < 0 (phone head UP): |rollRad| < π/2 -> gravity pulls toward -Y (bottom of tank)
  // - When gyScreen > 0 (phone head DOWN `手机头向下`): |rollRad| > π/2 (up to ±π) -> gravity pulls toward +Y (top of tank)!
  const rollRad =
    Math.abs(gxScreen) < 1e-6 && Math.abs(gyScreen) < 1e-6
      ? 0
      : Math.atan2(gxScreen, -gyScreen);

  const pitchRad = THREE.MathUtils.degToRad(THREE.MathUtils.clamp(pitchDeg, -55, 55));

  return {
    rollRad,
    pitchRad,
  };
}

export function attachDeviceSensorBridge(
  getState: () => SimulationState | null,
  onTiltUpdate?: (rollDeg: number, pitchDeg: number) => void
): SensorBridgeController {
  let active = false;

  const onOrientation = (e: DeviceOrientationEvent) => {
    if (e.beta === null && e.gamma === null) return;
    const state = getState();
    if (!state) return;

    const screenAngle = getScreenOrientationAngle();
    const { rollRad, pitchRad } = mapDeviceOrientationToTiltRadians(
      e.beta ?? 60,
      e.gamma ?? 0,
      screenAngle
    );
    setGravityFromTilt(state, rollRad, pitchRad);
    active = true;
    if (onTiltUpdate) {
      onTiltUpdate(
        Math.round(THREE.MathUtils.radToDeg(state.tiltRoll)),
        Math.round(THREE.MathUtils.radToDeg(state.tiltPitch))
      );
    }
  };

  const onMotion = (e: DeviceMotionEvent) => {
    const state = getState();
    if (!state || !e.rotationRate) return;

    // e.rotationRate: alpha (Z deg/s), beta (X deg/s), gamma (Y deg/s)
    const screenAngle = getScreenOrientationAngle();
    let ratePitchDeg = e.rotationRate.beta || 0;
    let rateRollDeg = e.rotationRate.gamma || 0;
    const rateYawDeg = e.rotationRate.alpha || 0;

    if (screenAngle === 90) {
      ratePitchDeg = e.rotationRate.gamma || 0;
      rateRollDeg = -(e.rotationRate.beta || 0);
    } else if (screenAngle === -90 || screenAngle === 270) {
      ratePitchDeg = -(e.rotationRate.gamma || 0);
      rateRollDeg = e.rotationRate.beta || 0;
    } else if (screenAngle === 180) {
      ratePitchDeg = -(e.rotationRate.beta || 0);
      rateRollDeg = -(e.rotationRate.gamma || 0);
    }

    setGyroscopeAngularVelocity(
      state,
      THREE.MathUtils.degToRad(ratePitchDeg),
      THREE.MathUtils.degToRad(rateRollDeg),
      THREE.MathUtils.degToRad(rateYawDeg)
    );
    active = true;
  };

  const requestPermissionAndStart = async (): Promise<boolean> => {
    if (typeof window === 'undefined') return false;

    try {
      const DevOrient = (window as unknown as {
        DeviceOrientationEvent?: { requestPermission?: () => Promise<'granted' | 'denied'> };
      }).DeviceOrientationEvent;
      if (DevOrient && typeof DevOrient.requestPermission === 'function') {
        const res = await DevOrient.requestPermission();
        if (res !== 'granted') return false;
      }

      const DevMotion = (window as unknown as {
        DeviceMotionEvent?: { requestPermission?: () => Promise<'granted' | 'denied'> };
      }).DeviceMotionEvent;
      if (DevMotion && typeof DevMotion.requestPermission === 'function') {
        await DevMotion.requestPermission().catch(() => 'denied');
      }

      window.addEventListener('deviceorientation', onOrientation);
      window.addEventListener('devicemotion', onMotion);
      active = true;
      return true;
    } catch {
      return false;
    }
  };

  const stop = () => {
    if (typeof window === 'undefined') return;
    window.removeEventListener('deviceorientation', onOrientation);
    window.removeEventListener('devicemotion', onMotion);
    active = false;
  };

  // Auto-bind non-iOS listeners passively
  if (typeof window !== 'undefined') {
    window.addEventListener('deviceorientation', onOrientation);
    window.addEventListener('devicemotion', onMotion);
  }

  return {
    requestPermissionAndStart,
    stop,
    isActive: () => active,
  };
}
