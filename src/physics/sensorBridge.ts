import * as THREE from 'three';
import {
  SimulationState,
  setGravityFromTilt,
  setGyroscopeAngularVelocity,
} from './waterPhysics';

/**
 * Hardware Mobile Sensor Bridge (`sensorBridge.ts`)
 * Connects:
 * 1. `deviceorientation` (beta, gamma, alpha) with screen orientation (0° / 90° / -90° / 180°) axis remapping
 * 2. `devicemotion` (`rotationRate.alpha`, `rotationRate.beta`, `rotationRate.gamma`) for real-time gyroscope angular velocity
 *    so shaking or tilting the phone drives whole-tank water inertial sloshing (`waterAngularVel`)!
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
 * Maps raw DeviceOrientation (beta, gamma) in degrees into console-local (rollRad, pitchRad)
 * accounting for Portrait (0°), Landscape-Left (90°), Landscape-Right (-90° / 270°), and Upside-Down (180°).
 */
export function mapDeviceOrientationToTiltRadians(
  betaDeg: number,
  gammaDeg: number,
  screenAngleDeg = 0
): { rollRad: number; pitchRad: number } {
  // Neutral handheld viewing angle is ~60° beta in portrait
  let rollDeg = gammaDeg;
  let pitchDeg = betaDeg - 60;

  if (screenAngleDeg === 90) {
    // Landscape primary: phone rotated 90° CCW
    rollDeg = -(betaDeg);
    pitchDeg = gammaDeg - 60;
  } else if (screenAngleDeg === -90 || screenAngleDeg === 270) {
    // Landscape secondary: phone rotated 90° CW
    rollDeg = betaDeg;
    pitchDeg = -(gammaDeg + 60);
  } else if (screenAngleDeg === 180) {
    rollDeg = -gammaDeg;
    pitchDeg = -(betaDeg - 60);
  }

  return {
    rollRad: THREE.MathUtils.degToRad(THREE.MathUtils.clamp(rollDeg, -150, 150)),
    pitchRad: THREE.MathUtils.degToRad(THREE.MathUtils.clamp(pitchDeg, -55, 55)),
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
