import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  createInitialSimulationState,
  loadSimulationStateFromStorage,
  saveSimulationStateToStorage,
  setPumpInput,
  setGravityFromTilt,
  stepWaterSimulation,
  SimulationState,
} from './physics/waterPhysics';
import {
  attachDeviceSensorBridge,
  SensorBridgeController,
} from './physics/sensorBridge';
import {
  buildTomyConsoleRig,
  TomyConsoleRig,
} from './scene/TomyConsoleBuilder';
import { waterAudio } from './audio/waterAudio';

export default function App() {
  const [leftCompression, setLeftCompression] = useState<number>(0);
  const [rightCompression, setRightCompression] = useState<number>(0);

  const canvasMountRef = useRef<HTMLDivElement | null>(null);
  const simStateRef = useRef<SimulationState | null>(null);
  const consoleRigRef = useRef<TomyConsoleRig | null>(null);
  const sensorBridgeRef = useRef<SensorBridgeController | null>(null);
  const sensorRequestedRef = useRef<boolean>(false);

  // Track touch/key press start timestamps so hold duration controls pump force continuously
  const leftPressStartRef = useRef<number | null>(null);
  const rightPressStartRef = useRef<number | null>(null);
  const keysHeldRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.body.dataset.state = 'playing';
    }
  }, []);

  // Wire hardware gyroscope & orientation sensor bridge (deviceorientation + devicemotion rotationRate)
  useEffect(() => {
    const controller = attachDeviceSensorBridge(() => simStateRef.current);
    sensorBridgeRef.current = controller;
    return () => {
      controller.stop();
    };
  }, []);

  const ensureSensorPermissionOnGesture = useCallback(() => {
    if (!sensorRequestedRef.current && sensorBridgeRef.current) {
      sensorRequestedRef.current = true;
      void sensorBridgeRef.current.requestPermissionAndStart();
    }
  }, []);

  // Begin pressing Left or Right pump button (initial gentle puff; holding longer ramps force up to 100%)
  const startPumpHold = useCallback(
    (side: 'left' | 'right') => {
      ensureSensorPermissionOnGesture();
      const sim = simStateRef.current;
      if (!sim) return;

      const ref = side === 'left' ? leftPressStartRef : rightPressStartRef;
      if (ref.current === null) {
        ref.current = performance.now();
        waterAudio.playPumpPress(side);
        setPumpInput(sim, side, true, 0.54);
      }
    },
    [ensureSensorPermissionOnGesture]
  );

  // Release Left or Right pump button (triggers proportional bellows expansion suction backflow)
  const endPumpHold = useCallback((side: 'left' | 'right') => {
    const sim = simStateRef.current;
    const ref = side === 'left' ? leftPressStartRef : rightPressStartRef;
    if (ref.current !== null) {
      ref.current = null;
      if (sim) {
        waterAudio.playPumpRelease();
        setPumpInput(sim, side, false);
        saveSimulationStateToStorage(sim);
      }
    }
  }, []);

  // Global keyboard support (Left: A / ArrowLeft / Z / X, Right: D / ArrowRight / M / Slash, Tilt: Q/E/W/S)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const code = e.code;
      if (keysHeldRef.current.has(code)) return;
      keysHeldRef.current.add(code);

      if (code === 'KeyA' || code === 'ArrowLeft' || code === 'KeyZ' || code === 'KeyX') {
        startPumpHold('left');
      } else if (
        code === 'KeyD' ||
        code === 'ArrowRight' ||
        code === 'KeyN' ||
        code === 'Slash' ||
        code === 'Period'
      ) {
        startPumpHold('right');
      }

      const sim = simStateRef.current;
      if (sim && (code === 'KeyQ' || code === 'KeyE' || code === 'KeyW' || code === 'KeyS')) {
        let rollDeg = THREE.MathUtils.radToDeg(sim.tiltRoll);
        let pitchDeg = THREE.MathUtils.radToDeg(sim.tiltPitch);
        if (code === 'KeyQ') rollDeg = Math.max(-135, rollDeg - 10);
        if (code === 'KeyE') rollDeg = Math.min(135, rollDeg + 10);
        if (code === 'KeyW') pitchDeg = Math.min(45, pitchDeg + 8);
        if (code === 'KeyS') pitchDeg = Math.max(-45, pitchDeg - 8);
        setGravityFromTilt(
          sim,
          THREE.MathUtils.degToRad(rollDeg),
          THREE.MathUtils.degToRad(pitchDeg)
        );
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const code = e.code;
      keysHeldRef.current.delete(code);
      if (code === 'KeyA' || code === 'ArrowLeft' || code === 'KeyZ' || code === 'KeyX') {
        endPumpHold('left');
      } else if (
        code === 'KeyD' ||
        code === 'ArrowRight' ||
        code === 'KeyN' ||
        code === 'Slash' ||
        code === 'Period'
      ) {
        endPumpHold('right');
      }
    };

    const onWindowBlur = () => {
      keysHeldRef.current.clear();
      endPumpHold('left');
      endPumpHold('right');
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWindowBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onWindowBlur);
    };
  }, [startPumpHold, endPumpHold]);

  // Initialize Full-Screen 3D Water Chamber & Hydrodynamic Loop
  useEffect(() => {
    const mountEl = canvasMountRef.current;
    if (!mountEl) return;

    const isSoftwareGl = (() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setSize(
      mountEl.clientWidth || window.innerWidth,
      mountEl.clientHeight || window.innerHeight
    );

    mountEl.innerHTML = '';
    mountEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x041326);
    scene.fog = new THREE.FogExp2(0x041326, 0.025);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    // Initialize 3D Water Physics State, restore saved state from localStorage if available, & build Scene Rig
    const simState = createInitialSimulationState(12);
    loadSimulationStateFromStorage(simState);
    saveSimulationStateToStorage(simState);
    simStateRef.current = simState;

    const consoleRig = buildTomyConsoleRig(simState, renderer, isSoftwareGl);
    consoleRigRef.current = consoleRig;
    scene.add(consoleRig.rootGroup);

    // Pure Perpendicular / Orthographic Projection Camera (`垂直投影，无透视盒子变形`)
    const camera = new THREE.OrthographicCamera(-2, 2, 2.2, -2.2, 0.1, 100);

    const updateCameraFraming = () => {
      const w = mountEl.clientWidth || window.innerWidth;
      const h = mountEl.clientHeight || window.innerHeight;
      const aspect = w / Math.max(1, h);

      let orthoHalfWidth = 1.8;
      let orthoHalfHeight = 2.25;
      let cameraOffsetY = 0.0;

      if (aspect < 0.98) {
        // Mobile Portrait (`竖屏`): fit the chamber width edge-to-edge and shift view slightly down so thumb buttons sit below the nozzles
        orthoHalfWidth = 1.44;
        orthoHalfHeight = Math.max(2.45, orthoHalfWidth / Math.max(0.36, aspect));
        cameraOffsetY = -0.34;

        simState.tank.halfWidthBottom = 1.34;
        simState.tank.halfWidthTop = 1.38;
        simState.nozzles[0].x = -1.04;
        simState.nozzles[1].x = 1.04;
      } else {
        // Mobile Landscape (`横屏`) & Desktop: fit the full chamber height vertically with thumb buttons on the left and right sides
        orthoHalfHeight = 2.22;
        orthoHalfWidth = orthoHalfHeight * aspect;
        cameraOffsetY = 0.0;

        simState.tank.halfWidthBottom = Math.min(1.58, Math.max(1.36, orthoHalfWidth - 0.55));
        simState.tank.halfWidthTop = Math.min(1.74, Math.max(1.42, orthoHalfWidth - 0.45));
        simState.nozzles[0].x = -Math.min(1.16, simState.tank.halfWidthBottom - 0.34);
        simState.nozzles[1].x = Math.min(1.16, simState.tank.halfWidthBottom - 0.34);
      }

      camera.left = -orthoHalfWidth;
      camera.right = orthoHalfWidth;
      camera.top = orthoHalfHeight;
      camera.bottom = -orthoHalfHeight;
      camera.position.set(0, cameraOffsetY, 10);
      camera.lookAt(0, cameraOffsetY, 0);
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    updateCameraFraming();

    // Underwater 3-Point Lighting Rig
    const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x082f49, 0.95);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.9);
    keyLight.position.set(4.5, 8.0, 7.5);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 25;
    const d = 4.2;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.3);
    rimLight.position.set(-5.5, 4.0, -5.0);
    scene.add(rimLight);

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    const ResizeObs = typeof ResizeObserver !== 'undefined' ? ResizeObserver : null;
    const ro = ResizeObs ? new ResizeObs(updateCameraFraming) : null;
    if (ro) ro.observe(mountEl);
    window.addEventListener('resize', updateCameraFraming);

    const persistNow = () => {
      if (simStateRef.current) {
        saveSimulationStateToStorage(simStateRef.current);
      }
    };
    window.addEventListener('pagehide', persistNow);
    window.addEventListener('beforeunload', persistNow);
    document.addEventListener('visibilitychange', persistNow);

    let frameId = 0;
    let lastTime = performance.now();
    let lastUiSync = 0;
    let lastStorageSave = performance.now();

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      if (!gl || typeof gl.fenceSync !== 'function') {
        frameId = window.setTimeout(cb, 80);
        return;
      }
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function tick() {
      const now = performance.now();
      const rawDt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;

      const activeSim = simStateRef.current;
      if (activeSim) {
        // Touch-duration force control: holding the button compresses the bellows over a finite stroke (0.54 -> 1.12 over 240ms).
        // Once targetDepth reaches 1.12, the bellows bottoms out and water stops spraying even if held!
        if (leftPressStartRef.current !== null) {
          const holdSec = (now - leftPressStartRef.current) / 1000;
          const force = Math.min(1.12, 0.54 + (holdSec / 0.24) * 0.58);
          setPumpInput(activeSim, 'left', true, force);
        }
        if (rightPressStartRef.current !== null) {
          const holdSec = (now - rightPressStartRef.current) / 1000;
          const force = Math.min(1.12, 0.54 + (holdSec / 0.24) * 0.58);
          setPumpInput(activeSim, 'right', true, force);
        }

        stepWaterSimulation(activeSim, rawDt);

        consoleRig.updateFromSimulation(activeSim);

        if (now - lastStorageSave > 350) {
          lastStorageSave = now;
          saveSimulationStateToStorage(activeSim);
        }

        if (now - lastUiSync > 45) {
          lastUiSync = now;
          setLeftCompression(Math.min(1, activeSim.leftPump.depressed / 1.15));
          setRightCompression(Math.min(1, activeSim.rightPump.depressed / 1.15));
        }
      }

      renderer.render(scene, camera);

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    }

    frameId = requestAnimationFrame(tick);

    return () => {
      persistNow();
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      if (ro) ro.disconnect();
      window.removeEventListener('resize', updateCameraFraming);
      window.removeEventListener('pagehide', persistNow);
      window.removeEventListener('beforeunload', persistNow);
      document.removeEventListener('visibilitychange', persistNow);
      consoleRig.dispose();
      renderer.dispose();
    };
  }, []);

  const ringCircumference = 2 * Math.PI * 40;

  return (
    <main className="relative w-full h-full overflow-hidden bg-[#041326] select-none touch-none">
      {/* Full-Screen 3D Water Chamber Canvas */}
      <div
        ref={canvasMountRef}
        data-testid="waterful-3d-canvas"
        className="absolute inset-0 w-full h-full"
      />

      {/* Left & Right Physical Pump Buttons (Touch Duration Controls Force) */}
      <div className="fixed inset-x-0 bottom-6 sm:bottom-9 z-20 pointer-events-none flex items-center justify-between px-6 sm:px-12">
        {/* LEFT WATER PUMP BUTTON */}
        <button
          type="button"
          data-testid="left-pump-button"
          aria-label="Left Water Pump"
          onPointerDown={(e) => {
            e.preventDefault();
            try {
              e.currentTarget.setPointerCapture?.(e.pointerId);
            } catch {
              // Ignore synthetic pointer events without active pointerId
            }
            startPumpHold('left');
          }}
          onPointerUp={() => endPumpHold('left')}
          onPointerCancel={() => endPumpHold('left')}
          onLostPointerCapture={() => endPumpHold('left')}
          className={`pointer-events-auto relative w-24 h-24 sm:w-28 sm:h-28 rounded-full flex items-center justify-center cursor-pointer focus:outline-none transition-transform duration-75 ${
            leftCompression > 0.05 ? 'scale-90 translate-y-1.5' : 'scale-100 translate-y-0'
          }`}
        >
          {/* Outer tactile bellows rim + touch-duration force ring */}
          <svg
            viewBox="0 0 96 96"
            className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none"
          >
            <circle
              cx="48"
              cy="48"
              r="40"
              fill="rgba(14, 116, 144, 0.22)"
              stroke="rgba(125, 211, 252, 0.28)"
              strokeWidth="4"
            />
            <circle
              cx="48"
              cy="48"
              r="40"
              fill="none"
              stroke="#38bdf8"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={ringCircumference}
              strokeDashoffset={ringCircumference * (1 - leftCompression)}
            />
          </svg>
          {/* Inner tactile rubber button dome */}
          <span className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-b from-sky-300/90 to-sky-600/90 shadow-[0_6px_20px_rgba(56,189,248,0.45),inset_0_2px_6px_rgba(255,255,255,0.65)] border border-sky-200/60" />
        </button>

        {/* RIGHT WATER PUMP BUTTON */}
        <button
          type="button"
          data-testid="right-pump-button"
          aria-label="Right Water Pump"
          onPointerDown={(e) => {
            e.preventDefault();
            try {
              e.currentTarget.setPointerCapture?.(e.pointerId);
            } catch {
              // Ignore synthetic pointer events without active pointerId
            }
            startPumpHold('right');
          }}
          onPointerUp={() => endPumpHold('right')}
          onPointerCancel={() => endPumpHold('right')}
          onLostPointerCapture={() => endPumpHold('right')}
          className={`pointer-events-auto relative w-24 h-24 sm:w-28 sm:h-28 rounded-full flex items-center justify-center cursor-pointer focus:outline-none transition-transform duration-75 ${
            rightCompression > 0.05 ? 'scale-90 translate-y-1.5' : 'scale-100 translate-y-0'
          }`}
        >
          {/* Outer tactile bellows rim + touch-duration force ring */}
          <svg
            viewBox="0 0 96 96"
            className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none"
          >
            <circle
              cx="48"
              cy="48"
              r="40"
              fill="rgba(14, 116, 144, 0.22)"
              stroke="rgba(125, 211, 252, 0.28)"
              strokeWidth="4"
            />
            <circle
              cx="48"
              cy="48"
              r="40"
              fill="none"
              stroke="#38bdf8"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={ringCircumference}
              strokeDashoffset={ringCircumference * (1 - rightCompression)}
            />
          </svg>
          {/* Inner tactile rubber button dome */}
          <span className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-b from-sky-300/90 to-sky-600/90 shadow-[0_6px_20px_rgba(56,189,248,0.45),inset_0_2px_6px_rgba(255,255,255,0.65)] border border-sky-200/60" />
        </button>
      </div>
    </main>
  );
}
