import * as THREE from 'three';
import {
  SimulationState,
  DEFAULT_TANK_BOUNDS,
  DEFAULT_PEGS,
  DEFAULT_NOZZLES,
  WAVE_NX,
  WAVE_NZ,
} from '../physics/waterPhysics';

export interface TomyConsoleRig {
  rootGroup: THREE.Group;
  waterSurfaceMesh: THREE.Mesh;
  meniscusRibbonMesh: THREE.Mesh;
  ringMeshes: THREE.Mesh[];
  bubbleInstancedMesh: THREE.InstancedMesh;
  flowParticleInstancedMesh: THREE.InstancedMesh;
  updateFromSimulation: (state: SimulationState) => void;
  dispose: () => void;
}

const _dummyObj = new THREE.Object3D();
const _upAxis = new THREE.Vector3(0, 1, 0);
const _velDir = new THREE.Vector3();
const _instanceColor = new THREE.Color();

/**
 * Pure Full-Screen 3D Hydrodynamic Scene:
 * - Full-bleed 2D Shallow-Water Wave Equation (SWE) free-surface mesh (`WAVE_NX x WAVE_NZ`)
 *   with Fresnel specular wave shader and front meniscus wave line
 * - 520 Lagrangian velocity-aligned fluid micro-particles (`flowParticleInstancedMesh`)
 *   that trace the 3D water velocity field, vortices, and suction backflow
 * - 360 Maxey-Riley cavitation micro-bubbles (`bubbleInstancedMesh`) rendered with a
 *   true Fresnel hollow-bubble rim + specular glint shader and Eötvös oblate deformation
 * - 2 submerged 3D tapered target needles and 12 3D plastic torus rings
 */
export function buildTomyConsoleRig(
  initialState: SimulationState,
  _renderer: THREE.WebGLRenderer,
  isSoftwareGl: boolean
): TomyConsoleRig {
  const rootGroup = new THREE.Group();

  const pegMat = new THREE.MeshPhysicalMaterial({
    color: 0xf8fafc,
    roughness: 0.16,
    metalness: 0.05,
    clearcoat: 1.0,
  });

  const nozzleMat = new THREE.MeshStandardMaterial({
    color: 0xcbd5e1,
    roughness: 0.25,
    metalness: 0.75,
  });

  const tank = DEFAULT_TANK_BOUNDS;
  const fullBleedSurfaceWidth = 10.0;
  const fullDepth = tank.halfDepth * 2;
  const baseSurfaceY = tank.halfHeight - 0.08;

  // --- 1. 2D Shallow-Water Equation (SWE) Dynamic Free-Surface Wave Band (Pure Orthographic, No 3D Perspective Box) ---
  const waterSurfaceGeo = new THREE.PlaneGeometry(
    fullBleedSurfaceWidth,
    fullDepth,
    WAVE_NX - 1,
    WAVE_NZ - 1
  );
  waterSurfaceGeo.rotateX(-Math.PI / 2);

  const waterSurfaceMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying float vWaveHeight;
      void main() {
        vWaveHeight = position.y;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying float vWaveHeight;
      void main() {
        vec3 N = normalize(vNormal);
        vec3 V = normalize(vec3(0.0, 0.25, 1.0));
        vec3 L = normalize(vec3(0.4, 0.85, 0.5));
        float fresnel = pow(1.0 - abs(dot(N, V)), 2.0);
        vec3 R = reflect(-L, N);
        float spec = pow(max(dot(R, V), 0.0), 28.0);
        float crestFoam = smoothstep(0.02, 0.14, abs(vWaveHeight));
        vec3 deepCol = vec3(0.03, 0.32, 0.56);
        vec3 rimCol = vec3(0.45, 0.86, 1.0);
        vec3 col = mix(deepCol, rimCol, fresnel * 0.75 + crestFoam * 0.45) + vec3(0.9, 0.98, 1.0) * spec * 0.75;
        float alpha = clamp(0.14 + crestFoam * 0.28 + spec * 0.35, 0.08, 0.55);
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });

  const waterSurfaceMesh = new THREE.Mesh(waterSurfaceGeo, waterSurfaceMat);
  waterSurfaceMesh.position.set(0, baseSurfaceY, 0);
  waterSurfaceMesh.renderOrder = 4;
  rootGroup.add(waterSurfaceMesh);

  // Crisp glowing front meniscus wave line across the top of the water
  const meniscusRibbonGeo = new THREE.PlaneGeometry(fullBleedSurfaceWidth, 0.024, WAVE_NX - 1, 1);
  const meniscusRibbonMat = new THREE.MeshBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity: 0.62,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const meniscusRibbonMesh = new THREE.Mesh(meniscusRibbonGeo, meniscusRibbonMat);
  meniscusRibbonMesh.position.set(0, baseSurfaceY, tank.halfDepth * 0.85);
  meniscusRibbonMesh.renderOrder = 5;
  rootGroup.add(meniscusRibbonMesh);

  // --- 2. Bottom Left & Right Upward Jet Nozzles (`喷水口`) + Separate Return Intake Vents (`独立回水口`) ---
  const nozzleGroups: THREE.Group[] = [];
  DEFAULT_NOZZLES.forEach((noz) => {
    const nozzleGroup = new THREE.Group();
    const nozzleConeGeo = new THREE.CylinderGeometry(0.055, 0.11, 0.14, 20);
    const nozzleMesh = new THREE.Mesh(nozzleConeGeo, nozzleMat);
    const orificeGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.16, 16);
    const orificeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const orificeMesh = new THREE.Mesh(orificeGeo, orificeMat);
    nozzleGroup.add(nozzleMesh, orificeMesh);
    nozzleGroup.position.set(noz.x, noz.y + 0.04, noz.z);
    nozzleGroup.rotation.z = -noz.dirX * 0.8;
    rootGroup.add(nozzleGroup);
    nozzleGroups.push(nozzleGroup);
  });

  // Separate Bottom-Center Return Intake Grate (`底部中央独立吸水回流孔`)
  const intakeMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.5,
    metalness: 0.4,
  });
  const centerIntakeGeo = new THREE.CylinderGeometry(0.14, 0.16, 0.05, 20);
  const centerIntakeMesh = new THREE.Mesh(centerIntakeGeo, intakeMat);
  centerIntakeMesh.position.set(0, -tank.halfHeight + 0.025, 0);
  rootGroup.add(centerIntakeMesh);

  // --- 3. Submerged 3D Tapered Target Needles (`立柱针`) ---
  DEFAULT_PEGS.forEach((peg) => {
    const pegGroup = new THREE.Group();
    pegGroup.position.set(peg.baseX, peg.baseY, peg.baseZ);

    const shaftGeo = new THREE.CylinderGeometry(peg.tipRadius, peg.baseRadius, peg.height, 24);
    shaftGeo.translate(0, peg.height / 2, 0);
    const shaftMesh = new THREE.Mesh(shaftGeo, pegMat);
    shaftMesh.castShadow = !isSoftwareGl;
    pegGroup.add(shaftMesh);

    const tipGeo = new THREE.SphereGeometry(peg.tipRadius, 16, 12);
    const tipMesh = new THREE.Mesh(tipGeo, pegMat);
    tipMesh.position.y = peg.height;
    pegGroup.add(tipMesh);

    const cupGeo = new THREE.CylinderGeometry(
      peg.holderCupRadius,
      peg.holderCupRadius * 0.65,
      0.075,
      24
    );
    const cupMesh = new THREE.Mesh(cupGeo, pegMat);
    cupMesh.position.y = peg.holderCupY;
    cupMesh.castShadow = !isSoftwareGl;
    pegGroup.add(cupMesh);

    rootGroup.add(pegGroup);
  });

  // --- 4. 3D Plastic Torus Rings (`塑料套圈` with compact center hole) Floating in Water ---
  const ringMeshes: THREE.Mesh[] = [];
  const sharedTorusGeo = new THREE.TorusGeometry(
    initialState.rings[0]?.majorRadius || 0.185,
    initialState.rings[0]?.minorRadius || 0.066,
    24,
    44
  );

  for (let i = 0; i < initialState.rings.length; i++) {
    const rState = initialState.rings[i];
    const rMat = new THREE.MeshPhysicalMaterial({
      color: rState.colorHex,
      roughness: 0.18,
      metalness: 0.04,
      clearcoat: 0.95,
      clearcoatRoughness: 0.08,
    });
    const rMesh = new THREE.Mesh(sharedTorusGeo, rMat);
    rMesh.castShadow = !isSoftwareGl;
    rMesh.receiveShadow = !isSoftwareGl;
    rMesh.position.copy(rState.position);
    rMesh.quaternion.copy(rState.quaternion);
    rootGroup.add(rMesh);
    ringMeshes.push(rMesh);
  }

  // --- 5. Subtle Velocity-Aligned Lagrangian Fluid Micro-Particles (`flowParticleInstancedMesh`) ---
  const flowParticleGeo = new THREE.SphereGeometry(1, 6, 6);
  const flowParticleMat = new THREE.MeshBasicMaterial({
    color: 0xbae6fd,
    transparent: true,
    opacity: 0.12,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const flowParticleInstancedMesh = new THREE.InstancedMesh(
    flowParticleGeo,
    flowParticleMat,
    initialState.flowParticles.length
  );
  flowParticleInstancedMesh.frustumCulled = false;
  flowParticleInstancedMesh.renderOrder = 2;
  rootGroup.add(flowParticleInstancedMesh);

  // --- 6. Minimal, Clean Maxey-Riley Cavitation Micro-Bubbles with True Fresnel Hollow-Bubble Shader ---
  const bubbleSphereGeo = new THREE.SphereGeometry(1, 14, 12);
  const bubbleShaderMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      void main() {
        #ifdef USE_INSTANCING
          vec4 mvPosition = viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.0);
          mat3 instNormalMat = normalMatrix * mat3(modelMatrix) * mat3(instanceMatrix);
        #else
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          mat3 instNormalMat = normalMatrix;
        #endif
        vNormal = normalize(instNormalMat * normal);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      void main() {
        vec3 N = normalize(vNormal);
        vec3 V = normalize(vViewPosition);
        float ndotv = clamp(abs(dot(N, V)), 0.0, 1.0);
        // Hollow air-bubble in water: transparent interior + crisp bright Fresnel refractive rim
        float rim = pow(1.0 - ndotv, 1.85);
        vec3 L1 = normalize(vec3(-0.45, 0.75, 0.65));
        vec3 H1 = normalize(L1 + V);
        float spec1 = pow(max(dot(N, H1), 0.0), 36.0);
        vec3 L2 = normalize(vec3(0.55, -0.45, 0.5));
        vec3 H2 = normalize(L2 + V);
        float spec2 = pow(max(dot(N, H2), 0.0), 18.0) * 0.45;

        vec3 coreCol = vec3(0.45, 0.85, 1.0);
        vec3 rimCol = vec3(0.86, 0.97, 1.0);
        vec3 finalCol = mix(coreCol, rimCol, rim) + vec3(1.0) * (spec1 * 1.15 + spec2);
        float alpha = clamp(0.08 + rim * 0.68 + spec1 * 0.85 + spec2 * 0.4, 0.06, 0.90);
        gl_FragColor = vec4(finalCol, alpha);
      }
    `,
  });

  const bubbleInstancedMesh = new THREE.InstancedMesh(
    bubbleSphereGeo,
    bubbleShaderMat,
    initialState.bubbles.length
  );
  bubbleInstancedMesh.frustumCulled = false;
  bubbleInstancedMesh.renderOrder = 3;
  rootGroup.add(bubbleInstancedMesh);

  const updateFromSimulation = (state: SimulationState) => {
    // 1. Update 2D Shallow-Water Wave Free-Surface Mesh vertices & normals
    const surfPosAttr = waterSurfaceGeo.attributes.position as THREE.BufferAttribute;
    const surfArr = surfPosAttr.array as Float32Array;
    const waveH = state.surfaceWaveHeight;

    for (let i = 0; i < waveH.length; i++) {
      surfArr[i * 3 + 1] = waveH[i];
    }
    surfPosAttr.needsUpdate = true;
    waterSurfaceGeo.computeVertexNormals();

    // Update front meniscus wave line to match wave profile
    const ribPosAttr = meniscusRibbonGeo.attributes.position as THREE.BufferAttribute;
    const ribArr = ribPosAttr.array as Float32Array;
    const midRowOffset = Math.floor(WAVE_NZ * 0.6) * WAVE_NX;
    for (let ix = 0; ix < WAVE_NX; ix++) {
      const hVal = waveH[midRowOffset + ix] || 0;
      ribArr[ix * 3 + 1] = hVal + 0.013;
      ribArr[(WAVE_NX + ix) * 3 + 1] = hVal - 0.013;
    }
    ribPosAttr.needsUpdate = true;

    // 2. Sync Nozzle Positions (for Portrait / Landscape responsive adaptation) & Update 3D Plastic Torus Rings
    for (let nIdx = 0; nIdx < state.nozzles.length; nIdx++) {
      const nGroup = nozzleGroups[nIdx];
      const noz = state.nozzles[nIdx];
      if (nGroup && noz) {
        nGroup.position.set(noz.x, noz.y + 0.04, noz.z);
      }
    }

    for (let i = 0; i < state.rings.length; i++) {
      const rState = state.rings[i];
      const rMesh = ringMeshes[i];
      if (!rMesh) continue;
      rMesh.position.copy(rState.position);
      rMesh.quaternion.copy(rState.quaternion);
    }

    // 3. Update 64 Lagrangian Fluid Micro-Particles (delicate hydro-motes aligned with velocity)
    for (let i = 0; i < state.flowParticles.length; i++) {
      const fp = state.flowParticles[i];
      const speed = fp.velocity.length();
      _dummyObj.position.copy(fp.position);

      if (speed > 0.04) {
        _velDir.copy(fp.velocity).divideScalar(speed);
        _dummyObj.quaternion.setFromUnitVectors(_upAxis, _velDir);
      } else {
        _dummyObj.quaternion.identity();
      }

      const r = fp.baseSize * 0.6;
      const streakLen = r * (1.0 + Math.min(2.2, speed * 0.55));
      _dummyObj.scale.set(r, streakLen, r);
      _dummyObj.updateMatrix();
      flowParticleInstancedMesh.setMatrixAt(i, _dummyObj.matrix);

      const speedNorm = Math.min(1.0, speed / 4.2);
      _instanceColor.setRGB(
        0.15 + speedNorm * 0.55,
        0.52 + speedNorm * 0.42,
        0.78 + speedNorm * 0.22
      );
      flowParticleInstancedMesh.setColorAt(i, _instanceColor);
    }
    flowParticleInstancedMesh.instanceMatrix.needsUpdate = true;
    if (flowParticleInstancedMesh.instanceColor) {
      flowParticleInstancedMesh.instanceColor.needsUpdate = true;
    }

    // 4. Update 32 Maxey-Riley Cavitation Micro-Bubbles with Eötvös oblate deformation
    for (let i = 0; i < state.bubbles.length; i++) {
      const b = state.bubbles[i];
      if (!b.active) {
        _dummyObj.position.set(0, -999, 0);
        _dummyObj.scale.setScalar(0.0001);
      } else {
        _dummyObj.position.copy(b.position);
        const speed = b.velocity.length();
        if (speed > 0.1) {
          _velDir.copy(b.velocity).divideScalar(speed);
          _dummyObj.quaternion.setFromUnitVectors(_upAxis, _velDir);
        } else {
          _dummyObj.quaternion.identity();
        }
        const fade = 1.0 - (b.life / b.maxLife) * 0.25;
        const r = b.radius * fade * 1.15;
        const oblate = Math.min(0.25, speed * 0.032);
        _dummyObj.scale.set(r * (1.0 + oblate), r * (1.0 - oblate * 0.75), r * (1.0 + oblate));
      }
      _dummyObj.updateMatrix();
      bubbleInstancedMesh.setMatrixAt(i, _dummyObj.matrix);
    }
    bubbleInstancedMesh.instanceMatrix.needsUpdate = true;
  };

  const dispose = () => {
    waterSurfaceGeo.dispose();
    waterSurfaceMat.dispose();
    meniscusRibbonGeo.dispose();
    meniscusRibbonMat.dispose();
    sharedTorusGeo.dispose();
    flowParticleGeo.dispose();
    flowParticleMat.dispose();
    bubbleSphereGeo.dispose();
    bubbleShaderMat.dispose();
  };

  return {
    rootGroup,
    waterSurfaceMesh,
    meniscusRibbonMesh,
    ringMeshes,
    bubbleInstancedMesh,
    flowParticleInstancedMesh,
    updateFromSimulation,
    dispose,
  };
}
