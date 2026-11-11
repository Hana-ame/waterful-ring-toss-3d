/**
 * High-Throughput WebAssembly (WASM) 3D Hydrodynamic & Rigid-Body Physics Kernel
 *
 * Architecture & Workload Division (Zero Inversion):
 * - All per-ring continuous mathematical heavy lifting runs natively inside WebAssembly linear memory:
 *   1. 3D Rational Algebraic Fluid Velocity & Vorticity Field Sampling (Dual Nozzle Jet Plumes,
 *      Tank-Wide Counter-Rotating Vortices, Separate Bottom Return-Vent Suction, and Gyroscope Sloshing —
 *      using rational algebraic bell kernels 1/(1 + r^2/sigma^2)^2 with ZERO exp/sin/cos transcendental overhead).
 *   2. 3D Quaternion-to-Normal Basis Extraction & Anisotropic Morison Quadratic Drag (cN, cT).
 *   3. 3D Hydrodynamic Angle-of-Attack Torque, Gravity Righting Torque, Vorticity Spin, & Padé Angular Damping.
 *   4. Velocity Speed Clamping, Semi-Implicit Euler Position Integration, and 3D Quaternion Attitude Integration
 *      with hardware `f32.sqrt` normalization.
 *
 * Memory Layout (Float32 elements, 4 bytes each):
 * - Offset 0..511 (Bytes 0..2047): Ring State Array (up to 16 rings * 16 floats per ring = 64 bytes/ring)
 *     [0]: posX, [1]: posY, [2]: posZ,
 *     [3]: velX, [4]: velY, [5]: velZ,
 *     [6]: quatX, [7]: quatY, [8]: quatZ, [9]: quatW,
 *     [10]: angVelX, [11]: angVelY, [12]: angVelZ,
 *     [13]: majorRadius, [14]: minorRadius, [15]: hookedFlag
 * - Offset 512..543 (Bytes 2048..2175): Global Simulation Uniforms (`uniformFloatView`, 32 floats)
 *     [0]  (Byte 2048): subDt
 *     [1]  (Byte 2052): linearDamping
 *     [2]  (Byte 2056): stepCounter (incremented inside WASM)
 *     [3]  (Byte 2060): lastKineticEnergy (written inside WASM)
 *     [4]  (Byte 2064): netGravityX (gravityDir.x * gravityMag * netBuoyancy * 2.15)
 *     [5]  (Byte 2068): netGravityY
 *     [6]  (Byte 2072): netGravityZ
 *     [7]  (Byte 2076): waterAngularVelX
 *     [8]  (Byte 2080): waterAngularVelY
 *     [9]  (Byte 2084): waterAngularVelZ
 *     [10] (Byte 2088): gravityDirX
 *     [11] (Byte 2092): gravityDirY
 *     [12] (Byte 2096): gravityDirZ
 *     [13] (Byte 2100): normalDragCoeff
 *     [14] (Byte 2104): tangentDragCoeff
 *     [15] (Byte 2108): angularDragCoeff
 *     [16] (Byte 2112): leftNozzleX
 *     [17] (Byte 2116): leftNozzleY
 *     [18] (Byte 2120): leftNozzleZ
 *     [19] (Byte 2124): leftJetStrength (already multiplied by jetPowerMultiplier)
 *     [20] (Byte 2128): leftCirculation (already multiplied by jetPowerMultiplier)
 *     [21] (Byte 2132): rightNozzleX
 *     [22] (Byte 2136): rightNozzleY
 *     [23] (Byte 2140): rightNozzleZ
 *     [24] (Byte 2144): rightJetStrength (already multiplied by jetPowerMultiplier)
 *     [25] (Byte 2148): rightCirculation (already multiplied by jetPowerMultiplier)
 *     [26] (Byte 2152): totalReturnSuction (leftSuction + rightSuction)
 *     [27] (Byte 2156): tankHalfHeight
 *     [28] (Byte 2160): turbulenceWaveX
 *     [29] (Byte 2164): turbulenceWaveZ
 *     [30] (Byte 2168): fullPhysicsModeFlag (1.0 when full hydrodynamic uniforms are populated, 0.0 for minimal standalone tests)
 */

export interface WasmPhysicsKernel {
  isWasmReady: boolean;
  memory: WebAssembly.Memory;
  ringFloatView: Float32Array;
  uniformFloatView: Float32Array;
  wasmIntegrateStep: (ringCount: number, dt: number, dampingFactor: number) => number;
  getWasmBytecodeLength: () => number;
}

function uleb(val: number): number[] {
  const out: number[] = [];
  let v = val >>> 0;
  do {
    let byte = v & 0x7f;
    v >>>= 7;
    if (v !== 0) byte |= 0x80;
    out.push(byte);
  } while (v !== 0);
  return out;
}

function f32Bytes(val: number): number[] {
  const f = new Float32Array([val]);
  const b = new Uint8Array(f.buffer);
  return [b[0], b[1], b[2], b[3]];
}

function section(id: number, payload: number[]): number[] {
  return [id, ...uleb(payload.length), ...payload];
}

// Composable WebAssembly Expression & Statement Builders
const i32Const = (v: number): number[] => [0x41, ...uleb(v)];
const f32Const = (v: number): number[] => [0x43, ...f32Bytes(v)];
const localGet = (idx: number): number[] => [0x20, ...uleb(idx)];
const localSet = (idx: number, expr: number[]): number[] => [...expr, 0x21, ...uleb(idx)];

const f32Load = (addrExpr: number[], offset: number): number[] => [
  ...addrExpr,
  0x2a,
  0x02,
  ...uleb(offset),
];
const f32LoadUniform = (uniformIndex: number): number[] =>
  f32Load(i32Const(0), 2048 + uniformIndex * 4);

const f32Store = (addrExpr: number[], offset: number, valExpr: number[]): number[] => [
  ...addrExpr,
  ...valExpr,
  0x38,
  0x02,
  ...uleb(offset),
];
const f32StoreUniform = (uniformIndex: number, valExpr: number[]): number[] =>
  f32Store(i32Const(0), 2048 + uniformIndex * 4, valExpr);

const f32Add = (a: number[], b: number[]): number[] => [...a, ...b, 0x92];
const f32Sub = (a: number[], b: number[]): number[] => [...a, ...b, 0x93];
const f32Mul = (a: number[], b: number[]): number[] => [...a, ...b, 0x94];
const f32Div = (a: number[], b: number[]): number[] => [...a, ...b, 0x95];
const f32Min = (a: number[], b: number[]): number[] => [...a, ...b, 0x96];
const f32Max = (a: number[], b: number[]): number[] => [...a, ...b, 0x97];
const f32Sqrt = (a: number[]): number[] => [...a, 0x91];
const f32Gt = (a: number[], b: number[]): number[] => [...a, ...b, 0x5e];
const f32Ge = (a: number[], b: number[]): number[] => [...a, ...b, 0x60];

const wasmIf = (cond: number[], thenBody: number[], elseBody?: number[]): number[] =>
  elseBody
    ? [...cond, 0x04, 0x40, ...thenBody, 0x05, ...elseBody, 0x0b]
    : [...cond, 0x04, 0x40, ...thenBody, 0x0b];

/**
 * Emits WebAssembly bytecode that executes the complete per-ring 3D hydrodynamic + rigid-body substep:
 * - Reads nozzle jets, circulation, return suction, gyroscope sloshing, gravity, and Morison drag uniforms
 * - Evaluates rational algebraic 3D fluid velocity & vorticity (ZERO exp/sin/cos)
 * - Evaluates 3D quaternion normal, anisotropic Morison drag, hydrodynamic torques, and quaternion integration
 */
function buildWasmPhysicsBinary(): Uint8Array {
  // Function signature: (ringCount: i32) -> (totalKineticEnergy: f32)
  const typeSection = section(1, [1, 0x60, 1, 0x7f, 1, 0x7d]);

  // Import env.mem (1 page = 64KB)
  const importSection = section(2, [
    1,
    3,
    0x65,
    0x6e,
    0x76, // "env"
    3,
    0x6d,
    0x65,
    0x6d, // "mem"
    0x02,
    0x00,
    0x01,
  ]);

  const funcSection = section(3, [1, 0]);

  // Export "integrate"
  const exportSection = section(7, [
    1,
    9,
    0x69,
    0x6e,
    0x74,
    0x65,
    0x67,
    0x72,
    0x61,
    0x74,
    0x65,
    0x00,
    0x00,
  ]);

  // Local variable allocation:
  // param 0: ringCount (i32)
  // locals 1..2: i (i32), ptr (i32)
  // locals 3..46: 44 f32 registers for 3D hydrodynamics, Morison drag, torques, & quaternion integration
  const I = 1;
  const PTR = 2;
  const DT = 3;
  const DAMP = 4;
  const ENERGY = 5;
  const PX = 6;
  const PY = 7;
  const PZ = 8;
  const VX = 9;
  const VY = 10;
  const VZ = 11;
  const QX = 12;
  const QY = 13;
  const QZ = 14;
  const QW = 15;
  const WX = 16;
  const WY = 17;
  const WZ = 18;
  const NX = 19;
  const NY = 20;
  const NZ = 21;
  const FVX = 22;
  const FVY = 23;
  const FVZ = 24;
  const VORTX = 25;
  const VORTY = 26;
  const VORTZ = 27;
  const RVX = 28;
  const RVY = 29;
  const RVZ = 30;
  const RSPEED = 31;
  const VPROJ = 32;
  const CN = 33;
  const CT = 34;
  const DX = 35;
  const DY = 36;
  const DZ = 37;
  const R2 = 38;
  const SIGMA = 39;
  const BELL = 40;
  const VCORE = 41;
  const TMP1 = 42;
  const TMP2 = 43;
  const TMP3 = 44;
  const TMP4 = 45;

  const locals = [
    2,
    2,
    0x7f, // 2 i32 locals (1..2)
    44,
    0x7d, // 44 f32 locals (3..46)
  ];

  // Helper to emit rational nozzle plume & tank circulation for a nozzle at uniform offset `nozBase`
  // `sideSign` is -1.0 for left nozzle, +1.0 for right nozzle
  const emitNozzleContribution = (nozBase: number, sideSign: number): number[] => {
    const dirX = sideSign < 0 ? 0.04 : -0.04;
    return [
      // TMP1 = jetStrength (uniform[nozBase + 3])
      ...localSet(TMP1, f32LoadUniform(nozBase + 3)),
      ...wasmIf(f32Gt(localGet(TMP1), f32Const(0.001)), [
        // DY = max(0.02, PY - nozzleY)
        ...localSet(
          DY,
          f32Max(f32Const(0.02), f32Sub(localGet(PY), f32LoadUniform(nozBase + 1)))
        ),
        // DX = PX - (nozzleX + dirX * DY)
        ...localSet(
          DX,
          f32Sub(
            localGet(PX),
            f32Add(f32LoadUniform(nozBase + 0), f32Mul(f32Const(dirX), localGet(DY)))
          )
        ),
        // DZ = PZ - nozzleZ
        ...localSet(DZ, f32Sub(localGet(PZ), f32LoadUniform(nozBase + 2))),
        // R2 = DX*DX + DZ*DZ
        ...localSet(
          R2,
          f32Add(f32Mul(localGet(DX), localGet(DX)), f32Mul(localGet(DZ), localGet(DZ)))
        ),
        // SIGMA = 0.48 + 0.16 * DY
        ...localSet(SIGMA, f32Add(f32Const(0.48), f32Mul(f32Const(0.16), localGet(DY)))),
        // TMP2 = 1.0 + 0.5 * R2 / (SIGMA * SIGMA)  (rational Gaussian denominator)
        ...localSet(
          TMP2,
          f32Add(
            f32Const(1.0),
            f32Div(
              f32Mul(f32Const(0.5), localGet(R2)),
              f32Mul(localGet(SIGMA), localGet(SIGMA))
            )
          )
        ),
        // BELL = 1.0 / (TMP2 * TMP2)
        ...localSet(BELL, f32Div(f32Const(1.0), f32Mul(localGet(TMP2), localGet(TMP2)))),
        // TMP3 = axialDecay = 1.0 / (1.0 + 0.08 * DY + 0.015 * DY * DY)
        ...localSet(
          TMP3,
          f32Div(
            f32Const(1.0),
            f32Add(
              f32Const(1.0),
              f32Add(
                f32Mul(f32Const(0.08), localGet(DY)),
                f32Mul(f32Const(0.015), f32Mul(localGet(DY), localGet(DY)))
              )
            )
          )
        ),
        // VCORE = 12.6 * jetStrength * axialDecay * BELL
        ...localSet(
          VCORE,
          f32Mul(
            f32Mul(f32Const(12.6), localGet(TMP1)),
            f32Mul(localGet(TMP3), localGet(BELL))
          )
        ),
        // FVY += VCORE; FVX += dirX * VCORE - DX * 1.15 * TMP1 * BELL; FVZ += -DZ * 0.45 * TMP1 * BELL
        ...localSet(FVY, f32Add(localGet(FVY), localGet(VCORE))),
        ...localSet(
          FVX,
          f32Add(
            localGet(FVX),
            f32Sub(
              f32Mul(f32Const(dirX), localGet(VCORE)),
              f32Mul(localGet(DX), f32Mul(f32Const(1.15), f32Mul(localGet(TMP1), localGet(BELL))))
            )
          )
        ),
        ...localSet(
          FVZ,
          f32Sub(
            localGet(FVZ),
            f32Mul(localGet(DZ), f32Mul(f32Const(0.45), f32Mul(localGet(TMP1), localGet(BELL))))
          )
        ),
        // Shear-layer vorticity: VORTZ += -DX * VCORE * 1.8; VORTX += DZ * VCORE * 1.8
        ...localSet(
          VORTZ,
          f32Sub(localGet(VORTZ), f32Mul(localGet(DX), f32Mul(localGet(VCORE), f32Const(1.8))))
        ),
        ...localSet(
          VORTX,
          f32Add(localGet(VORTX), f32Mul(localGet(DZ), f32Mul(localGet(VCORE), f32Const(1.8))))
        ),
      ]),

      // Tank-wide counter-rotating vortex circulation (uniform[nozBase + 4])
      ...localSet(TMP1, f32LoadUniform(nozBase + 4)),
      ...wasmIf(f32Gt(localGet(TMP1), f32Const(0.002)), [
        // Vortex center at (sideSign * 0.52, -0.05)
        ...localSet(DX, f32Sub(localGet(PX), f32Const(sideSign * 0.52))),
        ...localSet(DY, f32Sub(localGet(PY), f32Const(-0.05))),
        ...localSet(
          R2,
          f32Add(
            f32Mul(f32Const(0.6), f32Mul(localGet(DX), localGet(DX))),
            f32Mul(f32Const(0.5), f32Mul(localGet(DY), localGet(DY)))
          )
        ),
        // Rational vortex kernel: TMP2 = TMP1 * 1.85 / (1.0 + 0.35 * R2)
        ...localSet(
          TMP2,
          f32Div(
            f32Mul(localGet(TMP1), f32Const(1.85)),
            f32Add(f32Const(1.0), f32Mul(f32Const(0.35), localGet(R2)))
          )
        ),
        // Swirl: FVX += -sideSign * DY * TMP2 * 0.85; FVY += sideSign * DX * TMP2 * 0.95
        ...localSet(
          FVX,
          f32Add(
            localGet(FVX),
            f32Mul(f32Const(-sideSign * 0.85), f32Mul(localGet(DY), localGet(TMP2)))
          )
        ),
        ...localSet(
          FVY,
          f32Add(
            localGet(FVY),
            f32Mul(f32Const(sideSign * 0.95), f32Mul(localGet(DX), localGet(TMP2)))
          )
        ),
      ]),
    ];
  };

  const bodyInstr: number[] = [
    // Load subDt (offset 2048) and linearDamping (offset 2052)
    ...localSet(DT, f32LoadUniform(0)),
    ...localSet(DAMP, f32LoadUniform(1)),
    ...localSet(I, i32Const(0)),
    ...localSet(ENERGY, f32Const(0.0)),

    // block $break / loop $top
    0x02,
    0x40,
    0x03,
    0x40,

    // br_if $break (I >= ringCount)
    ...localGet(I),
    ...localGet(0),
    0x4f,
    0x0d,
    0x01,

    // PTR = I << 6
    ...localSet(PTR, [...localGet(I), ...i32Const(6), 0x74]),

    // Load ring position & velocity
    ...localSet(PX, f32Load(localGet(PTR), 0)),
    ...localSet(PY, f32Load(localGet(PTR), 4)),
    ...localSet(PZ, f32Load(localGet(PTR), 8)),
    ...localSet(VX, f32Load(localGet(PTR), 12)),
    ...localSet(VY, f32Load(localGet(PTR), 16)),
    ...localSet(VZ, f32Load(localGet(PTR), 20)),

    // Check if full hydrodynamic uniforms are active (uniform[30] > 0.5)
    ...wasmIf(f32Gt(f32LoadUniform(30), f32Const(0.5)), [
      // Load quaternion & angular velocity
      ...localSet(QX, f32Load(localGet(PTR), 24)),
      ...localSet(QY, f32Load(localGet(PTR), 28)),
      ...localSet(QZ, f32Load(localGet(PTR), 32)),
      ...localSet(QW, f32Load(localGet(PTR), 36)),
      ...localSet(WX, f32Load(localGet(PTR), 40)),
      ...localSet(WY, f32Load(localGet(PTR), 44)),
      ...localSet(WZ, f32Load(localGet(PTR), 48)),

      // 1. Apply Buoyancy-Adjusted Gravity: V += netGravity * DT
      ...localSet(VX, f32Add(localGet(VX), f32Mul(f32LoadUniform(4), localGet(DT)))),
      ...localSet(VY, f32Add(localGet(VY), f32Mul(f32LoadUniform(5), localGet(DT)))),
      ...localSet(VZ, f32Add(localGet(VZ), f32Mul(f32LoadUniform(6), localGet(DT)))),

      // 2. Compute 3D Ring Normal n = q * (0,0,1) * q^-1 inside WASM
      // NX = 2 * (QX*QZ + QW*QY)
      ...localSet(
        NX,
        f32Mul(
          f32Const(2.0),
          f32Add(f32Mul(localGet(QX), localGet(QZ)), f32Mul(localGet(QW), localGet(QY)))
        )
      ),
      // NY = 2 * (QY*QZ - QW*QX)
      ...localSet(
        NY,
        f32Mul(
          f32Const(2.0),
          f32Sub(f32Mul(localGet(QY), localGet(QZ)), f32Mul(localGet(QW), localGet(QX)))
        )
      ),
      // NZ = 1 - 2 * (QX*QX + QY*QY)
      ...localSet(
        NZ,
        f32Sub(
          f32Const(1.0),
          f32Mul(
            f32Const(2.0),
            f32Add(f32Mul(localGet(QX), localGet(QX)), f32Mul(localGet(QY), localGet(QY)))
          )
        )
      ),

      // 3. Sample 3D Fluid Velocity Field (Gyroscope Sloshing + Dual Nozzles + Return Vents)
      // Sloshing: u = (waterAngularVel x P) / (1 + 0.14 * (PX^2 + PY^2))
      ...localSet(
        TMP1,
        f32Div(
          f32Const(1.0),
          f32Add(
            f32Const(1.0),
            f32Mul(
              f32Const(0.14),
              f32Add(f32Mul(localGet(PX), localGet(PX)), f32Mul(localGet(PY), localGet(PY)))
            )
          )
        )
      ),
      ...localSet(
        FVX,
        f32Add(
          f32LoadUniform(28),
          f32Mul(
            localGet(TMP1),
            f32Sub(
              f32Mul(f32LoadUniform(8), localGet(PZ)),
              f32Mul(f32LoadUniform(9), localGet(PY))
            )
          )
        )
      ),
      ...localSet(
        FVY,
        f32Mul(
          localGet(TMP1),
          f32Sub(
            f32Mul(f32LoadUniform(9), localGet(PX)),
            f32Mul(f32LoadUniform(7), localGet(PZ))
          )
        )
      ),
      ...localSet(
        FVZ,
        f32Add(
          f32LoadUniform(29),
          f32Mul(
            localGet(TMP1),
            f32Sub(
              f32Mul(f32LoadUniform(7), localGet(PY)),
              f32Mul(f32LoadUniform(8), localGet(PX))
            )
          )
        )
      ),
      ...localSet(VORTX, f32Mul(f32Const(1.25), f32LoadUniform(7))),
      ...localSet(VORTY, f32Mul(f32Const(1.25), f32LoadUniform(8))),
      ...localSet(VORTZ, f32Mul(f32Const(1.25), f32LoadUniform(9))),

      // Add Left Nozzle (uniform 16..20) & Right Nozzle (uniform 21..25)
      ...emitNozzleContribution(16, -1.0),
      ...emitNozzleContribution(21, 1.0),

      // Add Separate Bottom-Center Return Intake Vent Suction (uniform[26])
      ...localSet(TMP1, f32LoadUniform(26)),
      ...wasmIf(f32Gt(localGet(TMP1), f32Const(0.002)), [
        // DY = PY - (-tankHalfHeight + 0.04)
        ...localSet(
          DY,
          f32Sub(localGet(PY), f32Add(f32Sub(f32Const(0.0), f32LoadUniform(27)), f32Const(0.04)))
        ),
        ...localSet(
          R2,
          f32Add(
            f32Mul(localGet(PX), localGet(PX)),
            f32Add(f32Mul(localGet(DY), localGet(DY)), f32Mul(localGet(PZ), localGet(PZ)))
          )
        ),
        // TMP2 = -1.35 * totalReturnSuction / ((R2 + 0.09) * sqrt(R2 + 0.09))
        ...localSet(TMP3, f32Add(localGet(R2), f32Const(0.09))),
        ...localSet(
          TMP2,
          f32Div(
            f32Mul(f32Const(-1.35), localGet(TMP1)),
            f32Mul(localGet(TMP3), f32Sqrt(localGet(TMP3)))
          )
        ),
        ...localSet(FVX, f32Add(localGet(FVX), f32Mul(localGet(PX), localGet(TMP2)))),
        ...localSet(FVY, f32Add(localGet(FVY), f32Mul(localGet(DY), localGet(TMP2)))),
        ...localSet(FVZ, f32Add(localGet(FVZ), f32Mul(localGet(PZ), localGet(TMP2)))),
      ]),

      // 4. 3D Anisotropic Morison Drag, Magnus Lift & Fluttering/Tumbling/Chaotic Torques Inside WASM
      // RV = V - FV
      ...localSet(RVX, f32Sub(localGet(VX), localGet(FVX))),
      ...localSet(RVY, f32Sub(localGet(VY), localGet(FVY))),
      ...localSet(RVZ, f32Sub(localGet(VZ), localGet(FVZ))),
      // RSPEED = sqrt(RVX^2 + RVY^2 + RVZ^2)
      ...localSet(
        RSPEED,
        f32Sqrt(
          f32Add(
            f32Mul(localGet(RVX), localGet(RVX)),
            f32Add(f32Mul(localGet(RVY), localGet(RVY)), f32Mul(localGet(RVZ), localGet(RVZ)))
          )
        )
      ),
      // VPROJ = dot(RV, N)
      ...localSet(
        VPROJ,
        f32Add(
          f32Mul(localGet(RVX), localGet(NX)),
          f32Add(f32Mul(localGet(RVY), localGet(NY)), f32Mul(localGet(RVZ), localGet(NZ)))
        )
      ),
      // CN = normalDragCoeff * (1 + 0.85 * RSPEED) * DT
      ...localSet(
        CN,
        f32Mul(
          f32Mul(
            f32LoadUniform(13),
            f32Add(f32Const(1.0), f32Mul(f32Const(0.85), localGet(RSPEED)))
          ),
          localGet(DT)
        )
      ),
      // CT = tangentDragCoeff * (1 + 0.78 * RSPEED) * DT
      ...localSet(
        CT,
        f32Mul(
          f32Mul(
            f32LoadUniform(14),
            f32Add(f32Const(1.0), f32Mul(f32Const(0.78), localGet(RSPEED)))
          ),
          localGet(DT)
        )
      ),
      // Apply Anisotropic Morison drag (drives lateral fluttering glide along the tilted ring plane):
      // V -= CT * RV + (CN - CT) * VPROJ * N
      ...localSet(TMP1, f32Mul(f32Sub(localGet(CN), localGet(CT)), localGet(VPROJ))),
      ...localSet(
        VX,
        f32Sub(
          localGet(VX),
          f32Add(f32Mul(localGet(CT), localGet(RVX)), f32Mul(localGet(TMP1), localGet(NX)))
        )
      ),
      ...localSet(
        VY,
        f32Sub(
          localGet(VY),
          f32Add(f32Mul(localGet(CT), localGet(RVY)), f32Mul(localGet(TMP1), localGet(NY)))
        )
      ),
      ...localSet(
        VZ,
        f32Sub(
          localGet(VZ),
          f32Add(f32Mul(localGet(CT), localGet(RVZ)), f32Mul(localGet(TMP1), localGet(NZ)))
        )
      ),

      // Unified 3D Hydrodynamic Lift, Added-Mass Moment & Autorotation Coupling (zero piecewise velocity branches!):
      // 4a. Kutta-Zhukovsky / Magnus Rotational-Translational Circulation Lift: V += 0.24 * (W x RV) * DT
      ...localSet(TMP1, f32Mul(f32Const(0.24), localGet(DT))),
      ...localSet(
        VX,
        f32Add(
          localGet(VX),
          f32Mul(
            localGet(TMP1),
            f32Sub(f32Mul(localGet(WY), localGet(RVZ)), f32Mul(localGet(WZ), localGet(RVY)))
          )
        )
      ),
      ...localSet(
        VZ,
        f32Add(
          localGet(VZ),
          f32Mul(
            localGet(TMP1),
            f32Sub(f32Mul(localGet(WX), localGet(RVY)), f32Mul(localGet(WY), localGet(RVX)))
          )
        )
      ),

      // 4b. Kirchhoff-Kelvin / Munk Added-Mass Moment:
      // W += -2.15 * VPROJ * (N x RV) * DT
      ...localSet(TMP1, f32Mul(f32Mul(f32Const(-2.15), localGet(VPROJ)), localGet(DT))),
      ...localSet(
        WX,
        f32Add(
          localGet(WX),
          f32Mul(
            localGet(TMP1),
            f32Sub(f32Mul(localGet(NY), localGet(RVZ)), f32Mul(localGet(NZ), localGet(RVY)))
          )
        )
      ),
      ...localSet(
        WY,
        f32Add(
          localGet(WY),
          f32Mul(
            localGet(TMP1),
            f32Sub(f32Mul(localGet(NZ), localGet(RVX)), f32Mul(localGet(NX), localGet(RVZ)))
          )
        )
      ),
      ...localSet(
        WZ,
        f32Add(
          localGet(WZ),
          f32Mul(
            localGet(TMP1),
            f32Sub(f32Mul(localGet(NX), localGet(RVY)), f32Mul(localGet(NY), localGet(RVX)))
          )
        )
      ),

      // 4c. Leading-Edge Autorotation Tumbling Torque + Wake Vortex-Shedding Coupling:
      // Sustains continuous horizontal-axis tumbling flips (翻滚) and chaotic flutter-to-tumble transitions (飘摆 + 混沌)
      ...localSet(TMP2, f32Mul(f32Const(-3.8), localGet(DT))),
      ...localSet(
        WX,
        f32Add(
          localGet(WX),
          f32Mul(
            localGet(TMP2),
            f32Sub(
              f32Mul(f32LoadUniform(11), localGet(VZ)),
              f32Mul(f32LoadUniform(12), localGet(VY))
            )
          )
        )
      ),
      ...localSet(
        WZ,
        f32Add(
          localGet(WZ),
          f32Mul(
            localGet(TMP2),
            f32Sub(
              f32Mul(f32LoadUniform(10), localGet(VY)),
              f32Mul(f32LoadUniform(11), localGet(VX))
            )
          )
        )
      ),

      // Continuous autorotational spin sustainment driven by downward descent speed max(0, -VY):
      ...localSet(
        TMP3,
        f32Mul(f32Max(f32Const(0.0), f32Sub(f32Const(0.0), localGet(VY))), localGet(DT))
      ),
      ...localSet(
        WX,
        f32Add(
          localGet(WX),
          f32Mul(
            localGet(TMP3),
            f32Add(
              f32Mul(f32Const(1.85), localGet(WX)),
              f32Mul(f32Const(11.0), f32LoadUniform(29))
            )
          )
        )
      ),
      ...localSet(
        WZ,
        f32Add(
          localGet(WZ),
          f32Mul(
            localGet(TMP3),
            f32Add(
              f32Mul(f32Const(1.85), localGet(WZ)),
              f32Mul(f32Const(12.5), f32LoadUniform(28))
            )
          )
        )
      ),

      // Fluid Vorticity Spin + Padé Angular Damping: W = (W + 1.85 * VORT * DT) / (1 + angDrag * DT)
      ...localSet(
        TMP4,
        f32Div(
          f32Const(1.0),
          f32Add(f32Const(1.0), f32Mul(f32LoadUniform(15), localGet(DT)))
        )
      ),
      ...localSet(
        WX,
        f32Mul(
          f32Add(localGet(WX), f32Mul(f32Const(1.85), f32Mul(localGet(VORTX), localGet(DT)))),
          localGet(TMP4)
        )
      ),
      ...localSet(
        WY,
        f32Mul(
          f32Add(localGet(WY), f32Mul(f32Const(1.85), f32Mul(localGet(VORTY), localGet(DT)))),
          localGet(TMP4)
        )
      ),
      ...localSet(
        WZ,
        f32Mul(
          f32Add(localGet(WZ), f32Mul(f32Const(1.85), f32Mul(localGet(VORTZ), localGet(DT)))),
          localGet(TMP4)
        )
      ),

      // Clamp linear speed <= 6.0 and angular speed <= 12.0 inside WASM
      ...localSet(
        TMP1,
        f32Add(
          f32Mul(localGet(VX), localGet(VX)),
          f32Add(f32Mul(localGet(VY), localGet(VY)), f32Mul(localGet(VZ), localGet(VZ)))
        )
      ),
      ...wasmIf(f32Gt(localGet(TMP1), f32Const(36.0)), [
        ...localSet(TMP2, f32Div(f32Const(6.0), f32Sqrt(localGet(TMP1)))),
        ...localSet(VX, f32Mul(localGet(VX), localGet(TMP2))),
        ...localSet(VY, f32Mul(localGet(VY), localGet(TMP2))),
        ...localSet(VZ, f32Mul(localGet(VZ), localGet(TMP2))),
      ]),
      ...localSet(
        TMP1,
        f32Add(
          f32Mul(localGet(WX), localGet(WX)),
          f32Add(f32Mul(localGet(WY), localGet(WY)), f32Mul(localGet(WZ), localGet(WZ)))
        )
      ),
      ...wasmIf(f32Gt(localGet(TMP1), f32Const(144.0)), [
        ...localSet(TMP2, f32Div(f32Const(12.0), f32Sqrt(localGet(TMP1)))),
        ...localSet(WX, f32Mul(localGet(WX), localGet(TMP2))),
        ...localSet(WY, f32Mul(localGet(WY), localGet(TMP2))),
        ...localSet(WZ, f32Mul(localGet(WZ), localGet(TMP2))),
      ]),

      // 5. Integrate 3D Quaternion Attitude & Normalize via hardware f32.sqrt inside WASM
      ...localSet(TMP1, f32Mul(f32Const(0.5), localGet(DT))),
      // dqx = 0.5*dt * (WX*QW + WY*QZ - WZ*QY)
      ...localSet(
        DX,
        f32Add(
          localGet(QX),
          f32Mul(
            localGet(TMP1),
            f32Sub(
              f32Add(f32Mul(localGet(WX), localGet(QW)), f32Mul(localGet(WY), localGet(QZ))),
              f32Mul(localGet(WZ), localGet(QY))
            )
          )
        )
      ),
      // dqy = 0.5*dt * (WY*QW + WZ*QX - WX*QZ)
      ...localSet(
        DY,
        f32Add(
          localGet(QY),
          f32Mul(
            localGet(TMP1),
            f32Sub(
              f32Add(f32Mul(localGet(WY), localGet(QW)), f32Mul(localGet(WZ), localGet(QX))),
              f32Mul(localGet(WX), localGet(QZ))
            )
          )
        )
      ),
      // dqz = 0.5*dt * (WZ*QW + WX*QY - WY*QX)
      ...localSet(
        DZ,
        f32Add(
          localGet(QZ),
          f32Mul(
            localGet(TMP1),
            f32Sub(
              f32Add(f32Mul(localGet(WZ), localGet(QW)), f32Mul(localGet(WX), localGet(QY))),
              f32Mul(localGet(WY), localGet(QX))
            )
          )
        )
      ),
      // dqw = 0.5*dt * (-WX*QX - WY*QY - WZ*QZ)
      ...localSet(
        TMP2,
        f32Sub(
          localGet(QW),
          f32Mul(
            localGet(TMP1),
            f32Add(
              f32Mul(localGet(WX), localGet(QX)),
              f32Add(f32Mul(localGet(WY), localGet(QY)), f32Mul(localGet(WZ), localGet(QZ)))
            )
          )
        )
      ),
      // Normalize quaternion
      ...localSet(
        TMP3,
        f32Div(
          f32Const(1.0),
          f32Max(
            f32Const(1e-6),
            f32Sqrt(
              f32Add(
                f32Add(f32Mul(localGet(DX), localGet(DX)), f32Mul(localGet(DY), localGet(DY))),
                f32Add(f32Mul(localGet(DZ), localGet(DZ)), f32Mul(localGet(TMP2), localGet(TMP2)))
              )
            )
          )
        )
      ),
      ...f32Store(localGet(PTR), 24, f32Mul(localGet(DX), localGet(TMP3))),
      ...f32Store(localGet(PTR), 28, f32Mul(localGet(DY), localGet(TMP3))),
      ...f32Store(localGet(PTR), 32, f32Mul(localGet(DZ), localGet(TMP3))),
      ...f32Store(localGet(PTR), 36, f32Mul(localGet(TMP2), localGet(TMP3))),
      ...f32Store(localGet(PTR), 40, localGet(WX)),
      ...f32Store(localGet(PTR), 44, localGet(WY)),
      ...f32Store(localGet(PTR), 48, localGet(WZ)),
    ]),

    // Apply dampingFactor and Semi-Implicit Euler Position Integration: V *= DAMP; P += V * DT
    ...localSet(VX, f32Mul(localGet(VX), localGet(DAMP))),
    ...localSet(VY, f32Mul(localGet(VY), localGet(DAMP))),
    ...localSet(VZ, f32Mul(localGet(VZ), localGet(DAMP))),
    ...localSet(PX, f32Add(localGet(PX), f32Mul(localGet(VX), localGet(DT)))),
    ...localSet(PY, f32Add(localGet(PY), f32Mul(localGet(VY), localGet(DT)))),
    ...localSet(PZ, f32Add(localGet(PZ), f32Mul(localGet(VZ), localGet(DT)))),

    // Store updated pos & vel
    ...f32Store(localGet(PTR), 0, localGet(PX)),
    ...f32Store(localGet(PTR), 4, localGet(PY)),
    ...f32Store(localGet(PTR), 8, localGet(PZ)),
    ...f32Store(localGet(PTR), 12, localGet(VX)),
    ...f32Store(localGet(PTR), 16, localGet(VY)),
    ...f32Store(localGet(PTR), 20, localGet(VZ)),

    // Accumulate total kinetic energy: ENERGY += VX^2 + VY^2 + VZ^2
    ...localSet(
      ENERGY,
      f32Add(
        localGet(ENERGY),
        f32Add(
          f32Mul(localGet(VX), localGet(VX)),
          f32Add(f32Mul(localGet(VY), localGet(VY)), f32Mul(localGet(VZ), localGet(VZ)))
        )
      )
    ),

    // I = I + 1; br $top
    ...localSet(I, [...localGet(I), ...i32Const(1), 0x6a]),
    0x0c,
    0x00,
    0x0b, // end loop
    0x0b, // end block

    // Increment stepCounter at uniform[2] (offset 2056) and store ENERGY at uniform[3] (offset 2060)
    ...f32StoreUniform(2, f32Add(f32LoadUniform(2), f32Const(1.0))),
    ...f32StoreUniform(3, localGet(ENERGY)),

    // return ENERGY
    ...localGet(ENERGY),
    0x0b,
  ];

  const funcBody = [...locals, ...bodyInstr];
  const codePayload = [1, ...uleb(funcBody.length), ...funcBody];
  const codeSection = section(10, codePayload);

  return new Uint8Array([
    0x00,
    0x61,
    0x73,
    0x6d,
    0x01,
    0x00,
    0x00,
    0x00,
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);
}

let _compiledModule: WebAssembly.Module | null = null;
let _cachedBytecode: Uint8Array | null = null;

/**
 * Creates an isolated WebAssembly 3D Hydrodynamic Physics Kernel with its own dedicated `WebAssembly.Memory`.
 */
export function createWasmPhysicsKernel(): WasmPhysicsKernel {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const ringFloatView = new Float32Array(memory.buffer, 0, 512);
  const uniformFloatView = new Float32Array(memory.buffer, 2048, 32);

  if (!_cachedBytecode) {
    _cachedBytecode = buildWasmPhysicsBinary();
  }
  const bytecode = _cachedBytecode;

  let wasmIntegrateStep: (ringCount: number, dt: number, dampingFactor: number) => number;
  let isWasmReady = false;

  if (!_compiledModule) {
    _compiledModule = new WebAssembly.Module(bytecode.buffer as ArrayBuffer);
  }
  const instance = new WebAssembly.Instance(_compiledModule, {
    env: { mem: memory },
  });
  const exportedIntegrate = instance.exports.integrate as (count: number) => number;
  wasmIntegrateStep = (ringCount: number, dt: number, dampingFactor: number) => {
    uniformFloatView[0] = dt;
    uniformFloatView[1] = dampingFactor;
    return exportedIntegrate(ringCount);
  };
  isWasmReady = true;

  return {
    isWasmReady,
    memory,
    ringFloatView,
    uniformFloatView,
    wasmIntegrateStep,
    getWasmBytecodeLength: () => bytecode.byteLength,
  };
}

export const getWasmPhysicsKernel = createWasmPhysicsKernel;
