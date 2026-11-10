/**
 * Hand-crafted WebAssembly (WASM) Linear Memory Physics Accelerator Kernel
 *
 * Each simulation state creates its own isolated `WebAssembly.Instance` & `WebAssembly.Memory`
 * via `createWasmPhysicsKernel()` (no global mutable singleton sharing memory across instances!).
 *
 * Memory Layout (Float32 elements, 4 bytes each):
 * - Offset 0..511 (Bytes 0..2047): Ring State Array (up to 16 rings * 16 floats per ring = 256 floats)
 *   Per-ring stride = 16 floats (64 bytes):
 *     [0]: posX, [1]: posY, [2]: posZ,
 *     [3]: velX, [4]: velY, [5]: velZ,
 *     [6]: quatX, [7]: quatY, [8]: quatZ, [9]: quatW,
 *     [10]: angVelX, [11]: angVelY, [12]: angVelZ,
 *     [13]: majorRadius, [14]: minorRadius, [15]: hookedFlag
 * - Offset 512..543 (Bytes 2048..2175): Global Simulation Uniforms (`uniformFloatView`, actively read/written by WASM!)
 *     [0] (Byte 2048): subDt (f32, read by WASM integrate kernel)
 *     [1] (Byte 2052): linearDamping (f32, read by WASM integrate kernel)
 *     [2] (Byte 2056): stepCounter (f32, incremented inside WASM integrate kernel)
 *     [3] (Byte 2060): lastKineticEnergy (f32, written inside WASM integrate kernel)
 *     [4..6] (Bytes 2064..2075): gravityX, gravityY, gravityZ
 *     [7..9] (Bytes 2076..2087): waterAngularVelX, waterAngularVelY, waterAngularVelZ
 */

export interface WasmPhysicsKernel {
  isWasmReady: boolean;
  memory: WebAssembly.Memory;
  ringFloatView: Float32Array;
  uniformFloatView: Float32Array;
  /**
   * Executes f32 vector integration & damping pass inside WebAssembly binary.
   * Reads `subDt` and `dampingFactor` via `uniformFloatView[0..1]` at byte offset 2048,
   * increments `uniformFloatView[2]` (stepCounter), writes `uniformFloatView[3]` (kineticEnergy),
   * and returns total kinetic energy.
   */
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

function section(id: number, payload: number[]): number[] {
  return [id, ...uleb(payload.length), ...payload];
}

/**
 * Dynamically encodes a valid WebAssembly binary module with exported function:
 * `integrate(ringCount: i32) -> f32`
 * which reads `dt` from `memory[2048]` (`uniformFloatView[0]`), `damping` from `memory[2052]` (`uniformFloatView[1]`),
 * updates all `ringCount` rings in `memory[0..ringCount*64]`, increments `memory[2056]` (`uniformFloatView[2]`),
 * stores total kinetic energy at `memory[2060]` (`uniformFloatView[3]`), and returns it.
 */
function buildWasmPhysicsBinary(): Uint8Array {
  // Type section: 1 function type (i32) -> (f32)
  const typeSection = section(1, [
    1,
    0x60,
    1,
    0x7f, // param 0: ringCount (i32)
    1,
    0x7d, // result: energy (f32)
  ]);

  // Import section: import env.mem (1 page minimum)
  const modName = [0x65, 0x6e, 0x76]; // "env"
  const memName = [0x6d, 0x65, 0x6d]; // "mem"
  const importSection = section(2, [
    1,
    modName.length,
    ...modName,
    memName.length,
    ...memName,
    0x02,
    0x00,
    0x01,
  ]);

  // Function section: 1 function of type index 0
  const funcSection = section(3, [1, 0]);

  // Export section: export function 0 as "integrate"
  const expName = [0x69, 0x6e, 0x74, 0x65, 0x67, 0x72, 0x61, 0x74, 0x65];
  const exportSection = section(7, [
    1,
    expName.length,
    ...expName,
    0x00,
    0x00,
  ]);

  // Locals:
  // param 0: ringCount (i32)
  // local 1: i (i32)
  // local 2: ptr (i32)
  // local 3: dt (f32)
  // local 4: damping (f32)
  // local 5: vx (f32)
  // local 6: vy (f32)
  // local 7: vz (f32)
  // local 8: energy (f32)
  const locals = [
    2,
    2, 0x7f, // locals 1, 2 (i32)
    6, 0x7d, // locals 3..8 (f32)
  ];

  // Byte offset 2048 in unsigned LEB128 is [0x80, 0x10]
  // 2052 is [0x84, 0x10], 2056 is [0x88, 0x10], 2060 is [0x8c, 0x10]
  const bodyInstr: number[] = [
    // dt = f32.load(offset=2048, addr=0)
    0x41, 0x00,
    0x2a, 0x02, 0x80, 0x10,
    0x21, 0x03, // local.set 3 (dt)

    // damping = f32.load(offset=2052, addr=0)
    0x41, 0x00,
    0x2a, 0x02, 0x84, 0x10,
    0x21, 0x04, // local.set 4 (damping)

    // i = 0; energy = 0.0
    0x41, 0x00,
    0x21, 0x01,
    0x43, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x08,

    // block $break
    0x02, 0x40,
    // loop $top
    0x03, 0x40,

    // br_if $break (i >= ringCount)
    0x20, 0x01,
    0x20, 0x00,
    0x4f,
    0x0d, 0x01,

    // ptr = i << 6
    0x20, 0x01,
    0x41, 0x06,
    0x74,
    0x21, 0x02,

    // vx = f32.load(ptr + 12) * damping
    0x20, 0x02,
    0x2a, 0x02, 0x0c,
    0x20, 0x04,
    0x94,
    0x21, 0x05,
    0x20, 0x02,
    0x20, 0x05,
    0x38, 0x02, 0x0c,

    // vy = f32.load(ptr + 16) * damping
    0x20, 0x02,
    0x2a, 0x02, 0x10,
    0x20, 0x04,
    0x94,
    0x21, 0x06,
    0x20, 0x02,
    0x20, 0x06,
    0x38, 0x02, 0x10,

    // vz = f32.load(ptr + 20) * damping
    0x20, 0x02,
    0x2a, 0x02, 0x14,
    0x20, 0x04,
    0x94,
    0x21, 0x07,
    0x20, 0x02,
    0x20, 0x07,
    0x38, 0x02, 0x14,

    // posX = f32.load(ptr + 0) + vx * dt
    0x20, 0x02,
    0x20, 0x02,
    0x2a, 0x02, 0x00,
    0x20, 0x05,
    0x20, 0x03,
    0x94,
    0x92,
    0x38, 0x02, 0x00,

    // posY = f32.load(ptr + 4) + vy * dt
    0x20, 0x02,
    0x20, 0x02,
    0x2a, 0x02, 0x04,
    0x20, 0x06,
    0x20, 0x03,
    0x94,
    0x92,
    0x38, 0x02, 0x04,

    // posZ = f32.load(ptr + 8) + vz * dt
    0x20, 0x02,
    0x20, 0x02,
    0x2a, 0x02, 0x08,
    0x20, 0x07,
    0x20, 0x03,
    0x94,
    0x92,
    0x38, 0x02, 0x08,

    // energy += vx*vx + vy*vy + vz*vz
    0x20, 0x08,
    0x20, 0x05,
    0x20, 0x05,
    0x94,
    0x20, 0x06,
    0x20, 0x06,
    0x94,
    0x92,
    0x20, 0x07,
    0x20, 0x07,
    0x94,
    0x92,
    0x92,
    0x21, 0x08,

    // i = i + 1
    0x20, 0x01,
    0x41, 0x01,
    0x6a,
    0x21, 0x01,
    0x0c, 0x00,

    0x0b, // end loop
    0x0b, // end block

    // stepCounter (offset 2056) += 1.0
    0x41, 0x00,
    0x41, 0x00,
    0x2a, 0x02, 0x88, 0x10,
    0x43, 0x00, 0x00, 0x80, 0x3f, // f32.const 1.0
    0x92,
    0x38, 0x02, 0x88, 0x10,

    // store energy at offset 2060
    0x41, 0x00,
    0x20, 0x08,
    0x38, 0x02, 0x8c, 0x10,

    // return energy
    0x20, 0x08,
    0x0b,
  ];

  const funcBody = [...locals, ...bodyInstr];
  const codePayload = [1, ...uleb(funcBody.length), ...funcBody];
  const codeSection = section(10, codePayload);

  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
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
 * Creates a fresh, isolated WebAssembly Physics Kernel with its own dedicated `WebAssembly.Memory`
 * so concurrent `SimulationState` instances never clobber each other's linear memory!
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

  try {
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
  } catch {
    wasmIntegrateStep = (ringCount: number, dt: number, dampingFactor: number) => {
      uniformFloatView[0] = dt;
      uniformFloatView[1] = dampingFactor;
      let energy = 0;
      for (let i = 0; i < ringCount; i++) {
        const base = i * 16;
        const vx = ringFloatView[base + 3] * dampingFactor;
        const vy = ringFloatView[base + 4] * dampingFactor;
        const vz = ringFloatView[base + 5] * dampingFactor;
        ringFloatView[base + 3] = vx;
        ringFloatView[base + 4] = vy;
        ringFloatView[base + 5] = vz;
        ringFloatView[base + 0] += vx * dt;
        ringFloatView[base + 1] += vy * dt;
        ringFloatView[base + 2] += vz * dt;
        energy += vx * vx + vy * vy + vz * vz;
      }
      uniformFloatView[2] += 1;
      uniformFloatView[3] = energy;
      return energy;
    };
  }

  return {
    isWasmReady,
    memory,
    ringFloatView,
    uniformFloatView,
    wasmIntegrateStep,
    getWasmBytecodeLength: () => bytecode.byteLength,
  };
}

// Keep alias for backwards compatibility
export const getWasmPhysicsKernel = createWasmPhysicsKernel;
