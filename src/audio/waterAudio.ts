/**
 * 100% Procedural Web Audio Synthesizer for 1976 Tomy Waterful Ring-Toss
 * Simulates mechanical rubber diaphragm pump squish, water jet turbulence, rising bubble pops,
 * underwater plastic ring clacks, and victory fanfare.
 */

class WaterSoundEngine {
  private ctx: AudioContext | null = null;
  private isMuted = false;

  private ensureContext(): AudioContext | null {
    if (this.isMuted) return null;
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  public setMuted(muted: boolean): void {
    this.isMuted = muted;
  }

  public getMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Plays mechanical rubber bellows squeeze + pressurized water jet burst + bubble pops
   */
  public playPumpPress(side: 'left' | 'right'): void {
    const ctx = this.ensureContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const pitchOffset = side === 'left' ? 0.96 : 1.04;

    // 1. Mechanical plastic button + rubber diaphragm thump
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(145 * pitchOffset, now);
    osc.frequency.exponentialRampToValueAtTime(52 * pitchOffset, now + 0.09);

    gain.gain.setValueAtTime(0.24, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.11);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.12);

    // 2. Pressurized water jet whoosh (bandpass filtered white noise)
    const bufferSize = Math.floor(ctx.sampleRate * 0.28);
    const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }

    const whiteNoise = ctx.createBufferSource();
    whiteNoise.buffer = noiseBuffer;

    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(420 * pitchOffset, now);
    filter.frequency.exponentialRampToValueAtTime(1180 * pitchOffset, now + 0.14);
    filter.frequency.exponentialRampToValueAtTime(540 * pitchOffset, now + 0.27);
    filter.Q.value = 3.2;

    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.01, now);
    noiseGain.gain.linearRampToValueAtTime(0.18, now + 0.04);
    noiseGain.gain.exponentialRampToValueAtTime(0.002, now + 0.27);

    whiteNoise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(ctx.destination);
    whiteNoise.start(now);
    whiteNoise.stop(now + 0.28);

    // 3. 2-3 Resonant underwater bubble blips
    for (let b = 0; b < 3; b++) {
      const bStart = now + 0.03 + b * 0.055 + Math.random() * 0.02;
      const bOsc = ctx.createOscillator();
      const bGain = ctx.createGain();
      bOsc.type = 'sine';
      const baseFreq = (480 + b * 140 + Math.random() * 90) * pitchOffset;
      bOsc.frequency.setValueAtTime(baseFreq, bStart);
      bOsc.frequency.exponentialRampToValueAtTime(baseFreq * 1.65, bStart + 0.045);

      bGain.gain.setValueAtTime(0.06, bStart);
      bGain.gain.exponentialRampToValueAtTime(0.001, bStart + 0.05);

      bOsc.connect(bGain);
      bGain.connect(ctx.destination);
      bOsc.start(bStart);
      bOsc.stop(bStart + 0.055);
    }
  }

  /**
   * Subtle spring return sound when releasing pump button
   */
  public playPumpRelease(): void {
    const ctx = this.ensureContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(75, now);
    osc.frequency.exponentialRampToValueAtTime(135, now + 0.06);

    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.065);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.07);
  }

  /**
   * Silent by design: no sound effect when a ring enters/hooks onto a needle ("插入的时候不要有任何音效")
   */
  public playRingHooked(_totalHooked?: number): void {
    // Intentionally silent
  }

  /**
   * Silent by design: no sound effect when rings are inserted
   */
  public playVictoryFanfare(): void {
    // Intentionally silent
  }
}

export const waterAudio = new WaterSoundEngine();
