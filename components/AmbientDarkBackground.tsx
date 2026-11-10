'use client';

import React, { useEffect, useRef } from 'react';
import { useTheme } from './ThemeProvider';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  baseAlpha: number;
  phase: number;
  pulseSpeed: number;
  color: string;
  glowColor: string;
}

interface LightFlare {
  baseX: number;
  baseY: number;
  orbitRx: number;
  orbitRy: number;
  speedX: number;
  speedY: number;
  phase: number;
  radius: number;
  colorInner: string;
  colorMid: string;
}

interface WaveConfig {
  yRatio: number;
  amplitude1: number;
  amplitude2: number;
  freq1: number;
  freq2: number;
  speed1: number;
  speed2: number;
  fillTop: string;
  fillBottom: string;
  strokeColor: string;
  glowColor: string;
  strokeWidth: number;
}

const INTERNAL_W = 1080;
const INTERNAL_H = 640;

const WAVE_LAYERS: WaveConfig[] = [
  {
    yRatio: 0.28,
    amplitude1: 42,
    amplitude2: 24,
    freq1: 0.0032,
    freq2: 0.0058,
    speed1: 0.95,
    speed2: -0.62,
    fillTop: 'rgba(116, 91, 255, 0.26)',
    fillBottom: 'rgba(5, 11, 24, 0.0)',
    strokeColor: 'rgba(155, 135, 255, 0.78)',
    glowColor: 'rgba(116, 91, 255, 0.28)',
    strokeWidth: 2.0,
  },
  {
    yRatio: 0.45,
    amplitude1: 52,
    amplitude2: 28,
    freq1: 0.0026,
    freq2: 0.0048,
    speed1: -1.05,
    speed2: 0.74,
    fillTop: 'rgba(37, 139, 255, 0.32)',
    fillBottom: 'rgba(5, 11, 24, 0.0)',
    strokeColor: 'rgba(53, 216, 242, 0.85)',
    glowColor: 'rgba(37, 139, 255, 0.34)',
    strokeWidth: 2.2,
  },
  {
    yRatio: 0.62,
    amplitude1: 46,
    amplitude2: 26,
    freq1: 0.0035,
    freq2: 0.0062,
    speed1: 1.18,
    speed2: 0.80,
    fillTop: 'rgba(53, 216, 242, 0.30)',
    fillBottom: 'rgba(5, 11, 24, 0.0)',
    strokeColor: 'rgba(53, 216, 242, 0.92)',
    glowColor: 'rgba(53, 216, 242, 0.36)',
    strokeWidth: 2.4,
  },
  {
    yRatio: 0.78,
    amplitude1: 38,
    amplitude2: 22,
    freq1: 0.0029,
    freq2: 0.0052,
    speed1: -0.88,
    speed2: -0.55,
    fillTop: 'rgba(87, 155, 255, 0.26)',
    fillBottom: 'rgba(5, 11, 24, 0.0)',
    strokeColor: 'rgba(87, 155, 255, 0.80)',
    glowColor: 'rgba(37, 139, 255, 0.30)',
    strokeWidth: 2.0,
  },
  {
    yRatio: 0.89,
    amplitude1: 32,
    amplitude2: 18,
    freq1: 0.0038,
    freq2: 0.0068,
    speed1: 1.02,
    speed2: -0.70,
    fillTop: 'rgba(53, 216, 242, 0.22)',
    fillBottom: 'rgba(5, 11, 24, 0.0)',
    strokeColor: 'rgba(116, 91, 255, 0.72)',
    glowColor: 'rgba(116, 91, 255, 0.25)',
    strokeWidth: 1.8,
  },
];

const LIGHT_FLARES: LightFlare[] = [
  {
    baseX: 0.18,
    baseY: 0.22,
    orbitRx: 0.14,
    orbitRy: 0.10,
    speedX: 0.42,
    speedY: 0.34,
    phase: 0.0,
    radius: 310,
    colorInner: 'rgba(37, 139, 255, 0.38)',
    colorMid: 'rgba(53, 216, 242, 0.16)',
  },
  {
    baseX: 0.82,
    baseY: 0.28,
    orbitRx: 0.15,
    orbitRy: 0.12,
    speedX: -0.38,
    speedY: 0.46,
    phase: 2.1,
    radius: 340,
    colorInner: 'rgba(53, 216, 242, 0.36)',
    colorMid: 'rgba(116, 91, 255, 0.18)',
  },
  {
    baseX: 0.50,
    baseY: 0.54,
    orbitRx: 0.22,
    orbitRy: 0.14,
    speedX: 0.33,
    speedY: -0.41,
    phase: 4.2,
    radius: 380,
    colorInner: 'rgba(116, 91, 255, 0.32)',
    colorMid: 'rgba(37, 139, 255, 0.16)',
  },
  {
    baseX: 0.24,
    baseY: 0.80,
    orbitRx: 0.16,
    orbitRy: 0.10,
    speedX: -0.44,
    speedY: -0.35,
    phase: 1.4,
    radius: 320,
    colorInner: 'rgba(53, 216, 242, 0.32)',
    colorMid: 'rgba(37, 139, 255, 0.14)',
  },
  {
    baseX: 0.78,
    baseY: 0.76,
    orbitRx: 0.14,
    orbitRy: 0.11,
    speedX: 0.36,
    speedY: 0.39,
    phase: 3.5,
    radius: 300,
    colorInner: 'rgba(37, 139, 255, 0.34)',
    colorMid: 'rgba(116, 91, 255, 0.16)',
  },
];

function createInitialParticles(width: number, height: number): Particle[] {
  const palette = [
    { color: '#35D8F2', glow: 'rgba(53, 216, 242, 0.32)' },
    { color: '#258BFF', glow: 'rgba(37, 139, 255, 0.32)' },
    { color: '#579BFF', glow: 'rgba(87, 155, 255, 0.28)' },
    { color: '#745BFF', glow: 'rgba(116, 91, 255, 0.28)' },
  ];

  const count = 42;
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    const s1 = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
    const r1 = s1 - Math.floor(s1);
    const s2 = Math.sin(i * 45.164 + 19.13) * 23421.631;
    const r2 = s2 - Math.floor(s2);
    const s3 = Math.sin(i * 93.712 + 51.42) * 15731.743;
    const r3 = s3 - Math.floor(s3);

    const pColor = palette[i % palette.length];
    particles.push({
      x: r1 * width,
      y: r2 * height,
      vx: (r3 - 0.45) * 22,
      vy: -(14 + r1 * 24),
      radius: 1.4 + (i % 4) * 0.65,
      baseAlpha: 0.42 + (i % 5) * 0.11,
      phase: r2 * Math.PI * 2,
      pulseSpeed: 1.4 + r3 * 1.8,
      color: pColor.color,
      glowColor: pColor.glow,
    });
  }
  return particles;
}

export default function AmbientDarkBackground() {
  const { isDark } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!isDark) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = INTERNAL_W;
    const height = INTERNAL_H;
    const particles = createInitialParticles(width, height);

    let rafId: number;
    let lastTime = performance.now();
    let elapsed = 0;

    const renderFrame = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      if (!document.hidden) {
        elapsed += dt;

        // 1. Base Midnight Navy Canvas (#050B18 -> #081326)
        const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
        bgGrad.addColorStop(0, '#050B18');
        bgGrad.addColorStop(0.5, '#081326');
        bgGrad.addColorStop(1, '#050B18');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // 2. Orbiting & Breathing Ambient Light Flares
        for (const flare of LIGHT_FLARES) {
          const cx =
            (flare.baseX +
              Math.sin(elapsed * flare.speedX + flare.phase) * flare.orbitRx) *
            width;
          const cy =
            (flare.baseY +
              Math.cos(elapsed * flare.speedY + flare.phase) * flare.orbitRy) *
            height;
          const pulse = 0.85 + 0.22 * Math.sin(elapsed * 1.1 + flare.phase);
          const r = flare.radius * pulse;

          const radGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
          radGrad.addColorStop(0, flare.colorInner);
          radGrad.addColorStop(0.5, flare.colorMid);
          radGrad.addColorStop(1, 'rgba(5, 11, 24, 0)');

          ctx.fillStyle = radGrad;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fill();
        }

        // 3. Flowing Multi-Layered Sine-Harmonic Aurora Waves (Hardware-Friendly Dual-Stroke Glow)
        const stepX = 16;
        for (const wave of WAVE_LAYERS) {
          const baseY = wave.yRatio * height;

          ctx.beginPath();
          for (let x = 0; x <= width + stepX; x += stepX) {
            const y =
              baseY +
              Math.sin(x * wave.freq1 + elapsed * wave.speed1) * wave.amplitude1 +
              Math.cos(x * wave.freq2 + elapsed * wave.speed2) * wave.amplitude2;
            if (x === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }

          // Wide soft outer glow pass (zero shadowBlur overhead)
          ctx.strokeStyle = wave.glowColor;
          ctx.lineWidth = wave.strokeWidth * 4.5;
          ctx.stroke();

          // Bright inner wave crest line
          ctx.strokeStyle = wave.strokeColor;
          ctx.lineWidth = wave.strokeWidth;
          ctx.stroke();

          // Fill the gradient curtain below the wave crest
          ctx.lineTo(width + stepX, height);
          ctx.lineTo(0, height);
          ctx.closePath();

          const waveGrad = ctx.createLinearGradient(
            0,
            baseY - wave.amplitude1 - 15,
            0,
            Math.min(height, baseY + 240)
          );
          waveGrad.addColorStop(0, wave.fillTop);
          waveGrad.addColorStop(1, wave.fillBottom);
          ctx.fillStyle = waveGrad;
          ctx.fill();
        }

        // 4. Floating & Twinkling Glowing Particles
        for (const p of particles) {
          p.x += p.vx * dt;
          p.y += p.vy * dt;

          if (p.y < -10) {
            p.y = height + 10;
            p.x = (p.x + width * 0.37) % width;
          }
          if (p.x < -10) p.x = width + 10;
          if (p.x > width + 10) p.x = -10;

          const twinkle =
            0.45 + 0.55 * Math.sin(elapsed * p.pulseSpeed + p.phase);
          const alpha = Math.max(0.15, Math.min(0.95, p.baseAlpha * twinkle));

          // Soft outer particle halo
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * 3.2, 0, Math.PI * 2);
          ctx.fillStyle = p.glowColor;
          ctx.fill();

          // Bright particle core
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.globalAlpha = alpha;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      rafId = requestAnimationFrame(renderFrame);
    };

    rafId = requestAnimationFrame(renderFrame);

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [isDark]);

  if (!isDark) {
    return null;
  }

  return (
    <div
      data-testid="ambient-dark-background"
      data-animated="true"
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#050B18] select-none"
    >
      <canvas
        ref={canvasRef}
        width={INTERNAL_W}
        height={INTERNAL_H}
        className="w-full h-full block"
      />
    </div>
  );
}
