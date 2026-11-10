'use client';

import React, { useMemo } from 'react';
import { useTheme } from './ThemeProvider';

interface ParticleSpec {
  id: number;
  left: string;
  top: string;
  size: number;
  color: string;
  glowColor: string;
  duration: string;
  delay: string;
}

interface FlareSpec {
  id: string;
  left: string;
  top: string;
  width: string;
  height: string;
  gradient: string;
  animation: string;
}

interface WaveRibbonSpec {
  id: string;
  fillGradientId: string;
  strokeGradientId: string;
  crestPath: string;
  curtainPath: string;
  glowStrokeWidth: number;
  coreStrokeWidth: number;
  glowOpacity: number;
  animation: string;
}

const FLARE_SPECS: FlareSpec[] = [
  {
    id: 'flare-top-left',
    left: '4%',
    top: '6%',
    width: '48vw',
    height: '48vw',
    gradient:
      'radial-gradient(circle, rgba(37, 139, 255, 0.36) 0%, rgba(53, 216, 242, 0.16) 45%, rgba(5, 11, 24, 0) 72%)',
    animation: 'neurolab-flare-orbit-a 18s ease-in-out infinite',
  },
  {
    id: 'flare-top-right',
    left: '56%',
    top: '8%',
    width: '50vw',
    height: '50vw',
    gradient:
      'radial-gradient(circle, rgba(53, 216, 242, 0.34) 0%, rgba(116, 91, 255, 0.18) 48%, rgba(5, 11, 24, 0) 72%)',
    animation: 'neurolab-flare-orbit-b 22s ease-in-out infinite',
  },
  {
    id: 'flare-center-violet',
    left: '28%',
    top: '28%',
    width: '54vw',
    height: '54vw',
    gradient:
      'radial-gradient(circle, rgba(116, 91, 255, 0.30) 0%, rgba(37, 139, 255, 0.15) 48%, rgba(5, 11, 24, 0) 72%)',
    animation: 'neurolab-flare-orbit-c 20s ease-in-out infinite',
  },
  {
    id: 'flare-bottom-left',
    left: '6%',
    top: '54%',
    width: '46vw',
    height: '46vw',
    gradient:
      'radial-gradient(circle, rgba(53, 216, 242, 0.30) 0%, rgba(37, 139, 255, 0.14) 46%, rgba(5, 11, 24, 0) 72%)',
    animation: 'neurolab-flare-orbit-b 25s ease-in-out infinite reverse',
  },
  {
    id: 'flare-bottom-right',
    left: '54%',
    top: '52%',
    width: '48vw',
    height: '48vw',
    gradient:
      'radial-gradient(circle, rgba(37, 139, 255, 0.32) 0%, rgba(116, 91, 255, 0.16) 46%, rgba(5, 11, 24, 0) 72%)',
    animation: 'neurolab-flare-orbit-a 21s ease-in-out infinite reverse',
  },
];

/**
 * Each wave ribbon spans 2880px horizontally (two identical 1440px periods)
 * so translating from 0% to -50% via translate3d loops seamlessly on the GPU compositor.
 */
const WAVE_RIBBONS: WaveRibbonSpec[] = [
  {
    id: 'wave-ribbon-1',
    fillGradientId: 'nl-wave-fill-violet',
    strokeGradientId: 'nl-wave-stroke-violet',
    crestPath:
      'M 0 240 C 360 140, 720 340, 1080 210 C 1260 150, 1350 290, 1440 240 C 1800 140, 2160 340, 2520 210 C 2700 150, 2790 290, 2880 240',
    curtainPath:
      'M 0 240 C 360 140, 720 340, 1080 210 C 1260 150, 1350 290, 1440 240 C 1800 140, 2160 340, 2520 210 C 2700 150, 2790 290, 2880 240 L 2880 900 L 0 900 Z',
    glowStrokeWidth: 9,
    coreStrokeWidth: 2.2,
    glowOpacity: 0.32,
    animation: 'neurolab-wave-flow-forward 22s linear infinite',
  },
  {
    id: 'wave-ribbon-2',
    fillGradientId: 'nl-wave-fill-electric',
    strokeGradientId: 'nl-wave-stroke-cyan',
    crestPath:
      'M 0 390 C 320 490, 680 270, 1040 410 C 1220 470, 1340 330, 1440 390 C 1760 490, 2120 270, 2480 410 C 2660 470, 2780 330, 2880 390',
    curtainPath:
      'M 0 390 C 320 490, 680 270, 1040 410 C 1220 470, 1340 330, 1440 390 C 1760 490, 2120 270, 2480 410 C 2660 470, 2780 330, 2880 390 L 2880 900 L 0 900 Z',
    glowStrokeWidth: 10,
    coreStrokeWidth: 2.4,
    glowOpacity: 0.36,
    animation: 'neurolab-wave-flow-reverse 18s linear infinite',
  },
  {
    id: 'wave-ribbon-3',
    fillGradientId: 'nl-wave-fill-cyan',
    strokeGradientId: 'nl-wave-stroke-cyan',
    crestPath:
      'M 0 540 C 380 430, 740 650, 1100 510 C 1270 440, 1360 590, 1440 540 C 1820 430, 2180 650, 2540 510 C 2710 440, 2800 590, 2880 540',
    curtainPath:
      'M 0 540 C 380 430, 740 650, 1100 510 C 1270 440, 1360 590, 1440 540 C 1820 430, 2180 650, 2540 510 C 2710 440, 2800 590, 2880 540 L 2880 900 L 0 900 Z',
    glowStrokeWidth: 11,
    coreStrokeWidth: 2.6,
    glowOpacity: 0.4,
    animation: 'neurolab-wave-flow-forward 15s linear infinite',
  },
  {
    id: 'wave-ribbon-4',
    fillGradientId: 'nl-wave-fill-softblue',
    strokeGradientId: 'nl-wave-stroke-electric',
    crestPath:
      'M 0 680 C 340 770, 710 580, 1060 700 C 1240 755, 1350 635, 1440 680 C 1780 770, 2150 580, 2500 700 C 2680 755, 2790 635, 2880 680',
    curtainPath:
      'M 0 680 C 340 770, 710 580, 1060 700 C 1240 755, 1350 635, 1440 680 C 1780 770, 2150 580, 2500 700 C 2680 755, 2790 635, 2880 680 L 2880 900 L 0 900 Z',
    glowStrokeWidth: 9,
    coreStrokeWidth: 2.1,
    glowOpacity: 0.34,
    animation: 'neurolab-wave-flow-reverse 20s linear infinite',
  },
  {
    id: 'wave-ribbon-5',
    fillGradientId: 'nl-wave-fill-violet',
    strokeGradientId: 'nl-wave-stroke-violet',
    crestPath:
      'M 0 790 C 360 710, 720 860, 1080 765 C 1260 720, 1350 825, 1440 790 C 1800 710, 2160 860, 2520 765 C 2700 720, 2790 825, 2880 790',
    curtainPath:
      'M 0 790 C 360 710, 720 860, 1080 765 C 1260 720, 1350 825, 1440 790 C 1800 710, 2160 860, 2520 765 C 2700 720, 2790 825, 2880 790 L 2880 900 L 0 900 Z',
    glowStrokeWidth: 8,
    coreStrokeWidth: 1.9,
    glowOpacity: 0.28,
    animation: 'neurolab-wave-flow-forward 25s linear infinite',
  },
];

function buildDeterministicParticles(): ParticleSpec[] {
  const palette = [
    { color: '#35D8F2', glow: 'rgba(53, 216, 242, 0.55)' },
    { color: '#258BFF', glow: 'rgba(37, 139, 255, 0.55)' },
    { color: '#579BFF', glow: 'rgba(87, 155, 255, 0.50)' },
    { color: '#745BFF', glow: 'rgba(116, 91, 255, 0.50)' },
  ];

  const items: ParticleSpec[] = [];
  const count = 36;
  for (let i = 0; i < count; i++) {
    const s1 = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
    const r1 = s1 - Math.floor(s1);
    const s2 = Math.sin(i * 45.164 + 19.13) * 23421.631;
    const r2 = s2 - Math.floor(s2);
    const s3 = Math.sin(i * 93.712 + 51.42) * 15731.743;
    const r3 = s3 - Math.floor(s3);

    const pColor = palette[i % palette.length];
    const size = 3 + (i % 4) * 1.5;
    const durationSec = (9 + r3 * 9).toFixed(2);
    const delaySec = (-r2 * 14).toFixed(2);

    items.push({
      id: i,
      left: `${(2 + r1 * 96).toFixed(2)}%`,
      top: `${(8 + r2 * 88).toFixed(2)}%`,
      size,
      color: pColor.color,
      glowColor: pColor.glow,
      duration: `${durationSec}s`,
      delay: `${delaySec}s`,
    });
  }
  return items;
}

export default function AmbientDarkBackground() {
  const { isDark } = useTheme();
  const particles = useMemo(() => buildDeterministicParticles(), []);

  if (!isDark) {
    return null;
  }

  return (
    <div
      data-testid="ambient-dark-background"
      data-animated="true"
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#050B18] select-none"
      style={{
        background:
          'linear-gradient(180deg, #050B18 0%, #081326 50%, #050B18 100%)',
        contain: 'strict',
      }}
    >
      {/* Layer 1: GPU-Composited Orbiting & Breathing Radial Light Flares */}
      <div className="absolute inset-0 overflow-hidden">
        {FLARE_SPECS.map((flare) => (
          <div
            key={flare.id}
            data-testid={flare.id}
            className="gpu-accelerated-layer absolute rounded-full"
            style={{
              left: flare.left,
              top: flare.top,
              width: flare.width,
              height: flare.height,
              background: flare.gradient,
              willChange: 'transform, opacity',
              transform: 'translate3d(0, 0, 0)',
              animation: flare.animation,
            }}
          />
        ))}
      </div>

      {/* Layer 2: GPU-Composited Multi-Layered Flowing Aurora Wave Ribbons */}
      <div className="absolute inset-0 overflow-hidden">
        {WAVE_RIBBONS.map((wave) => (
          <div
            key={wave.id}
            data-testid={wave.id}
            className="gpu-accelerated-layer absolute top-0 left-0 h-full w-[200%]"
            style={{
              willChange: 'transform',
              transform: 'translate3d(0, 0, 0)',
              animation: wave.animation,
            }}
          >
            <svg
              viewBox="0 0 2880 900"
              preserveAspectRatio="none"
              className="w-full h-full block"
            >
              <defs>
                <linearGradient id="nl-wave-fill-violet" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#745BFF" stopOpacity="0.28" />
                  <stop offset="65%" stopColor="#258BFF" stopOpacity="0.08" />
                  <stop offset="100%" stopColor="#050B18" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="nl-wave-fill-electric" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#258BFF" stopOpacity="0.34" />
                  <stop offset="65%" stopColor="#35D8F2" stopOpacity="0.10" />
                  <stop offset="100%" stopColor="#050B18" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="nl-wave-fill-cyan" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#35D8F2" stopOpacity="0.32" />
                  <stop offset="65%" stopColor="#258BFF" stopOpacity="0.10" />
                  <stop offset="100%" stopColor="#050B18" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="nl-wave-fill-softblue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#579BFF" stopOpacity="0.28" />
                  <stop offset="65%" stopColor="#258BFF" stopOpacity="0.08" />
                  <stop offset="100%" stopColor="#050B18" stopOpacity="0" />
                </linearGradient>

                <linearGradient id="nl-wave-stroke-cyan" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#258BFF" stopOpacity="0.88" />
                  <stop offset="25%" stopColor="#35D8F2" stopOpacity="0.96" />
                  <stop offset="50%" stopColor="#745BFF" stopOpacity="0.88" />
                  <stop offset="75%" stopColor="#35D8F2" stopOpacity="0.96" />
                  <stop offset="100%" stopColor="#258BFF" stopOpacity="0.88" />
                </linearGradient>
                <linearGradient id="nl-wave-stroke-violet" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#745BFF" stopOpacity="0.85" />
                  <stop offset="50%" stopColor="#35D8F2" stopOpacity="0.92" />
                  <stop offset="100%" stopColor="#745BFF" stopOpacity="0.85" />
                </linearGradient>
                <linearGradient id="nl-wave-stroke-electric" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#579BFF" stopOpacity="0.85" />
                  <stop offset="50%" stopColor="#258BFF" stopOpacity="0.92" />
                  <stop offset="100%" stopColor="#579BFF" stopOpacity="0.85" />
                </linearGradient>
              </defs>

              {/* Translucent Curtain Fill */}
              <path
                d={wave.curtainPath}
                fill={`url(#${wave.fillGradientId})`}
              />

              {/* Wide Soft Outer Glow Crest */}
              <path
                d={wave.crestPath}
                fill="none"
                stroke={`url(#${wave.strokeGradientId})`}
                strokeWidth={wave.glowStrokeWidth}
                strokeOpacity={wave.glowOpacity}
                strokeLinecap="round"
              />

              {/* Crisp Inner Aurora Crest */}
              <path
                d={wave.crestPath}
                fill="none"
                stroke={`url(#${wave.strokeGradientId})`}
                strokeWidth={wave.coreStrokeWidth}
                strokeLinecap="round"
              />
            </svg>
          </div>
        ))}
      </div>

      {/* Layer 3: GPU-Composited Drifting & Twinkling Luminous Particles */}
      <div className="absolute inset-0 overflow-hidden">
        {particles.map((p) => (
          <span
            key={p.id}
            data-testid={`ambient-particle-${p.id}`}
            className="gpu-accelerated-layer absolute rounded-full"
            style={{
              left: p.left,
              top: p.top,
              width: `${p.size}px`,
              height: `${p.size}px`,
              backgroundColor: p.color,
              boxShadow: `0 0 ${Math.round(p.size * 2.8)}px ${p.glowColor}`,
              willChange: 'transform, opacity',
              transform: 'translate3d(0, 0, 0)',
              animation: `neurolab-particle-rise ${p.duration} ease-in-out ${p.delay} infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
