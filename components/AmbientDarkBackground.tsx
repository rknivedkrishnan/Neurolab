'use client';

import React, { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { useTheme } from './ThemeProvider';

interface AmbientParticle {
  id: number;
  left: string;
  top: string;
  sizeClass: string;
  colorClass: string;
  duration: number;
  delay: number;
  driftX: number;
  driftY: number;
  maxOpacity: number;
}

const AMBIENT_PARTICLES: AmbientParticle[] = [
  { id: 1, left: '7%', top: '18%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#35D8F2] shadow-[0_0_8px_#35D8F2]', duration: 19, delay: 0, driftX: 24, driftY: -46, maxOpacity: 0.55 },
  { id: 2, left: '16%', top: '68%', sizeClass: 'w-1.5 h-1.5', colorClass: 'bg-[#258BFF] shadow-[0_0_10px_#258BFF]', duration: 24, delay: 2, driftX: -20, driftY: -54, maxOpacity: 0.45 },
  { id: 3, left: '25%', top: '34%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#35D8F2] shadow-[0_0_7px_#35D8F2]', duration: 21, delay: 4, driftX: 18, driftY: -38, maxOpacity: 0.5 },
  { id: 4, left: '34%', top: '82%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#579BFF] shadow-[0_0_8px_#579BFF]', duration: 26, delay: 1, driftX: 30, driftY: -60, maxOpacity: 0.4 },
  { id: 5, left: '43%', top: '22%', sizeClass: 'w-1.5 h-1.5', colorClass: 'bg-[#35D8F2] shadow-[0_0_10px_#35D8F2]', duration: 22, delay: 3, driftX: -26, driftY: -42, maxOpacity: 0.5 },
  { id: 6, left: '52%', top: '58%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#258BFF] shadow-[0_0_8px_#258BFF]', duration: 20, delay: 5, driftX: 22, driftY: -48, maxOpacity: 0.45 },
  { id: 7, left: '61%', top: '14%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#745BFF] shadow-[0_0_9px_#745BFF]', duration: 25, delay: 1.5, driftX: -18, driftY: 36, maxOpacity: 0.42 },
  { id: 8, left: '69%', top: '74%', sizeClass: 'w-1.5 h-1.5', colorClass: 'bg-[#35D8F2] shadow-[0_0_11px_#35D8F2]', duration: 23, delay: 2.5, driftX: 26, driftY: -52, maxOpacity: 0.52 },
  { id: 9, left: '78%', top: '42%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#258BFF] shadow-[0_0_8px_#258BFF]', duration: 18, delay: 0.8, driftX: -22, driftY: -40, maxOpacity: 0.48 },
  { id: 10, left: '87%', top: '64%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#35D8F2] shadow-[0_0_8px_#35D8F2]', duration: 27, delay: 3.5, driftX: -28, driftY: -58, maxOpacity: 0.45 },
  { id: 11, left: '92%', top: '26%', sizeClass: 'w-1.5 h-1.5', colorClass: 'bg-[#579BFF] shadow-[0_0_9px_#579BFF]', duration: 21, delay: 1.2, driftX: -20, driftY: -44, maxOpacity: 0.4 },
  { id: 12, left: '12%', top: '46%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#258BFF] shadow-[0_0_8px_#258BFF]', duration: 24, delay: 4.2, driftX: 20, driftY: -46, maxOpacity: 0.42 },
  { id: 13, left: '29%', top: '12%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#35D8F2] shadow-[0_0_8px_#35D8F2]', duration: 22, delay: 2.8, driftX: 16, driftY: 38, maxOpacity: 0.48 },
  { id: 14, left: '48%', top: '86%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#35D8F2] shadow-[0_0_9px_#35D8F2]', duration: 25, delay: 0.5, driftX: -24, driftY: -64, maxOpacity: 0.5 },
  { id: 15, left: '64%', top: '48%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#258BFF] shadow-[0_0_8px_#258BFF]', duration: 19, delay: 3.1, driftX: 22, driftY: -36, maxOpacity: 0.44 },
  { id: 16, left: '82%', top: '16%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#745BFF] shadow-[0_0_9px_#745BFF]', duration: 26, delay: 1.9, driftX: -24, driftY: 42, maxOpacity: 0.4 },
  { id: 17, left: '21%', top: '88%', sizeClass: 'w-1.5 h-1.5', colorClass: 'bg-[#35D8F2] shadow-[0_0_10px_#35D8F2]', duration: 23, delay: 4.6, driftX: 28, driftY: -56, maxOpacity: 0.48 },
  { id: 18, left: '74%', top: '84%', sizeClass: 'w-1 h-1', colorClass: 'bg-[#258BFF] shadow-[0_0_8px_#258BFF]', duration: 20, delay: 2.2, driftX: -18, driftY: -50, maxOpacity: 0.46 },
];

export default function AmbientDarkBackground() {
  const { isDark } = useTheme();
  const prefersReducedMotion = useReducedMotion();
  const [isTabHidden, setIsTabHidden] = useState(false);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const handleVisibility = () => {
      setIsTabHidden(document.hidden);
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);

  if (!isDark) {
    return null;
  }

  const shouldAnimate = !prefersReducedMotion && !isTabHidden;

  return (
    <div
      data-testid="ambient-dark-background"
      data-animated={shouldAnimate ? 'true' : 'false'}
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#050B18] select-none"
    >
      {/* D. Deep Ambient Gradient Lighting Behind Dashboard */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_85%_60%_at_50%_0%,rgba(18,36,64,0.72),rgba(8,19,38,0.45)_55%,rgba(5,11,24,1)_100%)]" />

      {/* B. Glowing Light Flares (Slowly shifting & breathing radial light sources) */}
      <motion.div
        animate={
          shouldAnimate
            ? {
                x: [0, 45, -25, 0],
                y: [0, 28, -18, 0],
                scale: [1, 1.12, 0.94, 1],
                opacity: [0.24, 0.36, 0.2, 0.24],
              }
            : { opacity: 0.24 }
        }
        transition={{
          duration: 24,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute -top-32 left-[8%] w-[540px] h-[540px] rounded-full bg-[radial-gradient(circle,rgba(37,139,255,0.42)_0%,rgba(53,216,242,0.16)_45%,transparent_72%)] blur-3xl"
      />

      <motion.div
        animate={
          shouldAnimate
            ? {
                x: [0, -50, 30, 0],
                y: [0, -35, 25, 0],
                scale: [1, 1.15, 0.92, 1],
                opacity: [0.2, 0.32, 0.18, 0.2],
              }
            : { opacity: 0.2 }
        }
        transition={{
          duration: 28,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute top-[18%] right-[4%] w-[600px] h-[600px] rounded-full bg-[radial-gradient(circle,rgba(53,216,242,0.34)_0%,rgba(116,91,255,0.18)_48%,transparent_74%)] blur-3xl"
      />

      <motion.div
        animate={
          shouldAnimate
            ? {
                x: [0, 38, -32, 0],
                y: [0, -24, 30, 0],
                scale: [1, 1.08, 0.95, 1],
                opacity: [0.18, 0.28, 0.15, 0.18],
              }
            : { opacity: 0.18 }
        }
        transition={{
          duration: 26,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute -bottom-40 left-[28%] w-[680px] h-[520px] rounded-full bg-[radial-gradient(circle,rgba(116,91,255,0.32)_0%,rgba(37,139,255,0.18)_50%,transparent_75%)] blur-3xl"
      />

      {/* A. Flowing Ambient Waves (Multi-layered organic curved light ribbons) */}
      <div className="absolute inset-0 flex items-center justify-center opacity-90">
        <svg
          viewBox="0 0 1600 900"
          preserveAspectRatio="none"
          className="w-full h-full"
        >
          <defs>
            <linearGradient id="neurolabWaveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#258BFF" stopOpacity="0.02" />
              <stop offset="28%" stopColor="#258BFF" stopOpacity="0.22" />
              <stop offset="55%" stopColor="#35D8F2" stopOpacity="0.28" />
              <stop offset="80%" stopColor="#745BFF" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#050B18" stopOpacity="0.02" />
            </linearGradient>

            <linearGradient id="neurolabWaveGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#745BFF" stopOpacity="0.03" />
              <stop offset="32%" stopColor="#35D8F2" stopOpacity="0.2" />
              <stop offset="65%" stopColor="#258BFF" stopOpacity="0.24" />
              <stop offset="100%" stopColor="#050B18" stopOpacity="0.02" />
            </linearGradient>

            <linearGradient id="neurolabWaveStroke1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#258BFF" stopOpacity="0" />
              <stop offset="35%" stopColor="#35D8F2" stopOpacity="0.45" />
              <stop offset="68%" stopColor="#258BFF" stopOpacity="0.38" />
              <stop offset="100%" stopColor="#745BFF" stopOpacity="0" />
            </linearGradient>

            <linearGradient id="neurolabWaveStroke2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#745BFF" stopOpacity="0" />
              <stop offset="40%" stopColor="#579BFF" stopOpacity="0.32" />
              <stop offset="75%" stopColor="#35D8F2" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#258BFF" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Wave Layer 1: Primary Electric Blue -> Cyan Ribbon */}
          <motion.g
            animate={
              shouldAnimate
                ? {
                    x: [-35, 35, -35],
                    y: [-16, 18, -16],
                    scaleY: [1, 1.06, 0.96, 1],
                  }
                : undefined
            }
            transition={{
              duration: 22,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <path
              d="M -160 620 C 220 460, 520 740, 880 540 C 1220 350, 1440 590, 1760 450 L 1760 940 L -160 940 Z"
              fill="url(#neurolabWaveGrad1)"
            />
            <path
              d="M -160 620 C 220 460, 520 740, 880 540 C 1220 350, 1440 590, 1760 450"
              fill="none"
              stroke="url(#neurolabWaveStroke1)"
              strokeWidth="2"
            />
          </motion.g>

          {/* Wave Layer 2: Secondary Cyan -> Violet Upper Horizon Wave */}
          <motion.g
            animate={
              shouldAnimate
                ? {
                    x: [40, -40, 40],
                    y: [14, -20, 14],
                    scaleY: [1, 0.94, 1.07, 1],
                  }
                : undefined
            }
            transition={{
              duration: 27,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <path
              d="M -160 360 C 180 520, 580 210, 960 390 C 1310 550, 1510 280, 1760 370 L 1760 940 L -160 940 Z"
              fill="url(#neurolabWaveGrad2)"
            />
            <path
              d="M -160 360 C 180 520, 580 210, 960 390 C 1310 550, 1510 280, 1760 370"
              fill="none"
              stroke="url(#neurolabWaveStroke2)"
              strokeWidth="1.5"
            />
          </motion.g>

          {/* Wave Layer 3: Subtle Deep Crest Filament */}
          <motion.g
            animate={
              shouldAnimate
                ? {
                    x: [-25, 28, -25],
                    y: [10, -14, 10],
                  }
                : undefined
            }
            transition={{
              duration: 31,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <path
              d="M -120 750 C 310 640, 690 830, 1090 670 C 1380 550, 1560 710, 1740 640"
              fill="none"
              stroke="url(#neurolabWaveStroke1)"
              strokeWidth="1.25"
              opacity="0.65"
            />
          </motion.g>
        </svg>
      </div>

      {/* C. Floating Glowing Particles (Low-density drifting blue & cyan micro-particles) */}
      <div className="absolute inset-0">
        {AMBIENT_PARTICLES.map((p) => (
          <motion.span
            key={p.id}
            style={{ left: p.left, top: p.top }}
            animate={
              shouldAnimate
                ? {
                    x: [0, p.driftX, 0],
                    y: [0, p.driftY, 0],
                    opacity: [0.12, p.maxOpacity, 0.12],
                  }
                : { opacity: p.maxOpacity * 0.6 }
            }
            transition={{
              duration: p.duration,
              delay: p.delay,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className={`absolute rounded-full ${p.sizeClass} ${p.colorClass}`}
          />
        ))}
      </div>
    </div>
  );
}
