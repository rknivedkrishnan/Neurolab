'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Info,
  Eye,
  ZapOff,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import {
  LayerActivations,
  NeuralNetworkState,
  NeuronAblationConfig,
  buildNetworkVisualizationSnapshot,
  computeNeuronSaliencyMap,
  VisualNode,
  H1_SIZE,
  H2_SIZE,
} from '@/lib/neural-engine';
import { useTheme } from './ThemeProvider';

interface NeuralNetworkVizProps {
  activations: LayerActivations;
  network: NeuralNetworkState;
  activeModelMode: 'pretrained' | 'visitor';
  onModelModeChange: (mode: 'pretrained' | 'visitor') => void;
  animationTrigger: number;
  ablationConfig: NeuronAblationConfig;
  onAblationChange: (cfg: NeuronAblationConfig) => void;
}

const STAGE_DESCRIPTIONS = [
  {
    stage: 1,
    title: '1. Your drawing enters the network',
    subtitle: 'Input Layer · 784 pixel brightness numbers (0.00 to 1.00)',
    explanation:
      'The 28 × 28 grid is unrolled into a column of 784 numbers. Bright ink pixels send strong initial signals forward.',
  },
  {
    stage: 2,
    title: '2. First hidden layer spots simple strokes',
    subtitle: 'Hidden Layer 1 · 128 ReLU neurons detect edges, loops & slants',
    explanation:
      'Each neuron multiplies incoming pixel values by its learned weights. Positive matches activate via ReLU.',
  },
  {
    stage: 3,
    title: '3. Second hidden layer combines patterns',
    subtitle: 'Hidden Layer 2 · 64 ReLU neurons assemble higher-level digit shapes',
    explanation:
      'Neurons here combine individual strokes (like a top loop plus a vertical stem) into recognizable digit parts.',
  },
  {
    stage: 4,
    title: '4. The AI chooses the most likely number',
    subtitle: 'Output Layer · 10 Softmax neurons score digits 0 through 9',
    explanation:
      'The final 10 neurons convert incoming evidence into percentages that add up to 100%. The highest score wins.',
  },
];

function SaliencyGridCanvas({
  values,
  inputPixels,
  mode,
  isDark,
}: {
  values: number[];
  inputPixels: number[];
  mode: 'receptive' | 'attribution';
  isDark: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 140;
    const cell = size / 28;
    ctx.clearRect(0, 0, size, size);

    // Background fill
    ctx.fillStyle = isDark ? '#060E1F' : '#F4F4F0';
    ctx.fillRect(0, 0, size, size);

    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const idx = r * 28 + c;
        const v = values[idx] ?? 0; // [-1, 1]
        const px = inputPixels[idx] ?? 0; // [0, 1]
        const x = c * cell;
        const y = r * cell;

        if (mode === 'receptive') {
          const mag = Math.min(1, Math.abs(v));
          if (mag > 0.04) {
            if (v > 0) {
              // Excitatory: Electric Blue -> Bright Cyan
              const alpha = Math.min(0.95, Math.pow(mag, 0.75) * 0.92);
              ctx.fillStyle = isDark
                ? `rgba(53, 216, 242, ${alpha.toFixed(3)})`
                : `rgba(37, 139, 255, ${alpha.toFixed(3)})`;
            } else {
              // Inhibitory: Deep Violet / Rose
              const alpha = Math.min(0.9, Math.pow(mag, 0.8) * 0.85);
              ctx.fillStyle = isDark
                ? `rgba(244, 63, 94, ${alpha.toFixed(3)})`
                : `rgba(225, 29, 72, ${alpha.toFixed(3)})`;
            }
            ctx.fillRect(x, y, cell + 0.35, cell + 0.35);
          }
          // Faint ghost outline of the user's current drawing for spatial reference
          if (px > 0.25) {
            ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.16)' : 'rgba(15, 23, 42, 0.14)';
            ctx.fillRect(x + 0.8, y + 0.8, Math.max(1, cell - 1.6), Math.max(1, cell - 1.6));
          }
        } else {
          // Attribution mode: draw user's stroke + highlight active excitatory/inhibitory hits
          if (px > 0.06) {
            ctx.fillStyle = isDark
              ? `rgba(148, 163, 184, ${(px * 0.28).toFixed(3)})`
              : `rgba(100, 116, 139, ${(px * 0.22).toFixed(3)})`;
            ctx.fillRect(x, y, cell, cell);
          }
          const mag = Math.min(1, Math.abs(v));
          if (mag > 0.05) {
            if (v > 0) {
              const alpha = Math.min(0.98, Math.pow(mag, 0.68) * 0.95);
              ctx.fillStyle = isDark
                ? `rgba(53, 216, 242, ${alpha.toFixed(3)})`
                : `rgba(16, 185, 129, ${alpha.toFixed(3)})`;
            } else {
              const alpha = Math.min(0.95, Math.pow(mag, 0.72) * 0.9);
              ctx.fillStyle = isDark
                ? `rgba(244, 63, 94, ${alpha.toFixed(3)})`
                : `rgba(225, 29, 72, ${alpha.toFixed(3)})`;
            }
            ctx.fillRect(x, y, cell + 0.35, cell + 0.35);
          }
        }
      }
    }

    // Subtle 4x4 grid guide lines
    ctx.strokeStyle = isDark ? 'rgba(32, 54, 83, 0.38)' : 'rgba(229, 229, 224, 0.65)';
    ctx.lineWidth = 0.5;
    for (let g = 7; g < 28; g += 7) {
      const pos = g * cell;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(size, pos);
      ctx.stroke();
    }
  }, [values, inputPixels, mode, isDark]);

  return (
    <canvas
      ref={canvasRef}
      width={140}
      height={140}
      className="w-[124px] h-[124px] sm:w-[136px] sm:h-[136px] rounded-lg border border-[#E5E5E0] block shrink-0"
    />
  );
}

export default function NeuralNetworkViz({
  activations,
  network,
  activeModelMode,
  onModelModeChange,
  animationTrigger,
  ablationConfig,
  onAblationChange,
}: NeuralNetworkVizProps) {
  const { isDark } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentStage, setCurrentStage] = useState<number>(4); // 1..4
  const [progress, setProgress] = useState<number>(1); // 0..1 overall flow progress
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [saliencyTarget, setSaliencyTarget] = useState<{
    layerIndex: 0 | 1 | 2 | 3;
    realIndex: number;
  } | null>(null);

  const snapshot = React.useMemo(
    () => buildNetworkVisualizationSnapshot(activations, network, ablationConfig),
    [activations, network, ablationConfig]
  );

  // Sync saliency target when user clicks a neuron on the canvas, or default to winning output digit
  const activeSaliencyTarget = React.useMemo(() => {
    if (selectedNode) {
      return {
        layerIndex: selectedNode.layerIndex,
        realIndex: selectedNode.realIndex,
      };
    }
    if (saliencyTarget) {
      return saliencyTarget;
    }
    return {
      layerIndex: 3 as const,
      realIndex: activations.predictedDigit,
    };
  }, [selectedNode, saliencyTarget, activations.predictedDigit]);

  const saliencyMap = React.useMemo(
    () =>
      computeNeuronSaliencyMap(
        network,
        activations,
        activeSaliencyTarget.layerIndex,
        activeSaliencyTarget.realIndex
      ),
    [network, activations, activeSaliencyTarget]
  );

  // Trigger smooth forward-pass signal animation whenever animationTrigger changes
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      setProgress(0);
      setCurrentStage(1);
      setIsPlaying(true);
    });
    return () => cancelAnimationFrame(id);
  }, [animationTrigger]);

  useEffect(() => {
    if (!isPlaying) return;

    let rafId: number;
    let lastTime = performance.now();

    const tick = (now: number) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      setProgress((prev) => {
        if (prev >= 1) {
          setCurrentStage(4);
          setIsPlaying(false);
          return 1;
        }
        const next = Math.min(1, prev + dt * 0.55); // ~1.8 seconds full traversal
        if (next < 0.25) setCurrentStage(1);
        else if (next < 0.55) setCurrentStage(2);
        else if (next < 0.85) setCurrentStage(3);
        else setCurrentStage(4);
        return next;
      });
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  const handleStepStage = () => {
    setIsPlaying(false);
    const nextStage = currentStage >= 4 ? 1 : currentStage + 1;
    setCurrentStage(nextStage);
    const stageProgressMap: Record<number, number> = {
      1: 0.18,
      2: 0.48,
      3: 0.78,
      4: 1.0,
    };
    setProgress(stageProgressMap[nextStage]);
  };

  const handleReplay = () => {
    setProgress(0);
    setCurrentStage(1);
    setIsPlaying(true);
  };

  const toggleSingleNeuronAblation = (layerIndex: 1 | 2, realIndex: number) => {
    if (layerIndex === 1) {
      const exists = ablationConfig.disabledH1.includes(realIndex);
      const nextH1 = exists
        ? ablationConfig.disabledH1.filter((id) => id !== realIndex)
        : [...ablationConfig.disabledH1, realIndex];
      onAblationChange({
        ...ablationConfig,
        disabledH1: nextH1,
      });
    } else {
      const exists = ablationConfig.disabledH2.includes(realIndex);
      const nextH2 = exists
        ? ablationConfig.disabledH2.filter((id) => id !== realIndex)
        : [...ablationConfig.disabledH2, realIndex];
      onAblationChange({
        ...ablationConfig,
        disabledH2: nextH2,
      });
    }
  };

  // Draw the 4-layer network onto high-DPI Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 620;
    const height = rect.height || 340;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const layerX = [width * 0.1, width * 0.37, width * 0.64, width * 0.88];
    const topPad = 42;
    const bottomPad = 24;
    const usableH = height - topPad - bottomPad;

    const getNodeCoords = (layerIdx: number, nodeIdx: number, totalNodes: number) => {
      const x = layerX[layerIdx];
      const step = usableH / Math.max(1, totalNodes - 1);
      const y = topPad + nodeIdx * step;
      return { x, y };
    };

    // Draw layer column titles
    const headers = [
      { title: 'INPUT', sub: '784 px' },
      { title: 'HIDDEN 1', sub: '128 ReLU' },
      { title: 'HIDDEN 2', sub: '64 ReLU' },
      { title: 'OUTPUT', sub: '0 – 9' },
    ];

    headers.forEach((h, idx) => {
      const isStageActive = currentStage === idx + 1;
      ctx.font = '600 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = isStageActive
        ? isDark
          ? '#35D8F2'
          : '#3478F6'
        : isDark
          ? '#A2B5D0'
          : '#777777';
      ctx.fillText(h.title, layerX[idx], 16);
      ctx.font = '400 10px JetBrains Mono, monospace';
      ctx.fillStyle = isDark ? '#7186A6' : '#777777';
      ctx.fillText(h.sub, layerX[idx], 29);
    });

    // Determine max signal per layer transition to normalize line opacity
    let maxSignal01 = 0.001;
    let maxSignal12 = 0.001;
    let maxSignal23 = 0.001;
    for (const syn of snapshot.synapses) {
      if (syn.fromLayer === 0 && syn.signalStrength > maxSignal01) maxSignal01 = syn.signalStrength;
      if (syn.fromLayer === 1 && syn.signalStrength > maxSignal12) maxSignal12 = syn.signalStrength;
      if (syn.fromLayer === 2 && syn.signalStrength > maxSignal23) maxSignal23 = syn.signalStrength;
    }

    // Draw synapses
    for (const syn of snapshot.synapses) {
      const fromNodes = snapshot.nodesByLayer[syn.fromLayer];
      const toNodes = snapshot.nodesByLayer[syn.toLayer];
      const fromNode = fromNodes[syn.fromVisualIndex];
      const toNode = toNodes[syn.toVisualIndex];
      const p1 = getNodeCoords(syn.fromLayer, syn.fromVisualIndex, fromNodes.length);
      const p2 = getNodeCoords(syn.toLayer, syn.toVisualIndex, toNodes.length);

      const maxSig =
        syn.fromLayer === 0 ? maxSignal01 : syn.fromLayer === 1 ? maxSignal12 : maxSignal23;
      const normSignal = Math.min(1, syn.signalStrength / maxSig);

      const transitionStart = syn.fromLayer === 0 ? 0.0 : syn.fromLayer === 1 ? 0.28 : 0.58;
      const transitionReach = Math.max(0, Math.min(1, (progress - transitionStart) / 0.3));

      const isConnectedToSelected =
        selectedNode &&
        ((selectedNode.layerIndex === syn.fromLayer &&
          selectedNode.nodeIndex === syn.fromVisualIndex) ||
          (selectedNode.layerIndex === syn.toLayer &&
            selectedNode.nodeIndex === syn.toVisualIndex));

      if (fromNode?.isAblated || toNode?.isAblated) {
        if (isConnectedToSelected) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = isDark ? 'rgba(244, 63, 94, 0.35)' : 'rgba(225, 29, 72, 0.28)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        continue;
      }

      if (normSignal < 0.12 && !isConnectedToSelected) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = isDark ? 'rgba(37, 139, 255, 0.055)' : 'rgba(32, 33, 36, 0.035)';
        ctx.lineWidth = 0.6;
        ctx.stroke();
        continue;
      }

      const alpha = isConnectedToSelected
        ? Math.max(0.28, normSignal * 0.9)
        : normSignal * 0.58 * (0.25 + 0.75 * transitionReach);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);

      if (syn.isPositive) {
        ctx.strokeStyle = isDark
          ? `rgba(53, 216, 242, ${alpha.toFixed(3)})`
          : `rgba(52, 120, 246, ${alpha.toFixed(3)})`;
      } else {
        ctx.strokeStyle = isDark
          ? `rgba(116, 91, 255, ${(alpha * 0.65).toFixed(3)})`
          : `rgba(140, 140, 148, ${(alpha * 0.55).toFixed(3)})`;
      }

      ctx.lineWidth = isConnectedToSelected ? 1.8 : 0.6 + normSignal * 1.6;
      ctx.stroke();

      // Animated traveling signal pulse along strong positive connections
      if (isPlaying && transitionReach > 0.05 && transitionReach < 0.98 && normSignal > 0.45) {
        const px = p1.x + (p2.x - p1.x) * transitionReach;
        const py = p1.y + (p2.y - p1.y) * transitionReach;
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = isDark ? '#35D8F2' : '#3478F6';
        ctx.fill();
      }
    }

    // Draw nodes for each layer
    snapshot.nodesByLayer.forEach((nodes, layerIdx) => {
      const layerRevealThreshold =
        layerIdx === 0 ? 0 : layerIdx === 1 ? 0.25 : layerIdx === 2 ? 0.55 : 0.82;
      const layerFactor = progress >= layerRevealThreshold ? 1 : 0.35;

      nodes.forEach((node, nIdx) => {
        const { x, y } = getNodeCoords(layerIdx, nIdx, nodes.length);
        const radius = layerIdx === 3 ? 9.5 : layerIdx === 0 ? 5.5 : 6.5;
        const act = Math.min(1, Math.max(0, node.activation)) * layerFactor;
        const isWinner = layerIdx === 3 && node.realIndex === activations.predictedDigit;
        const isSelected =
          selectedNode?.layerIndex === layerIdx && selectedNode?.nodeIndex === nIdx;
        const isSaliencyFocused =
          activeSaliencyTarget.layerIndex === layerIdx &&
          activeSaliencyTarget.realIndex === node.realIndex;

        // Node circle background
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);

        if (node.isAblated) {
          ctx.fillStyle = isDark ? '#1A162B' : '#FEF2F2';
          ctx.fill();
          ctx.lineWidth = isSelected ? 2.2 : 1.3;
          ctx.strokeStyle = isDark ? '#F43F5E' : '#E11D48';
          ctx.stroke();

          // Diagonal surgical ablation slash across the disabled neuron
          ctx.beginPath();
          ctx.moveTo(x - radius * 0.72, y - radius * 0.72);
          ctx.lineTo(x + radius * 0.72, y + radius * 0.72);
          ctx.strokeStyle = isDark ? '#FB7185' : '#E11D48';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          return;
        }

        if (isWinner) {
          if (isDark) {
            ctx.save();
            ctx.shadowColor = 'rgba(53, 216, 242, 0.75)';
            ctx.shadowBlur = 14;
            ctx.fillStyle = '#258BFF';
            ctx.fill();
            ctx.restore();
          } else {
            ctx.fillStyle = '#3478F6';
            ctx.fill();
          }
        } else {
          if (isDark) {
            const r = Math.round(13 + (53 - 13) * act);
            const g = Math.round(27 + (216 - 27) * act);
            const b = Math.round(51 + (242 - 51) * act);
            ctx.save();
            if (act > 0.55) {
              ctx.shadowColor = 'rgba(53, 216, 242, 0.45)';
              ctx.shadowBlur = 8;
            }
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fill();
            ctx.restore();
          } else {
            const r = Math.round(255 - (255 - 52) * act);
            const g = Math.round(255 - (255 - 120) * act);
            const b = Math.round(255 - (255 - 246) * act);
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fill();
          }
        }

        ctx.lineWidth = isSelected || isWinner || isSaliencyFocused ? 2.2 : 1.1;
        ctx.strokeStyle =
          isSelected || isWinner
            ? isDark
              ? '#35D8F2'
              : '#1D4ED8'
            : isSaliencyFocused
              ? isDark
                ? '#745BFF'
                : '#2563EB'
              : act > 0.35
                ? isDark
                  ? '#258BFF'
                  : '#3478F6'
                : isDark
                  ? '#203653'
                  : '#D4D4CE';
        ctx.stroke();

        // Output layer digit labels & probabilities
        if (layerIdx === 3) {
          ctx.font = isWinner ? '700 10px Inter, sans-serif' : '600 9px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle =
            isWinner || act > 0.55 ? '#FFFFFF' : isDark ? '#F1F6FF' : '#202124';
          ctx.fillText(`${node.realIndex}`, x, y + 0.5);

          const pct = `${Math.round(activations.probabilities[node.realIndex] * 100)}%`;
          ctx.font = isWinner
            ? '600 10px JetBrains Mono, monospace'
            : '400 9px JetBrains Mono, monospace';
          ctx.textAlign = 'left';
          ctx.fillStyle = isWinner
            ? isDark
              ? '#35D8F2'
              : '#3478F6'
            : isDark
              ? '#A2B5D0'
              : '#777777';
          ctx.fillText(pct, x + 14, y + 0.5);
        }
      });
    });

    ctx.restore();
  }, [
    snapshot,
    activations,
    progress,
    currentStage,
    isPlaying,
    selectedNode,
    activeSaliencyTarget,
    isDark,
  ]);

  const handleCanvasPointerMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const width = rect.width || 620;
    const height = rect.height || 340;
    const layerX = [width * 0.1, width * 0.37, width * 0.64, width * 0.88];
    const topPad = 42;
    const bottomPad = 24;
    const usableH = height - topPad - bottomPad;

    let found: VisualNode | null = null;
    snapshot.nodesByLayer.forEach((nodes, layerIdx) => {
      const step = usableH / Math.max(1, nodes.length - 1);
      nodes.forEach((node, nIdx) => {
        const x = layerX[layerIdx];
        const y = topPad + nIdx * step;
        const dist = Math.hypot(mx - x, my - y);
        if (dist <= 13) {
          found = node;
        }
      });
    });

    if (found) {
      setSelectedNode(found);
    }
  };

  const stageInfo = STAGE_DESCRIPTIONS[currentStage - 1] || STAGE_DESCRIPTIONS[3];
  const totalHiddenNeurons = H1_SIZE + H2_SIZE; // 192
  const totalAblatedNeurons = snapshot.ablatedH1Count + snapshot.ablatedH2Count;
  const hasAnyAblation =
    ablationConfig.dropoutRatio > 0 ||
    ablationConfig.disabledH1.length > 0 ||
    ablationConfig.disabledH2.length > 0;

  // Quick-select neurons for Saliency Inspector
  const sortedDigits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].sort(
    (a, b) => activations.probabilities[b] - activations.probabilities[a]
  );
  const winningDigit = sortedDigits[0];
  const runnerUpDigit = sortedDigits[1];
  const topH1Node =
    [...snapshot.nodesByLayer[1]].sort((a, b) => b.rawActivation - a.rawActivation)[0] ??
    snapshot.nodesByLayer[1][0];
  const topH2Node =
    [...snapshot.nodesByLayer[2]].sort((a, b) => b.rawActivation - a.rawActivation)[0] ??
    snapshot.nodesByLayer[2][0];

  return (
    <div className="bg-white rounded-2xl border border-[#E5E5E0] p-5 sm:p-6 shadow-xs flex flex-col justify-between h-full">
      {/* Header & Model Switcher */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-[#3478F6]/10 text-[#3478F6] text-xs font-semibold">
              2
            </span>
            <h2 className="text-base font-semibold tracking-tight text-[#202124]">
              Watch the Network Think
            </h2>
          </div>
          <p className="text-xs text-[#777777] mt-1">
            Live signals traveling across 4 layers (784 → 128 → 64 → 10). Hover or tap any neuron to inspect its 28×28 visual field.
          </p>
        </div>

        {/* Active Model Selector */}
        <div
          className="inline-flex rounded-xl bg-[#F3F3EE] p-1 border border-[#E5E5E0]"
          role="group"
          aria-label="Select neural network model"
        >
          <button
            type="button"
            onClick={() => onModelModeChange('pretrained')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap shrink-0 cursor-pointer ${
              activeModelMode === 'pretrained'
                ? 'bg-white text-[#202124] shadow-xs'
                : 'text-[#777777] hover:text-[#202124]'
            }`}
          >
            Pretrained Model
          </button>
          <button
            type="button"
            onClick={() => onModelModeChange('visitor')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap shrink-0 cursor-pointer ${
              activeModelMode === 'visitor'
                ? 'bg-white text-[#202124] shadow-xs'
                : 'text-[#777777] hover:text-[#202124]'
            }`}
          >
            Visitor-Trained Model
          </button>
        </div>
      </div>

      {/* Interactive Network Canvas */}
      <div className="relative rounded-xl bg-[#FAFAF8] border border-[#E5E5E0] overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseMove={handleCanvasPointerMove}
          onClick={handleCanvasPointerMove}
          data-testid="neural-network-canvas"
          aria-label="Interactive visualization of the 784 to 128 to 64 to 10 neural network"
          className="w-full h-[300px] sm:h-[330px] block cursor-pointer"
        />

        {/* Selected Neuron Inspector Overlay Pill with Surgical Ablation Toggle */}
        {selectedNode && (
          <div className="absolute bottom-3 left-3 right-3 bg-white/95 backdrop-blur-xs border border-[#E5E5E0] rounded-xl px-3.5 py-2 text-xs flex flex-wrap items-center justify-between gap-2 shadow-xs">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="font-semibold text-[#202124]">{selectedNode.label}</span>
              {selectedNode.isAblated && (
                <span className="px-1.5 py-0.5 rounded-md bg-rose-500/15 text-rose-500 font-mono text-[10px] font-semibold">
                  ABLATED (OFF)
                </span>
              )}
              <span className="text-[#777777]">·</span>
              <span className="font-mono text-[#3478F6] tabular-nums">
                Activation: {selectedNode.rawActivation.toFixed(3)}
              </span>
              {selectedNode.layerIndex === 3 && (
                <>
                  <span className="text-[#777777]">·</span>
                  <span className="font-mono text-[#202124] tabular-nums">
                    Probability: {(selectedNode.activation * 100).toFixed(1)}%
                  </span>
                </>
              )}
            </div>

            <div className="flex items-center gap-2">
              {(selectedNode.layerIndex === 1 || selectedNode.layerIndex === 2) && (
                <button
                  type="button"
                  data-testid="toggle-single-neuron-ablation"
                  onClick={() =>
                    toggleSingleNeuronAblation(
                      selectedNode.layerIndex as 1 | 2,
                      selectedNode.realIndex
                    )
                  }
                  className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors cursor-pointer ${
                    selectedNode.isAblated
                      ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-600'
                      : 'bg-rose-500/10 border-rose-500/30 text-rose-500 hover:bg-rose-500/20'
                  }`}
                >
                  <ZapOff className="w-3 h-3" />
                  {selectedNode.isAblated ? 'Restore Neuron' : 'Disable Neuron'}
                </button>
              )}
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="text-[#777777] hover:text-[#202124] font-medium cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Stage Stepper & Playback Controls */}
      <div className="mt-3.5 bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2.5 mb-2">
          <div className="flex items-center gap-1.5">
            {[1, 2, 3, 4].map((st) => (
              <button
                key={st}
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setCurrentStage(st);
                  setProgress(st === 1 ? 0.15 : st === 2 ? 0.45 : st === 3 ? 0.75 : 1.0);
                }}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  currentStage === st
                    ? 'bg-[#3478F6] text-white'
                    : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
                }`}
              >
                Step {st}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                if (progress >= 1) {
                  handleReplay();
                } else {
                  setIsPlaying((p) => !p);
                }
              }}
              aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              onClick={handleStepStage}
              aria-label="Next processing stage"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <StepForward className="w-3.5 h-3.5" />
              Next Step
            </button>
            <button
              type="button"
              onClick={handleReplay}
              aria-label="Replay neural network flow"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Replay
            </button>
          </div>
        </div>

        <div className="mt-1.5">
          <div className="flex items-baseline justify-between gap-2">
            <h3 className="text-xs sm:text-sm font-semibold text-[#202124]">{stageInfo.title}</h3>
            <span className="text-[11px] text-[#777777] font-mono hidden sm:inline">
              {stageInfo.subtitle}
            </span>
          </div>
          <p className="text-xs text-[#777777] mt-0.5 leading-relaxed">{stageInfo.explanation}</p>
        </div>
      </div>

      {/* #4 INTERACTIVE NEURON ABLATION ("Brain Surgery / Resilience Lab") */}
      <div
        data-testid="neuron-ablation-panel"
        className="mt-3.5 bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3.5"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#3478F6] shrink-0" />
            <span className="text-xs font-semibold text-[#202124]">
              Brain Resilience Lab · Interactive Neuron Ablation
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              data-testid="ablation-status-badge"
              className={`px-2 py-0.5 rounded-md font-mono text-[11px] font-semibold tabular-nums ${
                totalAblatedNeurons > 0
                  ? 'bg-rose-500/15 text-rose-500 border border-rose-500/30'
                  : 'bg-[#3478F6]/10 text-[#3478F6]'
              }`}
            >
              {totalAblatedNeurons} / {totalHiddenNeurons} Hidden Neurons Disabled (
              {Math.round((totalAblatedNeurons / totalHiddenNeurons) * 100)}%)
            </span>
            {hasAnyAblation && (
              <button
                type="button"
                data-testid="reset-ablation-btn"
                onClick={() =>
                  onAblationChange({ dropoutRatio: 0, disabledH1: [], disabledH2: [] })
                }
                className="text-[11px] font-medium text-[#3478F6] hover:underline cursor-pointer"
              >
                Restore All
              </button>
            )}
          </div>
        </div>

        <p className="text-[11px] text-[#777777] mb-2.5 leading-relaxed">
          Disable a percentage of hidden neurons (or click individual H1/H2 nodes above) to test distributed fault tolerance. Neural networks degrade gracefully until critical pathways collapse.
        </p>

        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1 flex items-center gap-2.5">
            <span className="text-[11px] font-mono text-[#777777] w-16 shrink-0">
              Ablate {Math.round(ablationConfig.dropoutRatio * 100)}%
            </span>
            <input
              type="range"
              min={0}
              max={90}
              step={5}
              data-testid="ablation-slider"
              aria-label="Percentage of hidden neurons to disable"
              value={Math.round(ablationConfig.dropoutRatio * 100)}
              onChange={(e) =>
                onAblationChange({
                  ...ablationConfig,
                  dropoutRatio: Number(e.target.value) / 100,
                })
              }
              className="w-full accent-[#3478F6] cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            {[
              { label: '0% Intact', val: 0 },
              { label: '35% Mild', val: 0.35 },
              { label: '65% Heavy', val: 0.65 },
              { label: '85% Critical', val: 0.85 },
            ].map((preset) => {
              const isActive = Math.abs(ablationConfig.dropoutRatio - preset.val) < 0.02;
              return (
                <button
                  key={preset.label}
                  type="button"
                  data-testid={`ablation-preset-${Math.round(preset.val * 100)}`}
                  onClick={() =>
                    onAblationChange({
                      ...ablationConfig,
                      dropoutRatio: preset.val,
                    })
                  }
                  className={`px-2 py-1 rounded-lg text-[11px] font-medium transition-colors cursor-pointer whitespace-nowrap ${
                    isActive
                      ? 'bg-[#3478F6] text-white'
                      : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* #1 NEURON SALIENCY & ATTRIBUTION HEATMAP ("What This Neuron Sees") */}
      <div
        data-testid="saliency-heatmap-panel"
        className="mt-3.5 bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3.5"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-[#3478F6] shrink-0" />
            <div>
              <span className="text-xs font-semibold text-[#202124]">
                What This Neuron Sees · 28×28 Saliency & Attribution Map
              </span>
            </div>
          </div>
          <span
            data-testid="saliency-target-label"
            className="px-2 py-0.5 rounded-md bg-[#3478F6]/10 text-[#3478F6] font-mono text-[11px] font-semibold"
          >
            {saliencyMap.label}
          </span>
        </div>

        {/* Quick-Select Target Pills */}
        <div className="flex flex-wrap items-center gap-1.5 mb-3">
          <span className="text-[11px] text-[#777777] mr-1">Inspect:</span>
          <button
            type="button"
            data-testid="saliency-select-winner"
            onClick={() => {
              setSelectedNode(null);
              setSaliencyTarget({ layerIndex: 3, realIndex: winningDigit });
            }}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors cursor-pointer ${
              activeSaliencyTarget.layerIndex === 3 &&
              activeSaliencyTarget.realIndex === winningDigit
                ? 'bg-[#3478F6] text-white'
                : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
            }`}
          >
            Winner Digit {winningDigit} ({Math.round(activations.probabilities[winningDigit] * 100)}
            %)
          </button>
          <button
            type="button"
            data-testid="saliency-select-runnerup"
            onClick={() => {
              setSelectedNode(null);
              setSaliencyTarget({ layerIndex: 3, realIndex: runnerUpDigit });
            }}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors cursor-pointer ${
              activeSaliencyTarget.layerIndex === 3 &&
              activeSaliencyTarget.realIndex === runnerUpDigit
                ? 'bg-[#3478F6] text-white'
                : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
            }`}
          >
            Runner-Up Digit {runnerUpDigit} (
            {Math.round(activations.probabilities[runnerUpDigit] * 100)}%)
          </button>
          {topH1Node && (
            <button
              type="button"
              data-testid="saliency-select-h1"
              onClick={() => {
                setSelectedNode(null);
                setSaliencyTarget({ layerIndex: 1, realIndex: topH1Node.realIndex });
              }}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors cursor-pointer ${
                activeSaliencyTarget.layerIndex === 1 &&
                activeSaliencyTarget.realIndex === topH1Node.realIndex
                  ? 'bg-[#3478F6] text-white'
                  : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
              }`}
            >
              Top H1 Stroke (#{topH1Node.realIndex})
            </button>
          )}
          {topH2Node && (
            <button
              type="button"
              data-testid="saliency-select-h2"
              onClick={() => {
                setSelectedNode(null);
                setSaliencyTarget({ layerIndex: 2, realIndex: topH2Node.realIndex });
              }}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors cursor-pointer ${
                activeSaliencyTarget.layerIndex === 2 &&
                activeSaliencyTarget.realIndex === topH2Node.realIndex
                  ? 'bg-[#3478F6] text-white'
                  : 'bg-white border border-[#E5E5E0] text-[#777777] hover:text-[#202124]'
              }`}
            >
              Top H2 Shape (#{topH2Node.realIndex})
            </button>
          )}
        </div>

        {/* Two 28x28 Heatmaps + Legend */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="bg-white rounded-xl border border-[#E5E5E0] p-3 flex items-center gap-3">
            <SaliencyGridCanvas
              values={saliencyMap.receptiveField}
              inputPixels={activations.input}
              mode="receptive"
              isDark={isDark}
            />
            <div className="min-w-0">
              <div className="text-xs font-semibold text-[#202124]">
                1. Learned Template (Receptive Field)
              </div>
              <p className="text-[11px] text-[#777777] mt-1 leading-relaxed">
                <span className="text-[#3478F6] font-semibold">Cyan/Blue</span> pixels excite this neuron (+weight).{' '}
                <span className="text-rose-500 font-semibold">Red/Rose</span> pixels inhibit it (−weight).
              </p>
              <div className="mt-2 flex items-center gap-2 text-[10px] font-mono text-[#777777]">
                <span className="inline-block w-2 h-2 rounded-xs bg-[#35D8F2]" /> +Excite
                <span className="inline-block w-2 h-2 rounded-xs bg-rose-500 ml-1" /> −Inhibit
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-[#E5E5E0] p-3 flex items-center gap-3">
            <SaliencyGridCanvas
              values={saliencyMap.inputAttribution}
              inputPixels={activations.input}
              mode="attribution"
              isDark={isDark}
            />
            <div className="min-w-0">
              <div className="text-xs font-semibold text-[#202124]">
                2. Active Drawing Attribution
              </div>
              <p className="text-[11px] text-[#777777] mt-1 leading-relaxed">
                Exact pixels of your drawing (<code className="font-mono">px × weight</code>) that triggered or penalized this neuron right now.
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] font-mono">
                <span className="text-[#3478F6] font-semibold">
                  +Hit: {saliencyMap.excitatoryScore.toFixed(1)}
                </span>
                <span className="text-rose-500 font-semibold">
                  −Penalty: {saliencyMap.inhibitoryScore.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Transparency Note on Visual Simplification */}
      <div className="mt-3 flex items-center justify-between gap-2 text-[11px] text-[#777777]">
        <span className="inline-flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-[#3478F6] shrink-0" />
          Display simplified to 50 representative nodes so individual connections stay clear. Full 784 → 128 → 64 → 10 math runs underneath.
        </span>
        <span className="font-mono tabular-nums shrink-0 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-[#3478F6]" />
          {activeModelMode === 'pretrained' ? 'Pretrained Reference' : 'Visitor-Trained Model'}
        </span>
      </div>
    </div>
  );
}
