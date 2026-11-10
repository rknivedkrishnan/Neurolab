'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, StepForward, Info } from 'lucide-react';
import {
  LayerActivations,
  NeuralNetworkState,
  buildNetworkVisualizationSnapshot,
  VisualNode,
} from '@/lib/neural-engine';

interface NeuralNetworkVizProps {
  activations: LayerActivations;
  network: NeuralNetworkState;
  activeModelMode: 'pretrained' | 'visitor';
  onModelModeChange: (mode: 'pretrained' | 'visitor') => void;
  animationTrigger: number;
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

export default function NeuralNetworkViz({
  activations,
  network,
  activeModelMode,
  onModelModeChange,
  animationTrigger,
}: NeuralNetworkVizProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentStage, setCurrentStage] = useState<number>(4); // 1..4
  const [progress, setProgress] = useState<number>(1); // 0..1 overall flow progress
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);

  const snapshot = React.useMemo(
    () => buildNetworkVisualizationSnapshot(activations, network),
    [activations, network]
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
        if (next >= 1) {
          setIsPlaying(false);
        }
        return next;
      });
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  const handleReplay = () => {
    setProgress(0);
    setCurrentStage(1);
    setIsPlaying(true);
  };

  const handleStepStage = () => {
    setIsPlaying(false);
    const nextStage = currentStage >= 4 ? 1 : currentStage + 1;
    setCurrentStage(nextStage);
    setProgress(nextStage === 1 ? 0.15 : nextStage === 2 ? 0.45 : nextStage === 3 ? 0.75 : 1.0);
  };

  // Draw the neural network onto the HTML Canvas using actual model weights and activations
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const layerX = [76, width * 0.36, width * 0.66, width - 82];
    const topPad = 48;
    const bottomPad = 32;
    const usableH = height - topPad - bottomPad;

    const getNodeCoords = (layerIdx: number, nodeIdx: number, count: number) => {
      const x = layerX[layerIdx];
      const step = count > 1 ? usableH / (count - 1) : 0;
      const y = topPad + nodeIdx * step;
      return { x, y };
    };

    // Layer Headers
    const headers = [
      { label: 'Input Layer', sub: '784 values (16 shown)' },
      { label: 'Hidden Layer 1', sub: `128 ReLU (${snapshot.activeH1Count} active)` },
      { label: 'Hidden Layer 2', sub: `64 ReLU (${snapshot.activeH2Count} active)` },
      { label: 'Output Layer', sub: '10 Digits (0–9)' },
    ];

    ctx.textAlign = 'center';
    headers.forEach((h, idx) => {
      const isStageActive = currentStage >= idx + 1;
      ctx.font = '600 12px sans-serif';
      ctx.fillStyle = isStageActive ? '#202124' : '#777777';
      ctx.fillText(h.label, layerX[idx], 18);
      ctx.font = '400 10px sans-serif';
      ctx.fillStyle = '#777777';
      ctx.fillText(h.sub, layerX[idx], 32);
    });

    // Compute max signal per layer pair so synapse opacity scales cleanly with actual activations
    let maxSig01 = 0.01;
    let maxSig12 = 0.01;
    let maxSig23 = 0.01;
    for (const syn of snapshot.synapses) {
      if (syn.fromLayer === 0 && syn.signalStrength > maxSig01) maxSig01 = syn.signalStrength;
      if (syn.fromLayer === 1 && syn.signalStrength > maxSig12) maxSig12 = syn.signalStrength;
      if (syn.fromLayer === 2 && syn.signalStrength > maxSig23) maxSig23 = syn.signalStrength;
    }

    // Draw Synapses (Actual model weights and activations)
    for (const syn of snapshot.synapses) {
      const fromNodes = snapshot.nodesByLayer[syn.fromLayer];
      const toNodes = snapshot.nodesByLayer[syn.toLayer];
      const p1 = getNodeCoords(syn.fromLayer, syn.fromVisualIndex, fromNodes.length);
      const p2 = getNodeCoords(syn.toLayer, syn.toVisualIndex, toNodes.length);

      // Determine if signal has reached this synapse segment
      const segStart = syn.fromLayer * 0.28;
      const segProgress = Math.max(0, Math.min(1, (progress - segStart) / 0.32));

      const layerMax =
        syn.fromLayer === 0 ? maxSig01 : syn.fromLayer === 1 ? maxSig12 : maxSig23;
      const relStrength = syn.signalStrength / layerMax;

      // Filter weak connections so the diagram remains clean and legible
      if (relStrength < 0.16) continue;

      const isWinningOutputSynapse =
        syn.toLayer === 3 &&
        syn.toRealIndex === activations.predictedDigit &&
        relStrength > 0.22;

      const alpha =
        segProgress > 0
          ? Math.min(0.82, (0.14 + relStrength * 0.65) * segProgress)
          : 0.06;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);

      if (isWinningOutputSynapse) {
        ctx.strokeStyle = `rgba(52, 120, 246, ${Math.max(0.42, alpha)})`;
        ctx.lineWidth = Math.min(2.8, 1.1 + relStrength * 1.8);
      } else if (syn.isPositive) {
        ctx.strokeStyle = `rgba(52, 120, 246, ${alpha})`;
        ctx.lineWidth = Math.min(2.2, 0.7 + relStrength * 1.4);
      } else {
        ctx.strokeStyle = `rgba(88, 86, 214, ${alpha * 0.9})`;
        ctx.lineWidth = Math.min(1.9, 0.6 + relStrength * 1.2);
      }
      ctx.stroke();

      // Animated pulse travelling along the strongest synapses during active flow
      if (segProgress > 0.05 && segProgress < 0.98 && relStrength > 0.35) {
        const px = p1.x + (p2.x - p1.x) * segProgress;
        const py = p1.y + (p2.y - p1.y) * segProgress;
        ctx.beginPath();
        ctx.arc(px, py, 2.8, 0, Math.PI * 2);
        ctx.fillStyle = syn.isPositive ? '#3478F6' : '#5856D6';
        ctx.fill();
      }
    }

    // Draw Neurons for each layer
    snapshot.nodesByLayer.forEach((nodes, layerIdx) => {
      const layerReached = progress >= layerIdx * 0.26;
      nodes.forEach((node) => {
        const { x, y } = getNodeCoords(layerIdx, node.nodeIndex, nodes.length);
        const radius = layerIdx === 3 ? 13 : layerIdx === 0 ? 7.5 : 9.5;
        const act = layerReached ? node.activation : 0.05;
        const isPredictedOutput =
          layerIdx === 3 && node.realIndex === activations.predictedDigit && layerReached;

        // Node circle fill
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);

        if (isPredictedOutput) {
          ctx.fillStyle = '#3478F6';
        } else if (act > 0.06) {
          // Interpolate between warm white and restrained electric blue / soft indigo
          const tint = Math.min(1, Math.pow(act, 0.75) * 0.95);
          ctx.fillStyle =
            layerIdx === 2
              ? `rgba(88, 86, 214, ${0.18 + tint * 0.78})`
              : `rgba(52, 120, 246, ${0.18 + tint * 0.78})`;
        } else {
          ctx.fillStyle = '#FFFFFF';
        }
        ctx.fill();

        ctx.lineWidth = isPredictedOutput ? 2.5 : act > 0.2 ? 1.6 : 1.2;
        ctx.strokeStyle = isPredictedOutput
          ? '#1D4ED8'
          : act > 0.2
          ? layerIdx === 2
            ? '#5856D6'
            : '#3478F6'
          : '#D4D4CE';
        ctx.stroke();

        // Output layer labels (digits 0-9 and percentages)
        if (layerIdx === 3) {
          ctx.font = '600 11px sans-serif';
          ctx.fillStyle = isPredictedOutput || act > 0.55 ? '#FFFFFF' : '#202124';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(node.realIndex), x, y);

          // Probability bar and text to the right of output neuron
          const probPct = Math.round(node.activation * 100);
          ctx.textAlign = 'left';
          ctx.font = isPredictedOutput ? '700 11px monospace' : '500 10px monospace';
          ctx.fillStyle = isPredictedOutput ? '#3478F6' : '#777777';
          ctx.fillText(`${probPct}%`, x + 18, y);
        }
      });
    });
  }, [snapshot, progress, currentStage, activations.predictedDigit]);

  // Allow clicking on a neuron in the canvas to inspect its exact numeric value
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    const width = canvas.width;
    const height = canvas.height;
    const layerX = [76, width * 0.36, width * 0.66, width - 82];
    const topPad = 48;
    const bottomPad = 32;
    const usableH = height - topPad - bottomPad;

    for (let l = 0; l < 4; l++) {
      const nodes = snapshot.nodesByLayer[l];
      const step = nodes.length > 1 ? usableH / (nodes.length - 1) : 0;
      for (let i = 0; i < nodes.length; i++) {
        const nx = layerX[l];
        const ny = topPad + i * step;
        if (Math.hypot(cx - nx, cy - ny) <= 16) {
          setSelectedNode(nodes[i]);
          return;
        }
      }
    }
  };

  const stageInfo = STAGE_DESCRIPTIONS[currentStage - 1] || STAGE_DESCRIPTIONS[3];

  return (
    <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 flex flex-col justify-between">
      {/* Header & Model Selector */}
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div>
          <p className="text-xs font-medium text-[#777777] tracking-wide mb-1">
            Stage 2 · Watch the AI Think
          </p>
          <h2 className="text-xl font-semibold text-[#202124]">
            Live Neural Network (784 → 128 → 64 → 10)
          </h2>
          <p className="text-sm text-[#777777] mt-0.5">
            Real activations and weights flow from left to right. Click any neuron to inspect its math.
          </p>
        </div>

        {/* Segmented Model Switcher */}
        <div className="flex items-center gap-1 p-1 bg-[#F2F2EF] rounded-xl border border-[#E5E5E0]">
          <button
            type="button"
            onClick={() => onModelModeChange('pretrained')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeModelMode === 'pretrained'
                ? 'bg-white text-[#202124] shadow-2xs'
                : 'text-[#777777] hover:text-[#202124]'
            }`}
          >
            Pretrained Model
          </button>
          <button
            type="button"
            onClick={() => onModelModeChange('visitor')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeModelMode === 'visitor'
                ? 'bg-white text-[#3478F6] shadow-2xs'
                : 'text-[#777777] hover:text-[#202124]'
            }`}
          >
            Visitor-Trained Model
          </button>
        </div>
      </div>

      {/* Interactive Network Canvas */}
      <div className="relative bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] overflow-hidden">
        <canvas
          ref={canvasRef}
          width={680}
          height={360}
          onClick={handleCanvasClick}
          data-testid="neural-network-canvas"
          aria-label="Interactive visualization of the 784 to 128 to 64 to 10 neural network"
          className="w-full h-[320px] sm:h-[350px] block cursor-pointer"
        />

        {/* Selected Neuron Inspector Overlay Pill */}
        {selectedNode && (
          <div className="absolute bottom-3 left-3 right-3 sm:right-auto bg-white/95 backdrop-blur-xs border border-[#E5E5E0] rounded-xl px-3.5 py-2 text-xs flex items-center justify-between gap-4 shadow-xs">
            <div>
              <span className="font-semibold text-[#202124]">{selectedNode.label}</span>
              <span className="mx-1.5 text-[#777777]">·</span>
              <span className="font-mono text-[#3478F6] tabular-nums">
                Activation: {selectedNode.rawActivation.toFixed(3)}
              </span>
              {selectedNode.layerIndex === 3 && (
                <>
                  <span className="mx-1.5 text-[#777777]">·</span>
                  <span className="font-mono text-[#202124] tabular-nums">
                    Probability: {(selectedNode.activation * 100).toFixed(1)}%
                  </span>
                </>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSelectedNode(null)}
              className="text-[#777777] hover:text-[#202124] font-medium"
            >
              Close
            </button>
          </div>
        )}
      </div>

      {/* Stage Stepper & Playback Controls */}
      <div className="mt-4 bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
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

          <div className="flex items-center gap-2">
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
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              onClick={handleStepStage}
              aria-label="Next processing stage"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <StepForward className="w-3.5 h-3.5" />
              Next Step
            </button>
            <button
              type="button"
              onClick={handleReplay}
              aria-label="Replay neural network flow"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Replay
            </button>
          </div>
        </div>

        <div className="mt-2">
          <div className="flex items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-[#202124]">{stageInfo.title}</h3>
            <span className="text-xs text-[#777777] font-mono">{stageInfo.subtitle}</span>
          </div>
          <p className="text-xs text-[#777777] mt-1 leading-relaxed">{stageInfo.explanation}</p>
        </div>
      </div>

      {/* Transparency Note on Visual Simplification */}
      <div className="mt-3 flex items-center justify-between gap-2 text-[11px] text-[#777777]">
        <span className="inline-flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-[#3478F6] shrink-0" />
          Display simplified to 50 representative nodes so individual connections stay clear. Full 784 → 128 → 64 → 10 math runs underneath.
        </span>
        <span className="font-mono tabular-nums shrink-0">
          {activeModelMode === 'pretrained' ? 'Pretrained Reference' : 'Visitor-Trained Model'}
        </span>
      </div>
    </div>
  );
}
