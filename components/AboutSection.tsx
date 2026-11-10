'use client';

import React, { useState } from 'react';
import {
  RotateCcw,
  Cpu,
  ShieldCheck,
  Layers,
  Eye,
  Code2,
  Terminal,
  Copy,
  Check,
  Award,
} from 'lucide-react';
import { TfModelRuntimeInfo } from '@/lib/tf-bridge';

interface AboutSectionProps {
  tfInfo: TfModelRuntimeInfo;
  onResetAll: () => void;
  onGoToExplore: () => void;
}

export default function AboutSection({
  tfInfo,
  onResetAll,
  onGoToExplore,
}: AboutSectionProps) {
  const [copiedCmd, setCopiedCmd] = useState(false);

  const handleCopyCloneSnippet = () => {
    const snippet = `git clone https://github.com/neurolab-exhibit/neurolab.git\ncd neurolab && npm install && npm run dev`;
    navigator.clipboard?.writeText(snippet).catch(() => {});
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2500);
  };

  return (
    <div className="space-y-8">
      {/* Hero Banner */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="max-w-2xl">
          <p className="text-xs font-medium text-[#3478F6] mb-1">
            About the Exhibition & Open-Source Project · NEUROLAB
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#202124] tracking-tight">
            Teach a machine. Watch it learn.
          </h1>
          <p className="text-sm sm:text-base text-[#777777] mt-2 leading-relaxed">
            NEUROLAB is an interactive educational science exhibit designed to make artificial intelligence transparent, approachable, and tangible for visitors of every age.
          </p>
          <p className="text-xs font-semibold text-[#202124] mt-3">
            Made by R.K Nived Krishnan &amp; Adithyan J
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={onGoToExplore}
            className="px-4 py-2.5 rounded-xl bg-[#3478F6] hover:bg-[#2563EB] text-xs font-medium text-white transition-colors whitespace-nowrap cursor-pointer"
          >
            Return to Explore
          </button>
          <button
            type="button"
            onClick={onResetAll}
            className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#F2F2EF] text-xs font-medium text-[#202124] transition-colors whitespace-nowrap cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Exhibit for Next Visitor
          </button>
        </div>
      </div>

      {/* Creators & Exhibition Attribution Card */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#3478F6]/10 border border-[#3478F6]/25 flex items-center justify-center shrink-0">
            <Award className="w-5 h-5 text-[#3478F6]" />
          </div>
          <div>
            <span className="text-xs font-medium text-[#777777]">
              Project Creators &amp; Machine Learning Engineering
            </span>
            <h2 className="text-lg font-semibold text-[#202124]">
              Made by R.K Nived Krishnan &amp; Adithyan J
            </h2>
            <p className="text-xs text-[#777777] mt-0.5">
              Designed and engineered for public science exhibition &amp; open-source GitHub release.
            </p>
          </div>
        </div>
        <div className="text-xs font-mono text-[#777777] tabular-nums shrink-0">
          1,000 Built-In Digit Examples · 109,386 Parameters
        </div>
      </div>

      {/* Why Two Models? Dual Architecture Transparency */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#3478F6]">
            <ShieldCheck className="w-4 h-4" />
            Model 1 · Pretrained MNIST Reference Model (800 Examples)
          </div>
          <h2 className="text-lg font-semibold text-[#202124]">
            Reliable Initial Digit Recognition
          </h2>
          <p className="text-sm text-[#777777] leading-relaxed">
            When you first draw a number in the Explore stage, our Pretrained Reference Model provides dependable digit recognition. It is trained with shuffled mini-batch Adam on 800 multi-archetype handwritten digits (0–9) with center-of-mass bounding-box normalisation and adaptive stroke scaling.
          </p>
        </div>

        <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#5856D6]">
            <Layers className="w-4 h-4" />
            Model 2 · Visitor-Trainable Neural Network
          </div>
          <h2 className="text-lg font-semibold text-[#202124]">
            Live Backpropagation &amp; Parameter Updates
          </h2>
          <p className="text-sm text-[#777777] leading-relaxed">
            To show how a machine actually learns from feedback without corrupting the reference recogniser, NEUROLAB maintains a separate 784 → 128 → 64 → 10 trainable network. When you teach the AI, real Adam backpropagation updates its 109,386 weights in your browser and evaluates against 100 unseen test digits.
          </p>
        </div>
      </div>

      {/* Technical Architecture & TensorFlow.js Table */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#202124] flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#3478F6]" />
              Neural Network Architecture (784 → 128 → 64 → 10)
            </h2>
            <p className="text-xs text-[#777777] mt-0.5">
              Synchronised with TensorFlow.js ({tfInfo.backend} backend · {tfInfo.totalParameters.toLocaleString()} total learnable parameters).
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#E5E5E0] text-[#777777]">
                <th className="py-2.5 pr-4 font-medium">Layer Name</th>
                <th className="py-2.5 px-4 font-medium">Dimensions</th>
                <th className="py-2.5 px-4 font-medium">Mathematical Activation</th>
                <th className="py-2.5 pl-4 font-medium">Learnable Weights + Biases</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E5E0]">
              {tfInfo.layersSummary.map((layer) => (
                <tr key={layer.name} className="text-[#202124]">
                  <td className="py-3 pr-4 font-medium">{layer.name}</td>
                  <td className="py-3 px-4 font-mono tabular-nums">{layer.shape}</td>
                  <td className="py-3 px-4">{layer.activation}</td>
                  <td className="py-3 pl-4 font-mono tabular-nums">
                    {layer.params.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* GitHub Repository & Open-Source Documentation Card */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-[#202124]">
              <Code2 className="w-4 h-4 text-[#3478F6]" />
              GitHub Repository Overview &amp; Quick Start
            </div>
            <h2 className="text-lg font-semibold text-[#202124] mt-1">
              neurolab — Interactive Machine Learning Experience
            </h2>
            <p className="text-xs text-[#777777] mt-0.5">
              Complete README.md, unit tests, and modular TypeScript architecture included in the repository root.
            </p>
          </div>

          <button
            type="button"
            onClick={handleCopyCloneSnippet}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#F2F2EF] text-xs font-medium text-[#202124] transition-colors cursor-pointer"
          >
            {copiedCmd ? (
              <>
                <Check className="w-3.5 h-3.5 text-[#3478F6]" />
                Copied Commands
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                Copy Setup Commands
              </>
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4 font-mono text-xs text-[#202124] space-y-1.5">
            <div className="text-[11px] font-sans font-semibold text-[#777777] flex items-center gap-1.5 mb-2">
              <Terminal className="w-3.5 h-3.5 text-[#3478F6]" />
              Local Development &amp; Verification
            </div>
            <div>$ npm install</div>
            <div>$ npm run dev</div>
            <div>$ npx --yes tsx --test lib/neural-engine.test.ts</div>
            <div>$ npm run build</div>
          </div>

          <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4 text-xs text-[#777777] space-y-2 leading-relaxed">
            <div className="font-semibold text-[#202124]">
              Repository Highlights
            </div>
            <p>
              • Pure TypeScript + Float32Array 784→128→64→10 neural network engine synchronized with TensorFlow.js.
            </p>
            <p>
              • 1,000-example multi-archetype dataset with MNIST 20×20 center-of-mass bounding-box normalization and adaptive stroke scaling.
            </p>
            <p>
              • Full offline reliability for school and science museum exhibitions.
            </p>
          </div>
        </div>
      </div>

      {/* Exhibition Accessibility & Offline Reliability */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6">
        <div className="flex items-center gap-2 text-xs font-semibold text-[#202124] mb-2">
          <Eye className="w-4 h-4 text-[#3478F6]" />
          Designed for Public Science &amp; School Exhibitions
        </div>
        <p className="text-xs text-[#777777] leading-relaxed">
          All neural network forward passes, ReLU activations, Softmax distributions, Cross-Entropy loss calculations, and Adam optimiser weight updates run 100% locally in the browser with zero external API keys or internet requirements once loaded. Session progress is saved automatically in local storage.
        </p>
      </div>
    </div>
  );
}
