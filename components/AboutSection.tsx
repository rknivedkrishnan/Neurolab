'use client';

import React from 'react';
import { RotateCcw, Cpu, ShieldCheck, Layers, Eye } from 'lucide-react';
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
  return (
    <div className="space-y-8">
      {/* Hero Banner */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="max-w-2xl">
          <p className="text-xs font-medium text-[#777777] mb-1">
            About the Exhibition · NEUROLAB
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#202124] tracking-tight">
            Teach a machine. Watch it learn.
          </h1>
          <p className="text-sm sm:text-base text-[#777777] mt-2 leading-relaxed">
            NEUROLAB is an interactive educational science exhibit designed to make artificial intelligence transparent, approachable, and tangible for visitors of every age.
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

      {/* Why Two Models? Dual Architecture Transparency */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#3478F6]">
            <ShieldCheck className="w-4 h-4" />
            Model 1 · Pretrained MNIST Reference Model
          </div>
          <h2 className="text-lg font-semibold text-[#202124]">
            Reliable Initial Digit Recognition
          </h2>
          <p className="text-sm text-[#777777] leading-relaxed">
            When you first draw a number in the Explore stage, our Pretrained Reference Model provides dependable digit recognition. It has already practiced on multi-style handwritten digits (0–9) with center-of-mass bounding-box normalisation.
          </p>
        </div>

        <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#5856D6]">
            <Layers className="w-4 h-4" />
            Model 2 · Visitor-Trainable Neural Network
          </div>
          <h2 className="text-lg font-semibold text-[#202124]">
            Live Backpropagation & Parameter Updates
          </h2>
          <p className="text-sm text-[#777777] leading-relaxed">
            To show how a machine actually learns from feedback without corrupting the reference recogniser, NEUROLAB maintains a separate 784 → 128 → 64 → 10 trainable network. When you teach the AI, real Adam backpropagation updates its 109,386 weights in your browser.
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

      {/* Exhibition Accessibility & Offline Reliability */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6">
        <div className="flex items-center gap-2 text-xs font-semibold text-[#202124] mb-2">
          <Eye className="w-4 h-4 text-[#3478F6]" />
          Designed for Public Science & School Exhibitions
        </div>
        <p className="text-xs text-[#777777] leading-relaxed">
          All neural network forward passes, ReLU activations, Softmax distributions, Cross-Entropy loss calculations, and Adam optimiser weight updates run 100% locally in the browser with zero external API keys or internet requirements once loaded. Session progress is saved automatically in local storage.
        </p>
      </div>
    </div>
  );
}
