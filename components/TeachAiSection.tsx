'use client';

import React, { useState } from 'react';
import {
  GraduationCap,
  Play,
  Plus,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  RotateCcw,
} from 'lucide-react';
import {
  LabeledDigitExample,
  TrainingSessionResult,
  synthesizeHandwrittenDigit,
} from '@/lib/neural-engine';
import DrawingPad from './DrawingPad';

interface TeachAiSectionProps {
  visitorDataset: LabeledDigitExample[];
  onAddExample: (pixels: number[], label: number) => void;
  onRunTraining: (epochs: number, learningRate: number, focusExample?: LabeledDigitExample) => void;
  lastSession: TrainingSessionResult | null;
  currentTestAccuracy: number;
  onResetVisitorModel: () => void;
}

export default function TeachAiSection({
  visitorDataset,
  onAddExample,
  onRunTraining,
  lastSession,
  currentTestAccuracy,
  onResetVisitorModel,
}: TeachAiSectionProps) {
  const [customPixels, setCustomPixels] = useState<number[]>(() =>
    synthesizeHandwrittenDigit(4, { thickness: 1.85, slant: 0.05 })
  );
  const [selectedLabel, setSelectedLabel] = useState<number>(4);
  const [epochs, setEpochs] = useState<number>(12);
  const [learningRate, setLearningRate] = useState<number>(0.005);
  const [justAddedToast, setJustAddedToast] = useState<string | null>(null);

  const visitorContributedCount = visitorDataset.filter((d) => d.source === 'visitor').length;

  // Count examples per digit (0-9)
  const digitCounts = new Array(10).fill(0);
  for (const ex of visitorDataset) {
    digitCounts[ex.label]++;
  }

  const handleSaveAndTeach = () => {
    const activeInk = customPixels.filter((p) => p > 0.1).length;
    if (activeInk === 0) return;

    onAddExample(customPixels, selectedLabel);
    const focusEx: LabeledDigitExample = {
      id: `temp-${Date.now()}`,
      pixels: customPixels,
      label: selectedLabel,
      createdAt: Date.now(),
      source: 'visitor',
    };
    onRunTraining(epochs, learningRate, focusEx);
    setJustAddedToast(
      `Added your drawing of "${selectedLabel}" and trained the network for ${epochs} epochs!`
    );
    setTimeout(() => setJustAddedToast(null), 5000);
  };

  const handleBatchAddVariety = (perDigitCount = 5) => {
    // Adds 10 or 50 fresh varied handwriting examples across all 4 archetypes (0-9) and trains
    let lastEx: LabeledDigitExample | undefined;
    const thicknesses = [1.0, 1.2, 1.35, 1.5, 1.65];
    for (let d = 0; d <= 9; d++) {
      for (let k = 0; k < perDigitCount; k++) {
        const px = synthesizeHandwrittenDigit(d, {
          thickness: thicknesses[k % thicknesses.length],
          slant: ((k % 5) - 2) * 0.06,
          scaleX: 0.9 + (k % 3) * 0.08,
          scaleY: 0.9 + ((k + 1) % 3) * 0.08,
          variant: (visitorDataset.length + k) % 4,
        });
        onAddExample(px, d);
        lastEx = {
          id: `batch-${d}-${k}`,
          pixels: px,
          label: d,
          createdAt: Date.now(),
          source: 'visitor',
        };
      }
    }
    const totalAdded = perDigitCount * 10;
    onRunTraining(epochs, learningRate, lastEx);
    setJustAddedToast(
      `Added +${totalAdded} diverse handwritten digits (0–9 across 4 handwriting styles) and completed live mini-batch backpropagation!`
    );
    setTimeout(() => setJustAddedToast(null), 5000);
  };

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="max-w-2xl">
          <p className="text-xs font-medium text-[#777777] mb-1">
            Interactive Training Studio · Real Mini-Batch Backpropagation
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#202124] tracking-tight">
            Teach the AI With a Bigger Dataset
          </h1>
          <p className="text-sm sm:text-base text-[#777777] mt-2 leading-relaxed">
            Draw a number, label what digit it truly is, or inject +50 diverse handwritten examples at once to watch the 784 → 128 → 64 → 10 network refine its 109,386 weights.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() => handleBatchAddVariety(5)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-[#3478F6] hover:bg-[#2563EB] text-xs font-semibold text-white transition-colors whitespace-nowrap cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            Expand Dataset (+50 Examples) & Train
          </button>
          <button
            type="button"
            onClick={() => handleBatchAddVariety(1)}
            className="inline-flex items-center gap-2 px-3.5 py-2.5 rounded-xl border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#F2F2EF] text-xs font-medium text-[#202124] transition-colors whitespace-nowrap cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5 text-[#3478F6]" />
            +10 Examples
          </button>
          <button
            type="button"
            onClick={onResetVisitorModel}
            className="inline-flex items-center gap-1.5 px-3.5 py-2.5 rounded-xl border border-[#E5E5E0] bg-white hover:bg-[#F2F2EF] text-xs font-medium text-[#777777] hover:text-[#202124] transition-colors whitespace-nowrap cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {justAddedToast && (
        <div className="bg-[#3478F6]/10 border border-[#3478F6]/30 text-[#202124] rounded-xl px-4 py-3 text-xs font-medium flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-[#3478F6] shrink-0" />
          <span>{justAddedToast}</span>
        </div>
      )}

      {/* Main Two-Column Training Studio */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Draw & Label Example */}
        <div className="lg:col-span-6 space-y-6">
          <DrawingPad
            pixels28x28={customPixels}
            onPixelsChange={(px) => setCustomPixels(px)}
            onRecognize={(px) => setCustomPixels(px)}
          />
        </div>

        {/* Right: Label Selection, Training Parameters & Live Weight Delta Results */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-5">
            <div>
              <span className="text-xs font-medium text-[#777777]">
                Step 1 · Label Your Example
              </span>
              <h2 className="text-lg font-semibold text-[#202124] mt-0.5">
                Which number did you draw?
              </h2>
              <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 mt-3">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setSelectedLabel(d)}
                    className={`h-11 rounded-xl font-semibold text-sm border transition-all tabular-nums cursor-pointer ${
                      selectedLabel === d
                        ? 'bg-[#3478F6] text-white border-[#3478F6] shadow-xs'
                        : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                    }`}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>

            {/* Training Rounds (Epochs) & Speed */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-[#E5E5E0]">
              <div>
                <span className="text-xs font-semibold text-[#202124] block mb-2">
                  Training Rounds (Epochs): {epochs}
                </span>
                <div className="flex items-center gap-1.5">
                  {[5, 12, 20].map((ep) => (
                    <button
                      key={ep}
                      type="button"
                      onClick={() => setEpochs(ep)}
                      className={`flex-1 py-2 rounded-lg text-xs font-medium border transition-colors cursor-pointer ${
                        epochs === ep
                          ? 'bg-[#202124] text-white border-[#202124]'
                          : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                      }`}
                    >
                      {ep} rounds
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <span className="text-xs font-semibold text-[#202124] block mb-2">
                  Step Size (Learning Rate)
                </span>
                <div className="flex items-center gap-1.5">
                  {[
                    { label: 'Gentle (0.002)', val: 0.002 },
                    { label: 'Balanced (0.005)', val: 0.005 },
                    { label: 'Bold (0.009)', val: 0.009 },
                  ].map((opt) => (
                    <button
                      key={opt.label}
                      type="button"
                      onClick={() => setLearningRate(opt.val)}
                      className={`flex-1 py-2 px-2 rounded-lg text-[11px] font-medium border transition-colors whitespace-nowrap truncate cursor-pointer ${
                        learningRate === opt.val
                          ? 'bg-[#202124] text-white border-[#202124]'
                          : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleSaveAndTeach}
              className="w-full py-3.5 px-5 rounded-xl bg-[#3478F6] hover:bg-[#2563EB] text-white font-semibold text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer"
            >
              <GraduationCap className="w-4 h-4" />
              Add Label “{selectedLabel}” & Train the AI Now
            </button>

            {/* Honest Scientific Generalisation Notice */}
            <div className="p-3.5 rounded-xl bg-[#FAFAF8] border border-[#E5E5E0] flex items-start gap-2.5 text-xs text-[#777777]">
              <AlertCircle className="w-4 h-4 text-[#3478F6] shrink-0 mt-0.5" />
              <div className="leading-relaxed">
                <strong className="text-[#202124]">Scientific honesty note:</strong>{' '}
                Teaching the AI a single drawing immediately boosts its confidence on that specific style, but true generalisation requires balanced examples across all digits (0–9). Current held-out test accuracy is{' '}
                <strong className="text-[#202124] font-mono tabular-nums">
                  {(currentTestAccuracy * 100).toFixed(1)}%
                </strong>
                .
              </div>
            </div>
          </div>

          {/* Latest Training Session Before/After Card */}
          {lastSession && (
            <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-medium text-[#3478F6]">
                    Latest Backpropagation Result
                  </span>
                  <h3 className="text-base font-semibold text-[#202124]">
                    How the Network Changed ({lastSession.epochs} Epochs)
                  </h3>
                </div>
                <span className="text-xs font-mono text-[#777777] tabular-nums">
                  {lastSession.examplesCount} training examples
                </span>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3">
                  <span className="text-[11px] text-[#777777]">Training Loss</span>
                  <p className="text-sm font-semibold font-mono text-[#202124] tabular-nums mt-0.5">
                    {lastSession.initialLoss.toFixed(3)} →{' '}
                    <span className="text-[#3478F6]">
                      {lastSession.finalLoss.toFixed(3)}
                    </span>
                  </p>
                </div>
                <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3">
                  <span className="text-[11px] text-[#777777]">Avg Weight Change</span>
                  <p className="text-sm font-semibold font-mono text-[#202124] tabular-nums mt-0.5">
                    ΔW {lastSession.meanAbsWeightDelta.toFixed(4)}
                  </p>
                </div>
                <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-3">
                  <span className="text-[11px] text-[#777777]">Test Set Accuracy</span>
                  <p className="text-sm font-semibold font-mono text-[#202124] tabular-nums mt-0.5">
                    {(lastSession.testAccuracyBefore * 100).toFixed(1)}% →{' '}
                    <span className="text-[#3478F6]">
                      {(lastSession.testAccuracyAfter * 100).toFixed(1)}%
                    </span>
                  </p>
                </div>
              </div>

              {lastSession.beforePrediction && lastSession.afterPrediction && (
                <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4">
                  <div className="text-xs font-semibold text-[#202124] mb-2">
                    Confidence on Digit “{lastSession.targetDigit}” Before vs. After Training:
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-[11px] text-[#777777]">Before Training</span>
                      <div className="text-sm font-mono font-semibold text-[#202124] tabular-nums">
                        Guessed {lastSession.beforePrediction.digit} (True digit{' '}
                        {lastSession.targetDigit}:{' '}
                        {(lastSession.beforePrediction.targetDigitProb * 100).toFixed(1)}%)
                      </div>
                    </div>
                    <div>
                      <span className="text-[11px] text-[#777777]">After Training</span>
                      <div className="text-sm font-mono font-semibold text-[#3478F6] tabular-nums">
                        Guessed {lastSession.afterPrediction.digit} (True digit{' '}
                        {lastSession.targetDigit}:{' '}
                        {(lastSession.afterPrediction.targetDigitProb * 100).toFixed(1)}%)
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Dataset Balance Across Digits 0-9 */}
          <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-[#202124]">
                  Visitor Training Dataset Balance ({visitorDataset.length} total ·{' '}
                  {visitorContributedCount} added by visitors)
                </h3>
                <p className="text-xs text-[#777777]">
                  Number of training examples stored for each digit (0–9).
                </p>
              </div>
            </div>
            <div className="grid grid-cols-10 gap-2 pt-2">
              {digitCounts.map((cnt, d) => (
                <div
                  key={d}
                  className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-2.5 text-center"
                >
                  <div className="text-xs font-semibold text-[#202124] tabular-nums">
                    #{d}
                  </div>
                  <div className="text-xs font-mono text-[#3478F6] font-semibold tabular-nums mt-1">
                    {cnt}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
