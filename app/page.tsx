'use client';

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  CheckCircle2,
  XCircle,
  Sparkles,
  GraduationCap,
  ArrowRight,
  Maximize2,
  Minimize2,
  RefreshCw,
  Sun,
  Moon,
} from 'lucide-react';
import {
  initializeNeurolabModels,
  forwardPass,
  trainNetworkOnBatch,
  cloneNeuralNetwork,
  synthesizeHandwrittenDigit,
  evaluateDataset,
  LabeledDigitExample,
  TrainingSessionResult,
  NeuralNetworkState,
} from '@/lib/neural-engine';
import { syncWithTensorFlowJs, TfModelRuntimeInfo } from '@/lib/tf-bridge';
import DrawingPad from '@/components/DrawingPad';
import NeuralNetworkViz from '@/components/NeuralNetworkViz';
import HowItWorksSection from '@/components/HowItWorksSection';
import TeachAiSection from '@/components/TeachAiSection';
import LearningProgressSection from '@/components/LearningProgressSection';
import AboutSection from '@/components/AboutSection';
import { useTheme } from '@/components/ThemeProvider';
import AmbientDarkBackground from '@/components/AmbientDarkBackground';

type NavSection = 'explore' | 'how-it-works' | 'teach' | 'progress' | 'about';

const STORAGE_KEY_SESSIONS = 'neurolab_exhibition_sessions_v1';
const STORAGE_KEY_VISITOR_EXAMPLES = 'neurolab_visitor_examples_v1';

export default function NeurolabApp() {
  const { isDark, toggleTheme } = useTheme();
  const [activeNav, setActiveNav] = useState<NavSection>('explore');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Initialize genuine 784->128->64->10 models & datasets deterministically
  const initialBundle = useMemo(() => initializeNeurolabModels(), []);

  const [pretrainedModel] = useState<NeuralNetworkState>(() => initialBundle.pretrainedModel);
  const [visitorModel, setVisitorModel] = useState<NeuralNetworkState>(
    () => initialBundle.visitorModel
  );
  const [visitorDataset, setVisitorDataset] = useState<LabeledDigitExample[]>(
    () => initialBundle.visitorDataset
  );
  const [testDataset] = useState<LabeledDigitExample[]>(() => initialBundle.testDataset);
  const [trainingSessions, setTrainingSessions] = useState<TrainingSessionResult[]>([]);

  // Active model mode for Stage 2 & 3 inspection ('pretrained' for reliable initial recognition, 'visitor' for visitor-trained network)
  const [activeModelMode, setActiveModelMode] = useState<'pretrained' | 'visitor'>('pretrained');

  // Stage 1: Current 28x28 drawing (starts with a crisp sample '7' so the exhibition screen is immediately alive)
  const [currentPixels, setCurrentPixels] = useState<number[]>(() =>
    synthesizeHandwrittenDigit(7, { thickness: 1.3, slant: 0.04, variant: 0 })
  );
  const [animationTrigger, setAnimationTrigger] = useState<number>(1);
  const [activeExploreStage, setActiveExploreStage] = useState<1 | 2 | 3 | 4>(3);

  // Stage 3: Visitor feedback on prediction ('correct' | 'incorrect' | null)
  const [predictionFeedback, setPredictionFeedback] = useState<'correct' | 'incorrect' | null>(
    null
  );

  // Stage 4: Teach the AI state
  const [correctDigitSelection, setCorrectDigitSelection] = useState<number | null>(7);
  const [isTrainingAnim, setIsTrainingAnim] = useState(false);
  const [latestExploreTraining, setLatestExploreTraining] =
    useState<TrainingSessionResult | null>(null);

  // TensorFlow.js runtime synchronization info computed cleanly via useMemo
  const tfInfo = useMemo<TfModelRuntimeInfo>(
    () =>
      syncWithTensorFlowJs(
        activeModelMode === 'pretrained' ? pretrainedModel : visitorModel
      ),
    [activeModelMode, pretrainedModel, visitorModel]
  );

  // Restore saved visitor examples & training sessions from localStorage on mount
  useEffect(() => {
    queueMicrotask(() => {
      try {
        const savedSessions = localStorage.getItem(STORAGE_KEY_SESSIONS);
        if (savedSessions) {
          const parsed = JSON.parse(savedSessions);
          if (Array.isArray(parsed)) setTrainingSessions(parsed);
        }
        const savedExamples = localStorage.getItem(STORAGE_KEY_VISITOR_EXAMPLES);
        if (savedExamples) {
          const parsedEx: LabeledDigitExample[] = JSON.parse(savedExamples);
          if (Array.isArray(parsedEx) && parsedEx.length > 0) {
            setVisitorDataset((prev) => {
              const existingIds = new Set(prev.map((p) => p.id));
              const merged = [...prev];
              for (const item of parsedEx) {
                if (!existingIds.has(item.id)) merged.push(item);
              }
              return merged;
            });
          }
        }
      } catch {
        // Ignore storage errors in private/restricted browsing
      }
    });
  }, []);

  // Compute live forward pass activations for both models
  const activeNetwork = activeModelMode === 'pretrained' ? pretrainedModel : visitorModel;
  const currentActivations = useMemo(
    () => forwardPass(currentPixels, activeNetwork),
    [currentPixels, activeNetwork]
  );
  const visitorActivations = useMemo(
    () => forwardPass(currentPixels, visitorModel),
    [currentPixels, visitorModel]
  );

  const visitorTestAccuracy = useMemo(
    () => evaluateDataset(visitorModel, testDataset).accuracy,
    [visitorModel, testDataset]
  );

  const handlePixelsChange = useCallback((pixels: number[], autoRecognize = false) => {
    setCurrentPixels(pixels);
    setPredictionFeedback(null);
    if (autoRecognize) {
      setAnimationTrigger((t) => t + 1);
      setActiveExploreStage(2);
      setTimeout(() => setActiveExploreStage(3), 650);
    } else {
      setActiveExploreStage(1);
    }
  }, []);

  const handleRecognizeClick = useCallback((pixels: number[]) => {
    setCurrentPixels(pixels);
    setPredictionFeedback(null);
    setAnimationTrigger((t) => t + 1);
    setActiveExploreStage(3);
  }, []);

  // Add a labeled example to the visitor dataset
  const handleAddVisitorExample = useCallback(
    (pixels: number[], label: number): LabeledDigitExample => {
      const newExample: LabeledDigitExample = {
        id: `visitor-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        pixels: [...pixels],
        label,
        createdAt: Date.now(),
        source: 'visitor',
        predictedBefore: currentActivations.predictedDigit,
      };

      setVisitorDataset((prev) => {
        const next = [...prev, newExample];
        try {
          const visitorOnly = next.filter((e) => e.source === 'visitor').slice(-40);
          localStorage.setItem(STORAGE_KEY_VISITOR_EXAMPLES, JSON.stringify(visitorOnly));
        } catch {
          // Ignore storage quota
        }
        return next;
      });

      return newExample;
    },
    [currentActivations.predictedDigit]
  );

  // Perform real Adam backpropagation on the Visitor-Trainable Model
  const handleRunTraining = useCallback(
    (epochs: number, learningRate: number, focusExample?: LabeledDigitExample) => {
      setIsTrainingAnim(true);

      const nextModel = cloneNeuralNetwork(visitorModel);
      // Keep a balanced 2-per-digit anchor from starter examples + visitor examples + 4x emphasis on focusExample
      // so the network learns the visitor's drawing rapidly without biasing toward any single digit class
      const visitorContributed = visitorDataset.filter((d) => d.source === 'visitor');
      const starterAnchor = visitorDataset.filter((d, i) => d.source === 'starter' && i % 5 < 2);
      const trainingPool = focusExample
        ? [
            ...starterAnchor,
            ...visitorContributed,
            focusExample,
            focusExample,
            focusExample,
            focusExample,
          ]
        : [...visitorDataset];

      const sessionResult = trainNetworkOnBatch(nextModel, trainingPool, {
        epochs,
        learningRate,
        testDataset,
        focusExample,
      });

      setVisitorModel(nextModel);
      setLatestExploreTraining(sessionResult);
      setTrainingSessions((prev) => {
        const updated = [sessionResult, ...prev].slice(0, 25);
        try {
          localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(updated));
        } catch {
          // Ignore storage error
        }
        return updated;
      });

      setTimeout(() => {
        setIsTrainingAnim(false);
      }, 350);
    },
    [visitorModel, visitorDataset, testDataset]
  );

  // Stage 4 handler: Visitor selects the true digit and trains the AI
  const handleTeachFromExplore = (trueDigit: number) => {
    setCorrectDigitSelection(trueDigit);
    setActiveExploreStage(4);
    const addedEx = handleAddVisitorExample(currentPixels, trueDigit);
    handleRunTraining(6, 0.004, addedEx);
  };

  // Expand Visitor Dataset by +50 diverse handwritten examples (5 per digit across all 4 styles)
  const handleExpandDatasetFromExplore = () => {
    setActiveExploreStage(4);
    const thicknesses = [1.0, 1.2, 1.35, 1.5, 1.65];
    const newBatch: LabeledDigitExample[] = [];
    for (let d = 0; d <= 9; d++) {
      for (let k = 0; k < 5; k++) {
        const px = synthesizeHandwrittenDigit(d, {
          thickness: thicknesses[k % thicknesses.length],
          slant: ((k % 5) - 2) * 0.06,
          scaleX: 0.9 + (k % 3) * 0.08,
          scaleY: 0.9 + ((k + 1) % 3) * 0.08,
          variant: (visitorDataset.length + k) % 4,
        });
        newBatch.push({
          id: `exp-${Date.now()}-${d}-${k}`,
          pixels: px,
          label: d,
          createdAt: Date.now(),
          source: 'visitor',
        });
      }
    }
    const mergedDataset = [...visitorDataset, ...newBatch];
    setVisitorDataset(mergedDataset);
    const nextModel = cloneNeuralNetwork(visitorModel);
    const sessionResult = trainNetworkOnBatch(nextModel, mergedDataset, {
      epochs: 5,
      learningRate: 0.0045,
      batchSize: 25,
      testDataset,
    });
    setVisitorModel(nextModel);
    setLatestExploreTraining(sessionResult);
    setTrainingSessions((prev) => [sessionResult, ...prev].slice(0, 25));
  };

  // Reset visitor model & session for the next exhibition guest
  const handleResetAll = () => {
    const fresh = initializeNeurolabModels();
    setVisitorModel(fresh.visitorModel);
    setVisitorDataset(fresh.visitorDataset);
    setTrainingSessions([]);
    setLatestExploreTraining(null);
    setPredictionFeedback(null);
    setActiveModelMode('pretrained');
    try {
      localStorage.removeItem(STORAGE_KEY_SESSIONS);
      localStorage.removeItem(STORAGE_KEY_VISITOR_EXAMPLES);
    } catch {
      // Ignore
    }
  };

  const toggleExhibitionFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
      setIsFullscreen(false);
    }
  };

  const hasInk = currentPixels.some((p) => p > 0.08);

  return (
    <div
      className={`min-h-screen flex flex-col relative ${
        isDark ? 'bg-transparent text-[#F1F6FF]' : 'bg-[#FAFAF8] text-[#202124]'
      }`}
    >
      {/* Animated Live-Wallpaper Background (Active in Dark Mode) */}
      <AmbientDarkBackground />

      {/* Strict 3-Zone Top Navigation Bar */}
      <header className="sticky top-0 z-30 bg-[#FAFAF8]/95 border-b border-[#E5E5E0] px-4 sm:px-6 py-4 flex items-center justify-between gap-4 sm:gap-8">
        {/* Zone 1: Brand Wordmark */}
        <button
          type="button"
          onClick={() => setActiveNav('explore')}
          className="text-lg font-bold tracking-tight text-[#202124] whitespace-nowrap shrink-0 cursor-pointer"
        >
          NEUROLAB
        </button>

        {/* Zone 2: 5 Concise Single-Line Navigation Links */}
        <nav
          aria-label="Main Navigation"
          className="flex items-center gap-4 sm:gap-7 overflow-x-auto text-sm font-medium text-[#777777]"
        >
          {[
            { id: 'explore', label: 'Explore' },
            { id: 'how-it-works', label: 'How It Works' },
            { id: 'teach', label: 'Teach the AI' },
            { id: 'progress', label: 'Learning Progress' },
            { id: 'about', label: 'About' },
          ].map((item) => {
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveNav(item.id as NavSection)}
                className={`py-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer border-b-2 ${
                  isActive
                    ? 'text-[#202124] font-semibold border-[#3478F6]'
                    : 'text-[#777777] hover:text-[#202124] border-transparent'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Zone 3: Theme Toggle + Exhibition Mode */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="relative group">
            <button
              type="button"
              onClick={toggleTheme}
              data-testid="theme-toggle"
              aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              className={`w-9 h-9 rounded-xl border flex items-center justify-center transition-all duration-200 cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#258BFF]/50 ${
                isDark
                  ? 'bg-[#122440] border-[#29476A] text-[#35D8F2] hover:bg-[#193156] hover:border-[#35D8F2]/60 shadow-[0_0_14px_rgba(53,216,242,0.18)]'
                  : 'bg-white border-[#E5E5E0] text-[#202124] hover:bg-[#F2F2EF]'
              }`}
            >
              {isDark ? (
                <Sun className="w-4 h-4 transition-transform duration-300 rotate-0 scale-100" />
              ) : (
                <Moon className="w-4 h-4 transition-transform duration-300 rotate-0 scale-100" />
              )}
            </button>
            <span
              role="tooltip"
              className={`pointer-events-none absolute right-0 top-full mt-2 px-2.5 py-1 rounded-lg text-[11px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 border shadow-md z-40 ${
                isDark
                  ? 'bg-[#122440] border-[#29476A] text-[#F1F6FF]'
                  : 'bg-[#202124] border-[#202124] text-white'
              }`}
            >
              {isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            </span>
          </div>

          <button
            type="button"
            onClick={toggleExhibitionFullscreen}
            className="inline-flex items-center gap-1.5 px-3.5 sm:px-4 py-2 text-xs font-medium text-white bg-[#202124] hover:bg-[#333438] rounded-xl transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Exit Fullscreen</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Exhibition Mode</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="relative z-10 flex-1 max-w-[1380px] w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {activeNav === 'explore' && (
          <div className="space-y-6">
            {/* Compact Exhibition Hero & 4-Stage Connected Progress Bar */}
            <section className="bg-white rounded-2xl border border-[#E5E5E0] p-6 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div>
                <p className="text-xs font-medium text-[#3478F6] mb-1">
                  Interactive Machine Learning Exhibit
                </p>
                <h1 className="text-2xl sm:text-3xl font-semibold text-[#202124] tracking-tight">
                  Teach a machine. Watch it learn.
                </h1>
                <p className="text-sm text-[#777777] mt-1 max-w-xl">
                  Draw a single number from 0 to 9, watch signals travel through a real neural network, and teach the AI when it makes a mistake.
                </p>
              </div>

              {/* 4 Connected Stages Indicator: DRAW -> WATCH -> SEE THE RESULT -> TEACH THE AI */}
              <div
                aria-label="Interactive Journey Stages"
                className="grid grid-cols-2 sm:grid-cols-4 gap-2 shrink-0"
              >
                {[
                  { step: 1, label: '1. Draw' },
                  { step: 2, label: '2. Watch' },
                  { step: 3, label: '3. See Result' },
                  { step: 4, label: '4. Teach AI' },
                ].map((st) => {
                  const isCurrent = activeExploreStage >= st.step;
                  return (
                    <div
                      key={st.step}
                      className={`px-3.5 py-2.5 rounded-xl border text-xs font-medium flex items-center gap-2 transition-colors whitespace-nowrap ${
                        isCurrent
                          ? 'bg-[#3478F6]/8 border-[#3478F6]/40 text-[#202124]'
                          : 'bg-[#FAFAF8] border-[#E5E5E0] text-[#777777]'
                      }`}
                    >
                      <span
                        className={`w-5 h-5 rounded-full text-[11px] font-semibold flex items-center justify-center shrink-0 ${
                          isCurrent
                            ? 'bg-[#3478F6] text-white'
                            : 'bg-[#E5E5E0] text-[#777777]'
                        }`}
                      >
                        {st.step}
                      </span>
                      <span>{st.label.split('. ')[1]}</span>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Row 1: Stage 1 (Draw a Number) + Stage 2 (Watch the AI Think) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              <div className="lg:col-span-5 flex flex-col">
                <DrawingPad
                  pixels28x28={currentPixels}
                  onPixelsChange={handlePixelsChange}
                  onRecognize={handleRecognizeClick}
                />
              </div>

              <div className="lg:col-span-7 flex flex-col">
                <NeuralNetworkViz
                  activations={currentActivations}
                  network={activeNetwork}
                  activeModelMode={activeModelMode}
                  onModelModeChange={setActiveModelMode}
                  animationTrigger={animationTrigger}
                />
              </div>
            </div>

            {/* Row 2: Stage 3 (See the Result) + Stage 4 (Teach the AI) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* Stage 3: See the Result */}
              <div className="lg:col-span-6 bg-white rounded-2xl border border-[#E5E5E0] p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between gap-2 mb-4">
                    <div>
                      <p className="text-xs font-medium text-[#777777] tracking-wide mb-1">
                        Stage 3 · See the Result
                      </p>
                      <h2 className="text-xl font-semibold text-[#202124]">
                        {hasInk
                          ? `I think you drew a ${currentActivations.predictedDigit}!`
                          : 'Draw a number above to see the AI prediction'}
                      </h2>
                    </div>
                    {hasInk && (
                      <div className="text-right">
                        <span className="text-xs text-[#777777] block">Confidence</span>
                        <span
                          data-testid="prediction-confidence"
                          className="text-2xl font-bold font-mono text-[#3478F6] tabular-nums"
                        >
                          {(currentActivations.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Hero Predicted Digit Display + Probability Distribution Bars (0-9) */}
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-center bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4">
                    {/* Big Predicted Number Box */}
                    <div className="sm:col-span-4 flex flex-col items-center justify-center bg-white rounded-xl border border-[#E5E5E0] py-6 px-4 text-center">
                      <span className="text-xs font-medium text-[#777777]">
                        Predicted Digit
                      </span>
                      <span
                        data-testid="predicted-digit-value"
                        className="text-6xl font-bold text-[#202124] font-mono tabular-nums my-2"
                      >
                        {hasInk ? currentActivations.predictedDigit : '–'}
                      </span>
                      <span className="text-[11px] text-[#777777]">
                        {activeModelMode === 'pretrained'
                          ? 'Pretrained Reference'
                          : 'Visitor-Trained Network'}
                      </span>
                    </div>

                    {/* 0-9 Probability Distribution Bars */}
                    <div className="sm:col-span-8 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px] text-[#777777] mb-1">
                        <span>Probability Distribution (Digits 0–9)</span>
                        <span className="font-mono">Softmax %</span>
                      </div>
                      {currentActivations.probabilities.map((prob, digit) => {
                        const pct = hasInk ? prob * 100 : 0;
                        const isWinner =
                          hasInk && digit === currentActivations.predictedDigit;
                        return (
                          <div key={digit} className="flex items-center gap-2 text-xs">
                            <span
                              className={`w-4 font-mono tabular-nums text-right ${
                                isWinner
                                  ? 'font-bold text-[#3478F6]'
                                  : 'text-[#777777]'
                              }`}
                            >
                              {digit}
                            </span>
                            <div className="flex-1 h-2.5 bg-[#E5E5E0]/70 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${
                                  isWinner
                                    ? isDark
                                      ? 'bg-gradient-to-r from-[#258BFF] to-[#35D8F2] shadow-[0_0_10px_rgba(53,216,242,0.45)]'
                                      : 'bg-[#3478F6]'
                                    : isDark
                                    ? 'bg-[#7186A6]/45'
                                    : 'bg-[#777777]/45'
                                }`}
                                style={{ width: `${Math.max(pct, 1.5)}%` }}
                              />
                            </div>
                            <span
                              className={`w-12 font-mono tabular-nums text-right text-[11px] ${
                                isWinner
                                  ? 'font-semibold text-[#202124]'
                                  : 'text-[#777777]'
                              }`}
                            >
                              {pct.toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Friendly Explanation & Interactive Visitor Verification */}
                <div className="mt-5 pt-4 border-t border-[#E5E5E0] space-y-3">
                  <p className="text-xs text-[#777777] leading-relaxed">
                    The AI compares all 10 output neurons. Because neuron{' '}
                    <strong className="text-[#202124]">
                      #{currentActivations.predictedDigit}
                    </strong>{' '}
                    received the strongest combined signal (
                    {(currentActivations.confidence * 100).toFixed(1)}%), it is the network’s top choice.
                  </p>

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs font-semibold text-[#202124]">
                      Did the AI get your number right?
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setPredictionFeedback('correct');
                          setCorrectDigitSelection(currentActivations.predictedDigit);
                        }}
                        className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border transition-colors cursor-pointer ${
                          predictionFeedback === 'correct'
                            ? 'bg-[#202124] text-white border-[#202124]'
                            : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                        }`}
                      >
                        <CheckCircle2 className="w-3.5 h-3.5 text-[#3478F6]" />
                        Yes, that’s right!
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setPredictionFeedback('incorrect');
                          setActiveExploreStage(4);
                        }}
                        className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border transition-colors cursor-pointer ${
                          predictionFeedback === 'incorrect'
                            ? 'bg-[#3478F6] text-white border-[#3478F6]'
                            : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                        }`}
                      >
                        <XCircle className="w-3.5 h-3.5" />
                        No, let me teach it
                      </button>
                    </div>
                  </div>

                  {predictionFeedback === 'correct' && (
                    <div className="p-3 rounded-xl bg-[#FAFAF8] border border-[#E5E5E0] text-xs text-[#202124] font-medium">
                      Great! I recognised your number. You can still reinforce this example in Stage 4 on the right!
                    </div>
                  )}
                  {predictionFeedback === 'incorrect' && (
                    <div className="p-3 rounded-xl bg-[#3478F6]/10 border border-[#3478F6]/30 text-xs text-[#202124] font-medium">
                      Oops! I didn’t get that right. Can you teach me by tapping the correct number in Stage 4?
                    </div>
                  )}
                </div>
              </div>

              {/* Stage 4: Teach the AI */}
              <div className="lg:col-span-6 bg-white rounded-2xl border border-[#E5E5E0] p-6 flex flex-col justify-between">
                <div className="space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs font-medium text-[#777777] tracking-wide mb-1">
                        Stage 4 · Teach the AI
                      </p>
                      <h2 className="text-xl font-semibold text-[#202124]">
                        Which number did you actually draw?
                      </h2>
                      <p className="text-sm text-[#777777] mt-0.5">
                        Select the true number below to add your drawing to the training dataset and run real backpropagation.
                      </p>
                    </div>
                  </div>

                  {/* 10 Large Selectable Number Buttons (0-9) */}
                  <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 pt-1">
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => {
                      const isSelected = correctDigitSelection === digit;
                      return (
                        <button
                          key={digit}
                          type="button"
                          onClick={() => setCorrectDigitSelection(digit)}
                          aria-label={`Select actual digit ${digit}`}
                          className={`h-12 rounded-xl font-semibold text-base border transition-all tabular-nums cursor-pointer ${
                            isSelected
                              ? 'bg-[#3478F6] text-white border-[#3478F6] shadow-xs'
                              : 'bg-[#FAFAF8] text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
                          }`}
                        >
                          {digit}
                        </button>
                      );
                    })}
                  </div>

                  {/* Primary Action to Trigger Genuine Model Training + Expand Dataset */}
                  <div className="flex flex-col sm:flex-row items-stretch gap-2.5">
                    <button
                      type="button"
                      disabled={correctDigitSelection === null || !hasInk || isTrainingAnim}
                      onClick={() => {
                        if (correctDigitSelection !== null) {
                          handleTeachFromExplore(correctDigitSelection);
                        }
                      }}
                      className="flex-1 py-3.5 px-4 rounded-xl bg-[#202124] hover:bg-[#333438] disabled:bg-[#E5E5E0] disabled:text-[#777777] text-white font-semibold text-sm flex items-center justify-center gap-2 transition-colors whitespace-nowrap cursor-pointer"
                    >
                      {isTrainingAnim ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Updating Weights...
                        </>
                      ) : (
                        <>
                          <GraduationCap className="w-4 h-4 text-[#3478F6]" />
                          Teach AI: “This Drawing is a {correctDigitSelection ?? '?'}”
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      disabled={isTrainingAnim}
                      onClick={handleExpandDatasetFromExplore}
                      className="py-3.5 px-4 rounded-xl border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#F2F2EF] text-xs font-semibold text-[#202124] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      +50 Dataset Examples
                    </button>
                  </div>

                  {/* Live Training Result & Before/After Comparison */}
                  {latestExploreTraining ? (
                    <div
                      data-testid="training-feedback-card"
                      className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-[#3478F6] flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5" />
                          You just gave the AI another example to learn from!
                        </span>
                        <span className="text-[11px] font-mono text-[#777777] tabular-nums">
                          {latestExploreTraining.epochs} epochs · ΔW{' '}
                          {latestExploreTraining.meanAbsWeightDelta.toFixed(4)}
                        </span>
                      </div>

                      {latestExploreTraining.beforePrediction &&
                        latestExploreTraining.afterPrediction && (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                            <div className="bg-white rounded-lg border border-[#E5E5E0] p-3">
                              <span className="text-[11px] text-[#777777] block">
                                Visitor Model Before Training
                              </span>
                              <span className="text-sm font-semibold font-mono text-[#202124] tabular-nums">
                                Digit {latestExploreTraining.targetDigit} Confidence:{' '}
                                {(
                                  latestExploreTraining.beforePrediction.targetDigitProb * 100
                                ).toFixed(1)}
                                %
                              </span>
                              <span className="text-[11px] text-[#777777] block mt-0.5 font-mono">
                                Loss: {latestExploreTraining.initialLoss.toFixed(3)}
                              </span>
                            </div>

                            <div className="bg-white rounded-lg border border-[#3478F6]/40 p-3">
                              <span className="text-[11px] text-[#3478F6] font-medium block">
                                Visitor Model After Training
                              </span>
                              <span className="text-sm font-semibold font-mono text-[#3478F6] tabular-nums">
                                Digit {latestExploreTraining.targetDigit} Confidence:{' '}
                                {(
                                  latestExploreTraining.afterPrediction.targetDigitProb * 100
                                ).toFixed(1)}
                                %
                              </span>
                              <span className="text-[11px] text-[#777777] block mt-0.5 font-mono">
                                Loss: {latestExploreTraining.finalLoss.toFixed(3)}
                              </span>
                            </div>
                          </div>
                        )}

                      <p className="text-[11px] text-[#777777] leading-relaxed">
                        Note: One example helps the Visitor-Trainable Model adapt to this specific drawing, but broad accuracy requires many examples across all digits. Switch Stage 2 to{' '}
                        <button
                          type="button"
                          onClick={() => setActiveModelMode('visitor')}
                          className="text-[#3478F6] font-medium underline cursor-pointer"
                        >
                          Visitor-Trained Model
                        </button>{' '}
                        to inspect your updated network live!
                      </p>
                    </div>
                  ) : (
                    <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4 text-xs text-[#777777] leading-relaxed">
                      <strong className="text-[#202124] block mb-1">
                        How Teaching Works:
                      </strong>
                      When you tap a number above and click Teach AI, your 28 × 28 drawing is added to the Visitor Dataset ({visitorDataset.length} examples currently) and the Visitor-Trainable Network runs real backpropagation to reduce its Cross-Entropy loss.
                      <div className="mt-2 flex items-center gap-4 text-[11px] font-mono text-[#202124] tabular-nums">
                        <span>Visitor Dataset: {visitorDataset.length} examples</span>
                        <span>·</span>
                        <span>
                          Visitor Model Test Accuracy: {(visitorTestAccuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Footer Links to Deeper Sections */}
                <div className="mt-4 pt-3 border-t border-[#E5E5E0] flex flex-wrap items-center justify-between gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => setActiveNav('progress')}
                    className="text-[#3478F6] font-medium hover:underline inline-flex items-center gap-1 cursor-pointer"
                  >
                    View Full Learning Progress & Loss Charts
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveNav('how-it-works')}
                    className="text-[#777777] hover:text-[#202124] font-medium cursor-pointer"
                  >
                    Learn How Backpropagation Works →
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeNav === 'how-it-works' && (
          <HowItWorksSection
            onGoToExplore={() => setActiveNav('explore')}
            onGoToTeach={() => setActiveNav('teach')}
          />
        )}

        {activeNav === 'teach' && (
          <TeachAiSection
            visitorDataset={visitorDataset}
            onAddExample={(px, lbl) => {
              handleAddVisitorExample(px, lbl);
            }}
            onRunTraining={handleRunTraining}
            lastSession={trainingSessions[0] ?? latestExploreTraining}
            currentTestAccuracy={visitorTestAccuracy}
            onResetVisitorModel={handleResetAll}
          />
        )}

        {activeNav === 'progress' && (
          <LearningProgressSection
            visitorModel={visitorModel}
            pretrainedModel={pretrainedModel}
            visitorDataset={visitorDataset}
            testDataset={testDataset}
            sessions={trainingSessions}
            onGoToTeach={() => setActiveNav('teach')}
          />
        )}

        {activeNav === 'about' && (
          <AboutSection
            tfInfo={tfInfo}
            onResetAll={handleResetAll}
            onGoToExplore={() => setActiveNav('explore')}
          />
        )}
      </main>

      {/* Exhibition Footer & Creator Watermark */}
      <footer className="relative z-10 border-t border-[#E5E5E0] bg-[#FAFAF8]/95 px-6 py-6 mt-12">
        <div className="max-w-[1380px] mx-auto flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4 text-xs text-[#777777]">
            <div>
              <strong className="text-[#202124] font-semibold">NEUROLAB</strong> · Interactive Machine Learning Exhibition Experience
            </div>
            <div className="flex items-center gap-4">
              <span>Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)</span>
              <button
                type="button"
                onClick={handleResetAll}
                className="text-[#202124] hover:text-[#3478F6] font-medium underline cursor-pointer"
              >
                Reset Session for Next Visitor
              </button>
            </div>
          </div>

          <div className="pt-4 border-t border-[#E5E5E0]/80 flex flex-col sm:flex-row items-center justify-between gap-3 pl-10 sm:pl-12">
            <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-[#777777]">
              <span>NEUROLAB Exhibition Build</span>
              <span>·</span>
              <a
                href="https://neurolab-three.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#3478F6] hover:underline font-medium"
              >
                https://neurolab-three.vercel.app/
              </a>
            </div>
            <div
              data-testid="creator-watermark"
              className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-lg bg-white border border-[#E5E5E0] text-xs font-semibold text-[#202124] shadow-[0_1px_2px_rgba(0,0,0,0.03)]"
            >
              <span className="w-2 h-2 rounded-full bg-[#3478F6]" />
              <span>Made by R.K Nived Krishnan &amp; Adithyan J</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
