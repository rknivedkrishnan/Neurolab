'use client';

import React, { useState } from 'react';
import {
  Brain,
  Sliders,
  ArrowRight,
  CheckCircle2,
  Sparkles,
  BookOpen,
  RefreshCw,
} from 'lucide-react';

interface ConceptModule {
  number: string;
  title: string;
  question: string;
  simpleSummary: string;
  analogyTitle: string;
  analogyText: string;
  deeperLook: string;
  visualType:
    | 'ai-vs-ml'
    | 'cat-analogy'
    | 'network-layers'
    | 'neuron-sandbox'
    | 'weights-biases'
    | 'forward-prop'
    | 'softmax-prediction'
    | 'loss-meter'
    | 'backprop-flow'
    | 'training-epochs';
}

const CONCEPT_MODULES: ConceptModule[] = [
  {
    number: '01',
    title: 'Artificial Intelligence (AI)',
    question: 'What is Artificial Intelligence?',
    simpleSummary:
      'Artificial Intelligence is the broad science of building computer programs that can perform tasks that usually require human intelligence — such as recognising handwriting, understanding speech, or playing chess.',
    analogyTitle: 'Everyday Analogy',
    analogyText:
      'Think of AI as the whole kitchen. Traditional software follows a strict recipe step-by-step ("if temperature is 180°C, bake for 20 minutes"). AI is a chef who can taste the soup and recognise what ingredient it needs.',
    deeperLook:
      'Early AI relied on thousands of hand-written rules created by programmers. However, writing rules for every possible way a human might scribble the number "7" is nearly impossible — which led scientists to Machine Learning.',
    visualType: 'ai-vs-ml',
  },
  {
    number: '02',
    title: 'Machine Learning (ML)',
    question: 'What is Machine Learning?',
    simpleSummary:
      'Machine Learning is a specific way of creating AI. Instead of a programmer writing rules by hand, the computer looks at hundreds of examples and discovers the patterns on its own.',
    analogyTitle: 'Teaching a Child to Recognise a Cat',
    analogyText:
      'Imagine teaching a child to recognise a cat. You do not give them a mathematical formula for ear angles or whisker lengths. Instead, you show them many pictures of cats and other animals. Eventually, they begin recognising patterns. Machine learning works in a similar way.',
    deeperLook:
      'In supervised machine learning, each training example is paired with a label (for example, a 28 × 28 drawing paired with the label "4"). The algorithm adjusts its internal numbers until its guesses match the labels.',
    visualType: 'cat-analogy',
  },
  {
    number: '03',
    title: 'Artificial Neural Networks',
    question: 'What is an artificial neural network?',
    simpleSummary:
      'An artificial neural network is a web of simple mathematical units arranged in layers. Information enters on the left, passes through hidden layers that spot patterns, and produces an answer on the right.',
    analogyTitle: 'A Team of Detectives Working in Stages',
    analogyText:
      'Imagine an assembly line of detectives. The first row only looks at tiny squares of ink. They pass notes to the second row, who spot curves and corners. The final row combines those clues to announce: "That is the number 8!"',
    deeperLook:
      'NEUROLAB uses a 4-layer network: 784 input nodes (one for each pixel in the 28 × 28 grid), 128 neurons in Hidden Layer 1, 64 neurons in Hidden Layer 2, and 10 output nodes for digits 0 through 9.',
    visualType: 'network-layers',
  },
  {
    number: '04',
    title: 'Artificial Neurons',
    question: 'What are artificial neurons?',
    simpleSummary:
      'An artificial neuron is a tiny calculator. It receives numbers from the previous layer, multiplies each number by a "weight", adds a "bias", and decides how strongly to fire.',
    analogyTitle: 'Deciding Whether to Go to the Park',
    analogyText:
      'Suppose you are deciding whether to go outside. You weigh different factors: Is it sunny? (+3), Are your friends going? (+2), Is it raining? (-4). If the total score is above zero, you go!',
    deeperLook:
      'In our hidden layers, each neuron uses the ReLU (Rectified Linear Unit) activation rule: if the weighted sum plus bias is negative, the neuron stays quiet (outputs 0). If positive, it passes the value straight through.',
    visualType: 'neuron-sandbox',
  },
  {
    number: '05',
    title: 'Weights and Biases',
    question: 'What are weights and biases?',
    simpleSummary:
      'Weights control how much importance a neuron gives to each incoming connection. Biases set the neuron’s baseline threshold for firing. Together, these are the "knobs" the AI tunes when it learns.',
    analogyTitle: 'Mixing Console Volume Sliders',
    analogyText:
      'Think of weights like the volume sliders on a music mixing board. Turning a weight up makes that input louder; making a weight negative turns that input into a blocker. The bias is the master threshold.',
    deeperLook:
      'Our 784 → 128 → 64 → 10 network contains 109,386 tunable weights and biases (100,480 in Layer 1, 8,256 in Layer 2, and 650 in the Output Layer). Every single memory the AI has is stored inside these numbers.',
    visualType: 'weights-biases',
  },
  {
    number: '06',
    title: 'Forward Propagation',
    question: 'How does forward propagation work?',
    simpleSummary:
      'Forward propagation is the step-by-step journey of your drawing flowing forward from the input pixels through each hidden layer until it reaches the 10 output digits.',
    analogyTitle: 'Ripples Travelling Across a Pond',
    analogyText:
      'When you drop a stone into water, ripples travel outward from ring to ring. When you draw a digit, the pixel brightness ripples forward through Layer 1, Layer 2, and finally the Output Layer.',
    deeperLook:
      'Mathematically, each layer performs matrix multiplication followed by a non-linear activation: h₁ = ReLU(x · W₁ + b₁), h₂ = ReLU(h₁ · W₂ + b₂), and z = h₂ · W₃ + b₃.',
    visualType: 'forward-prop',
  },
  {
    number: '07',
    title: 'Predictions & Confidence',
    question: 'What is a prediction?',
    simpleSummary:
      'Instead of outputting just one number, the AI assigns a confidence percentage to every digit from 0 to 9. All ten percentages add up to 100%, and the highest percentage is the AI’s prediction.',
    analogyTitle: 'A Weather Forecast',
    analogyText:
      'Just like a meteorologist says "85% chance of sun, 15% chance of clouds", the neural network might say "88% sure this is a 3, 9% sure it is an 8, and 3% sure it is a 5."',
    deeperLook:
      'The final layer uses the Softmax function, which exponentiates raw neuron scores and divides by their sum so that every output lies between 0.0 and 1.0 and the total equals 1.0 (100%).',
    visualType: 'softmax-prediction',
  },
  {
    number: '08',
    title: 'Measuring Mistakes (Loss)',
    question: 'What is loss?',
    simpleSummary:
      'Loss is a single number that measures how wrong the AI’s prediction was compared to the true answer. When the AI is confident and right, loss is near zero. When it is confident and wrong, loss is high.',
    analogyTitle: 'The "Hot or Cold" Guessing Game',
    analogyText:
      'Imagine searching for a hidden object while a friend tells you "freezing cold" (high loss) or "getting warmer... boiling hot!" (low loss). Loss gives the AI a clear signal of whether its adjustments are helping.',
    deeperLook:
      'NEUROLAB computes Categorical Cross-Entropy Loss: L = -ln(p_correct). If the network gives 95% probability to the true digit, loss is 0.05. If it gives only 5% probability to the true digit, loss jumps to 3.00.',
    visualType: 'loss-meter',
  },
  {
    number: '09',
    title: 'Backpropagation',
    question: 'How does backpropagation work?',
    simpleSummary:
      'Backpropagation is how the AI learns from a mistake. Starting from the error at the output layer, it works backward layer-by-layer to figure out which weights caused the mistake and nudges them in the right direction.',
    analogyTitle: 'Reviewing Game Tape After a Missed Basketball Shot',
    analogyText:
      'If a basketball player misses a shot to the left, their brain traces the motion backward: wrist angle, elbow height, knee bend — adjusting each muscle slightly so the next shot goes in.',
    deeperLook:
      'Using the chain rule of calculus, backpropagation computes the partial derivative of the loss with respect to every weight (∂L/∂W). Our Adam optimiser then nudges each weight opposite to its gradient.',
    visualType: 'backprop-flow',
  },
  {
    number: '10',
    title: 'Improving Through Training',
    question: 'How does an AI improve through training?',
    simpleSummary:
      'One example alone does not make an AI an expert. True learning happens when the network practices on many varied examples over multiple rounds (epochs), gradually lowering its loss and generalising to brand-new handwriting.',
    analogyTitle: 'Learning to Play a Musical Instrument',
    analogyText:
      'Playing a song once helps a little, but practicing across many sessions builds lasting muscle memory. Similarly, each training round refines the AI’s 109,386 weights.',
    deeperLook:
      'To test if an AI truly understands digits rather than just memorising one drawing, engineers evaluate it on a separate "Test Dataset" of drawings it has never trained on.',
    visualType: 'training-epochs',
  },
];

export default function HowItWorksSection({
  onGoToExplore,
  onGoToTeach,
}: {
  onGoToExplore: () => void;
  onGoToTeach: () => void;
}) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [showDeeper, setShowDeeper] = useState(false);

  // Interactive Neuron Sandbox state (used in Module 04 & 05)
  const [inputA, setInputA] = useState(0.8);
  const [weightA, setWeightA] = useState(1.2);
  const [inputB, setInputB] = useState(0.5);
  const [weightB, setWeightB] = useState(-0.6);
  const [bias, setBias] = useState(0.1);

  // Interactive Loss Sandbox state (used in Module 08)
  const [simulatedProb, setSimulatedProb] = useState(0.75);

  const weightedSum = inputA * weightA + inputB * weightB + bias;
  const reluOutput = Math.max(0, weightedSum);
  const simulatedLoss = -Math.log(Math.max(0.01, simulatedProb));

  const currentModule = CONCEPT_MODULES[activeIndex];

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="max-w-2xl">
          <p className="text-xs font-medium text-[#777777] mb-1">
            Interactive Science Guide · 10 Step-by-Step Concepts
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#202124] tracking-tight">
            How Does a Machine Actually Learn?
          </h1>
          <p className="text-sm sm:text-base text-[#777777] mt-2 leading-relaxed">
            No complicated jargon required. Explore each step below using everyday analogies and interactive visual sliders.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={onGoToExplore}
            className="px-4 py-2.5 rounded-xl border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#F2F2EF] text-xs font-medium text-[#202124] transition-colors whitespace-nowrap cursor-pointer"
          >
            Try Drawing a Number
          </button>
          <button
            type="button"
            onClick={onGoToTeach}
            className="px-4 py-2.5 rounded-xl bg-[#3478F6] hover:bg-[#2563EB] text-xs font-medium text-white transition-colors whitespace-nowrap cursor-pointer"
          >
            Teach the AI Live
          </button>
        </div>
      </div>

      {/* Main Two-Zone Educational Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Navigation Index (10 Topics) */}
        <div className="lg:col-span-4 bg-white rounded-2xl border border-[#E5E5E0] p-4 space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-[#777777]">
            10 Learning Chapters
          </div>
          {CONCEPT_MODULES.map((mod, idx) => {
            const isSelected = idx === activeIndex;
            return (
              <button
                key={mod.number}
                type="button"
                onClick={() => setActiveIndex(idx)}
                className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-medium flex items-center justify-between gap-2 transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-[#3478F6] text-white'
                    : 'text-[#202124] hover:bg-[#F2F2EF]'
                }`}
              >
                <span className="truncate">
                  <span
                    className={`font-mono mr-2 ${
                      isSelected ? 'text-white/80' : 'text-[#777777]'
                    }`}
                  >
                    {mod.number}.
                  </span>
                  {mod.title}
                </span>
                <ArrowRight
                  className={`w-3.5 h-3.5 shrink-0 ${
                    isSelected ? 'text-white' : 'text-[#777777]/50'
                  }`}
                />
              </button>
            );
          })}
        </div>

        {/* Right Interactive Concept Stage */}
        <div className="lg:col-span-8 bg-white rounded-2xl border border-[#E5E5E0] p-6 sm:p-8 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E5E5E0] pb-4">
            <div>
              <span className="text-xs font-mono text-[#3478F6] font-semibold">
                Chapter {currentModule.number} of 10
              </span>
              <h2 className="text-2xl font-semibold text-[#202124] mt-0.5">
                {currentModule.question}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={activeIndex === 0}
                onClick={() => setActiveIndex((i) => Math.max(0, i - 1))}
                className="px-3 py-1.5 rounded-lg border border-[#E5E5E0] text-xs font-medium text-[#202124] hover:bg-[#F2F2EF] disabled:opacity-40 cursor-pointer"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={activeIndex === CONCEPT_MODULES.length - 1}
                onClick={() =>
                  setActiveIndex((i) => Math.min(CONCEPT_MODULES.length - 1, i + 1))
                }
                className="px-3 py-1.5 rounded-lg bg-[#202124] text-xs font-medium text-white hover:bg-[#333438] disabled:opacity-40 cursor-pointer"
              >
                Next Concept
              </button>
            </div>
          </div>

          {/* Plain-English Explanation */}
          <div className="space-y-3">
            <p className="text-base text-[#202124] leading-relaxed">
              {currentModule.simpleSummary}
            </p>
          </div>

          {/* Everyday Analogy Box */}
          <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-5">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#3478F6] mb-1.5">
              <Sparkles className="w-4 h-4" />
              {currentModule.analogyTitle}
            </div>
            <p className="text-sm text-[#202124] leading-relaxed">
              “{currentModule.analogyText}”
            </p>
          </div>

          {/* Interactive Diagram / Sandbox tailored to the active topic */}
          <div className="bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-5">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold text-[#202124] flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#3478F6]" />
                Interactive Visual Demonstration
              </span>
              <span className="text-[11px] text-[#777777]">
                Try adjusting the controls below
              </span>
            </div>

            {/* Interactive Single Neuron & Weights/Biases Calculator */}
            {(currentModule.visualType === 'neuron-sandbox' ||
              currentModule.visualType === 'weights-biases') && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-white rounded-xl border border-[#E5E5E0] p-3.5">
                    <label className="block text-xs font-medium text-[#202124] mb-1">
                      Top Loop Pixel (Input 1):{' '}
                      <span className="font-mono tabular-nums text-[#3478F6]">
                        {inputA.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={inputA}
                      onChange={(e) => setInputA(parseFloat(e.target.value))}
                      className="w-full accent-[#3478F6]"
                    />
                    <label className="block text-xs font-medium text-[#202124] mt-3 mb-1">
                      Weight 1 (Importance):{' '}
                      <span className="font-mono tabular-nums text-[#5856D6]">
                        {weightA.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={weightA}
                      onChange={(e) => setWeightA(parseFloat(e.target.value))}
                      className="w-full accent-[#5856D6]"
                    />
                  </div>

                  <div className="bg-white rounded-xl border border-[#E5E5E0] p-3.5">
                    <label className="block text-xs font-medium text-[#202124] mb-1">
                      Bottom Stroke Pixel (Input 2):{' '}
                      <span className="font-mono tabular-nums text-[#3478F6]">
                        {inputB.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={inputB}
                      onChange={(e) => setInputB(parseFloat(e.target.value))}
                      className="w-full accent-[#3478F6]"
                    />
                    <label className="block text-xs font-medium text-[#202124] mt-3 mb-1">
                      Weight 2 (Importance):{' '}
                      <span className="font-mono tabular-nums text-[#5856D6]">
                        {weightB.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={weightB}
                      onChange={(e) => setWeightB(parseFloat(e.target.value))}
                      className="w-full accent-[#5856D6]"
                    />
                  </div>

                  <div className="bg-white rounded-xl border border-[#E5E5E0] p-3.5 flex flex-col justify-between">
                    <div>
                      <label className="block text-xs font-medium text-[#202124] mb-1">
                        Bias (Baseline Threshold):{' '}
                        <span className="font-mono tabular-nums text-[#202124]">
                          {bias.toFixed(2)}
                        </span>
                      </label>
                      <input
                        type="range"
                        min="-1"
                        max="1"
                        step="0.05"
                        value={bias}
                        onChange={(e) => setBias(parseFloat(e.target.value))}
                        className="w-full accent-[#202124]"
                      />
                    </div>
                    <div className="mt-3 pt-2 border-t border-[#E5E5E0]">
                      <div className="text-[11px] text-[#777777]">
                        Weighted Sum:{' '}
                        <span className="font-mono text-[#202124] tabular-nums">
                          {weightedSum.toFixed(2)}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-[#3478F6] mt-0.5 tabular-nums">
                        Neuron Output (ReLU): {reluOutput.toFixed(2)}{' '}
                        {reluOutput > 0 ? '● FIRING' : '○ QUIET'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Interactive Loss Meter (Module 08) */}
            {currentModule.visualType === 'loss-meter' && (
              <div className="space-y-4">
                <div className="bg-white rounded-xl border border-[#E5E5E0] p-4">
                  <label className="block text-xs font-semibold text-[#202124] mb-2">
                    Drag to change how confident the AI is in the CORRECT number:{' '}
                    <span className="font-mono text-[#3478F6] tabular-nums">
                      {Math.round(simulatedProb * 100)}%
                    </span>
                  </label>
                  <input
                    type="range"
                    min="0.02"
                    max="0.99"
                    step="0.01"
                    value={simulatedProb}
                    onChange={(e) => setSimulatedProb(parseFloat(e.target.value))}
                    className="w-full accent-[#3478F6]"
                  />
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-[#FAFAF8] border border-[#E5E5E0]">
                      <span className="text-xs text-[#777777]">AI Confidence in True Answer</span>
                      <p className="text-xl font-semibold text-[#202124] font-mono tabular-nums mt-0.5">
                        {(simulatedProb * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-[#FAFAF8] border border-[#E5E5E0]">
                      <span className="text-xs text-[#777777]">Resulting Cross-Entropy Loss</span>
                      <p className="text-xl font-semibold text-[#3478F6] font-mono tabular-nums mt-0.5">
                        {simulatedLoss.toFixed(2)}{' '}
                        <span className="text-xs font-normal text-[#777777]">
                          ({simulatedLoss < 0.3 ? 'Low Error — Great!' : simulatedLoss < 1.2 ? 'Moderate Error' : 'High Penalty!'})
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Visual Flow Diagram for remaining modules */}
            {currentModule.visualType !== 'neuron-sandbox' &&
              currentModule.visualType !== 'weights-biases' &&
              currentModule.visualType !== 'loss-meter' && (
                <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                  {[
                    {
                      step: 'Step 1',
                      label:
                        currentModule.visualType === 'backprop-flow'
                          ? '1. Spot Mistake at Output'
                          : '1. 28 × 28 Pixel Grid',
                      detail:
                        currentModule.visualType === 'backprop-flow'
                          ? 'Compare AI guess against your true label'
                          : '784 brightness values between 0 and 1',
                    },
                    {
                      step: 'Step 2',
                      label:
                        currentModule.visualType === 'backprop-flow'
                          ? '2. Trace Error to Layer 2'
                          : '2. Hidden Layer 1 (128)',
                      detail:
                        currentModule.visualType === 'backprop-flow'
                          ? 'Calculate which 64 neurons contributed'
                          : 'Detects simple edges and curves',
                    },
                    {
                      step: 'Step 3',
                      label:
                        currentModule.visualType === 'backprop-flow'
                          ? '3. Trace Error to Layer 1'
                          : '3. Hidden Layer 2 (64)',
                      detail:
                        currentModule.visualType === 'backprop-flow'
                          ? 'Measure impact of 128 stroke detectors'
                          : 'Combines strokes into loops & stems',
                    },
                    {
                      step: 'Step 4',
                      label:
                        currentModule.visualType === 'backprop-flow'
                          ? '4. Nudge 109,386 Weights'
                          : '4. 10 Output Probabilities',
                      detail:
                        currentModule.visualType === 'backprop-flow'
                          ? 'Adam optimiser updates weights slightly'
                          : 'Softmax percentages for digits 0–9',
                    },
                  ].map((card, i) => (
                    <div
                      key={i}
                      className="bg-white rounded-xl border border-[#E5E5E0] p-3.5 flex flex-col justify-between"
                    >
                      <div>
                        <span className="text-[11px] font-mono text-[#3478F6] font-semibold">
                          {card.step}
                        </span>
                        <h4 className="text-xs font-semibold text-[#202124] mt-1">
                          {card.label}
                        </h4>
                      </div>
                      <p className="text-[11px] text-[#777777] mt-2 leading-snug">
                        {card.detail}
                      </p>
                    </div>
                  ))}
                </div>
              )}
          </div>

          {/* Optional Deeper Technical Explanation Toggle */}
          <div className="pt-2">
            <button
              type="button"
              onClick={() => setShowDeeper((v) => !v)}
              className="inline-flex items-center gap-2 text-xs font-semibold text-[#3478F6] hover:underline cursor-pointer"
            >
              <BookOpen className="w-3.5 h-3.5" />
              {showDeeper
                ? 'Hide Technical Deep-Dive'
                : 'Curious how the math works? View Technical Deep-Dive'}
            </button>
            {showDeeper && (
              <div className="mt-3 p-4 rounded-xl bg-[#F2F2EF] border border-[#E5E5E0] text-xs text-[#202124] leading-relaxed">
                <strong className="block mb-1 font-semibold">
                  Under the Hood in NEUROLAB:
                </strong>
                {currentModule.deeperLook}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
