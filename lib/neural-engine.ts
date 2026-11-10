// Pure TypeScript + Float32Array 784 -> 128 -> 64 -> 10 Neural Network Engine
// Implements genuine Forward Propagation, ReLU, Softmax, Cross-Entropy Loss,
// Adam Optimizer Backpropagation, MNIST 20x20 Center-of-Mass Preprocessing,
// and Separate Training vs. Held-Out Test Dataset Evaluation.

export interface LabeledDigitExample {
  id: string;
  pixels: number[]; // 784 length (28x28 grayscale in [0, 1])
  label: number; // 0 - 9
  createdAt: number;
  source: 'visitor' | 'starter' | 'test';
  predictedBefore?: number;
}

export interface LayerActivations {
  input: number[]; // 784
  hidden1: number[]; // 128 (post-ReLU)
  hidden1Raw: number[]; // 128 (pre-ReLU)
  hidden2: number[]; // 64 (post-ReLU)
  hidden2Raw: number[]; // 64 (pre-ReLU)
  logits: number[]; // 10
  probabilities: number[]; // 10 (Softmax)
  predictedDigit: number;
  confidence: number;
}

export interface VisualNode {
  layerIndex: 0 | 1 | 2 | 3;
  nodeIndex: number; // Index in the visual subset
  realIndex: number; // Actual neuron index in the 784/128/64/10 layer
  activation: number; // Normalized [0, 1] for display
  rawActivation: number; // Actual numeric activation
  label: string;
}

export interface VisualSynapse {
  fromLayer: 0 | 1 | 2;
  toLayer: 1 | 2 | 3;
  fromVisualIndex: number;
  toVisualIndex: number;
  fromRealIndex: number;
  toRealIndex: number;
  weight: number; // Actual model weight
  signalStrength: number; // |activation * weight|
  isPositive: boolean;
}

export interface NetworkVisualizationSnapshot {
  nodesByLayer: [VisualNode[], VisualNode[], VisualNode[], VisualNode[]];
  synapses: VisualSynapse[];
  maxActivationH1: number;
  maxActivationH2: number;
  activeH1Count: number;
  activeH2Count: number;
}

export interface NeuralNetworkState {
  name: string;
  W1: Float32Array; // 784 * 128
  b1: Float32Array; // 128
  W2: Float32Array; // 128 * 64
  b2: Float32Array; // 64
  W3: Float32Array; // 64 * 10
  b3: Float32Array; // 10
  // Adam optimizer moments
  mW1: Float32Array;
  vW1: Float32Array;
  mb1: Float32Array;
  vb1: Float32Array;
  mW2: Float32Array;
  vW2: Float32Array;
  mb2: Float32Array;
  vb2: Float32Array;
  mW3: Float32Array;
  vW3: Float32Array;
  mb3: Float32Array;
  vb3: Float32Array;
  adamStep: number;
  totalTrainingSteps: number;
}

export interface TrainingSessionResult {
  sessionId: string;
  timestamp: number;
  epochs: number;
  examplesCount: number;
  initialLoss: number;
  finalLoss: number;
  lossHistory: number[];
  testAccuracyBefore: number;
  testAccuracyAfter: number;
  meanAbsWeightDelta: number;
  maxWeightDelta: number;
  layerDeltas: {
    layer1: number;
    layer2: number;
    layer3: number;
  };
  beforePrediction?: {
    digit: number;
    confidence: number;
    probabilities: number[];
    targetDigitProb: number;
  };
  afterPrediction?: {
    digit: number;
    confidence: number;
    probabilities: number[];
    targetDigitProb: number;
  };
  targetDigit?: number;
}

export const INPUT_SIZE = 784;
export const H1_SIZE = 128;
export const H2_SIZE = 64;
export const OUTPUT_SIZE = 10;

// Deterministic PRNG (Mulberry32) so initial weights and datasets are reproducible
export function createSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function createNeuralNetwork(name: string, seed = 42): NeuralNetworkState {
  const rng = createSeededRng(seed);

  // He initialization for ReLU layers: std = sqrt(2 / fan_in)
  const std1 = Math.sqrt(2.0 / INPUT_SIZE);
  const std2 = Math.sqrt(2.0 / H1_SIZE);
  const std3 = Math.sqrt(1.0 / H2_SIZE);

  const W1 = new Float32Array(INPUT_SIZE * H1_SIZE);
  const b1 = new Float32Array(H1_SIZE);
  const W2 = new Float32Array(H1_SIZE * H2_SIZE);
  const b2 = new Float32Array(H2_SIZE);
  const W3 = new Float32Array(H2_SIZE * OUTPUT_SIZE);
  const b3 = new Float32Array(OUTPUT_SIZE);

  for (let i = 0; i < W1.length; i++) W1[i] = gaussianRandom(rng) * std1;
  for (let i = 0; i < W2.length; i++) W2[i] = gaussianRandom(rng) * std2;
  for (let i = 0; i < W3.length; i++) W3[i] = gaussianRandom(rng) * std3;

  return {
    name,
    W1,
    b1,
    W2,
    b2,
    W3,
    b3,
    mW1: new Float32Array(INPUT_SIZE * H1_SIZE),
    vW1: new Float32Array(INPUT_SIZE * H1_SIZE),
    mb1: new Float32Array(H1_SIZE),
    vb1: new Float32Array(H1_SIZE),
    mW2: new Float32Array(H1_SIZE * H2_SIZE),
    vW2: new Float32Array(H1_SIZE * H2_SIZE),
    mb2: new Float32Array(H2_SIZE),
    vb2: new Float32Array(H2_SIZE),
    mW3: new Float32Array(H2_SIZE * OUTPUT_SIZE),
    vW3: new Float32Array(H2_SIZE * OUTPUT_SIZE),
    mb3: new Float32Array(OUTPUT_SIZE),
    vb3: new Float32Array(OUTPUT_SIZE),
    adamStep: 0,
    totalTrainingSteps: 0,
  };
}

export function cloneNeuralNetwork(net: NeuralNetworkState): NeuralNetworkState {
  return {
    name: net.name,
    W1: new Float32Array(net.W1),
    b1: new Float32Array(net.b1),
    W2: new Float32Array(net.W2),
    b2: new Float32Array(net.b2),
    W3: new Float32Array(net.W3),
    b3: new Float32Array(net.b3),
    mW1: new Float32Array(net.mW1),
    vW1: new Float32Array(net.vW1),
    mb1: new Float32Array(net.mb1),
    vb1: new Float32Array(net.vb1),
    mW2: new Float32Array(net.mW2),
    vW2: new Float32Array(net.vW2),
    mb2: new Float32Array(net.mb2),
    vb2: new Float32Array(net.vb2),
    mW3: new Float32Array(net.mW3),
    vW3: new Float32Array(net.vW3),
    mb3: new Float32Array(net.mb3),
    vb3: new Float32Array(net.vb3),
    adamStep: net.adamStep,
    totalTrainingSteps: net.totalTrainingSteps,
  };
}

/**
 * Genuine Forward Propagation: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
 */
export function forwardPass(
  input: number[] | Float32Array,
  net: NeuralNetworkState
): LayerActivations {
  const hidden1Raw = new Array<number>(H1_SIZE);
  const hidden1 = new Array<number>(H1_SIZE);

  for (let j = 0; j < H1_SIZE; j++) {
    let sum = net.b1[j];
    for (let i = 0; i < INPUT_SIZE; i++) {
      const x = input[i];
      if (x !== 0) {
        sum += x * net.W1[i * H1_SIZE + j];
      }
    }
    hidden1Raw[j] = sum;
    hidden1[j] = sum > 0 ? sum : 0; // ReLU
  }

  const hidden2Raw = new Array<number>(H2_SIZE);
  const hidden2 = new Array<number>(H2_SIZE);

  for (let k = 0; k < H2_SIZE; k++) {
    let sum = net.b2[k];
    for (let j = 0; j < H1_SIZE; j++) {
      const a1 = hidden1[j];
      if (a1 !== 0) {
        sum += a1 * net.W2[j * H2_SIZE + k];
      }
    }
    hidden2Raw[k] = sum;
    hidden2[k] = sum > 0 ? sum : 0; // ReLU
  }

  const logits = new Array<number>(OUTPUT_SIZE);
  let maxLogit = -Infinity;
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    let sum = net.b3[o];
    for (let k = 0; k < H2_SIZE; k++) {
      const a2 = hidden2[k];
      if (a2 !== 0) {
        sum += a2 * net.W3[k * OUTPUT_SIZE + o];
      }
    }
    logits[o] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }

  // Numerically stable Softmax
  const probabilities = new Array<number>(OUTPUT_SIZE);
  let expSum = 0;
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    const e = Math.exp(logits[o] - maxLogit);
    probabilities[o] = e;
    expSum += e;
  }

  let predictedDigit = 0;
  let confidence = 0;
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    probabilities[o] = probabilities[o] / (expSum || 1);
    if (probabilities[o] > confidence) {
      confidence = probabilities[o];
      predictedDigit = o;
    }
  }

  return {
    input: Array.from(input),
    hidden1,
    hidden1Raw,
    hidden2,
    hidden2Raw,
    logits,
    probabilities,
    predictedDigit,
    confidence,
  };
}

/**
 * Categorical Cross-Entropy Loss: -ln(p_target)
 */
export function computeCrossEntropyLoss(
  probabilities: number[] | Float32Array,
  targetDigit: number
): number {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probabilities[targetDigit] ?? 1e-7));
  return -Math.log(p);
}

/**
 * Evaluates average Cross-Entropy loss and classification accuracy on a dataset
 */
export function evaluateDataset(
  net: NeuralNetworkState,
  dataset: LabeledDigitExample[]
): { loss: number; accuracy: number; perDigitCorrect: number[]; perDigitTotal: number[] } {
  const perDigitCorrect = new Array(10).fill(0);
  const perDigitTotal = new Array(10).fill(0);

  if (dataset.length === 0) {
    return { loss: 0, accuracy: 0, perDigitCorrect, perDigitTotal };
  }

  let totalLoss = 0;
  let correct = 0;

  for (const ex of dataset) {
    const res = forwardPass(ex.pixels, net);
    totalLoss += computeCrossEntropyLoss(res.probabilities, ex.label);
    perDigitTotal[ex.label]++;
    if (res.predictedDigit === ex.label) {
      correct++;
      perDigitCorrect[ex.label]++;
    }
  }

  return {
    loss: totalLoss / dataset.length,
    accuracy: correct / dataset.length,
    perDigitCorrect,
    perDigitTotal,
  };
}

/**
 * Genuine Backpropagation + Adam Optimizer Update
 */
export function trainNetworkOnBatch(
  net: NeuralNetworkState,
  examples: LabeledDigitExample[],
  options: {
    epochs: number;
    learningRate?: number;
    testDataset?: LabeledDigitExample[];
    focusExample?: LabeledDigitExample;
  }
): TrainingSessionResult {
  const epochs = Math.max(1, options.epochs);
  const lr = options.learningRate ?? 0.0045;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;

  // Snapshot initial weights to compute genuine weight delta statistics
  const prevW1 = new Float32Array(net.W1);
  const prevW2 = new Float32Array(net.W2);
  const prevW3 = new Float32Array(net.W3);

  const testAccuracyBefore = options.testDataset
    ? evaluateDataset(net, options.testDataset).accuracy
    : 0;

  let beforePrediction: TrainingSessionResult['beforePrediction'] = undefined;
  if (options.focusExample) {
    const fp = forwardPass(options.focusExample.pixels, net);
    beforePrediction = {
      digit: fp.predictedDigit,
      confidence: fp.confidence,
      probabilities: [...fp.probabilities],
      targetDigitProb: fp.probabilities[options.focusExample.label],
    };
  }

  const lossHistory: number[] = [];
  const N = examples.length;

  if (N === 0) {
    return {
      sessionId: `session-${Date.now()}`,
      timestamp: Date.now(),
      epochs: 0,
      examplesCount: 0,
      initialLoss: 0,
      finalLoss: 0,
      lossHistory: [],
      testAccuracyBefore,
      testAccuracyAfter: testAccuracyBefore,
      meanAbsWeightDelta: 0,
      maxWeightDelta: 0,
      layerDeltas: { layer1: 0, layer2: 0, layer3: 0 },
    };
  }

  // Allocate gradient buffers
  const dW1 = new Float32Array(INPUT_SIZE * H1_SIZE);
  const db1 = new Float32Array(H1_SIZE);
  const dW2 = new Float32Array(H1_SIZE * H2_SIZE);
  const db2 = new Float32Array(H2_SIZE);
  const dW3 = new Float32Array(H2_SIZE * OUTPUT_SIZE);
  const db3 = new Float32Array(OUTPUT_SIZE);

  const delta3 = new Float32Array(OUTPUT_SIZE);
  const delta2 = new Float32Array(H2_SIZE);
  const delta1 = new Float32Array(H1_SIZE);

  for (let epoch = 0; epoch < epochs; epoch++) {
    dW1.fill(0);
    db1.fill(0);
    dW2.fill(0);
    db2.fill(0);
    dW3.fill(0);
    db3.fill(0);

    let epochLoss = 0;

    for (let idx = 0; idx < N; idx++) {
      const ex = examples[idx];
      const fwd = forwardPass(ex.pixels, net);
      epochLoss += computeCrossEntropyLoss(fwd.probabilities, ex.label);

      // Output layer gradient: dL/dz3 = probabilities - one_hot(label)
      for (let o = 0; o < OUTPUT_SIZE; o++) {
        delta3[o] = fwd.probabilities[o] - (o === ex.label ? 1.0 : 0.0);
        db3[o] += delta3[o];
      }

      // Accumulate dW3 and compute delta2
      for (let k = 0; k < H2_SIZE; k++) {
        const a2 = fwd.hidden2[k];
        let gradSum = 0;
        const rowOffset = k * OUTPUT_SIZE;
        for (let o = 0; o < OUTPUT_SIZE; o++) {
          const d3 = delta3[o];
          if (a2 !== 0) {
            dW3[rowOffset + o] += a2 * d3;
          }
          gradSum += d3 * net.W3[rowOffset + o];
        }
        // ReLU derivative on hidden2Raw
        delta2[k] = fwd.hidden2Raw[k] > 0 ? gradSum : 0;
        db2[k] += delta2[k];
      }

      // Accumulate dW2 and compute delta1
      for (let j = 0; j < H1_SIZE; j++) {
        const a1 = fwd.hidden1[j];
        let gradSum = 0;
        const rowOffset = j * H2_SIZE;
        for (let k = 0; k < H2_SIZE; k++) {
          const d2 = delta2[k];
          if (a1 !== 0) {
            dW2[rowOffset + k] += a1 * d2;
          }
          gradSum += d2 * net.W2[rowOffset + k];
        }
        // ReLU derivative on hidden1Raw
        delta1[j] = fwd.hidden1Raw[j] > 0 ? gradSum : 0;
        db1[j] += delta1[j];
      }

      // Accumulate dW1
      for (let i = 0; i < INPUT_SIZE; i++) {
        const x = ex.pixels[i];
        if (x !== 0) {
          const rowOffset = i * H1_SIZE;
          for (let j = 0; j < H1_SIZE; j++) {
            const d1 = delta1[j];
            if (d1 !== 0) {
              dW1[rowOffset + j] += x * d1;
            }
          }
        }
      }
    }

    lossHistory.push(epochLoss / N);

    // Apply Adam update
    net.adamStep += 1;
    net.totalTrainingSteps += 1;
    const t = net.adamStep;
    const biasCorr1 = 1 - Math.pow(beta1, t);
    const biasCorr2 = 1 - Math.pow(beta2, t);
    const scale = 1.0 / N;

    const updateParam = (
      param: Float32Array,
      grad: Float32Array,
      m: Float32Array,
      v: Float32Array
    ) => {
      for (let i = 0; i < param.length; i++) {
        const g = grad[i] * scale;
        const mi = beta1 * m[i] + (1 - beta1) * g;
        const vi = beta2 * v[i] + (1 - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        const mHat = mi / biasCorr1;
        const vHat = vi / biasCorr2;
        param[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    updateParam(net.W1, dW1, net.mW1, net.vW1);
    updateParam(net.b1, db1, net.mb1, net.vb1);
    updateParam(net.W2, dW2, net.mW2, net.vW2);
    updateParam(net.b2, db2, net.mb2, net.vb2);
    updateParam(net.W3, dW3, net.mW3, net.vW3);
    updateParam(net.b3, db3, net.mb3, net.vb3);
  }

  // Evaluate final loss after last weight update
  let finalLossSum = 0;
  for (const ex of examples) {
    const fwd = forwardPass(ex.pixels, net);
    finalLossSum += computeCrossEntropyLoss(fwd.probabilities, ex.label);
  }
  const finalLoss = finalLossSum / N;
  const initialLoss = lossHistory[0] ?? finalLoss;
  lossHistory.push(finalLoss);

  // Compute real weight deltas
  let sumDelta1 = 0;
  let sumDelta2 = 0;
  let sumDelta3 = 0;
  let maxWeightDelta = 0;

  for (let i = 0; i < net.W1.length; i++) {
    const d = Math.abs(net.W1[i] - prevW1[i]);
    sumDelta1 += d;
    if (d > maxWeightDelta) maxWeightDelta = d;
  }
  for (let i = 0; i < net.W2.length; i++) {
    const d = Math.abs(net.W2[i] - prevW2[i]);
    sumDelta2 += d;
    if (d > maxWeightDelta) maxWeightDelta = d;
  }
  for (let i = 0; i < net.W3.length; i++) {
    const d = Math.abs(net.W3[i] - prevW3[i]);
    sumDelta3 += d;
    if (d > maxWeightDelta) maxWeightDelta = d;
  }

  const totalWeights = net.W1.length + net.W2.length + net.W3.length;
  const meanAbsWeightDelta = (sumDelta1 + sumDelta2 + sumDelta3) / totalWeights;

  const testAccuracyAfter = options.testDataset
    ? evaluateDataset(net, options.testDataset).accuracy
    : 0;

  let afterPrediction: TrainingSessionResult['afterPrediction'] = undefined;
  if (options.focusExample) {
    const fp = forwardPass(options.focusExample.pixels, net);
    afterPrediction = {
      digit: fp.predictedDigit,
      confidence: fp.confidence,
      probabilities: [...fp.probabilities],
      targetDigitProb: fp.probabilities[options.focusExample.label],
    };
  }

  return {
    sessionId: `session-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    timestamp: Date.now(),
    epochs,
    examplesCount: N,
    initialLoss,
    finalLoss,
    lossHistory,
    testAccuracyBefore,
    testAccuracyAfter,
    meanAbsWeightDelta,
    maxWeightDelta,
    layerDeltas: {
      layer1: sumDelta1 / net.W1.length,
      layer2: sumDelta2 / net.W2.length,
      layer3: sumDelta3 / net.W3.length,
    },
    beforePrediction,
    afterPrediction,
    targetDigit: options.focusExample?.label,
  };
}

/**
 * Extracts a representative, authentic visual subset of neurons and connections
 * from actual model activations and weights for the interactive visualization.
 */
export function buildNetworkVisualizationSnapshot(
  activations: LayerActivations,
  net: NeuralNetworkState
): NetworkVisualizationSnapshot {
  // Visual subset sizes:
  // Input: 16 representative spatial patches / active pixels from 784
  // Hidden 1: 14 top/representative neurons from 128
  // Hidden 2: 10 top/representative neurons from 64
  // Output: All 10 digit neurons (0 - 9)
  const inputVisualCount = 16;
  const h1VisualCount = 14;
  const h2VisualCount = 10;

  // 1. Input layer: sample 16 regions across the center 20x20 area (or top active pixels)
  const inputNodes: VisualNode[] = [];
  for (let idx = 0; idx < inputVisualCount; idx++) {
    // Map 16 nodes to a 4x4 grid over the active 28x28 area
    const gridR = Math.floor(idx / 4);
    const gridC = idx % 4;
    const rStart = 4 + gridR * 5;
    const cStart = 4 + gridC * 5;
    let bestRealIndex = rStart * 28 + cStart;
    let maxVal = 0;
    for (let dr = 0; dr < 5; dr++) {
      for (let dc = 0; dc < 5; dc++) {
        const pIdx = (rStart + dr) * 28 + (cStart + dc);
        if (activations.input[pIdx] > maxVal) {
          maxVal = activations.input[pIdx];
          bestRealIndex = pIdx;
        }
      }
    }
    inputNodes.push({
      layerIndex: 0,
      nodeIndex: idx,
      realIndex: bestRealIndex,
      activation: maxVal,
      rawActivation: maxVal,
      label: `px #${bestRealIndex}`,
    });
  }

  // 2. Hidden Layer 1: pick 14 neurons spaced across the 128 neurons, biased toward strongest active neurons
  let maxH1 = 0.001;
  let activeH1Count = 0;
  for (let j = 0; j < H1_SIZE; j++) {
    if (activations.hidden1[j] > maxH1) maxH1 = activations.hidden1[j];
    if (activations.hidden1[j] > 0.01) activeH1Count++;
  }

  const h1Nodes: VisualNode[] = [];
  const stride1 = Math.floor(H1_SIZE / h1VisualCount);
  for (let idx = 0; idx < h1VisualCount; idx++) {
    const start = idx * stride1;
    const end = idx === h1VisualCount - 1 ? H1_SIZE : start + stride1;
    let bestJ = start;
    let bestVal = -1;
    for (let j = start; j < end; j++) {
      if (activations.hidden1[j] > bestVal) {
        bestVal = activations.hidden1[j];
        bestJ = j;
      }
    }
    h1Nodes.push({
      layerIndex: 1,
      nodeIndex: idx,
      realIndex: bestJ,
      activation: Math.min(1, Math.max(0, bestVal / maxH1)),
      rawActivation: bestVal,
      label: `H1 · #${bestJ}`,
    });
  }

  // 3. Hidden Layer 2: pick 10 neurons spaced across the 64 neurons
  let maxH2 = 0.001;
  let activeH2Count = 0;
  for (let k = 0; k < H2_SIZE; k++) {
    if (activations.hidden2[k] > maxH2) maxH2 = activations.hidden2[k];
    if (activations.hidden2[k] > 0.01) activeH2Count++;
  }

  const h2Nodes: VisualNode[] = [];
  const stride2 = Math.floor(H2_SIZE / h2VisualCount);
  for (let idx = 0; idx < h2VisualCount; idx++) {
    const start = idx * stride2;
    const end = idx === h2VisualCount - 1 ? H2_SIZE : start + stride2;
    let bestK = start;
    let bestVal = -1;
    for (let k = start; k < end; k++) {
      if (activations.hidden2[k] > bestVal) {
        bestVal = activations.hidden2[k];
        bestK = k;
      }
    }
    h2Nodes.push({
      layerIndex: 2,
      nodeIndex: idx,
      realIndex: bestK,
      activation: Math.min(1, Math.max(0, bestVal / maxH2)),
      rawActivation: bestVal,
      label: `H2 · #${bestK}`,
    });
  }

  // 4. Output Layer: all 10 digits (0-9)
  const outputNodes: VisualNode[] = [];
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    outputNodes.push({
      layerIndex: 3,
      nodeIndex: o,
      realIndex: o,
      activation: activations.probabilities[o],
      rawActivation: activations.logits[o],
      label: `${o}`,
    });
  }

  // 5. Actual Synapses between visual nodes using genuine weights from W1, W2, W3
  const synapses: VisualSynapse[] = [];

  // Input -> H1
  for (const inNode of inputNodes) {
    for (const h1Node of h1Nodes) {
      const w = net.W1[inNode.realIndex * H1_SIZE + h1Node.realIndex];
      const signal = Math.abs(inNode.activation * w);
      synapses.push({
        fromLayer: 0,
        toLayer: 1,
        fromVisualIndex: inNode.nodeIndex,
        toVisualIndex: h1Node.nodeIndex,
        fromRealIndex: inNode.realIndex,
        toRealIndex: h1Node.realIndex,
        weight: w,
        signalStrength: signal,
        isPositive: w >= 0,
      });
    }
  }

  // H1 -> H2
  for (const h1Node of h1Nodes) {
    for (const h2Node of h2Nodes) {
      const w = net.W2[h1Node.realIndex * H2_SIZE + h2Node.realIndex];
      const signal = Math.abs(h1Node.activation * w);
      synapses.push({
        fromLayer: 1,
        toLayer: 2,
        fromVisualIndex: h1Node.nodeIndex,
        toVisualIndex: h2Node.nodeIndex,
        fromRealIndex: h1Node.realIndex,
        toRealIndex: h2Node.realIndex,
        weight: w,
        signalStrength: signal,
        isPositive: w >= 0,
      });
    }
  }

  // H2 -> Output
  for (const h2Node of h2Nodes) {
    for (const outNode of outputNodes) {
      const w = net.W3[h2Node.realIndex * OUTPUT_SIZE + outNode.realIndex];
      const signal = Math.abs(h2Node.activation * w);
      synapses.push({
        fromLayer: 2,
        toLayer: 3,
        fromVisualIndex: h2Node.nodeIndex,
        toVisualIndex: outNode.nodeIndex,
        fromRealIndex: h2Node.realIndex,
        toRealIndex: outNode.realIndex,
        weight: w,
        signalStrength: signal,
        isPositive: w >= 0,
      });
    }
  }

  return {
    nodesByLayer: [inputNodes, h1Nodes, h2Nodes, outputNodes],
    synapses,
    maxActivationH1: maxH1,
    maxActivationH2: maxH2,
    activeH1Count,
    activeH2Count,
  };
}

/**
 * Normalizes any raw 2D grayscale grid (e.g., from a high-res canvas or 28x28 grid)
 * into a standard MNIST 28x28 image:
 * 1. Finds bounding box of non-zero pixels
 * 2. Scales longest dimension to 20 pixels while preserving aspect ratio
 * 3. Centers the digit by center-of-mass inside the 28x28 frame
 */
export function normalizeGridToMNIST28x28(
  rawGrid: number[] | Float32Array,
  width: number,
  height: number
): number[] {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const val = rawGrid[y * width + x];
      if (val > 0.05) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return new Array(784).fill(0);
  }

  const boxW = Math.max(1, maxX - minX + 1);
  const boxH = Math.max(1, maxY - minY + 1);
  const maxDim = Math.max(boxW, boxH);
  const targetSize = 20;
  const scale = targetSize / maxDim;

  const scaledW = Math.max(1, Math.round(boxW * scale));
  const scaledH = Math.max(1, Math.round(boxH * scale));

  // Resample into a temporary 28x28 centered box first
  const temp = new Float32Array(784);
  const offsetX = Math.floor((28 - scaledW) / 2);
  const offsetY = Math.floor((28 - scaledH) / 2);

  for (let ty = 0; ty < scaledH; ty++) {
    for (let tx = 0; tx < scaledW; tx++) {
      // Bilinear / area sample from rawGrid
      const srcX = minX + (tx + 0.5) / scale - 0.5;
      const srcY = minY + (ty + 0.5) / scale - 0.5;

      const x0 = Math.max(minX, Math.min(maxX, Math.floor(srcX)));
      const y0 = Math.max(minY, Math.min(maxY, Math.floor(srcY)));
      const x1 = Math.max(minX, Math.min(maxX, x0 + 1));
      const y1 = Math.max(minY, Math.min(maxY, y0 + 1));

      const fx = Math.max(0, Math.min(1, srcX - x0));
      const fy = Math.max(0, Math.min(1, srcY - y0));

      const v00 = rawGrid[y0 * width + x0];
      const v10 = rawGrid[y0 * width + x1];
      const v01 = rawGrid[y1 * width + x0];
      const v11 = rawGrid[y1 * width + x1];

      const v0 = v00 * (1 - fx) + v10 * fx;
      const v1 = v01 * (1 - fx) + v11 * fx;
      const val = v0 * (1 - fy) + v1 * fy;

      const dstX = offsetX + tx;
      const dstY = offsetY + ty;
      if (dstX >= 0 && dstX < 28 && dstY >= 0 && dstY < 28) {
        temp[dstY * 28 + dstX] = Math.min(1, Math.max(0, val));
      }
    }
  }

  // Compute center of mass to align to (13.5, 13.5)
  let massSum = 0;
  let comX = 0;
  let comY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const m = temp[y * 28 + x];
      massSum += m;
      comX += x * m;
      comY += y * m;
    }
  }

  const out = new Array<number>(784).fill(0);
  if (massSum <= 1e-5) {
    return out;
  }

  const shiftX = Math.round(13.5 - comX / massSum);
  const shiftY = Math.round(13.5 - comY / massSum);

  // Clamp shift so we never clip ink pixels off the 28x28 frame
  const clampedShiftX = Math.max(-4, Math.min(4, shiftX));
  const clampedShiftY = Math.max(-4, Math.min(4, shiftY));

  let maxVal = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = x - clampedShiftX;
      const srcY = y - clampedShiftY;
      if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
        const v = temp[srcY * 28 + srcX];
        out[y * 28 + x] = v;
        if (v > maxVal) maxVal = v;
      }
    }
  }

  // Normalize peak contrast to 1.0
  if (maxVal > 0.1) {
    for (let i = 0; i < 784; i++) {
      out[i] = Math.min(1, Math.round((out[i] / maxVal) * 1000) / 1000);
    }
  }

  return out;
}

/**
 * Procedural Handwritten Digit Rasterizer (0 - 9)
 * Generates realistic, varied 28x28 grayscale handwritten digits with configurable
 * slant, stroke thickness, scale, and jitter so we can build authentic training & test sets.
 */
export interface DigitStyleVariation {
  thickness: number; // e.g., 1.7 to 2.6
  slant: number; // e.g., -0.18 to +0.18
  scaleX: number; // e.g., 0.88 to 1.12
  scaleY: number; // e.g., 0.88 to 1.12
  offsetX: number; // e.g., -1.5 to 1.5
  offsetY: number; // e.g., -1.5 to 1.5
  variant: number; // 0, 1, 2 (structural handwriting variants, e.g., crossed 7, open 4, looped 2)
}

function drawThickSegment(
  grid: Float32Array,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  radius: number
) {
  const steps = Math.max(12, Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2.5));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const cx = x0 + (x1 - x0) * t;
    const cy = y0 + (y1 - y0) * t;
    const rCeil = Math.ceil(radius + 1);
    for (let dy = -rCeil; dy <= rCeil; dy++) {
      for (let dx = -rCeil; dx <= rCeil; dx++) {
        const px = Math.round(cx + dx);
        const py = Math.round(cy + dy);
        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
          const dist = Math.hypot(px - cx, py - cy);
          if (dist <= radius + 0.65) {
            const intensity =
              dist <= radius - 0.35
                ? 1.0
                : Math.max(0, 1.0 - (dist - (radius - 0.35)) / 1.0);
            const idx = py * 28 + px;
            if (intensity > grid[idx]) {
              grid[idx] = intensity;
            }
          }
        }
      }
    }
  }
}

function drawPolyline(
  grid: Float32Array,
  points: Array<[number, number]>,
  style: DigitStyleVariation
) {
  const transform = ([nx, ny]: [number, number]): [number, number] => {
    // nx, ny in [-1, 1]
    const sx = nx * style.scaleX + ny * style.slant;
    const sy = ny * style.scaleY;
    const px = 14 + sx * 8.5 + style.offsetX;
    const py = 14 + sy * 8.5 + style.offsetY;
    return [px, py];
  };

  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = transform(points[i]);
    const [x1, y1] = transform(points[i + 1]);
    drawThickSegment(grid, x0, y0, x1, y1, style.thickness);
  }
}

export function synthesizeHandwrittenDigit(
  digit: number,
  style: Partial<DigitStyleVariation> = {}
): number[] {
  const fullStyle: DigitStyleVariation = {
    thickness: style.thickness ?? 1.75,
    slant: style.slant ?? 0,
    scaleX: style.scaleX ?? 1.0,
    scaleY: style.scaleY ?? 1.0,
    offsetX: style.offsetX ?? 0,
    offsetY: style.offsetY ?? 0,
    variant: style.variant ?? 0,
  };

  const grid = new Float32Array(784);

  switch (digit) {
    case 0: {
      const pts: Array<[number, number]> = [];
      const steps = 20;
      for (let i = 0; i <= steps; i++) {
        const angle = (i / steps) * Math.PI * 2;
        pts.push([0.68 * Math.cos(angle), 0.92 * Math.sin(angle)]);
      }
      drawPolyline(grid, pts, fullStyle);
      break;
    }
    case 1: {
      if (fullStyle.variant === 1) {
        // Serif 1 with base
        drawPolyline(
          grid,
          [
            [-0.32, -0.58],
            [0.05, -0.95],
            [0.05, 0.92],
          ],
          fullStyle
        );
        drawPolyline(
          grid,
          [
            [-0.38, 0.92],
            [0.45, 0.92],
          ],
          fullStyle
        );
      } else {
        // Clean vertical stroke 1
        drawPolyline(
          grid,
          [
            [-0.18, -0.68],
            [0.04, -0.95],
            [0.02, 0.95],
          ],
          fullStyle
        );
      }
      break;
    }
    case 2: {
      drawPolyline(
        grid,
        [
          [-0.62, -0.52],
          [-0.35, -0.9],
          [0.25, -0.92],
          [0.62, -0.62],
          [0.58, -0.18],
          [0.12, 0.28],
          [-0.65, 0.9],
          [0.7, 0.9],
        ],
        fullStyle
      );
      break;
    }
    case 3: {
      drawPolyline(
        grid,
        [
          [-0.58, -0.78],
          [0.05, -0.95],
          [0.58, -0.7],
          [0.55, -0.25],
          [-0.12, -0.04],
          [0.62, 0.22],
          [0.64, 0.72],
          [0.08, 0.95],
          [-0.62, 0.75],
        ],
        fullStyle
      );
      break;
    }
    case 4: {
      if (fullStyle.variant === 1) {
        // Closed top 4
        drawPolyline(
          grid,
          [
            [0.28, 0.95],
            [0.28, -0.92],
            [-0.68, 0.28],
            [0.68, 0.28],
          ],
          fullStyle
        );
      } else {
        // Open top 4
        drawPolyline(
          grid,
          [
            [-0.52, -0.88],
            [-0.65, 0.22],
            [0.68, 0.22],
          ],
          fullStyle
        );
        drawPolyline(
          grid,
          [
            [0.26, -0.85],
            [0.24, 0.95],
          ],
          fullStyle
        );
      }
      break;
    }
    case 5: {
      drawPolyline(
        grid,
        [
          [0.58, -0.9],
          [-0.55, -0.9],
          [-0.62, -0.08],
          [0.12, -0.15],
          [0.62, 0.15],
          [0.6, 0.68],
          [0.05, 0.94],
          [-0.62, 0.78],
        ],
        fullStyle
      );
      break;
    }
    case 6: {
      drawPolyline(
        grid,
        [
          [0.38, -0.92],
          [-0.25, -0.58],
          [-0.62, 0.08],
          [-0.58, 0.68],
          [-0.05, 0.95],
          [0.55, 0.72],
          [0.58, 0.22],
          [0.05, -0.05],
          [-0.58, 0.18],
        ],
        fullStyle
      );
      break;
    }
    case 7: {
      drawPolyline(
        grid,
        [
          [-0.68, -0.88],
          [0.68, -0.88],
          [-0.18, 0.95],
        ],
        fullStyle
      );
      if (fullStyle.variant === 1) {
        drawPolyline(
          grid,
          [
            [-0.35, 0.05],
            [0.42, 0.05],
          ],
          fullStyle
        );
      }
      break;
    }
    case 8: {
      drawPolyline(
        grid,
        [
          [0.0, -0.02],
          [-0.55, -0.35],
          [-0.52, -0.75],
          [0.0, -0.95],
          [0.52, -0.75],
          [0.55, -0.35],
          [0.0, -0.02],
          [-0.62, 0.35],
          [-0.58, 0.78],
          [0.0, 0.95],
          [0.58, 0.78],
          [0.62, 0.35],
          [0.0, -0.02],
        ],
        fullStyle
      );
      break;
    }
    case 9: {
      drawPolyline(
        grid,
        [
          [0.55, -0.15],
          [-0.05, 0.08],
          [-0.58, -0.18],
          [-0.58, -0.68],
          [-0.02, -0.94],
          [0.56, -0.68],
          [0.58, 0.12],
          [0.22, 0.95],
        ],
        fullStyle
      );
      break;
    }
  }

  return normalizeGridToMNIST28x28(grid, 28, 28);
}

/**
 * Generates a curated Training Dataset and a strictly separate Held-Out Test Dataset.
 */
export function buildInitialDatasets(): {
  pretrainedTrainingSet: LabeledDigitExample[];
  visitorStarterSet: LabeledDigitExample[];
  testDataset: LabeledDigitExample[];
} {
  const rng = createSeededRng(1001);
  const pretrainedTrainingSet: LabeledDigitExample[] = [];
  const visitorStarterSet: LabeledDigitExample[] = [];
  const testDataset: LabeledDigitExample[] = [];

  // 8 variations per digit (80 total) for the Pretrained Reference Model
  for (let d = 0; d <= 9; d++) {
    for (let v = 0; v < 8; v++) {
      const style: DigitStyleVariation = {
        thickness: 1.5 + (v % 3) * 0.3,
        slant: ((v % 5) - 2) * 0.07,
        scaleX: 0.92 + rng() * 0.16,
        scaleY: 0.92 + rng() * 0.16,
        offsetX: (rng() - 0.5) * 1.2,
        offsetY: (rng() - 0.5) * 1.2,
        variant: v % 2,
      };
      const pixels = synthesizeHandwrittenDigit(d, style);
      pretrainedTrainingSet.push({
        id: `pre-${d}-${v}`,
        pixels,
        label: d,
        createdAt: 1700000000000 + d * 100 + v,
        source: 'starter',
      });
    }
  }

  // 2 baseline starter examples per digit (20 total) for the Visitor-Trainable Model
  // so it starts with partial recognition (~65-75%) and visibly improves as visitors teach it
  for (let d = 0; d <= 9; d++) {
    for (let v = 0; v < 2; v++) {
      const pixels = synthesizeHandwrittenDigit(d, {
        thickness: 1.7 + v * 0.25,
        slant: v === 0 ? -0.05 : 0.06,
        scaleX: 1.0,
        scaleY: 1.0,
        offsetX: 0,
        offsetY: 0,
        variant: v,
      });
      visitorStarterSet.push({
        id: `starter-${d}-${v}`,
        pixels,
        label: d,
        createdAt: 1700000005000 + d * 10 + v,
        source: 'starter',
      });
    }
  }

  // 5 held-out test examples per digit (50 total) with unseen slant/scale combinations
  for (let d = 0; d <= 9; d++) {
    for (let t = 0; t < 5; t++) {
      const pixels = synthesizeHandwrittenDigit(d, {
        thickness: 1.55 + (t % 3) * 0.3,
        slant: (t - 2) * 0.07,
        scaleX: 0.92 + t * 0.04,
        scaleY: 0.94 + ((t + 1) % 3) * 0.04,
        offsetX: (t - 2) * 0.4,
        offsetY: ((t % 2) - 0.5) * 0.6,
        variant: t % 2,
      });
      testDataset.push({
        id: `test-${d}-${t}`,
        pixels,
        label: d,
        createdAt: 1700000010000 + d * 10 + t,
        source: 'test',
      });
    }
  }

  return { pretrainedTrainingSet, visitorStarterSet, testDataset };
}

/**
 * Initializes both the Pretrained Reference Model and the Visitor-Trainable Model.
 */
export function initializeNeurolabModels(): {
  pretrainedModel: NeuralNetworkState;
  visitorModel: NeuralNetworkState;
  visitorDataset: LabeledDigitExample[];
  testDataset: LabeledDigitExample[];
} {
  const { pretrainedTrainingSet, visitorStarterSet, testDataset } = buildInitialDatasets();

  // 1. Pretrained Reference Model: trained thoroughly on the 80-example multi-style dataset
  const pretrainedModel = createNeuralNetwork('Pretrained MNIST Reference Model', 2026);
  trainNetworkOnBatch(pretrainedModel, pretrainedTrainingSet, {
    epochs: 12,
    learningRate: 0.008,
  });

  // 2. Visitor-Trainable Model: lightly initialized with 1 epoch on the small starter set
  // so visitors can observe genuine mistakes (~60% baseline accuracy), add their own drawings, and train it live.
  const visitorModel = createNeuralNetwork('Visitor-Trainable Neural Network', 7777);
  trainNetworkOnBatch(visitorModel, visitorStarterSet, {
    epochs: 1,
    learningRate: 0.0022,
  });

  return {
    pretrainedModel,
    visitorModel,
    visitorDataset: visitorStarterSet,
    testDataset,
  };
}
