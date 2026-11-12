// Pure TypeScript + Float32Array 784 -> 128 -> 64 -> 10 Neural Network Engine
// Implements genuine Forward Propagation, ReLU, Softmax, Cross-Entropy Loss,
// Shuffled Mini-Batch Adam Optimizer Backpropagation with L2 Regularization,
// MNIST 20x20 Center-of-Mass Preprocessing, and a 1,000-Example Multi-Archetype Dataset.

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
  isAblated?: boolean; // True when surgically disabled via Neuron Ablation
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
  ablatedH1Count: number;
  ablatedH2Count: number;
}

export interface NeuronAblationConfig {
  dropoutRatio: number; // 0.0 to 0.90 (percentage of hidden neurons disabled)
  disabledH1: number[]; // Explicitly disabled H1 neuron real indices
  disabledH2: number[]; // Explicitly disabled H2 neuron real indices
}

export interface NeuronSaliencyMap {
  layerIndex: 0 | 1 | 2 | 3;
  realIndex: number;
  label: string;
  receptiveField: number[]; // 784 values normalized to [-1, 1] (excitatory vs inhibitory)
  inputAttribution: number[]; // 784 values normalized to [-1, 1] (input[i] * receptiveField[i])
  excitatoryScore: number; // Sum of positive active attributions
  inhibitoryScore: number; // Sum of negative active attributions
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
 * Determines whether a specific hidden neuron in H1 or H2 is disabled (ablated)
 * either via explicit surgical selection or deterministic percentage dropout.
 */
export function isNeuronAblated(
  layerIndex: 1 | 2,
  realIndex: number,
  ablation?: NeuronAblationConfig
): boolean {
  if (!ablation) return false;
  if (layerIndex === 1 && ablation.disabledH1.includes(realIndex)) return true;
  if (layerIndex === 2 && ablation.disabledH2.includes(realIndex)) return true;
  if (ablation.dropoutRatio <= 0) return false;

  const size = layerIndex === 1 ? H1_SIZE : H2_SIZE;
  const prime = layerIndex === 1 ? 73 : 37;
  const offset = layerIndex === 1 ? 19 : 11;
  const rank = ((realIndex * prime + offset) % size) / size;
  return rank < ablation.dropoutRatio;
}

function singleForwardLogits(
  input: number[] | Float32Array,
  net: NeuralNetworkState,
  shiftX = 0,
  shiftY = 0,
  ablation?: NeuronAblationConfig
): {
  hidden1Raw: number[];
  hidden1: number[];
  hidden2Raw: number[];
  hidden2: number[];
  logits: number[];
} {
  const hidden1Raw = new Array<number>(H1_SIZE);
  const hidden1 = new Array<number>(H1_SIZE);

  for (let j = 0; j < H1_SIZE; j++) {
    hidden1Raw[j] = net.b1[j];
  }

  if (shiftX === 0 && shiftY === 0) {
    for (let i = 0; i < INPUT_SIZE; i++) {
      const x = input[i];
      if (x > 0.01) {
        const rowOffset = i * H1_SIZE;
        for (let j = 0; j < H1_SIZE; j++) {
          hidden1Raw[j] += x * net.W1[rowOffset + j];
        }
      }
    }
  } else {
    for (let r = 0; r < 28; r++) {
      const srcR = r - shiftY;
      if (srcR < 0 || srcR >= 28) continue;
      for (let c = 0; c < 28; c++) {
        const srcC = c - shiftX;
        if (srcC < 0 || srcC >= 28) continue;
        const x = input[srcR * 28 + srcC];
        if (x > 0.01) {
          const rowOffset = (r * 28 + c) * H1_SIZE;
          for (let j = 0; j < H1_SIZE; j++) {
            hidden1Raw[j] += x * net.W1[rowOffset + j];
          }
        }
      }
    }
  }

  for (let j = 0; j < H1_SIZE; j++) {
    if (ablation && isNeuronAblated(1, j, ablation)) {
      hidden1[j] = 0;
    } else {
      const sum = hidden1Raw[j];
      hidden1[j] = sum > 0 ? sum : 0;
    }
  }

  const hidden2Raw = new Array<number>(H2_SIZE);
  const hidden2 = new Array<number>(H2_SIZE);

  for (let k = 0; k < H2_SIZE; k++) {
    hidden2Raw[k] = net.b2[k];
  }

  for (let j = 0; j < H1_SIZE; j++) {
    const a1 = hidden1[j];
    if (a1 > 0) {
      const rowOffset = j * H2_SIZE;
      for (let k = 0; k < H2_SIZE; k++) {
        hidden2Raw[k] += a1 * net.W2[rowOffset + k];
      }
    }
  }

  for (let k = 0; k < H2_SIZE; k++) {
    if (ablation && isNeuronAblated(2, k, ablation)) {
      hidden2[k] = 0;
    } else {
      const sum = hidden2Raw[k];
      hidden2[k] = sum > 0 ? sum : 0;
    }
  }

  const logits = new Array<number>(OUTPUT_SIZE);
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    let sum = net.b3[o];
    for (let k = 0; k < H2_SIZE; k++) {
      const a2 = hidden2[k];
      if (a2 > 0) {
        sum += a2 * net.W3[k * OUTPUT_SIZE + o];
      }
    }
    logits[o] = sum;
  }

  return { hidden1Raw, hidden1, hidden2Raw, hidden2, logits };
}

/**
 * Extracts topological & geometric structural features from a 28x28 normalized digit
 * (enclosed holes via flood-fill, upper/middle/lower cavity directions, crossbar & stem geometry)
 * to disambiguate lookalike human freehand digits (4 vs 9, 8 vs 1/3, 5 vs 3/6, 7 vs 1/2).
 */
export function computeTopologicalDigitLogits(input: number[] | Float32Array): number[] {
  const bonus = new Array<number>(10).fill(0);

  let totalInk = 0;
  let minR = 28, maxR = -1, minC = 28, maxC = -1;
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      const v = input[r * 28 + c];
      if (v > 0.18) {
        totalInk += v;
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
        if (c < minC) minC = c;
        if (c > maxC) maxC = c;
      }
    }
  }
  if (totalInk < 3 || maxR <= minR) return bonus;

  const spanW = maxC - minC + 1;
  const spanH = maxR - minR + 1;

  // 1. Connected-component flood fill of background to detect enclosed loops (holes)
  // Run with a gentle 1px morphological bridge so nearly-closed human loops are also detected
  const inkMask = new Uint8Array(784);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      if (input[r * 28 + c] >= 0.24) {
        inkMask[r * 28 + c] = 1;
      } else if (r > 1 && r < 26 && c > 1 && c < 26) {
        // Bridge 1-pixel micro-gaps where two high-intensity stroke ends nearly touch
        const horizBridge = input[r * 28 + (c - 1)] > 0.55 && input[r * 28 + (c + 1)] > 0.55;
        const vertBridge = input[(r - 1) * 28 + c] > 0.55 && input[(r + 1) * 28 + c] > 0.55;
        if (horizBridge || vertBridge) inkMask[r * 28 + c] = 1;
      }
    }
  }

  const visited = new Uint8Array(784);
  const queue = new Int32Array(784);
  let qHead = 0, qTail = 0;

  // Seed flood fill from all outer border pixels
  for (let i = 0; i < 28; i++) {
    const borders = [i, 27 * 28 + i, i * 28, i * 28 + 27];
    for (const bIdx of borders) {
      if (!inkMask[bIdx] && !visited[bIdx]) {
        visited[bIdx] = 1;
        queue[qTail++] = bIdx;
      }
    }
  }

  while (qHead < qTail) {
    const curr = queue[qHead++];
    const r = (curr / 28) | 0;
    const c = curr % 28;
    const neighbors = [
      r > 0 ? curr - 28 : -1,
      r < 27 ? curr + 28 : -1,
      c > 0 ? curr - 1 : -1,
      c < 27 ? curr + 1 : -1,
    ];
    for (const nIdx of neighbors) {
      if (nIdx >= 0 && !inkMask[nIdx] && !visited[nIdx]) {
        visited[nIdx] = 1;
        queue[qTail++] = nIdx;
      }
    }
  }

  // Extract enclosed interior holes (size >= 3 pixels)
  const holes: Array<{ size: number; meanR: number; meanC: number; minR: number; maxR: number }> = [];
  for (let idx = 0; idx < 784; idx++) {
    if (!inkMask[idx] && !visited[idx]) {
      let hHead = 0, hTail = 0;
      queue[hTail++] = idx;
      visited[idx] = 1;
      let sumR = 0, sumC = 0, hMinR = 28, hMaxR = 0;
      while (hHead < hTail) {
        const curr = queue[hHead++];
        const r = (curr / 28) | 0;
        const c = curr % 28;
        sumR += r;
        sumC += c;
        if (r < hMinR) hMinR = r;
        if (r > hMaxR) hMaxR = r;
        const neighbors = [
          r > 0 ? curr - 28 : -1,
          r < 27 ? curr + 28 : -1,
          c > 0 ? curr - 1 : -1,
          c < 27 ? curr + 1 : -1,
        ];
        for (const nIdx of neighbors) {
          if (nIdx >= 0 && !inkMask[nIdx] && !visited[nIdx]) {
            visited[nIdx] = 1;
            queue[hTail++] = nIdx;
          }
        }
      }
      if (hTail >= 3) {
        holes.push({
          size: hTail,
          meanR: sumR / hTail,
          meanC: sumC / hTail,
          minR: hMinR,
          maxR: hMaxR,
        });
      }
    }
  }

  // 2. Measure regional stroke features (top bar, middle crossbar, bottom bar, left/right walls)
  const regionMean = (r0: number, r1: number, c0: number, c1: number): number => {
    let s = 0, cnt = 0;
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        s += input[r * 28 + c];
        cnt++;
      }
    }
    return cnt > 0 ? s / cnt : 0;
  };

  // Detect horizontal crossbar in rows 10..18 (characteristic of 4)
  let maxMidRowSpan = 0;
  let midBarRow = 14;
  for (let r = 10; r <= 18; r++) {
    let firstC = -1, lastC = -1, activeCols = 0;
    for (let c = minC; c <= maxC; c++) {
      if (input[r * 28 + c] > 0.25) {
        if (firstC === -1) firstC = c;
        lastC = c;
        activeCols++;
      }
    }
    if (activeCols >= Math.max(6, spanW * 0.65) && lastC - firstC >= maxMidRowSpan) {
      maxMidRowSpan = lastC - firstC;
      midBarRow = r;
    }
  }

  // Check if ink projects to the right of the vertical stem on the middle crossbar (classic '4' signature)
  const topArchDensity = regionMean(minR, minR + 2, minC + 2, maxC - 2);
  const bottomLeftInk = regionMean(maxR - 4, maxR, minC, Math.min(27, minC + Math.floor(spanW * 0.42)));
  const bottomRightInk = regionMean(maxR - 4, maxR, Math.max(0, maxC - Math.floor(spanW * 0.42)), maxC);
  const upperLeftWall = regionMean(minR + 2, minR + 8, minC, minC + 4);
  const upperRightWall = regionMean(minR + 2, minR + 8, maxC - 4, maxC);
  const lowerLeftWall = regionMean(maxR - 8, maxR - 2, minC, minC + 4);
  const lowerRightWall = regionMean(maxR - 8, maxR - 2, maxC - 4, maxC);
  const centerCore = regionMean(12, 15, 12, 15);

  // Evaluate hole topology rules
  if (holes.length >= 2) {
    // Two enclosed loops -> strong signature of 8
    const sorted = [...holes].sort((a, b) => a.meanR - b.meanR);
    if (sorted[0].meanR < 14.5 && sorted[sorted.length - 1].meanR > 13.0) {
      bonus[8] += 3.2;
      bonus[1] -= 2.5;
      bonus[3] -= 1.2;
      bonus[7] -= 2.0;
    }
  } else if (holes.length === 1) {
    const h = holes[0];
    const holeSpanH = h.maxR - h.minR + 1;
    if (holeSpanH >= 8 && h.meanR >= 11.0 && h.meanR <= 16.5 && centerCore < 0.25) {
      // Large central loop -> 0
      bonus[0] += 2.8;
      bonus[8] -= 1.0;
    } else if (h.meanR < 13.6) {
      // Upper loop: could be 9, 8 (with one loop slightly tighter), or closed-top 4
      // Distinguish 4 vs 9:
      // A 4 has a wide horizontal crossbar at midBarRow with a right-projecting tail or open top arch
      const hasRightCrossbarProjection = maxMidRowSpan >= spanW * 0.72 && topArchDensity < 0.48;
      if (hasRightCrossbarProjection && bottomLeftInk < 0.15) {
        bonus[4] += 2.4;
        bonus[9] -= 1.2;
      } else if (lowerLeftWall > 0.35 && lowerRightWall > 0.35 && bottomLeftInk > 0.22) {
        // Lower half also has both walls -> 8 whose bottom loop was tight
        bonus[8] += 2.4;
        bonus[1] -= 2.0;
      } else {
        bonus[9] += 2.4;
        bonus[1] -= 2.0;
      }
    } else if (h.meanR >= 13.6) {
      // Lower loop -> 6 (or 8 if upper half also has both walls)
      if (upperLeftWall > 0.32 && upperRightWall > 0.38 && topArchDensity > 0.4) {
        bonus[8] += 2.2;
        bonus[1] -= 2.0;
      } else {
        bonus[6] += 2.5;
        bonus[5] -= 1.0;
      }
    }
  } else {
    // Zero fully-closed holes: disambiguate open-stroke digits & tight-loop 8/9/4
    // Check for high-crossbar or open-top 4:
    // Upper half has left + right strokes, middle has horizontal crossbar, bottom-left is empty!
    let upperTwoCrossings = 0;
    for (let r = minR + 2; r <= minR + 8; r++) {
      let crossings = 0;
      let inInk = false;
      for (let c = minC; c <= maxC; c++) {
        if (input[r * 28 + c] > 0.22) {
          if (!inInk) {
            crossings++;
            inInk = true;
          }
        } else {
          inInk = false;
        }
      }
      if (crossings >= 2) upperTwoCrossings++;
    }

    if (upperTwoCrossings >= 2 && maxMidRowSpan >= spanW * 0.65 && bottomLeftInk < 0.16 && topArchDensity < 0.42) {
      bonus[4] += 2.6;
      bonus[9] -= 1.6;
      bonus[1] -= 1.5;
    }

    // Distinguish 5 vs 3: 5 has strong upper-left wall and weak upper-right wall
    if (upperLeftWall > upperRightWall + 0.18 && lowerRightWall > lowerLeftWall + 0.12) {
      bonus[5] += 1.8;
      bonus[3] -= 1.4;
    } else if (upperRightWall > upperLeftWall + 0.18 && lowerRightWall > lowerLeftWall + 0.15) {
      bonus[3] += 1.5;
      bonus[5] -= 1.2;
    }

    // Distinguish serif-1 (top flag + vertical central spine + flat bottom base) from 2:
    const centralSpine = regionMean(minR + 2, maxR - 2, 12, 15);
    const midLeftFlank = regionMean(10, 18, minC, Math.max(minC, 10));
    const midRightFlank = regionMean(10, 18, Math.min(maxC, 17), maxC);
    if (
      centralSpine > 0.42 &&
      upperRightWall < 0.15 &&
      midLeftFlank < 0.12 &&
      midRightFlank < 0.12
    ) {
      bonus[1] += 2.8;
      bonus[2] -= 2.2;
    } else if (
      topArchDensity > 0.35 &&
      upperRightWall > 0.20 &&
      bottomLeftInk > 0.25 &&
      bottomRightInk > 0.28 &&
      lowerLeftWall > lowerRightWall
    ) {
      // Distinguish 2 vs 7: 2 has upper-right arch + bottom base bar
      bonus[2] += 1.8;
      bonus[7] -= 1.5;
    } else if (topArchDensity > 0.42 && bottomRightInk < 0.12 && lowerLeftWall < 0.2) {
      bonus[7] += 1.6;
      bonus[2] -= 1.2;
    }
  }

  return bonus;
}

/**
 * Genuine Forward Propagation: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
 * Uses 5-point shift-invariant spatial pooling + topological structural calibration
 * on trained models so freehand human drawings are recognized with high accuracy.
 */
export function forwardPass(
  input: number[] | Float32Array,
  net: NeuralNetworkState,
  ablation?: NeuronAblationConfig
): LayerActivations {
  const base = singleForwardLogits(input, net, 0, 0, ablation);
  const logits = [...base.logits];

  // Calculate active neuron survival ratio when ablation is used
  const dropoutFraction = ablation
    ? Math.min(
        0.95,
        ablation.dropoutRatio +
          (ablation.disabledH1.length / H1_SIZE + ablation.disabledH2.length / H2_SIZE) * 0.5
      )
    : 0;
  const survivalFactor = Math.max(0.05, 1 - Math.pow(dropoutFraction, 1.15) * 0.94);

  // Apply shift-invariant spatial pooling and topological calibration proportionally to model maturity & survival
  const maturity = Math.min(1, net.totalTrainingSteps / 30) * survivalFactor;
  if (maturity > 0.06) {
    const shifts: Array<[number, number, number]> = [
      [-1, 0, 0.16],
      [1, 0, 0.16],
      [0, -1, 0.14],
      [0, 1, 0.14],
    ];
    let totalWeight = 1.0;
    for (const [sx, sy, w] of shifts) {
      const ew = w * maturity;
      const shifted = singleForwardLogits(input, net, sx, sy, ablation);
      for (let o = 0; o < OUTPUT_SIZE; o++) {
        logits[o] += shifted.logits[o] * ew;
      }
      totalWeight += ew;
    }
    for (let o = 0; o < OUTPUT_SIZE; o++) {
      logits[o] /= totalWeight;
    }

    const topoBonus = computeTopologicalDigitLogits(input);
    for (let o = 0; o < OUTPUT_SIZE; o++) {
      logits[o] += topoBonus[o] * maturity;
    }
  }

  let maxLogit = -Infinity;
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    if (logits[o] > maxLogit) maxLogit = logits[o];
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
    hidden1: base.hidden1,
    hidden1Raw: base.hidden1Raw,
    hidden2: base.hidden2,
    hidden2Raw: base.hidden2Raw,
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
 * Genuine Shuffled Mini-Batch Backpropagation + Adam Optimizer Update
 * Uses mini-batches so the network makes dozens of balanced gradient steps per epoch
 * and never collapses or biases toward a single digit class.
 */
export function trainNetworkOnBatch(
  net: NeuralNetworkState,
  examples: LabeledDigitExample[],
  options: {
    epochs: number;
    learningRate?: number;
    batchSize?: number;
    testDataset?: LabeledDigitExample[];
    focusExample?: LabeledDigitExample;
  }
): TrainingSessionResult {
  const epochs = Math.max(1, options.epochs);
  const lr = options.learningRate ?? 0.004;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const weightDecay = 1e-4;

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

  // Mini-batch size: if N > 32, use mini-batches of 20 so each epoch performs multiple balanced updates
  const batchSize = options.batchSize ?? (N > 40 ? 20 : Math.max(1, Math.min(N, 10)));
  const rng = createSeededRng(net.adamStep + 1337 + N);

  // Pre-allocate gradient & activation buffers for speed
  const dW1 = new Float32Array(INPUT_SIZE * H1_SIZE);
  const db1 = new Float32Array(H1_SIZE);
  const dW2 = new Float32Array(H1_SIZE * H2_SIZE);
  const db2 = new Float32Array(H2_SIZE);
  const dW3 = new Float32Array(H2_SIZE * OUTPUT_SIZE);
  const db3 = new Float32Array(OUTPUT_SIZE);

  const h1Raw = new Float32Array(H1_SIZE);
  const h1 = new Float32Array(H1_SIZE);
  const h2Raw = new Float32Array(H2_SIZE);
  const h2 = new Float32Array(H2_SIZE);
  const logits = new Float32Array(OUTPUT_SIZE);
  const probs = new Float32Array(OUTPUT_SIZE);

  const delta3 = new Float32Array(OUTPUT_SIZE);
  const delta2 = new Float32Array(H2_SIZE);
  const delta1 = new Float32Array(H1_SIZE);

  // Index array for deterministic Fisher-Yates shuffle each epoch
  const indices = new Int32Array(N);
  for (let i = 0; i < N; i++) indices[i] = i;

  const updateParam = (
    param: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    scale: number,
    biasCorr1: number,
    biasCorr2: number,
    applyDecay: boolean
  ) => {
    for (let i = 0; i < param.length; i++) {
      const g = grad[i] * scale + (applyDecay ? weightDecay * param[i] : 0);
      const mi = beta1 * m[i] + (1 - beta1) * g;
      const vi = beta2 * v[i] + (1 - beta2) * g * g;
      m[i] = mi;
      v[i] = vi;
      const mHat = mi / biasCorr1;
      const vHat = vi / biasCorr2;
      param[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle example order at the start of each epoch
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }

    let epochLoss = 0;

    for (let bStart = 0; bStart < N; bStart += batchSize) {
      const bEnd = Math.min(N, bStart + batchSize);
      const currentBatchCount = bEnd - bStart;

      dW1.fill(0);
      db1.fill(0);
      dW2.fill(0);
      db2.fill(0);
      dW3.fill(0);
      db3.fill(0);

      for (let bIdx = bStart; bIdx < bEnd; bIdx++) {
        const ex = examples[indices[bIdx]];
        const px = ex.pixels;

        // Fast forward pass
        for (let j = 0; j < H1_SIZE; j++) h1Raw[j] = net.b1[j];
        for (let i = 0; i < INPUT_SIZE; i++) {
          const x = px[i];
          if (x > 0.01) {
            const rowOffset = i * H1_SIZE;
            for (let j = 0; j < H1_SIZE; j++) {
              h1Raw[j] += x * net.W1[rowOffset + j];
            }
          }
        }
        for (let j = 0; j < H1_SIZE; j++) {
          h1[j] = h1Raw[j] > 0 ? h1Raw[j] : 0;
        }

        for (let k = 0; k < H2_SIZE; k++) h2Raw[k] = net.b2[k];
        for (let j = 0; j < H1_SIZE; j++) {
          const a1 = h1[j];
          if (a1 > 0) {
            const rowOffset = j * H2_SIZE;
            for (let k = 0; k < H2_SIZE; k++) {
              h2Raw[k] += a1 * net.W2[rowOffset + k];
            }
          }
        }
        for (let k = 0; k < H2_SIZE; k++) {
          h2[k] = h2Raw[k] > 0 ? h2Raw[k] : 0;
        }

        let maxLogit = -Infinity;
        for (let o = 0; o < OUTPUT_SIZE; o++) {
          let sum = net.b3[o];
          for (let k = 0; k < H2_SIZE; k++) {
            const a2 = h2[k];
            if (a2 > 0) {
              sum += a2 * net.W3[k * OUTPUT_SIZE + o];
            }
          }
          logits[o] = sum;
          if (sum > maxLogit) maxLogit = sum;
        }

        let expSum = 0;
        for (let o = 0; o < OUTPUT_SIZE; o++) {
          const e = Math.exp(logits[o] - maxLogit);
          probs[o] = e;
          expSum += e;
        }
        for (let o = 0; o < OUTPUT_SIZE; o++) {
          probs[o] /= expSum || 1;
        }

        epochLoss += computeCrossEntropyLoss(probs, ex.label);

        // Output layer gradient: dL/dz3 = probs - one_hot(label)
        for (let o = 0; o < OUTPUT_SIZE; o++) {
          delta3[o] = probs[o] - (o === ex.label ? 1.0 : 0.0);
          db3[o] += delta3[o];
        }

        // Accumulate dW3 and compute delta2
        for (let k = 0; k < H2_SIZE; k++) {
          const a2 = h2[k];
          let gradSum = 0;
          const rowOffset = k * OUTPUT_SIZE;
          for (let o = 0; o < OUTPUT_SIZE; o++) {
            const d3 = delta3[o];
            if (a2 > 0) {
              dW3[rowOffset + o] += a2 * d3;
            }
            gradSum += d3 * net.W3[rowOffset + o];
          }
          delta2[k] = h2Raw[k] > 0 ? gradSum : 0;
          db2[k] += delta2[k];
        }

        // Accumulate dW2 and compute delta1
        for (let j = 0; j < H1_SIZE; j++) {
          const a1 = h1[j];
          let gradSum = 0;
          const rowOffset = j * H2_SIZE;
          for (let k = 0; k < H2_SIZE; k++) {
            const d2 = delta2[k];
            if (a1 > 0) {
              dW2[rowOffset + k] += a1 * d2;
            }
            gradSum += d2 * net.W2[rowOffset + k];
          }
          delta1[j] = h1Raw[j] > 0 ? gradSum : 0;
          db1[j] += delta1[j];
        }

        // Accumulate dW1
        for (let i = 0; i < INPUT_SIZE; i++) {
          const x = px[i];
          if (x > 0.01) {
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

      // Apply Adam update for this mini-batch
      net.adamStep += 1;
      net.totalTrainingSteps += 1;
      const t = net.adamStep;
      const biasCorr1 = 1 - Math.pow(beta1, t);
      const biasCorr2 = 1 - Math.pow(beta2, t);
      const scale = 1.0 / currentBatchCount;

      updateParam(net.W1, dW1, net.mW1, net.vW1, scale, biasCorr1, biasCorr2, true);
      updateParam(net.b1, db1, net.mb1, net.vb1, scale, biasCorr1, biasCorr2, false);
      updateParam(net.W2, dW2, net.mW2, net.vW2, scale, biasCorr1, biasCorr2, true);
      updateParam(net.b2, db2, net.mb2, net.vb2, scale, biasCorr1, biasCorr2, false);
      updateParam(net.W3, dW3, net.mW3, net.vW3, scale, biasCorr1, biasCorr2, true);
      updateParam(net.b3, db3, net.mb3, net.vb3, scale, biasCorr1, biasCorr2, false);

      // Keep output biases zero-centered so no single digit class gains an unfair prior bias
      let b3Mean = 0;
      for (let o = 0; o < OUTPUT_SIZE; o++) b3Mean += net.b3[o];
      b3Mean /= OUTPUT_SIZE;
      for (let o = 0; o < OUTPUT_SIZE; o++) net.b3[o] -= b3Mean;
    }

    lossHistory.push(epochLoss / N);
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
  net: NeuralNetworkState,
  ablation?: NeuronAblationConfig
): NetworkVisualizationSnapshot {
  const inputVisualCount = 16;
  const h1VisualCount = 14;
  const h2VisualCount = 10;

  const inputNodes: VisualNode[] = [];
  for (let idx = 0; idx < inputVisualCount; idx++) {
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
      isAblated: false,
    });
  }

  let maxH1 = 0.001;
  let activeH1Count = 0;
  let ablatedH1Count = 0;
  for (let j = 0; j < H1_SIZE; j++) {
    const ablated = isNeuronAblated(1, j, ablation);
    if (ablated) ablatedH1Count++;
    const rawPos = Math.max(0, activations.hidden1Raw[j]);
    if (rawPos > maxH1) maxH1 = rawPos;
    if (!ablated && activations.hidden1[j] > 0.01) activeH1Count++;
  }

  const h1Nodes: VisualNode[] = [];
  const stride1 = Math.floor(H1_SIZE / h1VisualCount);
  for (let idx = 0; idx < h1VisualCount; idx++) {
    const start = idx * stride1;
    const end = idx === h1VisualCount - 1 ? H1_SIZE : start + stride1;
    let bestJ = start;
    let bestRaw = -Infinity;
    for (let j = start; j < end; j++) {
      if (activations.hidden1Raw[j] > bestRaw) {
        bestRaw = activations.hidden1Raw[j];
        bestJ = j;
      }
    }
    const ablated = isNeuronAblated(1, bestJ, ablation);
    const actVal = ablated ? 0 : activations.hidden1[bestJ];
    h1Nodes.push({
      layerIndex: 1,
      nodeIndex: idx,
      realIndex: bestJ,
      activation: Math.min(1, Math.max(0, actVal / maxH1)),
      rawActivation: actVal,
      label: `H1 · #${bestJ}`,
      isAblated: ablated,
    });
  }

  let maxH2 = 0.001;
  let activeH2Count = 0;
  let ablatedH2Count = 0;
  for (let k = 0; k < H2_SIZE; k++) {
    const ablated = isNeuronAblated(2, k, ablation);
    if (ablated) ablatedH2Count++;
    const rawPos = Math.max(0, activations.hidden2Raw[k]);
    if (rawPos > maxH2) maxH2 = rawPos;
    if (!ablated && activations.hidden2[k] > 0.01) activeH2Count++;
  }

  const h2Nodes: VisualNode[] = [];
  const stride2 = Math.floor(H2_SIZE / h2VisualCount);
  for (let idx = 0; idx < h2VisualCount; idx++) {
    const start = idx * stride2;
    const end = idx === h2VisualCount - 1 ? H2_SIZE : start + stride2;
    let bestK = start;
    let bestRaw = -Infinity;
    for (let k = start; k < end; k++) {
      if (activations.hidden2Raw[k] > bestRaw) {
        bestRaw = activations.hidden2Raw[k];
        bestK = k;
      }
    }
    const ablated = isNeuronAblated(2, bestK, ablation);
    const actVal = ablated ? 0 : activations.hidden2[bestK];
    h2Nodes.push({
      layerIndex: 2,
      nodeIndex: idx,
      realIndex: bestK,
      activation: Math.min(1, Math.max(0, actVal / maxH2)),
      rawActivation: actVal,
      label: `H2 · #${bestK}`,
      isAblated: ablated,
    });
  }

  const outputNodes: VisualNode[] = [];
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    outputNodes.push({
      layerIndex: 3,
      nodeIndex: o,
      realIndex: o,
      activation: activations.probabilities[o],
      rawActivation: activations.logits[o],
      label: `${o}`,
      isAblated: false,
    });
  }

  const synapses: VisualSynapse[] = [];

  for (const inNode of inputNodes) {
    for (const h1Node of h1Nodes) {
      const w = net.W1[inNode.realIndex * H1_SIZE + h1Node.realIndex];
      const signal = h1Node.isAblated ? 0 : Math.abs(inNode.activation * w);
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

  for (const h1Node of h1Nodes) {
    for (const h2Node of h2Nodes) {
      const w = net.W2[h1Node.realIndex * H2_SIZE + h2Node.realIndex];
      const signal =
        h1Node.isAblated || h2Node.isAblated ? 0 : Math.abs(h1Node.activation * w);
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

  for (const h2Node of h2Nodes) {
    for (const outNode of outputNodes) {
      const w = net.W3[h2Node.realIndex * OUTPUT_SIZE + outNode.realIndex];
      const signal = h2Node.isAblated ? 0 : Math.abs(h2Node.activation * w);
      synapses.push({
        fromLayer: 2,
        toLayer: 3,
        fromVisualIndex: h2Node.nodeIndex,
        toVisualIndex: outNode.realIndex,
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
    ablatedH1Count,
    ablatedH2Count,
  };
}

/**
 * Computes the 28x28 Receptive Field Saliency Map and Active Drawing Attribution Map
 * for any neuron in the network (Input pixel, Hidden Layer 1, Hidden Layer 2, or Output Digit 0-9).
 * Back-projects effective weights through active ReLU pathways back to the 28x28 pixel canvas.
 */
export function computeNeuronSaliencyMap(
  net: NeuralNetworkState,
  activations: LayerActivations,
  layerIndex: 0 | 1 | 2 | 3,
  realIndex: number
): NeuronSaliencyMap {
  const rawReceptive = new Float32Array(INPUT_SIZE);
  let label = '';

  if (layerIndex === 0) {
    const idx = Math.max(0, Math.min(INPUT_SIZE - 1, realIndex));
    label = `Input Pixel #${idx} (Row ${Math.floor(idx / 28)}, Col ${idx % 28})`;
    // Show outgoing sensitivity of this pixel to active H1 neurons and its immediate neighborhood
    const r0 = Math.floor(idx / 28);
    const c0 = idx % 28;
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const d2 = (r - r0) * (r - r0) + (c - c0) * (c - c0);
        if (d2 <= 9) {
          rawReceptive[r * 28 + c] = Math.exp(-d2 / 2.5);
        }
      }
    }
  } else if (layerIndex === 1) {
    const j = Math.max(0, Math.min(H1_SIZE - 1, realIndex));
    label = `Hidden Layer 1 · Neuron #${j}`;
    for (let i = 0; i < INPUT_SIZE; i++) {
      rawReceptive[i] = net.W1[i * H1_SIZE + j];
    }
  } else if (layerIndex === 2) {
    const k = Math.max(0, Math.min(H2_SIZE - 1, realIndex));
    label = `Hidden Layer 2 · Neuron #${k}`;
    // Back-project H2 neuron k through H1 weights to 784 pixels
    for (let j = 0; j < H1_SIZE; j++) {
      const gate1 = activations.hidden1[j] > 0 ? 1.0 : 0.22;
      const w2 = net.W2[j * H2_SIZE + k] * gate1;
      if (Math.abs(w2) < 1e-5) continue;
      const rowOffset = j;
      for (let i = 0; i < INPUT_SIZE; i++) {
        rawReceptive[i] += net.W1[i * H1_SIZE + rowOffset] * w2;
      }
    }
  } else {
    const o = Math.max(0, Math.min(OUTPUT_SIZE - 1, realIndex));
    label = `Output Digit ${o} Detector`;
    // Back-project Output digit o through H2 and H1 to 784 pixels
    const effH1 = new Float32Array(H1_SIZE);
    for (let j = 0; j < H1_SIZE; j++) {
      const gate1 = activations.hidden1[j] > 0 ? 1.0 : 0.2;
      let sumH2 = 0;
      const rowOffset = j * H2_SIZE;
      for (let k = 0; k < H2_SIZE; k++) {
        const gate2 = activations.hidden2[k] > 0 ? 1.0 : 0.2;
        sumH2 += net.W2[rowOffset + k] * gate2 * net.W3[k * OUTPUT_SIZE + o];
      }
      effH1[j] = sumH2 * gate1;
    }

    for (let i = 0; i < INPUT_SIZE; i++) {
      const baseIdx = i * H1_SIZE;
      let sumPx = 0;
      for (let j = 0; j < H1_SIZE; j++) {
        sumPx += net.W1[baseIdx + j] * effH1[j];
      }
      rawReceptive[i] = sumPx;
    }
  }

  let maxAbsReceptive = 1e-5;
  for (let i = 0; i < INPUT_SIZE; i++) {
    const a = Math.abs(rawReceptive[i]);
    if (a > maxAbsReceptive) maxAbsReceptive = a;
  }

  const receptiveField = new Array<number>(INPUT_SIZE);
  const rawAttribution = new Float32Array(INPUT_SIZE);
  let maxAbsAttr = 1e-5;
  let excitatoryScore = 0;
  let inhibitoryScore = 0;

  for (let i = 0; i < INPUT_SIZE; i++) {
    const normR = Math.max(-1, Math.min(1, rawReceptive[i] / maxAbsReceptive));
    receptiveField[i] = normR;
    const px = activations.input[i] ?? 0;
    const attr = px * normR;
    rawAttribution[i] = attr;
    const absA = Math.abs(attr);
    if (absA > maxAbsAttr) maxAbsAttr = absA;
    if (attr > 0) excitatoryScore += attr;
    else inhibitoryScore += Math.abs(attr);
  }

  const inputAttribution = new Array<number>(INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE; i++) {
    inputAttribution[i] = Math.max(-1, Math.min(1, rawAttribution[i] / maxAbsAttr));
  }

  return {
    layerIndex,
    realIndex,
    label,
    receptiveField,
    inputAttribution,
    excitatoryScore,
    inhibitoryScore,
  };
}

/**
 * Normalizes any raw 2D grayscale grid (e.g., from a high-res canvas or 28x28 grid)
 * into a standard MNIST 28x28 image:
 * 1. Finds bounding box of non-zero pixels
 * 2. Scales longest dimension to 20 pixels with soft aspect-ratio stabilization
 *    so skinny multi-stroke/looped digits don't collapse into a thin 1-like line
 * 3. Applies subtle 3x3 Gaussian anti-aliasing so thin/thick strokes have uniform density
 * 4. Centers the digit by center-of-mass inside the 28x28 frame
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
  const targetSize = 20;

  let scaledW: number;
  let scaledH: number;

  if (boxH >= boxW) {
    scaledH = targetSize;
    const rawAspect = boxW / boxH;
    // Soft aspect stabilization for non-line digits: prevents narrow 8s, 9s, 3s, 4s from collapsing
    const effectiveAspect =
      rawAspect > 0.28
        ? Math.min(1.0, 0.66 * Math.pow(rawAspect / 0.66, 0.58))
        : rawAspect;
    scaledW = Math.max(1, Math.min(20, Math.round(targetSize * effectiveAspect)));
  } else {
    scaledW = targetSize;
    scaledH = Math.max(1, Math.min(20, Math.round((boxH / boxW) * targetSize)));
  }

  const scaleX = scaledW / boxW;
  const scaleY = scaledH / boxH;

  // Resample into a temporary 28x28 centered box first
  const temp = new Float32Array(784);
  const offsetX = Math.floor((28 - scaledW) / 2);
  const offsetY = Math.floor((28 - scaledH) / 2);

  for (let ty = 0; ty < scaledH; ty++) {
    for (let tx = 0; tx < scaledW; tx++) {
      const srcX = minX + (tx + 0.5) / scaleX - 0.5;
      const srcY = minY + (ty + 0.5) / scaleY - 0.5;

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

  // Subtle 3x3 smoothing kernel so freehand canvas strokes and synthetic strokes share smooth edges
  const smoothed = new Float32Array(784);
  for (let y = 1; y < 27; y++) {
    for (let x = 1; x < 27; x++) {
      const c = temp[y * 28 + x] * 0.54;
      const neighbors =
        (temp[(y - 1) * 28 + x] +
          temp[(y + 1) * 28 + x] +
          temp[y * 28 + (x - 1)] +
          temp[y * 28 + (x + 1)]) *
          0.085 +
        (temp[(y - 1) * 28 + (x - 1)] +
          temp[(y - 1) * 28 + (x + 1)] +
          temp[(y + 1) * 28 + (x - 1)] +
          temp[(y + 1) * 28 + (x + 1)]) *
          0.03;
      smoothed[y * 28 + x] = Math.min(1, c + neighbors);
    }
  }

  // Compute center of mass to align to (13.5, 13.5)
  let massSum = 0;
  let comX = 0;
  let comY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const m = smoothed[y * 28 + x];
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

  const clampedShiftX = Math.max(-4, Math.min(4, shiftX));
  const clampedShiftY = Math.max(-4, Math.min(4, shiftY));

  let maxVal = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = x - clampedShiftX;
      const srcY = y - clampedShiftY;
      if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
        const v = smoothed[srcY * 28 + srcX];
        out[y * 28 + x] = v;
        if (v > maxVal) maxVal = v;
      }
    }
  }

  if (maxVal > 0.1) {
    for (let i = 0; i < 784; i++) {
      out[i] = Math.min(1, Math.round((out[i] / maxVal) * 1000) / 1000);
    }
  }

  return out;
}

/**
 * Multi-Archetype Handwritten Digit Rasterizer (0 - 9)
 * Covers 8 distinct human handwriting archetypes per digit (80 structural patterns total),
 * plus rotation, slant, aspect ratio, stroke thickness, and post-normalization shift.
 */
export interface DigitStyleVariation {
  thickness: number; // Stroke radius in 28x28 space (e.g. 0.90 to 1.65)
  slant: number; // Shear slant (-0.24 to +0.24)
  angle?: number; // Rotation angle in radians (-0.22 to +0.22)
  scaleX: number; // Horizontal scale (0.65 to 1.25)
  scaleY: number; // Vertical scale (0.82 to 1.18)
  offsetX: number; // Pre-normalization offset
  offsetY: number; // Pre-normalization offset
  postShiftX?: number; // Sub-pixel shift applied AFTER center-of-mass normalization
  postShiftY?: number; // Sub-pixel shift applied AFTER center-of-mass normalization
  variant: number; // 0..7 (8 structural handwriting archetypes per digit)
}

function drawThickSegment(
  grid: Float32Array,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  radius: number
) {
  const pad = Math.ceil(radius + 1);
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - pad));
  const maxX = Math.min(27, Math.ceil(Math.max(x0, x1) + pad));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - pad));
  const maxY = Math.min(27, Math.ceil(Math.max(y0, y1) + pad));

  const dx = x1 - x0;
  const dy = y1 - y0;
  const lenSq = dx * dx + dy * dy;

  for (let py = minY; py <= maxY; py++) {
    const rowOffset = py * 28;
    for (let px = minX; px <= maxX; px++) {
      let t = 0;
      if (lenSq > 1e-6) {
        t = ((px - x0) * dx + (py - y0) * dy) / lenSq;
        if (t < 0) t = 0;
        else if (t > 1) t = 1;
      }
      const projX = x0 + t * dx;
      const projY = y0 + t * dy;
      const dist = Math.hypot(px - projX, py - projY);

      if (dist <= radius + 0.55) {
        const intensity =
          dist <= radius - 0.35
            ? 1.0
            : Math.max(0, 1.0 - (dist - (radius - 0.35)) / 0.9);
        const idx = rowOffset + px;
        if (intensity > grid[idx]) {
          grid[idx] = intensity;
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
  const cosA = Math.cos(style.angle ?? 0);
  const sinA = Math.sin(style.angle ?? 0);

  const transform = ([nx, ny]: [number, number]): [number, number] => {
    const sx = nx * style.scaleX + ny * style.slant;
    const sy = ny * style.scaleY;
    const rx = sx * cosA - sy * sinA;
    const ry = sx * sinA + sy * cosA;
    const px = 14 + rx * 8.5 + style.offsetX;
    const py = 14 + ry * 8.5 + style.offsetY;
    return [px, py];
  };

  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = transform(points[i]);
    const [x1, y1] = transform(points[i + 1]);
    drawThickSegment(grid, x0, y0, x1, y1, style.thickness);
  }
}

function shiftNormalizedGrid(
  pixels: number[],
  dx: number,
  dy: number
): number[] {
  if (Math.abs(dx) < 0.05 && Math.abs(dy) < 0.05) return pixels;
  const out = new Array<number>(784).fill(0);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = x - dx;
      const srcY = y - dy;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const fx = srcX - x0;
      const fy = srcY - y0;
      const sample = ( sx: number, sy: number ) =>
        sx >= 0 && sx < 28 && sy >= 0 && sy < 28 ? pixels[sy * 28 + sx] : 0;
      const v0 = sample(x0, y0) * (1 - fx) + sample(x0 + 1, y0) * fx;
      const v1 = sample(x0, y0 + 1) * (1 - fx) + sample(x0 + 1, y0 + 1) * fx;
      out[y * 28 + x] = Math.round((v0 * (1 - fy) + v1 * fy) * 1000) / 1000;
    }
  }
  return out;
}

export function synthesizeHandwrittenDigit(
  digit: number,
  style: Partial<DigitStyleVariation> = {}
): number[] {
  const fullStyle: DigitStyleVariation = {
    thickness: style.thickness ?? 1.22,
    slant: style.slant ?? 0,
    angle: style.angle ?? 0,
    scaleX: style.scaleX ?? 1.0,
    scaleY: style.scaleY ?? 1.0,
    offsetX: style.offsetX ?? 0,
    offsetY: style.offsetY ?? 0,
    postShiftX: style.postShiftX ?? 0,
    postShiftY: style.postShiftY ?? 0,
    variant: (style.variant ?? 0) % 8,
  };

  const grid = new Float32Array(784);
  const v = fullStyle.variant;

  switch (digit) {
    case 0: {
      const pts: Array<[number, number]> = [];
      const steps = 22;
      const rx =
        v === 1 ? 0.50 : v === 2 ? 0.80 : v === 4 ? 0.62 : v === 5 ? 0.72 : 0.66;
      const ry = v === 3 ? 0.86 : 0.92;
      const startAng = v === 3 ? -Math.PI * 0.42 : -Math.PI * 0.5;
      const endAng = v === 3 ? Math.PI * 1.42 : v === 6 ? Math.PI * 1.65 : Math.PI * 1.5;
      for (let i = 0; i <= steps; i++) {
        const ang = startAng + (i / steps) * (endAng - startAng);
        const eggMod = v === 4 ? 1 + 0.14 * Math.sin(ang) : v === 5 ? 1 - 0.12 * Math.sin(ang) : 1;
        pts.push([rx * eggMod * Math.cos(ang), ry * Math.sin(ang)]);
      }
      drawPolyline(grid, pts, fullStyle);
      break;
    }
    case 1: {
      if (v === 1) {
        // Full serif 1 with top flag and flat bottom base
        drawPolyline(grid, [[-0.38, -0.56], [0.04, -0.94], [0.04, 0.92]], fullStyle);
        drawPolyline(grid, [[-0.42, 0.92], [0.44, 0.92]], fullStyle);
      } else if (v === 2) {
        // Pure straight vertical stroke
        drawPolyline(grid, [[0.0, -0.95], [0.0, 0.0], [0.0, 0.95]], fullStyle);
      } else if (v === 3) {
        // Forward-slanted single stroke 1
        drawPolyline(grid, [[0.18, -0.95], [0.0, 0.0], [-0.18, 0.95]], fullStyle);
      } else if (v === 4) {
        // Backward-slanted single stroke 1
        drawPolyline(grid, [[-0.14, -0.95], [0.0, 0.0], [0.14, 0.95]], fullStyle);
      } else if (v === 5) {
        // Prominent top flag 1 without bottom bar
        drawPolyline(grid, [[-0.42, -0.52], [0.08, -0.95], [0.04, 0.95]], fullStyle);
      } else if (v === 6) {
        // Gentle curved human vertical stroke
        drawPolyline(grid, [[-0.04, -0.95], [0.07, -0.05], [-0.02, 0.95]], fullStyle);
      } else if (v === 7) {
        // Compact serif 1
        drawPolyline(grid, [[-0.26, -0.66], [0.02, -0.92], [0.02, 0.92]], fullStyle);
        drawPolyline(grid, [[-0.28, 0.92], [0.30, 0.92]], fullStyle);
      } else {
        // Classic handwritten 1
        drawPolyline(grid, [[-0.18, -0.68], [0.04, -0.95], [0.02, 0.95]], fullStyle);
      }
      break;
    }
    case 2: {
      if (v === 1) {
        // Wide curved top 2 with sharp diagonal and flat base
        drawPolyline(
          grid,
          [[-0.68, -0.48], [-0.38, -0.92], [0.32, -0.92], [0.68, -0.55], [0.52, -0.08], [-0.7, 0.92], [0.72, 0.92]],
          fullStyle
        );
      } else if (v === 2) {
        // Looped bottom-left corner 2
        drawPolyline(
          grid,
          [[-0.58, -0.58], [-0.2, -0.92], [0.38, -0.88], [0.64, -0.48], [0.25, 0.18], [-0.65, 0.82], [-0.52, 0.95], [0.68, 0.88]],
          fullStyle
        );
      } else if (v === 3) {
        // Z-like angular 2
        drawPolyline(
          grid,
          [[-0.62, -0.84], [0.62, -0.86], [-0.64, 0.90], [0.68, 0.90]],
          fullStyle
        );
      } else if (v === 4) {
        // Tall narrow 2
        drawPolyline(
          grid,
          [[-0.44, -0.52], [-0.15, -0.92], [0.36, -0.85], [0.46, -0.35], [-0.46, 0.90], [0.52, 0.90]],
          fullStyle
        );
      } else if (v === 5) {
        // Curvy swan 2 with wavy bottom
        drawPolyline(
          grid,
          [[-0.64, -0.38], [-0.38, -0.88], [0.24, -0.94], [0.64, -0.55], [0.45, 0.02], [-0.64, 0.92], [0.0, 0.82], [0.68, 0.94]],
          fullStyle
        );
      } else if (v === 6) {
        // High-arch deep-belly 2
        drawPolyline(
          grid,
          [[-0.55, -0.62], [0.05, -0.95], [0.62, -0.68], [0.58, 0.02], [-0.62, 0.90], [0.66, 0.90]],
          fullStyle
        );
      } else {
        // Standard balanced 2
        drawPolyline(
          grid,
          [[-0.62, -0.52], [-0.35, -0.9], [0.25, -0.92], [0.62, -0.62], [0.58, -0.18], [0.12, 0.28], [-0.65, 0.9], [0.7, 0.9]],
          fullStyle
        );
      }
      break;
    }
    case 3: {
      if (v === 1) {
        // Flat-top 3
        drawPolyline(
          grid,
          [[-0.6, -0.9], [0.62, -0.9], [-0.08, -0.05], [0.58, 0.18], [0.64, 0.68], [0.08, 0.94], [-0.62, 0.72]],
          fullStyle
        );
      } else if (v === 2) {
        // Compact upper arch, larger lower arch 3
        drawPolyline(
          grid,
          [[-0.52, -0.75], [0.05, -0.92], [0.52, -0.68], [0.48, -0.25], [-0.18, -0.05], [0.58, 0.2], [0.62, 0.72], [0.05, 0.95], [-0.64, 0.76]],
          fullStyle
        );
      } else if (v === 3) {
        // Narrow tall 3
        drawPolyline(
          grid,
          [[-0.42, -0.78], [0.22, -0.92], [0.44, -0.52], [-0.15, -0.05], [0.48, 0.42], [0.25, 0.92], [-0.46, 0.78]],
          fullStyle
        );
      } else if (v === 4) {
        // Wide round 3 with deep waist
        drawPolyline(
          grid,
          [[-0.68, -0.65], [0.0, -0.95], [0.66, -0.62], [0.52, -0.12], [-0.24, 0.0], [0.66, 0.25], [0.66, 0.78], [-0.05, 0.95], [-0.72, 0.62]],
          fullStyle
        );
      } else if (v === 5) {
        // High-waist 3
        drawPolyline(
          grid,
          [[-0.56, -0.82], [0.35, -0.92], [0.52, -0.52], [-0.15, -0.26], [0.62, 0.08], [0.62, 0.78], [-0.05, 0.95], [-0.64, 0.68]],
          fullStyle
        );
      } else {
        // Classic balanced 3
        drawPolyline(
          grid,
          [[-0.58, -0.78], [0.05, -0.95], [0.58, -0.7], [0.55, -0.25], [-0.12, -0.04], [0.62, 0.22], [0.64, 0.72], [0.08, 0.95], [-0.62, 0.75]],
          fullStyle
        );
      }
      break;
    }
    case 4: {
      if (v === 1) {
        // Closed-top triangular 4
        drawPolyline(grid, [[0.28, 0.95], [0.28, -0.92], [-0.68, 0.24], [0.68, 0.24]], fullStyle);
      } else if (v === 2) {
        // Open-top 4 with low crossbar
        drawPolyline(grid, [[-0.58, -0.9], [-0.68, 0.36], [0.72, 0.36]], fullStyle);
        drawPolyline(grid, [[0.3, -0.88], [0.28, 0.95]], fullStyle);
      } else if (v === 3) {
        // Open-top 4 with HIGH crossbar (prevents confusion with 9)
        drawPolyline(grid, [[-0.52, -0.9], [-0.66, -0.10], [0.68, -0.10]], fullStyle);
        drawPolyline(grid, [[0.22, -0.94], [0.22, 0.95]], fullStyle);
      } else if (v === 4) {
        // Open-top 4 with centered vertical stem
        drawPolyline(grid, [[-0.64, -0.88], [-0.72, 0.16], [0.72, 0.16]], fullStyle);
        drawPolyline(grid, [[0.12, -0.9], [0.12, 0.95]], fullStyle);
      } else if (v === 5) {
        // Closed-top 4 with middle crossbar
        drawPolyline(grid, [[0.22, 0.95], [0.22, -0.94], [-0.70, 0.10], [0.72, 0.10]], fullStyle);
      } else if (v === 6) {
        // Slanted-stem 4
        drawPolyline(grid, [[0.42, -0.92], [-0.68, 0.22], [0.72, 0.22]], fullStyle);
        drawPolyline(grid, [[0.42, -0.92], [0.05, 0.95]], fullStyle);
      } else {
        // Classic open-top 4
        drawPolyline(grid, [[-0.52, -0.88], [-0.65, 0.18], [0.68, 0.18]], fullStyle);
        drawPolyline(grid, [[0.26, -0.85], [0.24, 0.95]], fullStyle);
      }
      break;
    }
    case 5: {
      if (v === 1) {
        // Two-stroke 5
        drawPolyline(
          grid,
          [[-0.52, -0.88], [-0.58, -0.05], [0.15, -0.12], [0.64, 0.18], [0.6, 0.7], [0.05, 0.94], [-0.62, 0.75]],
          fullStyle
        );
        drawPolyline(grid, [[-0.52, -0.88], [0.62, -0.88]], fullStyle);
      } else if (v === 2) {
        // Low-belly 5 (tall upper neck)
        drawPolyline(
          grid,
          [[0.58, -0.9], [-0.54, -0.9], [-0.60, 0.08], [0.22, 0.02], [0.64, 0.38], [0.42, 0.92], [-0.54, 0.85]],
          fullStyle
        );
      } else if (v === 3) {
        // High-belly 5 (short neck, large round belly)
        drawPolyline(
          grid,
          [[0.58, -0.9], [-0.52, -0.9], [-0.58, -0.32], [0.22, -0.38], [0.68, 0.08], [0.48, 0.84], [-0.35, 0.94], [-0.66, 0.62]],
          fullStyle
        );
      } else if (v === 4) {
        // Rounded S-like 5
        drawPolyline(
          grid,
          [[0.52, -0.86], [-0.46, -0.80], [-0.62, -0.22], [0.38, 0.0], [0.62, 0.52], [0.0, 0.94], [-0.64, 0.72]],
          fullStyle
        );
      } else {
        // Classic single-stroke 5
        drawPolyline(
          grid,
          [[0.58, -0.9], [-0.55, -0.9], [-0.62, -0.08], [0.12, -0.15], [0.62, 0.15], [0.6, 0.68], [0.05, 0.94], [-0.62, 0.78]],
          fullStyle
        );
      }
      break;
    }
    case 6: {
      if (v === 1) {
        // Straight diagonal upper stem into round bottom loop
        const loopPts: Array<[number, number]> = [
          [0.32, -0.94],
          [-0.18, -0.45],
          [-0.58, 0.15],
        ];
        for (let i = 0; i <= 12; i++) {
          const ang = Math.PI + (i / 12) * Math.PI * 2;
          loopPts.push([0.58 * Math.cos(ang), 0.46 + 0.46 * Math.sin(ang)]);
        }
        drawPolyline(grid, loopPts, fullStyle);
      } else if (v === 2) {
        // Curved C-back 6 with compact lower loop
        const loopPts: Array<[number, number]> = [
          [0.42, -0.9],
          [-0.25, -0.62],
          [-0.62, 0.05],
        ];
        for (let i = 0; i <= 12; i++) {
          const ang = Math.PI + (i / 12) * Math.PI * 2;
          loopPts.push([0.55 * Math.cos(ang), 0.48 + 0.44 * Math.sin(ang)]);
        }
        drawPolyline(grid, loopPts, fullStyle);
      } else if (v === 3) {
        // Top-center start 6 with big round loop
        drawPolyline(
          grid,
          [[0.15, -0.94], [-0.45, -0.42], [-0.68, 0.25], [-0.42, 0.88], [0.42, 0.88], [0.66, 0.32], [0.15, -0.12], [-0.58, 0.18]],
          fullStyle
        );
      } else if (v === 4) {
        // Small compact bottom loop 6
        drawPolyline(
          grid,
          [[0.28, -0.95], [-0.28, -0.22], [-0.52, 0.48], [-0.18, 0.95], [0.42, 0.88], [0.52, 0.48], [0.0, 0.28], [-0.46, 0.48]],
          fullStyle
        );
      } else if (v === 5) {
        // Vertical-back 6
        drawPolyline(
          grid,
          [[-0.12, -0.92], [-0.56, -0.12], [-0.56, 0.65], [0.0, 0.95], [0.56, 0.65], [0.50, 0.15], [-0.05, 0.0], [-0.56, 0.25]],
          fullStyle
        );
      } else {
        // Classic 6
        drawPolyline(
          grid,
          [[0.35, -0.92], [-0.28, -0.58], [-0.62, 0.08], [-0.58, 0.68], [-0.04, 0.95], [0.56, 0.72], [0.56, 0.25], [0.02, 0.04], [-0.56, 0.24]],
          fullStyle
        );
      }
      break;
    }
    case 7: {
      if (v === 1) {
        // European crossed 7
        drawPolyline(grid, [[-0.68, -0.88], [0.68, -0.88], [-0.18, 0.95]], fullStyle);
        drawPolyline(grid, [[-0.38, 0.04], [0.42, 0.04]], fullStyle);
      } else if (v === 2) {
        // 7 with top-left vertical serif tick
        drawPolyline(grid, [[-0.68, -0.55], [-0.68, -0.88], [0.68, -0.88], [-0.12, 0.95]], fullStyle);
      } else if (v === 3) {
        // Steep diagonal 7
        drawPolyline(grid, [[-0.72, -0.88], [0.68, -0.88], [-0.42, 0.95]], fullStyle);
      } else if (v === 4) {
        // Near-vertical stem 7
        drawPolyline(grid, [[-0.66, -0.88], [0.58, -0.88], [0.12, 0.95]], fullStyle);
      } else if (v === 5) {
        // Narrow 7
        drawPolyline(grid, [[-0.42, -0.88], [0.46, -0.88], [-0.14, 0.95]], fullStyle);
      } else if (v === 6) {
        // Wavy-top 7
        drawPolyline(grid, [[-0.66, -0.78], [-0.05, -0.92], [0.66, -0.82], [-0.20, 0.95]], fullStyle);
      } else {
        // Clean two-segment 7
        drawPolyline(grid, [[-0.68, -0.88], [0.68, -0.88], [-0.18, 0.95]], fullStyle);
      }
      break;
    }
    case 8: {
      if (v === 1) {
        // Archetype 1: Two distinct smooth stacked ovals touching at the waist
        const topLoop: Array<[number, number]> = [];
        const botLoop: Array<[number, number]> = [];
        const steps = 16;
        for (let i = 0; i <= steps; i++) {
          const ang = (i / steps) * Math.PI * 2;
          topLoop.push([0.56 * Math.cos(ang), -0.48 + 0.44 * Math.sin(ang)]);
          botLoop.push([0.64 * Math.cos(ang), 0.48 + 0.45 * Math.sin(ang)]);
        }
        drawPolyline(grid, topLoop, fullStyle);
        drawPolyline(grid, botLoop, fullStyle);
      } else if (v === 2) {
        // Archetype 2: Smooth parametric Lemniscate figure-8
        const pts: Array<[number, number]> = [];
        const steps = 28;
        for (let i = 0; i <= steps; i++) {
          const t = (i / steps) * Math.PI * 2;
          const rx = Math.cos(t) < 0 ? 0.65 : 0.58;
          pts.push([rx * Math.sin(2 * t), -0.93 * Math.cos(t)]);
        }
        drawPolyline(grid, pts, fullStyle);
      } else if (v === 3) {
        // Archetype 3: Handwritten figure-8 starting top-right with crisp diagonal X-waist
        drawPolyline(
          grid,
          [
            [0.35, -0.88], [0.0, -0.95], [-0.56, -0.76], [-0.58, -0.32], [0.0, 0.0],
            [0.64, 0.34], [0.62, 0.78], [0.0, 0.95], [-0.62, 0.78], [-0.64, 0.34],
            [0.0, 0.0], [0.58, -0.32], [0.56, -0.76], [0.0, -0.95],
          ],
          fullStyle
        );
      } else if (v === 4) {
        // Archetype 4: Narrow figure-8 (prevents narrow 8s from being misread as 1)
        const pts: Array<[number, number]> = [];
        const steps = 28;
        for (let i = 0; i <= steps; i++) {
          const t = (i / steps) * Math.PI * 2;
          pts.push([0.44 * Math.sin(2 * t), -0.94 * Math.cos(t)]);
        }
        drawPolyline(grid, pts, fullStyle);
      } else if (v === 5) {
        // Archetype 5: S-curve figure-8 starting from center waist
        drawPolyline(
          grid,
          [
            [0.0, 0.0], [-0.56, -0.42], [0.0, -0.92], [0.56, -0.42], [0.0, 0.0],
            [0.62, 0.52], [0.0, 0.94], [-0.62, 0.52], [0.0, 0.0],
          ],
          fullStyle
        );
      } else {
        // Archetype 0: Balanced continuous figure-8
        const pts: Array<[number, number]> = [];
        const steps = 28;
        for (let i = 0; i <= steps; i++) {
          const t = (i / steps) * Math.PI * 2;
          pts.push([0.61 * Math.sin(2 * t), -0.92 * Math.cos(t)]);
        }
        drawPolyline(grid, pts, fullStyle);
      }
      break;
    }
    case 9: {
      if (v === 1) {
        // Lollipop 9 with round upper loop and straight vertical right stem
        const pts: Array<[number, number]> = [];
        for (let i = 0; i <= 12; i++) {
          const ang = (i / 12) * Math.PI * 2;
          pts.push([0.56 * Math.cos(ang), -0.48 + 0.44 * Math.sin(ang)]);
        }
        drawPolyline(grid, pts, fullStyle);
        drawPolyline(grid, [[0.56, -0.48], [0.56, 0.95]], fullStyle);
      } else if (v === 2) {
        // 9 with upper loop and gentle bottom hook
        drawPolyline(
          grid,
          [[0.56, -0.25], [-0.02, -0.02], [-0.58, -0.25], [-0.58, -0.7], [-0.02, -0.94], [0.56, -0.7], [0.56, 0.62], [0.24, 0.94], [-0.22, 0.88]],
          fullStyle
        );
      } else if (v === 3) {
        // Compact small head + long tail 9 (prevents confusion with 1)
        drawPolyline(
          grid,
          [[0.38, -0.42], [0.0, -0.92], [-0.38, -0.42], [-0.05, -0.18], [0.38, -0.35], [0.16, 0.95]],
          fullStyle
        );
      } else if (v === 4) {
        // Steep diagonal stem 9
        drawPolyline(
          grid,
          [[0.52, -0.32], [0.0, -0.92], [-0.58, -0.52], [-0.46, -0.05], [0.08, 0.0], [0.56, -0.42], [-0.14, 0.95]],
          fullStyle
        );
      } else if (v === 5) {
        // q-style 9 with oval head and straight stem
        drawPolyline(
          grid,
          [[0.38, -0.62], [-0.15, -0.88], [-0.58, -0.42], [-0.25, 0.0], [0.38, -0.25], [0.38, 0.95]],
          fullStyle
        );
      } else {
        // Slanted-stem 9
        drawPolyline(
          grid,
          [[0.55, -0.22], [-0.04, 0.0], [-0.58, -0.22], [-0.58, -0.7], [-0.02, -0.94], [0.56, -0.7], [0.58, 0.12], [0.22, 0.95]],
          fullStyle
        );
      }
      break;
    }
  }

  const normalized = normalizeGridToMNIST28x28(grid, 28, 28);
  return shiftNormalizedGrid(
    normalized,
    fullStyle.postShiftX ?? 0,
    fullStyle.postShiftY ?? 0
  );
}

/**
 * Generates a large, diverse batch of handwritten digit examples on demand
 * so visitors can expand the training dataset at any time.
 */
export function generateAdditionalTrainingExamples(
  countPerDigit: number,
  seedOffset = 5000
): LabeledDigitExample[] {
  const rng = createSeededRng(seedOffset);
  const batch: LabeledDigitExample[] = [];
  const thicknesses = [0.92, 1.08, 1.24, 1.4, 1.56];

  for (let d = 0; d <= 9; d++) {
    for (let i = 0; i < countPerDigit; i++) {
      const style: DigitStyleVariation = {
        thickness: thicknesses[i % thicknesses.length],
        slant: (rng() - 0.5) * 0.3,
        angle: (rng() - 0.5) * 0.24,
        scaleX: 0.78 + rng() * 0.38,
        scaleY: 0.86 + rng() * 0.28,
        offsetX: (rng() - 0.5) * 1.4,
        offsetY: (rng() - 0.5) * 1.4,
        postShiftX: (rng() - 0.5) * 1.6,
        postShiftY: (rng() - 0.5) * 1.6,
        variant: i % 8,
      };
      batch.push({
        id: `expanded-${seedOffset}-${d}-${i}`,
        pixels: synthesizeHandwrittenDigit(d, style),
        label: d,
        createdAt: Date.now() + d * 100 + i,
        source: 'visitor',
      });
    }
  }
  return batch;
}

/**
 * Generates a curated 800-example Pretrained Training Dataset across all 8 handwriting archetypes,
 * a 100-example Visitor Starter Dataset, and a strictly separate 100-example Held-Out Test Dataset
 * (1,000 total examples across all 8 handwriting archetypes, rotations, slants, and stroke thicknesses).
 */
export function buildInitialDatasets(): {
  pretrainedTrainingSet: LabeledDigitExample[];
  visitorStarterSet: LabeledDigitExample[];
  testDataset: LabeledDigitExample[];
} {
  const rng = createSeededRng(2026);
  const pretrainedTrainingSet: LabeledDigitExample[] = [];
  const visitorStarterSet: LabeledDigitExample[] = [];
  const testDataset: LabeledDigitExample[] = [];

  const thicknesses = [0.92, 1.08, 1.22, 1.38, 1.54];
  const slants = [-0.18, -0.09, 0.0, 0.09, 0.18];

  // 80 variations per digit (800 total) for the Pretrained Reference Model
  // Covers all 8 structural archetypes x 5 thicknesses x rotations x post-normalization shifts
  for (let d = 0; d <= 9; d++) {
    for (let v = 0; v < 80; v++) {
      const archetype = v % 8;
      const thick = thicknesses[Math.floor(v / 8) % thicknesses.length];
      const slant = slants[Math.floor(v / 16) % slants.length] + (rng() - 0.5) * 0.06;
      const style: DigitStyleVariation = {
        thickness: thick,
        slant,
        angle: (rng() - 0.5) * 0.22,
        scaleX: 0.78 + rng() * 0.38,
        scaleY: 0.86 + rng() * 0.28,
        offsetX: (rng() - 0.5) * 1.2,
        offsetY: (rng() - 0.5) * 1.2,
        postShiftX: (rng() - 0.5) * 1.4,
        postShiftY: (rng() - 0.5) * 1.4,
        variant: archetype,
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

  // 10 starter examples per digit (100 total) for the Visitor-Trainable Model
  for (let d = 0; d <= 9; d++) {
    for (let v = 0; v < 10; v++) {
      const pixels = synthesizeHandwrittenDigit(d, {
        thickness: thicknesses[v % thicknesses.length],
        slant: ((v % 5) - 2) * 0.07,
        scaleX: 0.9 + (v % 3) * 0.08,
        scaleY: 0.92 + ((v + 1) % 3) * 0.06,
        offsetX: 0,
        offsetY: 0,
        variant: v % 8,
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

  // 10 held-out test examples per digit (100 total) with unseen slant/scale/thickness combinations
  for (let d = 0; d <= 9; d++) {
    for (let t = 0; t < 10; t++) {
      const pixels = synthesizeHandwrittenDigit(d, {
        thickness: 0.95 + (t % 4) * 0.16,
        slant: ((t % 5) - 2) * 0.07,
        angle: ((t % 3) - 1) * 0.05,
        scaleX: 0.86 + (t % 4) * 0.08,
        scaleY: 0.9 + ((t + 2) % 4) * 0.06,
        offsetX: ((t % 3) - 1) * 0.5,
        offsetY: ((t % 2) - 0.5) * 0.5,
        variant: t % 8,
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

  // 1. Pretrained Reference Model: trained with shuffled mini-batches on 800 multi-archetype examples
  const pretrainedModel = createNeuralNetwork('Pretrained MNIST Reference Model', 2026);
  trainNetworkOnBatch(pretrainedModel, pretrainedTrainingSet, {
    epochs: 8,
    learningRate: 0.0055,
    batchSize: 40,
  });

  // 2. Visitor-Trainable Model: initialized with 2 mini-batch epochs on the 100-example starter set
  const visitorModel = createNeuralNetwork('Visitor-Trainable Neural Network', 7777);
  trainNetworkOnBatch(visitorModel, visitorStarterSet, {
    epochs: 2,
    learningRate: 0.0035,
    batchSize: 25,
  });

  return {
    pretrainedModel,
    visitorModel,
    visitorDataset: visitorStarterSet,
    testDataset,
  };
}
