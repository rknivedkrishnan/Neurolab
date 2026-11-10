import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createNeuralNetwork,
  forwardPass,
  computeCrossEntropyLoss,
  trainNetworkOnBatch,
  evaluateDataset,
  normalizeGridToMNIST28x28,
  synthesizeHandwrittenDigit,
  initializeNeurolabModels,
  buildNetworkVisualizationSnapshot,
  INPUT_SIZE,
  H1_SIZE,
  H2_SIZE,
  OUTPUT_SIZE,
} from './neural-engine';

test('784 -> 128 -> 64 -> 10 network architecture and forwardPass invariants', () => {
  const net = createNeuralNetwork('TestNet', 123);
  assert.equal(net.W1.length, INPUT_SIZE * H1_SIZE);
  assert.equal(net.b1.length, H1_SIZE);
  assert.equal(net.W2.length, H1_SIZE * H2_SIZE);
  assert.equal(net.b2.length, H2_SIZE);
  assert.equal(net.W3.length, H2_SIZE * OUTPUT_SIZE);
  assert.equal(net.b3.length, OUTPUT_SIZE);

  const sampleDigit = synthesizeHandwrittenDigit(7);
  assert.equal(sampleDigit.length, 784);

  const result = forwardPass(sampleDigit, net);
  assert.equal(result.hidden1.length, 128);
  assert.equal(result.hidden2.length, 64);
  assert.equal(result.probabilities.length, 10);

  // ReLU outputs must be non-negative
  for (const h of result.hidden1) assert.ok(h >= 0);
  for (const h of result.hidden2) assert.ok(h >= 0);

  // Softmax probabilities must sum to 1.0
  const probSum = result.probabilities.reduce((acc, v) => acc + v, 0);
  assert.ok(Math.abs(probSum - 1.0) < 1e-5, `Expected sum ~1.0, got ${probSum}`);
  assert.ok(result.predictedDigit >= 0 && result.predictedDigit <= 9);
  assert.equal(result.confidence, result.probabilities[result.predictedDigit]);
});

test('Adam backpropagation genuinely reduces cross-entropy loss and updates weights', () => {
  const net = createNeuralNetwork('TrainTestNet', 99);
  const digit3 = synthesizeHandwrittenDigit(3);
  const digit8 = synthesizeHandwrittenDigit(8);

  const examples = [
    { id: 'ex-1', pixels: digit3, label: 3, createdAt: 1, source: 'visitor' as const },
    { id: 'ex-2', pixels: digit8, label: 8, createdAt: 2, source: 'visitor' as const },
  ];

  const session = trainNetworkOnBatch(net, examples, {
    epochs: 10,
    learningRate: 0.008,
    focusExample: examples[0],
  });

  assert.ok(
    session.finalLoss < session.initialLoss,
    `Expected finalLoss (${session.finalLoss}) < initialLoss (${session.initialLoss})`
  );
  assert.ok(session.meanAbsWeightDelta > 0, 'Weights must change during backpropagation');
  assert.ok(
    session.afterPrediction!.targetDigitProb > session.beforePrediction!.targetDigitProb,
    'Target digit probability must increase after training on that example'
  );
});

test('Pretrained Reference Model achieves high recognition accuracy on held-out test set', () => {
  const { pretrainedModel, visitorModel, testDataset } = initializeNeurolabModels();
  const preEval = evaluateDataset(pretrainedModel, testDataset);
  const visEval = evaluateDataset(visitorModel, testDataset);

  assert.ok(
    preEval.accuracy >= 0.88,
    `Expected pretrained model accuracy >= 88%, got ${(preEval.accuracy * 100).toFixed(1)}%`
  );
  assert.ok(
    preEval.accuracy > visEval.accuracy,
    'Pretrained model should start with higher accuracy than lightly initialized visitor model'
  );
});

test('MNIST 28x28 center-of-mass preprocessing normalizes off-center drawings', () => {
  const raw = new Float32Array(56 * 56);
  // Draw a small vertical stroke in the bottom-right corner
  for (let y = 38; y <= 50; y++) {
    for (let x = 42; x <= 45; x++) {
      raw[y * 56 + x] = 1.0;
    }
  }
  const normalized = normalizeGridToMNIST28x28(raw, 56, 56);
  assert.equal(normalized.length, 784);
  // Center region around (14, 14) should now have non-zero ink
  let centerMass = 0;
  for (let y = 10; y <= 18; y++) {
    for (let x = 10; x <= 18; x++) {
      centerMass += normalized[y * 28 + x];
    }
  }
  assert.ok(centerMass > 3.0, `Expected centered ink mass > 3.0, got ${centerMass}`);
});

test('Visualization snapshot extracts authentic weights and activations from network', () => {
  const net = createNeuralNetwork('VisNet', 42);
  const digit0 = synthesizeHandwrittenDigit(0);
  const act = forwardPass(digit0, net);
  const snap = buildNetworkVisualizationSnapshot(act, net);

  assert.equal(snap.nodesByLayer[0].length, 16);
  assert.equal(snap.nodesByLayer[1].length, 14);
  assert.equal(snap.nodesByLayer[2].length, 10);
  assert.equal(snap.nodesByLayer[3].length, 10);
  assert.ok(snap.synapses.length > 100);
});
