import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createNeuralNetwork,
  forwardPass,
  trainNetworkOnBatch,
  evaluateDataset,
  normalizeGridToMNIST28x28,
  synthesizeHandwrittenDigit,
  initializeNeurolabModels,
  buildNetworkVisualizationSnapshot,
  buildInitialDatasets,
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

  for (const h of result.hidden1) assert.ok(h >= 0);
  for (const h of result.hidden2) assert.ok(h >= 0);

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

test('Expanded 1,000-example dataset (800 pretrained + 100 starter + 100 test) achieves >94% accuracy with zero digit-3 bias', () => {
  const datasets = buildInitialDatasets();
  assert.equal(datasets.pretrainedTrainingSet.length, 800);
  assert.equal(datasets.visitorStarterSet.length, 100);
  assert.equal(datasets.testDataset.length, 100);

  const { pretrainedModel, visitorModel, testDataset } = initializeNeurolabModels();
  const preEval = evaluateDataset(pretrainedModel, testDataset);
  const visEval = evaluateDataset(visitorModel, testDataset);

  assert.ok(
    preEval.accuracy >= 0.94,
    `Expected pretrained model accuracy >= 94%, got ${(preEval.accuracy * 100).toFixed(1)}%`
  );
  assert.ok(
    visEval.accuracy >= 0.72,
    `Expected visitor model starter accuracy >= 72%, got ${(visEval.accuracy * 100).toFixed(1)}%`
  );

  // Verify every single digit 0..9 is recognized across thin/medium/thick/slanted strokes without defaulting to 3
  let countPredicted3OnNon3 = 0;
  let totalNon3 = 0;
  for (let d = 0; d <= 9; d++) {
    for (const variant of [0, 1, 2, 3]) {
      const sample = synthesizeHandwrittenDigit(d, {
        thickness: 1.15,
        slant: 0.04,
        variant,
      });
      const res = forwardPass(sample, pretrainedModel);
      if (d !== 3) {
        totalNon3++;
        if (res.predictedDigit === 3) countPredicted3OnNon3++;
      }
    }
  }
  assert.ok(
    countPredicted3OnNon3 <= 1,
    `Expected almost zero false '3' predictions on non-3 digits, got ${countPredicted3OnNon3}/${totalNon3}`
  );

  // Verify digit 8 is recognized reliably across all 4 archetypes (figure-8, stacked ovals, S-curve, X-waist)
  // and lookalike digits (0, 3, 6, 9) do not get confused with 8
  let correct8 = 0;
  const total8Tests = 8;
  for (let v = 0; v < 4; v++) {
    for (const thick of [1.05, 1.35]) {
      const sample8 = synthesizeHandwrittenDigit(8, {
        thickness: thick,
        slant: (v - 1.5) * 0.05,
        variant: v,
      });
      const res8 = forwardPass(sample8, pretrainedModel);
      if (res8.predictedDigit === 8) correct8++;
    }
  }
  assert.ok(
    correct8 >= 7,
    `Expected digit 8 recognition >= 7/8 across all archetypes, got ${correct8}/${total8Tests}`
  );

  for (const lookalike of [0, 3, 6, 9]) {
    for (let v = 0; v < 4; v++) {
      const sample = synthesizeHandwrittenDigit(lookalike, {
        thickness: 1.2,
        slant: 0.02,
        variant: v,
      });
      const res = forwardPass(sample, pretrainedModel);
      assert.equal(
        res.predictedDigit,
        lookalike,
        `Expected lookalike digit ${lookalike} (variant ${v}) to be recognized as ${lookalike}, got ${res.predictedDigit}`
      );
    }
  }
});

test('MNIST 28x28 center-of-mass preprocessing normalizes off-center drawings', () => {
  const raw = new Float32Array(56 * 56);
  for (let y = 38; y <= 50; y++) {
    for (let x = 42; x <= 45; x++) {
      raw[y * 56 + x] = 1.0;
    }
  }
  const normalized = normalizeGridToMNIST28x28(raw, 56, 56);
  assert.equal(normalized.length, 784);
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
