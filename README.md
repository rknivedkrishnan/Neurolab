# NEUROLAB — Interactive Machine Learning Experience

> **Teach a machine. Watch it learn.**  
> *An Apple-inspired interactive exhibition platform for exploring handwritten digit recognition, real-time neural network forward propagation, and live backpropagation training.*

**Made by R.K Nived Krishnan & Adithyan J**

---

## Overview

**NEUROLAB** is a browser-based interactive science exhibition application designed to demystify Artificial Intelligence and Machine Learning for visitors of all ages—from school students and parents to educators and engineers.

Instead of hiding machine learning behind a black box or a complicated technical dashboard, NEUROLAB guides visitors through a clear, tactile four-stage journey:

1. **Draw a Number (0–9)** — Visitors sketch any digit on an interactive touch/mouse canvas and watch it transform into a normalized `28 × 28` numerical grayscale matrix (`784` values between `0.00` and `1.00`).
2. **Watch the AI Think** — A live `784 → 128 → 64 → 10` artificial neural network animates real neuron activations and synaptic weights across layers, complete with step-by-step playback controls (`Play`, `Pause`, `Next Step`, `Replay`) and clickable neuron inspection.
3. **See the Result** — A clean prediction card displays the top recognized digit, confidence percentage, and full Softmax probability distribution across digits `0–9`.
4. **Teach the AI** — Visitors can correct or reinforce the AI by selecting the true digit and running genuine **shuffled mini-batch Adam backpropagation** in the browser, observing real-time reductions in Cross-Entropy loss, weight changes (`ΔW`), and before-vs-after prediction comparisons.

---

## Key Features

- **Genuine Neural Network Mathematics**:
  - **Architecture**: `784` Input Nodes → `128` Hidden Layer 1 Neurons (`ReLU`) → `64` Hidden Layer 2 Neurons (`ReLU`) → `10` Output Layer Neurons (`Softmax`).
  - **Total Learnable Parameters**: `109,386` weights and biases synchronized with **TensorFlow.js**.
  - **Training Engine**: Categorical Cross-Entropy Loss (`L = -ln(p_target)`), analytical backpropagation gradients, L2 weight decay regularization, zero-mean output bias centering, and the **Adam Optimiser** (`β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸`).
- **MNIST-Standard 28×28 Preprocessing**:
  - Automatic ink bounding-box extraction, aspect-ratio-preserving scaling into a `20 × 20` box, adaptive stroke-width normalization, 3×3 Gaussian edge smoothing, and **center-of-mass translation** into the `28 × 28` frame.
- **Dual-Model Exhibition Architecture**:
  - **Pretrained Reference Model**: Trained on **800 multi-archetype handwritten digit examples** across 4 distinct structural handwriting styles per digit (`0–9`) for dependable initial recognition.
  - **Visitor-Trainable Model**: A separate `784 → 128 → 64 → 10` network that visitors can train live and evaluate against a strictly separate **100-example Held-Out Test Dataset**.
- **Interactive Science Guide (*How It Works*)**:
  - 10 plain-English learning chapters with everyday analogies, interactive **Single-Neuron & Weights/Biases Sliders**, and an interactive **Cross-Entropy Loss Meter**.
- **Exhibition-Ready Design**:
  - Warm minimalist visual identity (`#FAFAF8` canvas, `#FFFFFF` cards, `#202124` charcoal typography, `#3478F6` electric blue accents), zero external API keys required, offline-capable once loaded, and one-click **Exhibition Mode** fullscreen support.

---

## Neural Network Architecture

| Layer | Dimensions | Activation Function | Learnable Parameters | Role in Digit Recognition |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | `28 × 28 = 784` | Normalized Grayscale `[0, 1]` | `0` | Receives pixel brightness values from the visitor's drawing |
| **Hidden Layer 1** | `128 Neurons` | `ReLU(z) = max(0, z)` | `100,480` (`784×128 + 128`) | Detects local stroke edges, slants, corners, and curves |
| **Hidden Layer 2** | `64 Neurons` | `ReLU(z) = max(0, z)` | `8,256` (`128×64 + 64`) | Combines strokes into higher-level loops, stems, and intersections |
| **Output Layer** | `10 Neurons (0–9)` | `Softmax(z_i) = e^{z_i} / Σ e^{z_j}` | `650` (`64×10 + 10`) | Converts evidence into a probability distribution summing to `100%` |

---

## Tech Stack

- **Framework**: Next.js 15 (App Router) & React 19
- **Language**: TypeScript 5
- **Machine Learning**: Custom `Float32Array` Neural Engine + TensorFlow.js (`@tensorflow/tfjs`)
- **Visualizations**: HTML5 Canvas (2D Context) & Chart.js (`chart.js`, `react-chartjs-2`)
- **Styling**: Tailwind CSS v4 (`Plus Jakarta Sans` & `JetBrains Mono`)
- **Icons**: Lucide React (`lucide-react`)

---

## Getting Started

### Prerequisites

- **Node.js** `v18.18+` or `v20+`
- **npm** `v9+`

### Installation & Local Development

```bash
# 1. Clone the repository
git clone https://github.com/your-username/neurolab.git
cd neurolab

# 2. Install dependencies
npm install

# 3. Start the development server on http://localhost:3000
npm run dev
```

### Running Unit Tests & Production Build

```bash
# Run the deterministic neural network & preprocessing test suite
npx --yes tsx --test lib/neural-engine.test.ts

# Run ESLint verification
npm run lint

# Create an optimized production build
npm run build
```

---

## Project Structure

```text
├── app/
│   ├── globals.css                  # Tailwind CSS v4 imports
│   ├── layout.tsx                   # Root typography & metadata configuration
│   └── page.tsx                     # Main 4-stage Explore flow & section router
├── components/
│   ├── DrawingPad.tsx               # Interactive canvas & 28×28 pixel grid inspector
│   ├── NeuralNetworkViz.tsx         # Live 784→128→64→10 network visualization
│   ├── HowItWorksSection.tsx        # 10-chapter interactive ML guide & sandboxes
│   ├── TeachAiSection.tsx           # Interactive training studio & dataset expander
│   ├── LearningProgressSection.tsx  # Loss curves, per-digit accuracy & session logs
│   └── AboutSection.tsx             # Architecture specs, GitHub docs & reset controls
└── lib/
    ├── neural-engine.ts             # Core 784→128→64→10 engine, Adam & MNIST normalizer
    ├── neural-engine.test.ts        # Unit tests for forward/backward passes & accuracy
    └── tf-bridge.ts                 # TensorFlow.js Sequential model synchronization
```

---

## Credits & Authors

**Made by R.K Nived Krishnan & Adithyan J**

Built for interactive science and school exhibitions to make machine learning transparent, hands-on, and intuitive for everyone.
