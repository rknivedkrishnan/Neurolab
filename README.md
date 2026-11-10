<div align="center">

# NEUROLAB — Interactive Machine Learning Experience

### **Teach a machine. Watch it learn.**

*An interactive science exhibition platform for exploring handwritten digit recognition, real-time neural network forward propagation, and live backpropagation training in the browser.*

[![Live Website](https://img.shields.io/badge/Live_Website-neurolab--three.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://neurolab-three.vercel.app/)
[![Status](https://img.shields.io/badge/Deployment-Production_Ready-10B981?style=for-the-badge&logo=checkmarx&logoColor=white)](https://neurolab-three.vercel.app/)
[![Next.js 15](https://img.shields.io/badge/Next.js_15-111111?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![TypeScript 5.9](https://img.shields.io/badge/TypeScript_5.9-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Tailwind CSS 4](https://img.shields.io/badge/Tailwind_CSS_4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js_4-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![Architecture](https://img.shields.io/badge/MLP_Architecture-784_→_128_→_64_→_10-5856D6?style=for-the-badge)](https://neurolab-three.vercel.app/)
[![Parameters](https://img.shields.io/badge/Learnable_Params-109,386-3478F6?style=for-the-badge)](https://neurolab-three.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

**Live Demo:** [**https://neurolab-three.vercel.app/**](https://neurolab-three.vercel.app/)

**Made by R.K Nived Krishnan & Adithyan J**

</div>

---

## Overview

**NEUROLAB** ([neurolab-three.vercel.app](https://neurolab-three.vercel.app/)) is a browser-based interactive machine learning exhibition application built to make neural networks transparent and hands-on for visitors of all ages.

Visitors interact with a live multi-layer perceptron through four stages:

1. **Draw a Number (`0–9`)** — Sketch any digit on the touch/mouse canvas and inspect how it is normalized into a `28 × 28` numerical grayscale matrix (`784` values between `0.00` and `1.00`).
2. **Watch the Network Process** — Observe a `784 → 128 → 64 → 10` neural network compute layer-by-layer activations and weighted connections, with step-by-step playback controls (`Play`, `Pause`, `Next Step`, `Replay`) and single-neuron inspection.
3. **See the Prediction** — View the top predicted digit, confidence score, and full Softmax probability distribution across digits `0–9`.
4. **Teach the Model** — Provide feedback by selecting the true digit and running shuffled mini-batch Adam backpropagation directly in the browser, tracking Cross-Entropy loss reductions, weight updates (`ΔW`), and before-vs-after predictions.

---

## Live Deployment

The project is deployed at:

**[https://neurolab-three.vercel.app/](https://neurolab-three.vercel.app/)**

- **100% Client-Side Execution**: All forward passes, preprocessing, and backpropagation weight updates run locally in the browser with no external server dependencies.
- **Exhibition Mode**: Includes a fullscreen kiosk toggle (`Exhibition Mode`) and a `Reset Session for Next Visitor` control.

---

## Key Features

- **Neural Network Engine**:
  - **Architecture**: `784` Input Nodes → `128` Hidden Layer 1 Neurons (`ReLU`) → `64` Hidden Layer 2 Neurons (`ReLU`) → `10` Output Layer Neurons (`Softmax`).
  - **Total Parameters**: `109,386` learnable weights and biases synchronized with **TensorFlow.js** (`@tensorflow/tfjs`).
  - **Optimization**: Categorical Cross-Entropy Loss ($L = -\ln(p_{\text{target}})$), analytical backpropagation gradients, L2 weight decay regularization, zero-mean output bias centering, and the **Adam Optimiser** ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **MNIST `28 × 28` Preprocessing Pipeline**:
  - Ink bounding-box extraction, aspect-ratio-preserving scaling into a `20 × 20` box, adaptive stroke-width normalization, `3 × 3` Gaussian smoothing, and **center-of-mass translation** into the `28 × 28` grid.
- **Multi-Archetype Dataset**:
  - Built-in **1,000-example dataset** (`800` Pretrained Reference examples + `100` Visitor Starter examples + `100` Held-Out Test examples) spanning 4 handwriting archetypes per digit (`0–9`).
- **Dual-Model Setup**:
  - **Pretrained Reference Model**: Trained on 800 balanced multi-archetype digits for baseline recognition.
  - **Visitor-Trainable Model**: A separate `784 → 128 → 64 → 10` network that visitors train live in the browser and benchmark against 100 held-out test digits.
- **Interactive Guide (*How It Works*)**:
  - 10 chapters covering pixels, neurons, weights, biases, ReLU, Softmax, Cross-Entropy loss, and backpropagation, paired with interactive math sandboxes.

---

## Architecture Specification

| Layer | Dimensions | Activation Function | Learnable Parameters | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | `28 × 28 = 784` | Normalized Grayscale `[0.00, 1.00]` | `0` | Reads pixel brightness values from the canvas |
| **Hidden Layer 1** | `128 Neurons` | $\text{ReLU}(z) = \max(0, z)$ | `100,480` (`784×128 + 128`) | Detects local stroke edges, slants, and curves |
| **Hidden Layer 2** | `64 Neurons` | $\text{ReLU}(z) = \max(0, z)$ | `8,256` (`128×64 + 64`) | Combines strokes into loops, stems, and intersections |
| **Output Layer** | `10 Neurons (0–9)` | $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | `650` (`64×10 + 10`) | Outputs a probability distribution summing to `100%` |
| **Total** | **986 Neurons** | **Forward + Adam Backprop** | **109,386 Parameters** | **Handwritten Digit Classifier** |

---

## Tech Stack

| Category | Technology | Badge |
| :--- | :--- | :--- |
| **Framework** | Next.js 15 (App Router) & React 19 | ![Next.js](https://img.shields.io/badge/Next.js-15.4-000000?style=flat-square&logo=nextdotjs) ![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black) |
| **Language** | TypeScript 5.9 | ![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=flat-square&logo=typescript&logoColor=white) |
| **ML Engine** | Custom `Float32Array` Engine + TensorFlow.js | ![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |
| **Data Visualization** | HTML5 2D Canvas + Chart.js 4 | ![Chart.js](https://img.shields.io/badge/Chart.js-4.5-FF6384?style=flat-square&logo=chartdotjs&logoColor=white) |
| **Styling & UI** | Tailwind CSS v4 + Lucide Icons | ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.1-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white) |
| **Deployment** | Vercel | [![Vercel](https://img.shields.io/badge/Vercel-Live-000000?style=flat-square&logo=vercel)](https://neurolab-three.vercel.app/) |

---

## Getting Started

### Prerequisites

- **Node.js** `v18.18+` or `v20+`
- **npm** `v9+`

### Installation & Local Development

```bash
git clone https://github.com/your-username/neurolab.git
cd neurolab
npm install
npm run dev
```

### Running Tests & Production Build

```bash
# Run unit tests
npx --yes tsx --test lib/neural-engine.test.ts

# Run linter
npm run lint

# Build for production
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
│   ├── DrawingPad.tsx               # Interactive canvas & 28×28 grid inspector
│   ├── NeuralNetworkViz.tsx         # Live 784→128→64→10 network visualization
│   ├── HowItWorksSection.tsx        # 10-chapter interactive ML guide & sandboxes
│   ├── TeachAiSection.tsx           # Interactive training studio & dataset expander
│   ├── LearningProgressSection.tsx  # Loss curves, per-digit accuracy & session logs
│   └── AboutSection.tsx             # Architecture specs, GitHub badges & reset controls
├── lib/
│   ├── neural-engine.ts             # Core 784→128→64→10 engine, Adam & MNIST normalizer
│   ├── neural-engine.test.ts        # Unit tests for forward/backward passes & accuracy
│   └── tf-bridge.ts                 # TensorFlow.js Sequential model synchronization
└── README.md                        # Repository documentation
```

---

## Authors

<div align="center">

### **Made by R.K Nived Krishnan & Adithyan J**

**Live Website:** [https://neurolab-three.vercel.app/](https://neurolab-three.vercel.app/)

</div>
