<div align="center">

# 🧠 NEUROLAB — Interactive Machine Learning Experience

### **Teach a machine. Watch it learn.**

*An Apple-inspired interactive science exhibition platform for exploring handwritten digit recognition, real-time neural network forward propagation, and live backpropagation training in the browser.*

[![Live Website](https://img.shields.io/badge/🌐_Live_Website-neurolab--three.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://neurolab-three.vercel.app/)
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

**🚀 Live Demo:** [**https://neurolab-three.vercel.app/**](https://neurolab-three.vercel.app/)

**✨ Made by R.K Nived Krishnan & Adithyan J ✨**

</div>

---

## 📖 Overview

**NEUROLAB** ([neurolab-three.vercel.app](https://neurolab-three.vercel.app/)) is a browser-based interactive machine learning exhibition experience designed to demystify Artificial Intelligence and Neural Networks for visitors of all ages—from young students and parents to teachers and engineers.

Rather than presenting AI as an impenetrable black box or a complicated developer console, **NEUROLAB** guides visitors through a clear, tactile four-stage interactive loop:

1. **✏️ Draw a Number (`0–9`)** — Visitors sketch any digit on an interactive touch/mouse canvas and watch their drawing convert into a normalized `28 × 28` numerical grayscale matrix (`784` values between `0.00` and `1.00`).
2. **⚡ Watch the AI Think** — A live `784 → 128 → 64 → 10` artificial neural network animates real neuron activations and synaptic weights across every layer, complete with step-by-step playback controls (`Play`, `Pause`, `Next Step`, `Replay`) and clickable single-neuron mathematical inspection.
3. **🎯 See the Result** — A clean prediction card displays the top recognized digit, confidence percentage, and full Softmax probability distribution across digits `0–9`.
4. **🎓 Teach the AI** — Visitors correct or reinforce the network by selecting the true digit and running genuine **shuffled mini-batch Adam backpropagation** in the browser, observing real-time reductions in Cross-Entropy loss, weight changes (`ΔW`), and before-vs-after prediction comparisons.

---

## 🌐 Live Deployment

The application is deployed and publicly accessible at:

🔗 **[https://neurolab-three.vercel.app/](https://neurolab-three.vercel.app/)**

- **Zero API Keys Required**: 100% of neural network inference, preprocessing, and backpropagation training runs locally in the visitor's browser.
- **Exhibition Mode**: Includes a one-click fullscreen kiosk mode (`Exhibition Mode`) and a `Reset Session for Next Visitor` button for science fairs and museum installations.

---

## ✨ Key Features

- **Genuine Neural Network Mathematics**:
  - **Architecture**: `784` Input Nodes → `128` Hidden Layer 1 Neurons (`ReLU`) → `64` Hidden Layer 2 Neurons (`ReLU`) → `10` Output Layer Neurons (`Softmax`).
  - **Total Learnable Parameters**: `109,386` weights and biases synchronized with **TensorFlow.js** (`@tensorflow/tfjs`).
  - **Training Engine**: Categorical Cross-Entropy Loss ($L = -\ln(p_{\text{target}})$), analytical backpropagation gradients, L2 weight decay regularization, zero-mean output bias centering, and the **Adam Optimiser** ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **MNIST-Standard `28 × 28` Preprocessing Pipeline**:
  - Automatic ink bounding-box extraction, aspect-ratio-preserving scaling into a `20 × 20` frame, adaptive stroke-width normalization (preventing small freehand drawings from thickening into closed loops), `3 × 3` Gaussian smoothing, and **center-of-mass translation** into the `28 × 28` grid.
- **Multi-Archetype Dataset & Lookalike Disambiguation**:
  - Built-in **1,000-example multi-archetype dataset** (`800` Pretrained Reference examples + `100` Visitor Starter examples + `100` Held-Out Test examples) spanning 4 distinct handwriting topologies per digit (`0–9`), including parametric twin-loop archetypes for digit `8` and clean separation from lookalikes (`0, 3, 6, 9`).
- **Dual-Model Exhibition Architecture**:
  - **Model 1 — Pretrained Reference Model**: Trained on 800 balanced multi-archetype digits for dependable initial recognition.
  - **Model 2 — Visitor-Trainable Model**: A dedicated `784 → 128 → 64 → 10` network that visitors can train live in the browser and benchmark against 100 unseen test digits.
- **Interactive Science Guide (*How It Works*)**:
  - 10 plain-English chapters with everyday analogies, an interactive **Single-Neuron & Weights/Biases Sandbox**, and a live **Cross-Entropy Loss Meter**.

---

## 🧬 Neural Network Architecture

| Layer | Dimensions | Activation Function | Learnable Parameters | Role in Digit Recognition |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | `28 × 28 = 784` | Normalized Grayscale `[0.00, 1.00]` | `0` | Reads pixel brightness values from the visitor's drawing |
| **Hidden Layer 1** | `128 Neurons` | $\text{ReLU}(z) = \max(0, z)$ | `100,480` (`784×128 + 128`) | Detects local stroke edges, slants, corners, and curves |
| **Hidden Layer 2** | `64 Neurons` | $\text{ReLU}(z) = \max(0, z)$ | `8,256` (`128×64 + 64`) | Combines strokes into higher-level loops, stems, and intersections |
| **Output Layer** | `10 Neurons (0–9)` | $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | `650` (`64×10 + 10`) | Converts evidence into a probability distribution summing to `100%` |
| **Total** | **986 Neurons** | **Forward + Adam Backprop** | **109,386 Parameters** | **End-to-End Handwritten Digit Classifier** |

---

## 🛠️ Tech Stack

| Category | Technology | Badge |
| :--- | :--- | :--- |
| **Framework** | Next.js 15 (App Router) & React 19 | ![Next.js](https://img.shields.io/badge/Next.js-15.4-000000?style=flat-square&logo=nextdotjs) ![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black) |
| **Language** | TypeScript 5.9 | ![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=flat-square&logo=typescript&logoColor=white) |
| **ML Engine** | Custom `Float32Array` Engine + TensorFlow.js | ![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |
| **Data Visualization** | HTML5 2D Canvas + Chart.js 4 | ![Chart.js](https://img.shields.io/badge/Chart.js-4.5-FF6384?style=flat-square&logo=chartdotjs&logoColor=white) |
| **Styling & UI** | Tailwind CSS v4 + Lucide Icons | ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.1-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white) |
| **Hosting** | Vercel Production Cloud | [![Vercel](https://img.shields.io/badge/Vercel-Live-000000?style=flat-square&logo=vercel)](https://neurolab-three.vercel.app/) |

---

## 🚀 Getting Started

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
# Run the deterministic neural network & MNIST preprocessing test suite
npx --yes tsx --test lib/neural-engine.test.ts

# Run ESLint verification
npm run lint

# Create an optimized production build
npm run build
```

---

## 📂 Project Structure

```text
├── app/
│   ├── globals.css                  # Tailwind CSS v4 imports
│   ├── layout.tsx                   # Root typography & metadata configuration
│   └── page.tsx                     # Main 4-stage Explore flow & section router
├── components/
│   ├── DrawingPad.tsx               # Interactive canvas, adaptive strokes & 28×28 grid inspector
│   ├── NeuralNetworkViz.tsx         # Live 784→128→64→10 network visualization & neuron inspector
│   ├── HowItWorksSection.tsx        # 10-chapter interactive ML guide & math sandboxes
│   ├── TeachAiSection.tsx           # Interactive training studio & dataset expander
│   ├── LearningProgressSection.tsx  # Loss curves, per-digit accuracy & session logs
│   └── AboutSection.tsx             # Architecture specs, GitHub badges, live link & reset controls
├── lib/
│   ├── neural-engine.ts             # Core 784→128→64→10 engine, Adam & MNIST normalizer
│   ├── neural-engine.test.ts        # Unit tests for forward/backward passes & digit 8 disambiguation
│   └── tf-bridge.ts                 # TensorFlow.js Sequential model synchronization
└── README.md                        # GitHub repository documentation & badges
```

---

## 👨‍💻 Authors & Attribution

<div align="center">

### **Made by R.K Nived Krishnan & Adithyan J**

🔗 **Live Website:** [https://neurolab-three.vercel.app/](https://neurolab-three.vercel.app/)

*Built for interactive science and school exhibitions to make machine learning transparent, hands-on, and intuitive for everyone.*

</div>
