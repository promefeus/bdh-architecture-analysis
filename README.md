# BDH Architecture Analysis: Component-Level Study

## 📌 Overview
This project presents a controlled, component-level analysis of the Baby Dragon Hatchling (BDH) architecture. The objective is to identify which architectural components contribute meaningfully to model performance by isolating individual design choices under a strictly controlled experimental setup.

## 🎯 Research Objective
The study investigates the contribution of the following architectural elements:

- **Latent Dimensionality**: Performance within reduced latent spaces.
- **Multiplicative Interaction**: Multiplication vs. standard addition efficacy.
- **Activation Function / Sparsity**: Impact of GELU vs. ReLU.
- **Attention Mechanism**: Core attention logic within the BDH framework.

Each experiment modifies exactly one component at a time, ensuring differences are causally interpretable.

## ⚙️ Experimental Setup

| Parameter        | Value                  |
|------------------|------------------------|
| Dataset          | WikiText-2 (byte-level) |
| Vocabulary       | 256                    |
| Layers           | 6                      |
| Embedding Dim    | 256                    |
| Attention Heads  | 4                      |
| Block Size       | 128                    |
| Batch Size       | 8                      |

## 🧠 Models Evaluated

### Baseline
- **Transformer**: Standard architecture used as the primary benchmark.

### BDH Variants
- **BDH (Base)**: The full, original BDH architecture.
- **BDH (No Multiplication)**: Multiplicative interaction replaced with addition.
- **BDH (Low Dimension)**: Evaluates the model with reduced latent dimensionality.
- **BDH (Improved)**: Utilizes GELU activation instead of standard ReLU.

## 📊 Evaluation Metrics & Outputs
The pipeline tracks:
- Final / Validation Loss
- Stability (Last 50 steps)
- Component Impact (Δ Loss)

### Automated Visualizations (stored in `results/plots/`)
- Training and smoothed loss curves
- Model comparison bar charts
- Component impact visualizations

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install torch numpy matplotlib datasets
```

### 2. Run Experiments
```bash
python -m experiments.runner
```

### 3. Analyze Results
```bash
python analyze.py
```

## 📂 Project Structure
```plaintext
bdh-research/
├── configs/          # Experiment configurations
├── data/             # Dataset loading and preprocessing
├── models/           # BDH and Transformer implementations
├── train/            # Core training logic
├── experiments/      # Runner scripts and variant definitions
├── utils/            # Logging, metrics, and plotting tools
├── analyze.py        # Post-training analysis script
└── README.md
```

## 🔍 Key Insight
This project addresses the central question:

> Which components of BDH are responsible for its performance, and what is their relative impact?

## 👤 Author
**Nitin Saini**