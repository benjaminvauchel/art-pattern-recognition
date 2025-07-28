# ART Network Pattern Recognition — Binary Letter Classification

This mini-project implements an **Adaptive Resonance Theory (ART) network** for binary pattern recognition, specifically designed to classify 8×8 pixel representations of letters A through T. The implementation is built **from scratch** and explores the network's performance under various conditions including clean and noisy pattern recognition.

The project was created as part of the COEN 6331 Neural Networks course at Concordia University.

## 📌 Features

- **Pure ART1 implementation** solving the plasticity-stability dilemma
- **Binary pattern recognition** for 20 letter patterns (A-T)
- **Noise robustness testing** with configurable noise ratios
- **Advanced similarity calculation** with penalty mechanism
- **Comprehensive evaluation metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrices
  - Performance visualization
- **Hyperparameter analysis**:
  - Vigilance parameter (δ) tuning
  - Penalty rate optimization
  - Noise tolerance evaluation

## Problem Description

The task is to train an ART network to recognize and classify 20 different letter patterns (A-T) represented as 8×8 binary matrices (64 pixels total). Each pattern is flattened into a 1D array before being fed into the network.

**Key challenges addressed:**
- **Pattern similarity**: Letters like 'C' and 'P' are subsets of 'B', 'F' is a subset of 'E'
- **Noise robustness**: Network must handle corrupted input patterns
- **Plasticity-stability balance**: Learn new patterns without forgetting old ones

The ART network's unique architecture allows it to:
- Create new categories dynamically during training
- Maintain learned patterns without catastrophic forgetting
- Control classification strictness through the vigilance parameter

## Network Architecture

The ART1 network consists of:
- **Input Layer**: 64 neurons (8×8 flattened pattern)
- **Comparison Layer (F₁)**: Compares input with learned prototypes
- **Recognition Layer (F₂)**: Category neurons representing pattern classes
- **Two weight matrices**:
  - **Bottom-up weights**: Input → Recognition layer
  - **Top-down weights**: Recognition → Input layer (pattern memory)

## Key Innovations

### Enhanced Similarity Calculation
The standard ART similarity measure was coupled with a **penalty mechanism**:
- Penalizes categories when input has 0 where stored pattern has 1
- Prevents over-generalization (e.g., 'B' containing 'C' and 'P')
- Controlled by penalty rate parameter (p)

### Vigilance Parameter Optimization
- **Low vigilance**: More tolerance, better generalization
- **High vigilance**: Stricter categorization, risk of over-rejection (δ > 0.7 leads to a drop in accuracy)
- **Optimal range**: δ = 0.5 with penalty rate p = 0.001

## Some Performance Results

### Clean Pattern Recognition
| Configuration | Overall Accuracy |
|---------------|-----------------|
| δ = 0.5, p = 0 | 80% |
| δ = 0.95, p = 0 | 80% |
| δ = 0.5, p = 0.001 | **100%** |

### Noisy Pattern Recognition (10% noise)
| Configuration | Overall Accuracy |
|---------------|-----------------|
| δ = 0.5, p = 0 | 69.90% |
| δ = 0.8, p = 0 | 48.17% |
| δ = 0.5, p = 0.001 | **84.94%** |

### Noise Tolerance Analysis (with p=0.001 and δ=0.5)
- **5% noise**: ~93% accuracy
- **10% noise**: ~85% accuracy  
- **20% noise**: ~70% accuracy
- **25% noise**: ~50% accuracy

## Project Structure

```
art-pattern-recognition/
├── Report (academic).pdf    # Complete academic analysis
├── README.md               # This file :)
└── main.ipynb  # ART network code and simulations
```

## Key Findings

1. **Standard ART limitations**: Basic similarity measure causes over-generalization
2. **Penalty mechanism effectiveness**: Dramatic improvement in pattern discrimination
3. **Vigilance range**: δ = 0.5 provides optimal balance
4. **Noise robustness**: Network maintains reasonable performance up to 25% noise
5. **Training dynamics**: Multiple epochs needed for complex pattern sets

## Evaluation Metrics

The project uses comprehensive evaluation including:
- **Overall Accuracy**: Correct predictions / Total predictions
- **Per-class metrics**: Precision, Recall, F1-score
- **Confusion matrices**: Detailed classification analysis
- **Performance curves**: Accuracy vs. vigilance/noise/penalty rate

## Learning Outcomes

- Understanding ART network architecture and dynamics
- Implementing the plasticity-stability solution
- Analyzing pattern recognition under noise
- Hyperparameter optimization in neural networks
- Comprehensive performance evaluation methodologies

## Technical Implementation

**Weight Initialization:**
- Bottom-up weights: 1/(1 + n_features)
- Top-down weights: 1 (neutral state)

**Learning Rule:**
- Top-down: W_new = λ × min(X, W_old) + (1-λ) × W_old
- Bottom-up: Normalized based on updated top-down weights

**Similarity Calculation:**
- Enhanced with penalty for mismatched features
- Vigilance threshold determines category acceptance

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Future Enhancements

- Dynamic vigilance adjustment during classification
- ART2 implementation for continuous patterns
- Real-world dataset evaluation

## License

This project is provided for academic purposes. Feel free to explore, modify, or build upon it for educational use.
