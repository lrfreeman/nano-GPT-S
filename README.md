# Transformer Decoder-Only Architecture

## Overview

This project implements a transformer decoder-only architecture, closely following the "Attention is All You Need" paper by Vaswani et al. It leverages the PyTorch library and is inspired by the "Zero to Hero" series by Andrej Karpathy. The primary goal is to provide a foundational understanding of transformers and to serve as a basis for applying neural data to transformers and conducting mechanistic interpretability.

## Features

- **Transformer Decoder-Only Architecture**: Implements a transformer model focusing solely on the decoder.
- **Multi-Head Attention**: Multiple heads of self-attention in parallel for capturing different aspects of the input.
- **Feed-Forward Neural Networks**: For each token, allowing complex relationships within the sequence.
- **Layer Normalization**: Applied before attention and feed-forward layers for stability.
- **Dropout Regularization**: Used to prevent overfitting.
- **Text Generation**: Capability to generate new tokens based on a given context.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/transformer-decoder.git
    cd transformer-decoder
    ```
## Bigram note - As recommended by Anthropic

There is a seperate script which removes any attentional layers and MLP to soley focus on the bigram statistics. Which, as stated in "A Mathematical Framework for Transformer Circuits" is a good step towards pulling apart transformers. 
