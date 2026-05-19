# Pixel Transformer 🎮🤖

A decoder-only Transformer model that learns to generate pixel-art sprites row-by-row using self-attention.

## Overview

This project explores how Transformer architectures can be applied beyond text generation by training a model to generate 16×16 pixel-art sprites.

The model learns:

* Spatial patterns
* Colour relationships
* Sequential row generation
* Autoregressive prediction

## Features

* Custom Transformer implementation in PyTorch
* Multi-Head Self-Attention
* Positional Encoding
* Decoder-only architecture
* Pixel-aware Row Encoder
* Autoregressive sprite generation

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

## Architecture

Input Sprite
↓
Row Tokenization
↓
Row Encoder
↓
Positional Encoding
↓
Transformer Decoder Blocks
↓
Next Row Prediction

## Training

The model is trained using:

* Cross-Entropy Loss
* Adam Optimizer
* Cosine Annealing Learning Rate Scheduler

## Example Outputs

Generated pixel sprites after training:

(Add screenshots here)

## Run Project

```bash
pip install -r requirements.txt
python main.py
```

## Concepts Used

* Self-Attention
* Multi-Head Attention
* Causal Masking
* Transformer Decoder
* Embeddings
* Autoregressive Generation

## Future Improvements

* Larger sprite datasets
* Conditional generation
* Diffusion hybrid models
* GAN comparison
* Interactive web demo

