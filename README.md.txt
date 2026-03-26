# Latent Persona Disentanglement in Dialogue Systems

## Overview

This project focuses on learning **latent persona representations** from conversational dialogue without using explicit persona labels.

We first cluster speakers using semantic embeddings and then train a BERT-based model to learn **disentangled persona and content representations**.

---

## Key Features

- Unsupervised persona discovery using clustering
- BERT-based representation learning
- Adversarial training for disentanglement
- Persona leakage analysis
- Visualization of embedding space

---

## Pipeline

1. Extract speaker utterances
2. Generate embeddings using MiniLM
3. Perform K-Means clustering (K=14)
4. Create pseudo persona labels
5. Train BERT model with disentanglement
6. Evaluate using accuracy and leakage test

---

## Results

- Best Validation Accuracy: **16.93%**
- Persona Leakage Accuracy: **15.05%**
- Demonstrates **partial disentanglement**

---

## Model Architecture

- Encoder: BERT (bert-base-uncased)
- Persona embedding: 128-dim
- Content embedding: 128-dim
- Loss:
  - Classification Loss
  - Adversarial Loss
  - Orthogonality Loss

---

## Visualizations

All analysis plots are available in the `results/` folder.

---

## How to Run

```bash
# Step 1: Build dataset
python persona_full_pipeline.py

# Step 2: Train model
python -m src.train

# Step 3: Analyze results
python -m src.analyze_results