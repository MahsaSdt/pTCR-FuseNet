# pTCR-FuseNet: Accurate Peptide-TCR Binding Prediction using a Tri-branch Network and Dual-Embedding Fusion

This repository provides the implementation of **pTCR-FuseNet**, a deep learning model that integrates **ESM-2** and **ProtBERT** embeddings to predict peptide–TCR interactions.  

---

## Features
- Builds peptide and TCRα/β embeddings using **ESM-2** and **ProtBERT**
- Fuses multimodal embeddings via a custom neural network (**pTCR-FuseNet**)
- Loads pretrained weights (`pTCR_FuseNet_pretrained_model.h5`)
- Runs predictions on new datasets in CSV format
- Outputs probabilities and binary predictions (threshold = 0.5)

---

## Input Format
The input file must be a CSV (e.g., `case_study.csv`) with the following columns:

- `Peptide_Sequence`  
- `CDR1_Alpha`, `CDR2_Alpha`, `CDR3_Alpha`  
- `CDR1_Beta`, `CDR2_Beta`, `CDR3_Beta`  
- `Label` *(optional, for evaluation)*  

---

## Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/MahsaSdt/pTCR-FuseNet/
cd pTCR-FuseNet
pip install -r requirements.txt
