# multimodal-cancer-classification


## Project Description
This repository implements and benchmarks multimodal fusion strategies for head and neck cancer classification using the HANCOCK dataset. We train a multi-task neural network that simultaneously predicts three clinically relevant targets (HPV association, primary tumor site, and tumor grading) from two complementary modalities: structured clinical, pathological, and blood data, and free-text surgery reports encoded via Bio_ClinicalBERT. Five model configurations are compared: a tabular unimodal baseline, a text unimodal baseline, two late fusion variants (weighted probability averaging and a learned meta-classifier), and an end-to-end attention-based fusion model that learns per-patient modality weights. The goal is to evaluate whether multimodal integration meaningfully improves classification performance over single-modality models, and whether adaptive attention-based fusion outperforms simpler combination strategies.

## Repository Structure
project/
|-- README.md 
|-- requirements.txt 
|-- data
|-- src
|-- experiments
|-- results

## Environment Setup
### Requirements
### Installation

## Dataset Setup

## Reproducing Results
### 1. Tabular Unimodal Baseline
### 2. Late Fusion Pipeline
### 3. Attention-Based Fusion
### Full Results Table

## Expected Runtime and Hardware
### Runtime by Script
### Minimum Hardware Requirements

## Results Summary

## Citation

## License

## Contact
