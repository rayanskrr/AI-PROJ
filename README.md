# Egyptian Hieroglyph Recognition

Implementation of the paper: **"Recognition of Egyptian hieroglyphic texts through focused generic segmentation and cross-validation voting"** (Fuentes-Ferrer et al., 2025).

This repository contains a pipeline for recognizing ancient Egyptian hieroglyphs from stone stelae using both classical computer vision (Canny Edge) and deep learning (Segment Anything Model) approaches for segmentation, combined with a ConvNeXt-tiny backbone utilizing Cross-Validation Voting (CVV) for classification.

## Project Structure
* `src/`: Core pipeline modules (segmentation, classification, CVV loop).
* `notebooks/`: Google Colab-ready notebooks for training and demonstration.
* `config.py`: Global hyperparameter and path definitions.
