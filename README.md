
# Multimodal Aspect-Based Fake Review Detection

This project implements a multimodal transformer-based model (MAC) integrating **Variational Autoencoders (VAE)**, **ResNet-152**, and **Sentence Transformers** for:

1. **Complaint Aspect Detection**
2. **Sentiment Analysis**
3. **Summary Generation**

The model was applied across **9 e-commerce domains**, enhancing review authenticity detection and extracting actionable insights.

## Features

- **Multimodality**: Combines textual and visual modalities.
- **Aspect Detection**: Identifies specific aspects of complaints in reviews.
- **Sentiment Analysis**: Analyzes sentiment polarity (positive/negative/neutral).
- **Summarization**: Generates concise summaries of reviews.

## Directory Structure

- `multi_aspect_classification_vae.py`: Core script implementing multimodal aspect classification.
- `vae_gru_final.py`: Final VAE-GRU model implementation.
- `Summarizer_Domain.ipynb`: Domain-specific summarization logic.
- `Label_identification.ipynb`: Label identification methodology.
- `model_notebook.ipynb`: Model training and evaluation workflows.
- `dataset/`: Contains relevant datasets (e.g., `ICDAR.csv`, `Aspect_Complain_web_entity_clip.csv`).

## Installation

### Prerequisites

Ensure you have Python 3.8 or above installed.

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Fake-Review-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python multi_aspect_classification_vae.py
   ```

## Technologies Used

- **PyTorch**: For deep learning model implementation.
- **Sentence-Transformers**: For embedding textual data.
- **ResNet-152**: For visual feature extraction.
- **VAE and GRU**: For generating embeddings and sequence modeling.
- **Jupyter Notebooks**: For exploratory analysis and model fine-tuning.

## Dataset

The `dataset` folder contains e-commerce reviews datasets in CSV format. Example datasets include:
- `ICDAR.csv`
- `Aspect_Complain_web_entity_clip.csv`

## Authors

Developed by AI-ML-NLP Lab (Jan 2024 - Mar 2024).


