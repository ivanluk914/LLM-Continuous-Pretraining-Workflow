# Large Language Model Pretraining Workflow

This repository outlines the workflow for preparing datasets and training/upscaling large language models (LLMs) using the Hugging Face ecosystem. It includes steps to preprocess custom datasets, adapt existing models, and perform continuous pretraining.

## Key Features
- **Dataset Preprocessing**: Combines cleaned datasets with Python code datasets to prepare robust training data.
- **Advanced Data Cleaning**: Filters for short texts, removes repetitions, deduplicates entries, and applies language-based quality checks.
- **Model Upscaling**: Adapts pre-trained smaller models (e.g., 12-layer models) to larger architectures (e.g., 16-layer models) by leveraging existing layers.
- **Efficient Training Setup**: Implements memory-efficient dataset sharding, tokenization, and training configurations.
- **Hugging Face Integration**: Seamlessly integrates with Hugging Face tools for tokenizer setup, dataset management, and model optimization.

---

## Workflow Overview

### 1. Data Preparation
- **Datasets Used**: 
  - Subset of the Red Pajama dataset (1 trillion tokens).
  - Python script datasets scraped from GitHub repositories.
- **Preprocessing Includes**:
  - Data cleaning for quality assurance.
  - Filtering short or repetitive text.
  - Deduplication of entries.
  - Inclusion of only English-language documents.

### 2. Dataset Tokenization
- Input text is tokenized using pre-trained Hugging Face tokenizers.
- `bos` and `eos` tokens are added to each example.
- Packaged into fixed sequence lengths for training (e.g., 32 tokens).

### 3. Model Upscaling
- Initial Model: `TinySolar-248m-4k` pre-trained LLaMA-based model.
- Target Model: 16-layer configuration derived by:
  - Merging and copying specific layers from the smaller pretrained model.
  - Retaining embedding and classifier layers.

### 4. Training
- Training involves efficient gradient accumulation and memory optimization:
  - **Optimizations**: bfloat16 precision, gradient checkpointing.
  - **Logging**: Custom callbacks log training metrics.
- Final checkpoints are saved for downstream use.

---

## Setup and Installation

### Prerequisites
Ensure the following are installed:
- Python 3.9+
- Required Python dependencies:
  ```bash
  pip install numpy pandas torch transformers datasets fasttext
  ```

### Steps to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/ivanluk914/Pretraining-with-Hugging-Face-Transformers.git
   ```
2. Preprocess datasets by running the Jupyter Notebook file `pretraining.ipynb`.
3. Download datasets (e.g., Red Pajama) or use provided Python scraping scripts to gather data.
4. Train the model using customized Hugging Face Trainer settings.

---

## Results
- Preprocessed and tokenized datasets are saved in Parquet format.
- A pretrained LLaMA-like model is upscaled efficiently.
- Continuous pretraining results in improved model size and performance.

---

## Contributing
If you'd like to contribute, feel free to open an issue or create a pull request.

---

## References
- [Red Pajama Dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- Hugging Face Documentation:
  - [Datasets](https://huggingface.co/docs/datasets)
  - [Transformers](https://huggingface.co/docs/transformers)
