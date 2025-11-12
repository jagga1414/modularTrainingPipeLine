# Modular Training Pipeline for LLM Fine-tuning

A modular training pipeline for fine-tuning large language models (LLMs) on custom datasets. This project organizes the training code from a Jupyter notebook into a clean, modular Python structure.

## Project Structure

```
modularTrainingPipeLine1/
├── dataset/
│   └── dataset.py          # Dataset classes (MyDataset, MyCollate, Batch_D)
├── model/
│   └── model_loader.py     # Model and tokenizer loading functions
├── train/
│   └── trainer.py          # Training loop functions
├── scripts/
│   ├── train.py           # Main training script
│   └── demo.py            # Inference/demo script
└── pesto/
    └── pesto.ipynb        # Original notebook
```

## Modules

### `dataset/dataset.py`
- **`Batch_D`**: Container class for batch data and configurations
- **`MyDataset`**: PyTorch Dataset that loads CSV data and applies chat templates
- **`MyCollate`**: Collate function for batching and tokenizing text

### `model/model_loader.py`
- **`setup_tokenizer()`**: Loads and configures tokenizer with special tokens and chat template
- **`load_model()`**: Loads the microsoft/phi-1_5 model from HuggingFace

### `train/trainer.py`
- **`train_epoch()`**: Runs one training epoch
- **`train()`**: Main training loop with optimizer setup

## Usage

### Training

```bash
cd scripts
python train.py
```

This will:
1. Load the phi-1_5 model and tokenizer
2. Load training data from `Customer-Support.csv`
3. Train for 4 epochs with batch size 4
4. Save the trained model to `model.pt`

### Inference

```bash
cd scripts
python demo.py
```

This will load the trained model and generate a response for a test query.

## Requirements

```bash
pip install torch transformers pandas
```

## Data Format

Training data should be a CSV file with two columns:
- `query`: User questions
- `response`: Assistant responses

## Configuration

Default settings in `scripts/train.py`:
- Model: microsoft/phi-1_5
- Epochs: 4
- Batch size: 4
- Learning rate: 0.00001
- Device: cpu

You can modify these values directly in the script files.
