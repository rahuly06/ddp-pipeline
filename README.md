# Distributed GPU Training Pipeline

A learning project for training ResNet18 on CIFAR-10 with PyTorch, CUDA, DistributedDataParallel (DDP), and MLflow.

## Features

- ImageNet-pretrained ResNet18 adapted for 32x32 images and 10 classes.
- Fixed training/validation split with training-only augmentation.
- One training process per GPU, launched with torchrun.
- Validation and checkpoint saving on rank 0.
- Test evaluation once after all epochs, using the best validation checkpoint.
- MLflow logging for parameters, losses, timing, GPU resource metrics, and test results.

## Setup

Use Linux or WSL with NVIDIA GPU drivers and a compatible CUDA-enabled PyTorch installation. Run all commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies are not version-pinned yet. Choose a PyTorch/torchvision build compatible with the host driver; the default installation may require a newer driver. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Check GPU access before training:

```bash
nvidia-smi -L
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count())"
```

CIFAR-10 and pretrained weights are downloaded automatically on the first run.

## Train

Single GPU:

```bash
torchrun --standalone --nproc_per_node=1 train.py --batch-size 16 --epochs 2
```

Two GPUs on one machine:

```bash
torchrun --standalone --nproc_per_node=2 train.py --batch-size 8 --epochs 2
```

Batch size is per GPU: two GPUs with batch size 8 give a global training batch size of 16. Defaults are batch size 16 and 30 epochs. Learning rate is currently fixed at 0.0001.

The script requires torchrun, even with one GPU. It assumes a single machine with a shared dataset and pretrained-weight cache. Multi-machine deployment needs additional setup.

## Results and MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000, select **Model training**, then **CIFAR-10_DDP**.

- Best local weights: `models/<run_id>/best.pth`.
- Best weights attached to the run: `best_checkpoint/best.pth`.
- Test metrics: `test_loss`, `test_accuracy_percent`, and `test_samples`.
- Epoch timing includes training and rank-0 validation; peak tensor memory is for rank 0.
- GPU system monitoring requires working NVIDIA monitoring support.

The database and artifact files are stored separately. Preserve both when moving results off a rented machine. The current script does not generate a confusion matrix.

## Project files

- `train.py`: DDP training, validation, checkpointing, and final testing.
- `notebooks/`: earlier single-GPU exploration.
- `requirements.txt`: Python dependencies.

## Status

Single-worker DDP and MLflow logging have been verified locally. Two-GPU benchmarking, Docker packaging, and cloud deployment are the next milestones. Saved checkpoints contain model weights for evaluation; optimizer-state recovery is not implemented.
