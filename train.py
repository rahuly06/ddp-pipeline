from pathlib import Path
import mlflow
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import time
import argparse
import os
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

PROJECT_DIR = Path(".").resolve()
ARTIFACT_DIR = PROJECT_DIR / "mlartifacts"


# CREATE DATASETS   
def create_datasets():
    data_statictics = ((0.5,0.5,0.5), (0.5,0.5,0.5))

    transformed_train_images = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(*data_statictics, inplace=True), 
    ])

    transformed_test_images = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*data_statictics, inplace=True), 
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="data/downloads", 
        download=True,
        train=True,
        transform=transformed_train_images
    )

    total_size = len(train_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size 

    train_split, val_split = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root="data/downloads",
        train=True,
        download=False,
        transform=transformed_test_images,
    )

    val_split = torch.utils.data.Subset(
        val_dataset,
        val_split.indices,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/downloads", 
        download=True,
        train=False,
        transform=transformed_test_images
    )

    return train_split, val_split, test_dataset


def create_dataloaders(batch_size):
    train_split, val_split, test_dataset = create_datasets()
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_split,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_split,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader


# DEVICE SELECTION
def get_device():
    if torch.cuda.is_available():
        print("Cuda is Available!!", torch.cuda.get_device_name())
        return torch.device("cuda:0")
    else:
        raise RuntimeError("CUDA is required for this GPU training run.")

def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)

    return (
        device,
        local_rank,
        dist.get_rank(),
        dist.get_world_size(),
    )

# CREATE MODEL
def create_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Optimize for 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model = model.to(device)
    return model


# CREATE OPTIMIZERS AND LOSS CRITERIA
def opt_cri(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def parameters(batch_size, device, epochs):
    params = {
        "model": "resnet18",
        "dataset": "CIFAR-10",
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "epochs": epochs,
        "optimizer": "Adam",
        "model_type": "resnet18",
        "gpu": torch.cuda.get_device_name(device),
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }
    return params


def evaluate_test(model, test_loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += n
    return {
        "test_loss": loss_sum / total,
        "test_accuracy_percent": 100 * correct / total,
        "test_samples": total,
    }


# TRAINING LOOP WITH MLFLOW
def main(batch_size, epochs):
    device, local_rank, rank, world_size = setup_ddp()
    is_main = rank == 0

    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Rank 0 downloads data and pretrained weights first.
        # This assumes one machine with a shared filesystem/cache.
        if is_main:
            datasets = create_datasets()
            models.ResNet18_Weights.DEFAULT.get_state_dict(
                progress=True
            )

        dist.barrier()

        if not is_main:
            datasets = create_datasets()

        train_split, val_split, test_dataset = datasets

        train_sampler = DistributedSampler(
            train_split,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )

        train_loader = torch.utils.data.DataLoader(
            train_split,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
        )

        # Only rank 0 evaluates the complete validation set.
        val_loader = None
        if is_main:
            val_loader = torch.utils.data.DataLoader(
                val_split,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

        model = create_model(device)
        model = DDP(model, device_ids=[local_rank])

        optimizer, criterion = opt_cri(
            model,
            learning_rate=1e-4,
        )

        if is_main:
            mlflow.set_tracking_uri(
                f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
            )
            mlflow.set_experiment("CIFAR-10_DDP")

        # Other ranks use an empty context instead of opening runs.
        run_context = (
            mlflow.start_run(
                run_name=f"ddp-{world_size}gpu-batch{batch_size}",
                log_system_metrics=True,
            )
            if is_main
            else nullcontext()
        )

        with run_context as run:
            best_val_loss = float("inf")

            if is_main:
                params = parameters(
                    batch_size=batch_size,
                    device=device,
                    epochs=epochs,
                )
                params.update({
                    "world_size": world_size,
                    "per_gpu_batch_size": batch_size,
                    "global_batch_size": batch_size * world_size,
                    "split_seed": 42,
                    "validation_strategy": "rank0_full_dataset",
                })
                mlflow.log_params(params)

                checkpoint_dir = PROJECT_DIR / "models" / run.info.run_id
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "best.pth"

            for epoch in range(epochs):
                # Different shuffle each epoch, coordinated across ranks.
                train_sampler.set_epoch(epoch)

                dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                epoch_start = time.perf_counter()

                model.train()

                # Accumulate loss sum and example count on the GPU.
                train_stats = torch.zeros(
                    2, dtype=torch.float64, device=device
                )

                for images, labels in train_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # DDP synchronizes gradients here.
                    loss.backward()
                    optimizer.step()

                    n = labels.size(0)
                    train_stats[0] += loss.detach().double() * n
                    train_stats[1] += n

                # DDP synchronizes gradients, not your logged metrics.
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
                avg_train_loss = (
                    train_stats[0] / train_stats[1]
                ).item()

                if is_main:
                    # Use the underlying model for rank-0-only validation.
                    eval_model = model.module
                    eval_model.eval()

                    val_loss_sum = 0.0
                    val_count = 0

                    with torch.no_grad():
                        for images, labels in val_loader:
                            images = images.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                            outputs = eval_model(images)
                            loss = criterion(outputs, labels)

                            n = labels.size(0)
                            val_loss_sum += loss.item() * n
                            val_count += n

                    avg_val_loss = val_loss_sum / val_count

                # Everyone waits until validation finishes.
                dist.barrier()
                torch.cuda.synchronize(device)
                epoch_seconds = time.perf_counter() - epoch_start

                if is_main:
                    mlflow.log_metrics({
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "train_val_seconds": epoch_seconds,
                        "rank0_peak_tensor_memory_mib":
                            torch.cuda.max_memory_allocated(device)
                            / 1024**2,
                    }, step=epoch + 1)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss

                        # Save without DDP's wrapper.
                        torch.save(
                            model.module.state_dict(),
                            checkpoint_path,
                        )

                    print(
                        f"Epoch {epoch + 1}/{epochs} "
                        f"| Train: {avg_train_loss:.4f} "
                        f"| Val: {avg_val_loss:.4f} "
                        f"| Time: {epoch_seconds:.2f}s"
                    )

                # Rank 0 finishes logging/saving before the next epoch.
                dist.barrier()

            if is_main:
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_artifact(
                    str(checkpoint_path),
                    artifact_path="best_checkpoint",
                )

                # Test once after all epochs using the best saved checkpoint.
                test_model = model.module
                test_model.load_state_dict(
                    torch.load(
                        checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    )
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                )
                test_metrics = evaluate_test(
                    test_model, test_loader, criterion, device
                )
                mlflow.log_metrics(test_metrics)
                mlflow.set_tag("test_checkpoint", "best_checkpoint/best.pth")
                print("Best checkpoint test results:", test_metrics)

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on CIFAR-10"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training and validation batch size per GPU (default: 16)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Define the Epochs for Training (defaulf: 30)"
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")

    main(batch_size=args.batch_size, epochs=args.epochs)
