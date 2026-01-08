"""
Training Script for Point-Supervised Segmentation
Implements training loop with MLflow tracking, early stopping, and checkpointing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from loss import get_loss_function
from metrics import SegmentationMetrics
from model import get_model
from data_loader import get_dataloaders
from point_sampling import PointSampler
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    """
    Trainer class for point-supervised segmentation.

    Handles:
    - Training loop with mixed precision
    - Validation and metric tracking
    - MLflow experiment logging
    - Model checkpointing
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        config: dict,
        experiment_dir: Path,
        use_mlflow: bool = True,
    ):
        """
        Args:
            model: Segmentation model
            loss_fn: Loss function
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            config: Configuration dictionary
            experiment_dir: Directory to save experiment results
            use_mlflow: Whether to use MLflow tracking
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.experiment_dir = experiment_dir
        self.use_mlflow = use_mlflow

        # Create metrics tracker
        self.metrics = SegmentationMetrics(
            num_classes=config["num_classes"],
            class_names=config["class_names"],
            ignore_index=config.get("ignore_index", -1),
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "learning_rate": [],
        }

        # Best model tracking
        self.best_val_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # Mixed precision training
        self.use_amp = config.get("mixed_precision", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "reduce_on_plateau")

        if scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.get("scheduler_factor", 0.5),
                patience=self.config.get("scheduler_patience", 5),
                # verbose=True,
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]",
            leave=False,
        )

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)

                    # Compute loss
                    if hasattr(self.loss_fn, "forward") and "consistency" in str(
                        type(self.loss_fn)
                    ):
                        loss, loss_dict = self.loss_fn(predictions, masks)
                    else:
                        loss = self.loss_fn(predictions, masks)
                        loss_dict = {"total_loss": loss.item()}

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions = self.model(images)

                if hasattr(self.loss_fn, "forward") and "consistency" in str(
                    type(self.loss_fn)
                ):
                    loss, loss_dict = self.loss_fn(predictions, masks)
                else:
                    loss = self.loss_fn(predictions, masks)
                    loss_dict = {"total_loss": loss.item()}

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config['num_epochs']} [Val]",
            leave=False,
        )

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            if hasattr(self.loss_fn, "forward") and "consistency" in str(
                type(self.loss_fn)
            ):
                loss, _ = self.loss_fn(predictions, masks)
            else:
                loss = self.loss_fn(predictions, masks)

            total_loss += loss.item()

            # Update metrics
            self.metrics.update(predictions, masks)

        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute_metrics()
        metrics["val_loss"] = avg_loss

        return metrics

    def train(self):
        """
        Main training loop.
        """
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"{'='*80}\n")

        # Initialize MLflow
        if self.use_mlflow:
            mlflow.start_run(run_name=self.config.get("experiment_name", "experiment"))
            mlflow.log_params(self.config)

        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_iou"].append(val_metrics["mean_iou"])
            self.history["val_f1"].append(val_metrics["mean_f1"])
            self.history["learning_rate"].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Val F1: {val_metrics['mean_f1']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["val_loss"],
                        "val_iou": val_metrics["mean_iou"],
                        "val_f1": val_metrics["mean_f1"],
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mean_iou"])
                else:
                    self.scheduler.step()

            # Save best model
            if val_metrics["mean_iou"] > self.best_val_iou:
                self.best_val_iou = val_metrics["mean_iou"]
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ New best model! IoU: {self.best_val_iou:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            early_stopping_patience = self.config.get("early_stopping_patience", 10)
            if self.patience_counter >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
                print(f"{'='*80}\n")
                break

            # Save checkpoint every N epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        # Save final results
        self.save_results()

        # End MLflow run
        if self.use_mlflow:
            mlflow.end_run()

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
        print(f"{'='*80}\n")

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "history": self.history,
        }

        if is_best:
            path = self.experiment_dir / "best_model.pth"
        else:
            path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save(checkpoint, path)

        # Log to MLflow
        if self.use_mlflow and is_best:
            mlflow.pytorch.log_model(self.model, "best_model")

    def save_results(self):
        """Save training results and plots."""
        # Save history
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Plot training curves
        self.plot_training_curves()

        # Plot confusion matrix
        self.plot_confusion_matrix()

        # Save final metrics
        final_metrics = {
            "best_val_iou": self.best_val_iou,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
        }

        metrics_path = self.experiment_dir / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(epochs, self.history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # IoU curve
        axes[0, 1].plot(epochs, self.history["val_iou"], label="Val IoU", color="green")
        axes[0, 1].axhline(
            y=self.best_val_iou,
            color="r",
            linestyle="--",
            label=f"Best: {self.best_val_iou:.4f}",
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("IoU")
        axes[0, 1].set_title("Validation IoU")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # F1 curve
        axes[1, 0].plot(epochs, self.history["val_f1"], label="Val F1", color="orange")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("Validation F1 Score")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Learning rate
        axes[1, 1].plot(
            epochs, self.history["learning_rate"], label="Learning Rate", color="purple"
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        save_path = self.experiment_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_artifact(str(save_path))

    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        fig = self.metrics.plot_confusion_matrix(
            save_path=self.experiment_dir / "confusion_matrix.png",
            normalize=True,
        )
        plt.close(fig)

        # Also save non-normalized version
        fig = self.metrics.plot_confusion_matrix(
            save_path=self.experiment_dir / "confusion_matrix_raw.png",
            normalize=False,
        )
        plt.close(fig)

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_artifact(str(self.experiment_dir / "confusion_matrix.png"))


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
