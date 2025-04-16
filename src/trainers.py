"""
This module defines the CircleTrainer class which is responsible for training, validating,
and evaluating a regression model that predicts circle parameters from noisy images.

It includes logic for training loop, early stopping, validation, and evaluation using IoU metrics.
The training procedure saves checkpoints, logs metrics, and visualizes the loss and performance.
"""

# Python modules
import os
from pathlib import Path
from typing import Callable

# Libraries
import matplotlib.pyplot as plt
from tqdm import tqdm

# DL libraries
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Local modules
from src.dataloader import get_datasets
from src.metrics import iou_torch
from src.model import CircleRegression


class CircleTrainer:
    """
    Trainer class to handle model training, evaluation, checkpointing, and visualization.
    Accepts configuration and model setup, uses validation loss and IoU for early stopping.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.every = 5  # frequency of validation/checkpointing

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.train_loader, self.val_loader, self.test_loader = get_datasets(config)

        self.model = CircleRegression(
            input_shape=config.input_shape,
            channels=config.channels,
            kernels=config.kernels,
            pools=config.pools,
            strides=config.strides,
            FFN_dims=config.FFN_dims
        ).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.checkpoints = []
        self.best_val_iou = -1.0

        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.test_iou = None

    def _create_optimizer(self):
        """Creates the optimizer from configuration."""
        opt_type = self.config.optimizer_config.type.lower()
        params = self.config.optimizer_config.params
        if opt_type == 'sgd':
            return SGD(self.model.parameters(), **params)
        elif opt_type == 'adam':
            return Adam(self.model.parameters(), **params)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _create_scheduler(self):
        """Creates the learning rate scheduler from configuration."""
        sch_type = self.config.scheduler_config.type.lower()
        params = self.config.scheduler_config.params
        if sch_type == 'steplr':
            return StepLR(self.optimizer, **params)
        elif sch_type == 'reducelronplateau':
            return ReduceLROnPlateau(self.optimizer, **params)
        else:
            raise ValueError(f"Unsupported scheduler: {sch_type}")

    def train(self):
        """
        Main training loop. Performs training, validation, logging, early stopping,
        and saves the best model checkpoint.
        """
        max_epochs = self.config.data_config.num_epochs

        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0

            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            if epoch % self.every == 0:
                self._log_loss(avg_loss, epoch, phase='train')
                val_loss, val_iou = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.val_ious.append(val_iou)
                self._check_early_stop(val_iou, epoch)
                if self._early_stop_condition():
                    print("Early stopping triggered.")
                    break

        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{self.config.model_name}_final.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Load best model from top 3 checkpoints and evaluate
        best_ckpt = max(self.checkpoints[-3:], key=lambda x: x[1])  # x[1] is val_iou
        self.model.load_state_dict(torch.load(best_ckpt[2]))
        self.test_iou = self.evaluate(metric_fn=iou_torch)
        self._plot_metrics()
        self._compute_threshold_accuracy()

    def validate(self, epoch: int) -> tuple[float, float]:
        """Evaluate model on validation set and log metrics."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                ious = iou_torch(outputs, targets)
                total_iou += ious.sum().item()
                total_samples += outputs.size(0)

        avg_val_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / total_samples
        self._log_loss(avg_val_loss, epoch, phase='val')
        self._log_iou(avg_iou, epoch, phase='val')
        return avg_val_loss, avg_iou

    def evaluate(self, metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        """Evaluate final model on test set using IoU metric."""
        self.model.eval()
        total_score = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, targets in self.test_loader:
                outputs = self.model(images)
                score = metric_fn(outputs, targets)
                total_score += score.sum().item()
                total_samples += outputs.size(0)

        avg_score = total_score / total_samples
        print(f"Test IoU Score: {avg_score:.4f}")
        return avg_score

    def _compute_threshold_accuracy(self):
        """Computes thresholded IoU accuracy over test set (e.g., ≥ 0.5, ≥ 0.75...)."""
        self.model.eval()
        thresholds = self.config.thresholds
        counts = {th: 0 for th in thresholds}
        total = 0

        with torch.no_grad():
            for images, targets in self.test_loader:
                outputs = self.model(images)
                ious = iou_torch(outputs, targets)
                for th in thresholds:
                    counts[th] += (ious >= th).sum().item()
                total += outputs.size(0)

        print("\nThreshold Accuracy on Test Set:")
        for th in thresholds:
            acc = 100.0 * counts[th] / total
            print(f"IoU ≥ {th:.2f}: {acc:.2f}%")

    def _log_loss(self, loss: float, epoch: int, phase: str):
        """Saves training or validation loss to log file."""
        log_path = Path(self.config.log_dir) / f"{self.config.model_name}_{phase}_loss.txt"
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {loss:.6f}\n")

    def _log_iou(self, iou_score: float, epoch: int, phase: str):
        """Saves validation IoU to log file."""
        log_path = Path(self.config.log_dir) / f"{self.config.model_name}_{phase}_iou.txt"
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch}, IoU: {iou_score:.6f}\n")

    def _check_early_stop(self, val_iou: float, epoch: int):
        """Save checkpoint and prune old ones."""
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"{self.config.model_name}_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        self.checkpoints.append((epoch, val_iou, ckpt_path))

        if len(self.checkpoints) > 3:
            old_epoch, _, old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def _early_stop_condition(self):
        """
        Determines whether to trigger early stopping based on validation IoU and loss.
        Stops if:
        - IoU has dropped in the last 3 evaluations, or
        - Validation loss has increased, or
        - Metrics are flat (no significant change)
        """
        if len(self.val_ious) < 3 or len(self.train_losses) < 3:
            return False

        val_iou1, val_iou2, val_iou3 = self.val_ious[-3:]
        val_loss1, val_loss2, val_loss3 = self.val_losses[-3:]

        iou_decreasing = val_iou1 > val_iou2 > val_iou3
        val_loss_increasing = val_loss1 < val_loss2 < val_loss3
        flat = max(abs(val_iou1 - val_iou2), abs(val_iou3 - val_iou2)) < 0.005

        return iou_decreasing or val_loss_increasing or flat

    def _plot_metrics(self):
        """Visualizes training loss, validation loss/IoU, and test IoU."""
        plt.figure(figsize=(16, 5))

        # Plot the Losses
        plt.subplot(1, 3, 1)
        val_epochs = list(range(0, len(self.train_losses), self.every))[:len(self.val_losses)]
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(val_epochs, self.val_losses, label='Val Loss')
        plt.xlabel(f'Validation Epoch (every {self.every})')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        # Plot the validations metrics
        plt.subplot(1, 3, 2)
        plt.plot(val_epochs, self.val_ious, label='Val IoU', color='green')
        plt.xlabel(f'Validation Epoch (every {self.every})')
        plt.ylabel('IoU')
        plt.title('Validation IoU')
        plt.legend()

        # Plot the test metrics
        plt.subplot(1, 3, 3)
        if self.test_iou is not None:
            plt.plot([0], [self.test_iou], label='Test IoU', color='purple', marker='o')
        plt.xlabel('Run')
        plt.ylabel('IoU')
        plt.title('Test IoU')
        plt.legend()

        plt.tight_layout()
        plt.savefig(Path(self.config.log_dir) / f"{self.config.model_name}_metrics.png")
        plt.close()
