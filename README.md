Circle Detection Challenge (Regression-based Deep Learning)

## 1. Challenge Description

The goal of this challenge is to build a model that predicts the location and radius of a circle embedded in a noisy grayscale image. The task is formulated as a regression problem: given an image, the model must predict the coordinates (x, y) of the circle's center and its radius.

The challenge emphasizes robustness to noise and encourages simple, interpretable solutions. Evaluation is based on how well the predicted circle overlaps with the ground-truth circle using Intersection over Union (IoU), both as a raw metric and using threshold-based accuracy (e.g., % of predictions with IoU ≥ 0.75).

## 2. Requirements

This project is written in Python 3.8.

To install the required libraries:

```pip install torch numpy matplotlib scikit-image tqdm```

All code is self-contained and runs on CPU or GPU (via PyTorch). GPU usage is automatically detected with torch.cuda.

#### Running the Code

python src/main.py

This will, load the config, train the model, save logs and plots, and evaluate on the test set

## 3. Repository Structure

The `src/` directory contains all the code components:

- `src/config.py`: Configuration parser and validator using dataclasses.
- `src/config_benchmarking.json`: Main training configuration.
- `src/config_debugging.json`: Lightweight configuration for fast debugging.
- `src/config_robust.json`: Configuration with stronger noise for robustness testing.
- `src/dataloader.py`: Defines PyTorch datasets and loaders for training, validation, and testing.
- `src/metrics.py`: Implements IoU computation (both NumPy and vectorized PyTorch versions).
- `src/model.py`: Defines the `CircleRegression` CNN architecture.
- `src/tools.py`: Contains core geometric functions like circle drawing and noise addition.
- `src/trainers.py`: Implements the `CircleTrainer` class for training, validation, and early stopping.
- `main.py`: Entry point script that loads a config file and starts training.


## 4. Approach to the Challenge

#### Architecture

A lightweight CNN (CircleRegression) is used with three convolutional blocks followed by a fully connected head 
for regressing (row, col, radius). GELU activations are applied, and custom weight initialization is used (Kaiming for GELU).

#### Regression Objective

Training uses Smooth L1 Loss to predict circle parameters. 
Validation and test performance is evaluated via IoU (intersection-over-union) between predicted and true circles.

A vectorized PyTorch version of the IoU metric is used for speed and GPU compatibility.

## 5. Configurations

There are three configuration files:

1. config_benchmarking.json: main config (default)
2. config_debugging.json: quick runs for debugging
3. config_robust.json: increased noise to test robustness

They control image properties, model architecture, optimizer settings, and training duration.
To use a config, change the file in main.py:

config = Config.from_json("src/config_benchmarking.json")

#### Output & Logging

The trainers.py module creates:

1. A saved/logs/ folder with loss and IoU logs per epoch.
2. A saved/checkpoints/ folder with saved models (only 3 kept).

A plot saved in logs with:
1. Train and validation loss
2. Validation IoU
3. Final test IoU

Early stopping is based on validation IoU trends.

## 6. Project Notes & Reflections

#### Initial Steps

At first, I explored non-deep-learning approaches like clustering with DBSCAN, 
hoping to isolate circle edges from noisy images. Even at low noise (0.3), performance was poor and inconsistent.

#### Chosen Path

I shifted to a direct regression approach using CNNs, 
which quickly outperformed the traditional methods. 
I considered separating the denoising and regression steps with a Denoising Autoencoder (DAE), 
but focused on end-to-end learning first. The DAE idea could still be valuable—either as a pre-module 
or integrated via intermediate loss—but would require tuning multiple hyperparameters.

#### On Metrics

I discussed pixel-based IoU vs. circle-parameter IoU. 
The former is more precise in segmentation contexts, 
but for this regression task, parameterized IoU (as implemented) is more intuitive and appropriate.

#### If I Had More Time...

Threshold-based Metric: Implemented fully in training for easier selection of best-performing checkpoints.

Noise Diversity: Vary noise distributions and types (e.g. salt-and-pepper, blur). A DAE could help here.


