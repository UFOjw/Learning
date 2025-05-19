# Diffusion Model Project

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation using PyTorch. It is structured for modularity and extensibility, including utilities for data loading, training visualization, and model architecture customization (notably, a UNet with self-attention layers).

## Overview

This repository provides a PyTorch implementation of DDPMs, capable of training on custom image datasets. The core model architecture is a UNet with integrated self-attention layers to improve generation quality.

## Core Components

* **UNet**: The main model, implemented in `modules.py` with support for self-attention and time embedding.
* **SelfAttention**: Improves expressivity for global context, also in `modules.py`.
* **Utils**: Data pipeline, plotting, and logging helpers (`utils.py`).
* **Notebook**: The workflow for training, evaluating, and sampling images (`DDPM.ipynb`)
