# U-Net Segmentation Project

This repository contains a PyTorch implementation of the U-Net architecture for image segmentation tasks, particularly focused on grayscale medical or biological images with corresponding binary masks.

## 📁 Project Structure

```
.
├── model.py              # U-Net architecture implementation
├── train.py              # Training script
├── utils.py              # Utility functions (checkpointing, evaluation, saving predictions)
├── dataset.py            # Custom dataset class using torch.utils.data.Dataset
├── imageResizing.py      # Resizes input images to 256x256
├── data/                 # Folder with training/validation images and masks
├── saved_images/         # Saved predictions after validation
```

## 🚀 Getting Started

### 1. Prepare Dataset

Organize your dataset as follows:

```
data/
├── train_images/
├── train_masks/
├── val_images/
├── val_masks/
```

Images should be in `.jpg` format. Masks should be in `.png` format with a corresponding `_mask` suffix (e.g., `img1.jpg` -> `img1_mask.png`).

### 2. Resize Images (Optional)

If needed, run the `imageResizing.py` script to resize all images to 256x256.

### 3. Train the Model

Run:

```bash
python train.py
```

You can configure hyperparameters inside `train.py`, including:

* Learning rate
* Batch size
* Epoch count

### 4. Evaluation

During training, the script will:

* Save model checkpoints to `my_checkpoint.pth.tar`
* Evaluate accuracy and Dice score
* Save predicted masks to `saved_images/`

## 🧠 Model Architecture

The model is based on U-Net and uses:

* Encoder/Decoder with double convolution blocks
* Skip connections
* Batch normalization and ReLU activations
* `BCEWithLogitsLoss` for binary segmentation
