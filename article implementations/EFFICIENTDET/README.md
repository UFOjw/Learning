# EfficientDet Object Detection Project

This project is a PyTorch-based implementation of EfficientDet, an object detection model that combines the EfficientNet backbone with a BiFPN feature network and focal loss. It supports training on COCO-format datasets, evaluation on videos and images, and export of predictions.

---

## 📁 Project Structure

```
├── train.py              # Training script
├── test_dataset.py       # Evaluation on image dataset
├── main.py               # Evaluation on videos
├── model.py              # Model architecture
├── loss.py               # Focal loss definition
├── dataset.py            # COCO dataset handling
├── utils.py              # Anchors, box transformation
├── config.py             # Class labels and colors
```

---

## 🚀 Getting Started

### 1. Train the model

```bash
python train.py --data_path coco2017 --num_epochs 10
```

Outputs:

* Trained model in `trained_models/`
* TensorBoard logs in `tensorboard/`

### 2. Evaluate on validation images

```bash
python test_dataset.py --data_path coco2017 --output predictions/
```

### 3. Evaluate on a video

```bash
python main.py --input test_videos/14.mp4 --output test_videos/output.mp4
```

---

## ⚙️ Configuration

Key arguments:

* `--image_size`: Resize all input images to this size (default: 512)
* `--batch_size`: Number of images per batch (default: 4)
* `--cls_threshold`: Score threshold to filter weak detections (default: 0.5)
* `--nms_threshold`: IoU threshold for Non-Maximum Suppression (default: 0.5)

---

## 🧠 Model Details

* **Backbone**: EfficientNet-B0
* **Neck**: Bi-directional Feature Pyramid Network (BiFPN)
* **Head**: Regressor + Classifier
* **Loss**: Focal Loss for classification and smooth L1 for regression

---

---
