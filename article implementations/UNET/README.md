# Segmentation

The repository contains:
* Implementation of the UNET model (`model.py`);
* File for training `train.py`;
* Two auxiliary files

The model is classical except for the addition of paddings in convolutional layers of size 1 to prevent image reduction. As well as resizing when concatenating via skip_connections to match.

The code was borrowed [from](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet)

[Paper](https://arxiv.org/abs/1301.3781v3)
