Detection model based on EfficientNet backbone.

Paper: https://arxiv.org/pdf/1911.09070

Implementation uses trimmed COCO dataset with adding one class (apples with different defects).
Model formed on D1 size using FocalLoss, AdamW optimizer, and ReduceLROnPlateau for scaling learning rate.
