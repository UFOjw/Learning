import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import torch.optim as optim
from tqdm.auto import tqdm
from model import InferenceModel
from dataloader import RSNADataLoader
from utils import plot_loss
import time

train_path = 'data/train_images'
train_csv = 'data/train.csv'
train_series = 'data/train_series_descriptions.csv'
train_labels = 'data/train_label_coordinates.csv'

train_labels_df = pd.read_csv(train_labels)
train_series_df = pd.read_csv(train_series)
train_df = pd.read_csv(train_csv)

cond1_list = ["Right Neural Foraminal Narrowing", "Left Neural Foraminal Narrowing"]
cond2_list = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
foraminal_data = train_df[['study_id', 'left_neural_foraminal_narrowing_l1_l2',
                           'left_neural_foraminal_narrowing_l2_l3',
                           'left_neural_foraminal_narrowing_l3_l4',
                           'left_neural_foraminal_narrowing_l4_l5',
                           'left_neural_foraminal_narrowing_l5_s1',
                           'right_neural_foraminal_narrowing_l1_l2',
                           'right_neural_foraminal_narrowing_l2_l3',
                           'right_neural_foraminal_narrowing_l3_l4',
                           'right_neural_foraminal_narrowing_l4_l5',
                           'right_neural_foraminal_narrowing_l5_s1']]
mapping = {'Not defined': 0, 'Normal/Mild': 1, 'Moderate': 2, 'Severe': 3}

used_samples = train_labels_df[['study_id', 'series_id', 'instance_number']].drop_duplicates().reset_index(drop=True)
saggital_plane = used_samples.merge(train_series_df[train_series_df['series_description'] == 'Sagittal T1'], how='inner', on=['study_id', 'series_id'])

labels = saggital_plane[['study_id', 'series_id', 'instance_number']]
for i1, cond1 in enumerate(cond1_list):
    for i2, cond2 in enumerate(cond2_list):
        temp = train_labels_df[(train_labels_df['condition'] == cond1) & (train_labels_df['level'] == cond2)][['study_id', 'series_id', 'instance_number']]
        temp = temp.merge(foraminal_data.iloc[:, [0, 1 + i1 * 5 + i2]], how='left', on='study_id')
        labels = labels.merge(temp, how='left', on=['study_id', 'series_id', 'instance_number'])

labels = labels.fillna("Not defined").replace(mapping)


study = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]

series = []
for fold in study:
    path = os.path.join(train_path, fold)
    series.append([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

prompt = 'An MRI sagittal view of the lumbar spine'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = InferenceModel(num_inference_steps=50,
                       input_dim=4,
                       hidden1=10,
                       kernel=3,
                       hidden2=10,
                       linear_output_dim=4,
                       pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5"),
                       scheduler=DDIMScheduler,
                       device=device)

train_steps = 10
probabilities = {0: 0., 1: 0.4, 2: 0.3, 3: 0.3}

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
loader = RSNADataLoader(labels, train_path)

losses = []
for _ in tqdm(range(train_steps)):
    optimizer.zero_grad()
    status = np.random.choice(4, p=[p for _, p in probabilities.items()])

    gt_proba = torch.zeros(4).to(device)
    gt_proba[status] = 1

    args = (status, 'left_neural_foraminal_narrowing_l4_l5')
    sample = loader[args]
    image = sample['image']

    image = Image.fromarray(image).convert("RGB").resize((512, 512))
    image = (torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

    predict = model(image)

    loss = criterion(predict, gt_proba)
    losses.append(loss.item())
    plot_loss(losses)
    time.sleep(0.1)
    loss.backward()
    optimizer.step()

