# aesthetic_score.py

import os
import torch
import clip
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def compute_aesthetic_scores(image_folder, mlp_weights_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize MLP model with pre-trained weights
    mlp_model = MLP(768)
    mlp_state_dict = torch.load(mlp_weights_path)
    mlp_model.load_state_dict(mlp_state_dict)
    mlp_model.to(device)
    mlp_model.eval()

    # Initialize CLIP model
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    scores = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            pil_image = Image.open(img_path)
            image = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image)

            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = mlp_model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

            scores.append(prediction.item())

    if scores:
        average_score = sum(scores) / len(scores)
        return average_score
    else:
        return None
