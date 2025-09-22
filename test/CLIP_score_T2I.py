import os
import torch
import clip
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_T2I_clip_score(image_folder: str, csv_file: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    df = pd.read_csv(csv_file)
    total_score = 0
    count = 0
    for index, row in df.iterrows():
        image_filename = str(row['id']) + '.png'  # 将 id 转换为字符串
        image_path = os.path.join(image_folder, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_filename} not found.")
            continue
        
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(str([row['desc']])).to(device)
        
        with torch.no_grad():
            
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
       
        score = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())[0, 0]
        total_score += score
        count += 1

    average_score = total_score / count if count > 0 else 0
    return average_score

