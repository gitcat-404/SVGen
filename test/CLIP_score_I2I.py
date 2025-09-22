import os
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def calculate_I2I_clip_score(folder1, folder2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用 clip 库加载模型
    model, preprocess = clip.load("ViT-L/14", device=device)

    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    common_files = set(files1).intersection(set(files2))
    
    if not common_files:
        raise ValueError("no such file")
    
    total_score = 0
    count = 0
    
    def get_image_feature(image_path):
        # 图像预处理
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            # 提取图像特征
            features = model.encode_image(image)
        return features.cpu().numpy()

    for file_name in common_files:
        feature1 = get_image_feature(os.path.join(folder1, file_name))
        feature2 = get_image_feature(os.path.join(folder2, file_name))
        
        # 计算特征之间的余弦相似性
        score = cosine_similarity(feature1, feature2)[0, 0]
        total_score += score
        count += 1
    
    average_clip_score = total_score / count if count > 0 else 0
    return average_clip_score

