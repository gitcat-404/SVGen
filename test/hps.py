import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv
import os
from tqdm import tqdm
import pdb
def calculate_hps(hpc, image_folder, meta_file, batch_size=8, num_workers=16, device='cuda'):
    """
    Evaluate HPS using the provided parameters.

    :param hpc: Path to the hpc checkpoint.
    :param image_folder: Path to the image folder.
    :param meta_file: Path to the CSV file containing image descriptions.
    :param batch_size: Size of each data batch.
    :param num_workers: Number of worker threads for data loading.
    :param device: Device to be used for computation ('cuda' or 'cpu').
    :return: The average HPS score.
    """
    
    # Load CLIP model
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14", device=device)
    params = torch.load(hpc)['state_dict']
    model.load_state_dict(params)

    class ImageTextDataset(Dataset):
        def __init__(self, meta_file, image_folder, transforms, tokenizer):
            self.datalist = []
            with open(meta_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = str(row['id']) + '.png'
                    file_path = os.path.join(image_folder, file_name)
                    if os.path.exists(file_path):  # Check if image file exists
                        self.datalist.append({
                            'file_name': file_name,
                            'caption': row['desc']
                        })
                    else:
                        print(f"Warning: {file_name} not found in image folder, skipping this entry.")
            
            self.folder = image_folder
            self.transforms = transforms
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.datalist)

        def __getitem__(self, idx):
            try:
                images = self.transforms(Image.open(os.path.join(self.folder, self.datalist[idx]['file_name'])))
                input_ids = self.tokenizer(self.datalist[idx]['caption'], context_length=77, truncate=True)[0]
                return images, input_ids
            except Exception as e:
                print(f"Error: Failed to load data index {idx}. Error: {e}")
                return None, None

    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    dataset = ImageTextDataset(meta_file, image_folder, preprocess, clip.tokenize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)

    scores = []
    with torch.no_grad():
        for i, (images, text) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            text = text.to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T
            #pdb.set_trace()
            hps = hps.diagonal()
            scores.extend(hps.squeeze().tolist())

    if len(scores) > 0:
        average_hps = sum(scores) / len(scores)
        return average_hps
    else:
        print("No valid image-text pairs were found.")
        return None

# Example function call
# average_hps = evaluate_hps('path/to/hpc_checkpoint.pth', 'path/to/image_folder', 'path/to/meta_file.csv')
