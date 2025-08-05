import json
import os
import pandas as pd
from tqdm import tqdm

def create_json_from_csv(csv_path, folder_path, output_json_path):
    df = pd.read_csv(csv_path)
    data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        file_name = row['id']
        desc = row['desc']
        svg_file_path = os.path.join(folder_path, f"{file_name}.svg")
        if os.path.exists(svg_file_path):
            with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
                svg_content = svg_file.read().strip()
                if not svg_content:
                    continue
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Please generate an SVG icon that meets the following description: {desc}"
                        },
                        {
                            "from": "gpt",
                            "value": svg_content
                        }
                    ]
                }
                data.append(conversation)

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# 使用示例
csv_path = r'C:\Users\Lenovo\Desktop\api_inference\labels\colored_1.csv'
folder_path = r'D:\icon_download\SVG_and_PNG_Icon\SVG_changed_2'
output_json_path = r'C:\Users\Lenovo\Desktop\api_inference\json_data\color_1_t.json'
create_json_from_csv(csv_path, folder_path, output_json_path)