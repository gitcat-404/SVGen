import pandas as pd
import os
import json
from tqdm import tqdm

SYSTEM_PROMPT = "Please generate the reasoning process and final SVG code separately, according to the following format:<think>\n...\n</think>\n``svg\n...\n``"

def create_json_from_csv(csv_path, reason_csv_path, folder_path, output_json_path):
    df = pd.read_csv(csv_path)
    reason_df = pd.read_csv(reason_csv_path)
    reason_dict = {row['id']: row['desc'] for index, row in reason_df.iterrows()}
    csv_ids = set(df['id'])
    reason_ids = set(reason_df['id'])
    common_ids = csv_ids.intersection(reason_ids)
    data = []
    
    for file_id in tqdm(common_ids, desc="Processing"):
        desc = df[df['id'] == file_id]['desc'].values[0]
        reason = reason_dict[file_id]
        svg_file_path = os.path.join(folder_path, f"{file_id}.svg")
        svg_content = ""
        
        if os.path.exists(svg_file_path):
            with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
                svg_content = svg_file.read().strip()
        if not svg_content:
            continue  # Skip if SVG content is not available
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please generate an SVG icon that meets the following description: {desc}"
                },
                {
                    "from": "gpt",
                    "value": f"<think>\n{reason}\n</think>\n``svg\n{svg_content}\n``"
                }
            ],
            "system": SYSTEM_PROMPT
        }
        data.append(conversation)
    
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

csv_path = '/gemini/space/thu/zhaozhiyuan/wfy/colored_1.csv'  # 替换成实际的CSV文件路径
reason_csv_path = '/gemini/space/thu/zhaozhiyuan/wfy/merged_cot_file.csv'  # reason的CSV文件路径
folder_path = '/gemini/space/thu/zhaozhiyuan/wfy/svg_colored'
output_json_path = '/gemini/space/thu/zhaozhiyuan/wfy/json_data/color_reasons.json'

create_json_from_csv(csv_path, reason_csv_path, folder_path, output_json_path)