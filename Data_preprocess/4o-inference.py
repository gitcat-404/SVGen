import csv
import os
import base64
import asyncio
from openai import AsyncAzureOpenAI  # 异步客户端
from PIL import Image
from tqdm.asyncio import tqdm_asyncio  # 异步进度条
import time

# Azure配置参数
api_base = ""
api_key = ""
deployment_name = ""
api_version = ""

# 创建异步客户端
aclient = AsyncAzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

async def process_images(folder_path, output_csv, existing_csv=None, max_concurrent=8):
    """异步处理图片"""
    semaphore = asyncio.Semaphore(max_concurrent)  # 控制并发数
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')]
    
    # 如果提供了现有的 CSV 文件，读取已经处理过的文件名
    processed_ids = set()
    if existing_csv is not None and os.path.exists(existing_csv):
        with open(existing_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            processed_ids = {row['id'] for row in reader}

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'desc'])
        writer.writeheader()

        async def process_file(filename):
            """单个文件处理协程"""
            file_id = os.path.splitext(filename)[0]
            if file_id in processed_ids:
                print(f"Skipping {filename} as it is already processed.")
                return None

            async with semaphore:
                try:
                    img_path = os.path.join(folder_path, filename)
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    description = await analyze_image(img_path)
                    return {
                        'id': file_id,
                        'desc': description.replace('\n', ' ')
                    }
                except Exception as e:
                    print(f"处理 {filename} 失败: {str(e)}")
                    return None

        # 批量处理并显示进度条
        tasks = [process_file(f) for f in image_files]
        results = []
        for f in tqdm_asyncio.as_completed(tasks, desc="Processing", unit="img"):
            result = await f
            if result:
                writer.writerow(result)
                results.append(result)

async def analyze_image(image_path):
    """异步图片分析"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    data_url = f"data:{mime_type};base64,{base64_image}"

    response = await aclient.chat.completions.create(
        model=deployment_name,
        messages=[
            { "role": "system", "content": "You are an image analysis assistant."},
            { "role": "user", "content": [
                { "type": "text", "text": "I need to design an SVG based on this icon and analyze its components. Describe the design process with numbered points like '1...2...', connecting components into a complete process. Ensure to use numbers for points and limit to 2-6. Fewer points are better if the process is clear. Include shape names and fill colors. Make sure each sentence is complete." },
                { "type": "image_url", "image_url": {"url": data_url} }
            ]}
        ],
        max_tokens=150,
        temperature=0.1
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(process_images(
        folder_path=r"D:\icon_download\SVG_and_PNG_Icon\PNG_divide\colored",
        output_csv=r"C:\Users\Lenovo\Desktop\api_inference\labels\cot_colored_1_2.csv",
        existing_csv=r"C:\Users\Lenovo\Desktop\api_inference\labels\cot_colored_1.csv"  # 设定为已经存在的 CSV 文件路径
    ))
    print(f"Total time: {time.time()-start_time:.2f}s")


# import csv 
# import os 
# import base64 
# import asyncio 
# from openai import AsyncAzureOpenAI  # 更改为异步客户端 
# from PIL import Image 
# from tqdm.asyncio  import tqdm_asyncio  # 异步进度条 
# import time 
 
# # Azure配置参数 
# api_base = "https://aicenter1-01-gpt4-2.openai.azure.com" 
# api_key = "c9832c46fa9e47ac927ce7a5526aa0d9"
# deployment_name = 'gpt-4o'
# api_version = '2024-02-15-preview'
 
# # 创建异步客户端 
# aclient = AsyncAzureOpenAI(
#     api_key=api_key,
#     api_version=api_version,
#     base_url=f"{api_base}/openai/deployments/{deployment_name}"
# )
 
# async def process_images(folder_path, output_csv, max_concurrent=10):
#     """异步处理图片"""
#     semaphore = asyncio.Semaphore(max_concurrent)  # 控制并发数 
#     image_files = [f for f in os.listdir(folder_path) 
#                    if os.path.splitext(f)[1].lower()  in ('.jpg', '.jpeg', '.png')]
 
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=['id', 'desc'])
#         writer.writeheader() 
 
#         async def process_file(filename):
#             """单个文件处理协程"""
#             async with semaphore:
#                 try:
#                     img_path = os.path.join(folder_path,  filename)
#                     with Image.open(img_path)  as img:
#                         img.verify() 
                    
#                     description = await analyze_image(img_path)
#                     return {
#                         'id': os.path.splitext(filename)[0], 
#                         'desc': description.replace('\n',  ' ')
#                     }
#                 except Exception as e:
#                     print(f"处理 {filename} 失败: {str(e)}")
#                     return None 
 
#         # 批量处理并显示进度条 
#         tasks = [process_file(f) for f in image_files]
#         results = []
#         for f in tqdm_asyncio.as_completed(tasks,  desc="Processing", unit="img"):
#             result = await f 
#             if result:
#                 writer.writerow(result) 
#                 results.append(result) 
 
# async def analyze_image(image_path):
#     """异步图片分析"""
#     with open(image_path, "rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode('utf-8') 
 
#     mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg',  '.jpeg')) else "image/png"
#     data_url = f"data:{mime_type};base64,{base64_image}"
 
#     response = await aclient.chat.completions.create( 
#         model=deployment_name,
#         messages=[
#             { "role": "system", "content": "You are an image analysis assistant."},
#             { "role": "user", "content": [
#                 { "type": "text", "text": "I need to design an SVG based on this icon and analyze its components. Describe the design process with numbered points like '1...2...', connecting components into a complete process. Ensure to use numbers for points and limit to 2-6. Fewer points are better if the process is clear. Include shape names and fill colors. Make sure each sentence is complete." },
#                 { "type": "image_url", "image_url": {"url": data_url} }
#             ]}
#         ],
#         max_tokens=150,
#         temperature=0.1
#     )
#     return response.choices[0].message.content  
 
# if __name__ == "__main__":
#     start_time = time.time() 
#     asyncio.run(process_images( 
#         folder_path=r"D:\icon_download\SVG_and_PNG_Icon\PNG_divide\colored",
#         output_csv=r"C:\Users\Lenovo\Desktop\api_inference\labels\cot_colored_1.csv" 
#     ))
#     print(f"Total time: {time.time()-start_time:.2f}s") 


# # Please help me briefly describe the provided icon. Make sure that everyone can clearly imagine the content and style of this icon through your description. The description text is as short and clear as possible, and no more than 20 Tokens is allowed. Please note that the default background color of the icon is white, so there is no need to describe the white color.
# # Use 1-2 phrases to simply describe this icon. If this icon is composed of multiple components, please describe their relative positions. The description text is as short and clear as possible, and no more than 10 Tokens is allowed. Please note that the default background color of the icon is white, so there is no need to describe the white color.
        