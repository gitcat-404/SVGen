import os, csv, base64, time, asyncio, sys
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from config import CONFIG
from client_factory import get_client

# -------------------------------
# 1. 提示词模板
# -------------------------------
PROMPT_TEMPLATES = {
    "cot": {
        "text": (
            "I need to design an SVG based on this icon and analyze its components. "
            "Describe the design process with numbered points like '1...2...'. "
            "Ensure to use numbers for points and limit to 2-6. Fewer points are better "
            "if the process is clear. Include shape names and fill colors. Make sure each sentence is complete."
        )
    },
    "desc": {
        "text": (
            "Please help me briefly describe the provided icon. Make sure that everyone can clearly imagine "
            "the content and style of this icon through your description. "
            "The description text is as short and clear as possible, and no more than 30 Tokens is allowed."
        )
    }
}

# -------------------------------
# 2. 提示词类型通过命令行参数选择
# -------------------------------
if len(sys.argv) < 2 or sys.argv[1] not in PROMPT_TEMPLATES:
    print("Usage: python main.py [cot|desc]")
    sys.exit(1)

PROMPT_TYPE = sys.argv[1]

# -------------------------------
# 3. 初始化客户端
# -------------------------------
client = get_client(CONFIG)
provider = CONFIG["provider"]
deployment_name = CONFIG[provider].get("deployment_name", "gpt-4")

# -------------------------------
# 4. 图像分析函数
# -------------------------------
async def analyze_image(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    data_url = f"data:{mime_type};base64,{base64_image}"

    prompt = PROMPT_TEMPLATES[PROMPT_TYPE]["text"]

    messages = [
        { "role": "system", "content": "You are an image analysis assistant." },
        { "role": "user", "content": [
            { "type": "text", "text": prompt },
            { "type": "image_url", "image_url": {"url": data_url} }
        ]}
    ]

    response = await client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=150,
        temperature=0.1
    )
    return response.choices[0].message.content

# -------------------------------
# 5. 主处理流程
# -------------------------------
async def process_images(folder_path, output_csv, existing_csv=None, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')]

    processed_ids = set()
    if existing_csv and os.path.exists(existing_csv):
        with open(existing_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            processed_ids = {row['id'] for row in reader}

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'desc'])
        writer.writeheader()

        async def process_file(filename):
            file_id = os.path.splitext(filename)[0]
            if file_id in processed_ids:
                print(f"Skipping {filename}")
                return None

            async with semaphore:
                try:
                    img_path = os.path.join(folder_path, filename)
                    with Image.open(img_path) as img:
                        img.verify()

                    description = await analyze_image(img_path)
                    return {'id': file_id, 'desc': description.replace('\n', ' ')}
                except Exception as e:
                    print(f"Failed {filename}: {str(e)}")
                    return None

        tasks = [process_file(f) for f in image_files]
        for coro in tqdm_asyncio.as_completed(tasks, desc="Processing", unit="img"):
            result = await coro
            if result:
                writer.writerow(result)

# -------------------------------
# 6. 启动入口
# -------------------------------
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(process_images(
        folder_path=r"D:\icon_download\SVG_and_PNG_Icon\PNG_divide\colored",
        output_csv=fr"C:\Users\Lenovo\Desktop\api_inference\labels\{PROMPT_TYPE}_colored_output.csv",
        existing_csv=None
    ))
    print(f"Total time: {time.time()-start_time:.2f}s")
