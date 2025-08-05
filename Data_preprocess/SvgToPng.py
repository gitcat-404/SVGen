import os
import cairosvg
from tqdm import tqdm

def convert_svg_to_png(src_folder, dst_folder, width, height):
    # 检查源文件夹是否存在
    if not os.path.exists(src_folder):
        print(f"源文件夹 {src_folder} 不存在！")
        return

    # 创建目标文件夹，如果不存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 获取所有SVG文件
    svg_files = [f for f in os.listdir(src_folder) if f.endswith('.svg')]

    # 使用tqdm添加进度条
    for filename in tqdm(svg_files, desc="Converting SVGs to PNGs", unit="file"):
        svg_path = os.path.join(src_folder, filename)
        png_path = os.path.join(dst_folder, filename.replace('.svg', '.png'))

        try:
            # 使用cairosvg将svg转换为png
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width, output_height=height,background_color='white')
        except Exception as e:
            print(f"\n转换 {filename} 失败: {e}")

if __name__ == '__main__':
    source_folder = 'D:\icon_download\SVG_and_PNG_Icon\SVG'  # 替换为实际的SVG文件所在目录
    destination_folder = 'D:\icon_download\SVG_and_PNG_Icon\PNG'  # 替换为希望保存PNG文件的目录
    img_width = 200
    img_height = 200

    convert_svg_to_png(source_folder, destination_folder, img_width, img_height)

