import os

def sync_png_with_svg(svg_folder, png_folder):
    # 获取 SVG 文件夹中的所有文件名（不带扩展名）
    svg_files = {os.path.splitext(f)[0] for f in os.listdir(svg_folder) if f.endswith('.png')}
    
    # 遍历 PNG 文件夹中的所有文件
    for png_file in os.listdir(png_folder):
        if png_file.endswith('.svg'):
            png_name = os.path.splitext(png_file)[0]
            # 如果 PNG 文件名不在 SVG 文件名列表中，则删除
            if png_name not in svg_files:
                os.remove(os.path.join(png_folder, png_file))
                print(f"Deleted: {png_file}")

# 使用示例
svg_folder_path = r'D:\icon_download\SVG_and_PNG_Icon\PNG'
png_folder_path = r'D:\icon_download\SVG_and_PNG_Icon\SVG'

sync_png_with_svg(svg_folder_path, png_folder_path)
