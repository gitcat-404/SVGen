import os
from xml.dom.minidom import parse
from deepsvg.svglib.svg import SVG
from tqdm import tqdm
from deepsvg.svglib.svg_command import SVGCommandMove, SVGCommandLine, SVGCommandBezier, SVGCommandArc
from deepsvg.svglib.svg_path import SVGPath
import numpy as np

def arc_to_bezier(arc):
    # 将椭圆弧转换为贝塞尔曲线
    # 这里使用了svgpathtools库的内部转换方法
    return arc.to_beziers()

def create_svg_path_data(svg):
    path_data_list = []
    for path_group in svg.svg_path_groups:
        path_data = []
        for path in path_group.paths:
            for i, command in enumerate(path.all_commands()):
                if isinstance(command, SVGCommandMove):
                    if i == 0 or not path_data or not path_data[-1].startswith("M"):
                        path_data.append(f"M {int(command.end_pos.x)} {int(command.end_pos.y)}")
                elif isinstance(command, SVGCommandLine):
                    path_data.append(f"L {int(command.end_pos.x)} {int(command.end_pos.y)}")
                elif isinstance(command, SVGCommandBezier):
                    c1 = command.control1
                    c2 = command.control2
                    end = command.end_pos
                    path_data.append(f"C {int(c1.x)} {int(c1.y)} {int(c2.x)} {int(c2.y)} {int(end.x)} {int(end.y)}")
                elif isinstance(command, SVGCommandArc):
                    # 将椭圆弧转换为贝塞尔曲线
                    beziers = arc_to_bezier(command)
                    for bezier in beziers:
                        c1 = bezier.control1
                        c2 = bezier.control2
                        end = bezier.end_pos
                        path_data.append(f"C {int(c1.x)} {int(c1.y)} {int(c2.x)} {int(c2.y)} {int(end.x)} {int(end.y)}")
        path_data_list.append(" ".join(path_data))
    return path_data_list

def replace_svg_path_data(original_svg_file, path_data_list, output_file):
    doc = parse(original_svg_file)
    path_elements = doc.getElementsByTagName("path")

    # 确保path_data_list和path_elements长度相同
    if len(path_data_list) != len(path_elements):
        raise ValueError("路径数据列表的长度与SVG文件中的<path>元素数量不匹配。")

    for i, path_element in enumerate(path_elements):
        path_element.setAttribute("d", path_data_list[i])

    with open(output_file, "w") as f:
        f.write(doc.documentElement.toxml())

def process_svg_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    svg_files = [f for f in os.listdir(input_folder) if f.endswith('.svg')]

    for svg_file in tqdm(svg_files, desc="Processing SVG files"):
        input_file = os.path.join(input_folder, svg_file)
        output_file = os.path.join(output_folder, svg_file)

        if os.path.getsize(input_file) == 0:
            print(f"文件 {input_file} 是空的，跳过。")
            continue

        try:
            svg = SVG.load_svg(input_file)
            path_data_list = create_svg_path_data(svg)

            replace_svg_path_data(input_file, path_data_list, output_file)
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {e}")

if __name__ == "__main__":
    input_folder = r"C:\Users\Lenovo\Downloads\IconShop-main\orenge"
    output_folder = r"C:\Users\Lenovo\Downloads\IconShop-main\orenge"
    process_svg_files(input_folder, output_folder)
    print("所有文件已处理完成。")


# import os
# from xml.dom.minidom import parse
# from deepsvg.svglib.svg import SVG
# from tqdm import tqdm
# from deepsvg.svglib.svg_command import SVGCommandMove, SVGCommandLine, SVGCommandBezier
# def create_svg_path_data(svg):
#     path_data_list = []
#     for path_group in svg.svg_path_groups:
#         path_data = []
#         for path in path_group.paths:
#             for i, command in enumerate(path.all_commands()):
#                 if isinstance(command, SVGCommandMove):
#                     if i == 0 or not path_data or not path_data[-1].startswith("M"):
#                         path_data.append(f"M {int(command.end_pos.x)} {int(command.end_pos.y)}")
#                 elif isinstance(command, SVGCommandLine):
#                     path_data.append(f"L {int(command.end_pos.x)} {int(command.end_pos.y)}")
#                 elif isinstance(command, SVGCommandBezier):
#                     c1 = command.control1
#                     c2 = command.control2
#                     end = command.end_pos
#                     path_data.append(f"C {int(c1.x)} {int(c1.y)} {int(c2.x)} {int(c2.y)} {int(end.x)} {int(end.y)}")
#         path_data_list.append(" ".join(path_data))
#     return path_data_list

# def replace_svg_path_data(original_svg_file, path_data_list, output_file):
#     doc = parse(original_svg_file)
#     path_elements = doc.getElementsByTagName("path")

#     # 确保path_data_list和path_elements长度相同
#     if len(path_data_list) != len(path_elements):
#         raise ValueError("路径数据列表的长度与SVG文件中的<path>元素数量不匹配。")

#     for i, path_element in enumerate(path_elements):
#         path_element.setAttribute("d", path_data_list[i])

#     with open(output_file, "w") as f:
#         f.write(doc.documentElement.toxml())

# def process_svg_files(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder, exist_ok=True)

#     svg_files = [f for f in os.listdir(input_folder) if f.endswith('.svg')]

#     for svg_file in tqdm(svg_files, desc="Processing SVG files"):
#         input_file = os.path.join(input_folder, svg_file)
#         output_file = os.path.join(output_folder, svg_file)

#         if os.path.getsize(input_file) == 0:
#             print(f"文件 {input_file} 是空的，跳过。")
#             continue

#         try:
#             svg = SVG.load_svg(input_file)
#             path_data_list = create_svg_path_data(svg)

#             replace_svg_path_data(input_file, path_data_list, output_file)
#         except Exception as e:
#             print(f"处理文件 {input_file} 时出错: {e}")

# if __name__ == "__main__":
#     input_folder = r"D:\icon_download\SVG_and_PNG_Icon\SVG"
#     output_folder = r"D:\icon_download\SVG_and_PNG_Icon\SVG_changed"
#     input_folder = r"D:\data_iconshop\test"
#     output_folder = r"D:\data_iconshop\test_result"
#     process_svg_files(input_folder, output_folder)
#     print("所有文件已处理完成。")
