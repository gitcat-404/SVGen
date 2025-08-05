import requests  # 爬虫所需
import re  # Re检索
import time
import os
pic_TotalNum = 10000000  # 图片总数量

Icon_sort = {"所有图标库": 3, "官方图标库": 1, "多彩图标库": 3, "单色图标库": 1}  # 图标库，修改图标库的时候，注意有的库图片不多，要减少爬取量
pic_Num = 0  # 计算图片数量

for Icon_Page in range(1, 100000):  # page翻页
    time.sleep(0.5)  # 防止爬的过快
    print(f"此种图标第{Icon_Page}页")
    # 找到网页url
    url_Icon_main = f"https://www.iconfont.cn/api/collections.json?type={Icon_sort['所有图标库']}&sort=time&limit=9&page={Icon_Page}&keyword=&t=1730297800301&ctoken=0w_OmWRDvG8f5cVaMFUI0Q1r"
    Icon_main_text = requests.get(url_Icon_main)  # 访问链接

    Icon_obj_main = re.compile(r'"id":\s*(?P<ID>\d*),\s*"created_at":', re.S)  # 正则表达式
    Icon_Res_main = Icon_obj_main.finditer(Icon_main_text.text)  # 通过正则表达式进行Re搜索
    # 循环遍历迭代器
    for i in Icon_Res_main:
        url_Icon = f"https://www.iconfont.cn/api/collection/detail.json?id={i.group('ID')}&t=1730297800301&ctoken=0w_OmWRDvG8f5cVaMFUI0Q1r"
        Icon_text = requests.get(url_Icon)  # 访问链接

        Icon_obj = re.compile(r'"show_svg":"\u003C(?P<svg_Code>.*?)svg\u003E",', re.S)  # 正则表达式
        Icon_Res = Icon_obj.finditer(Icon_text.text)  # 通过正则表达式进行Re搜索
        # 循环遍历迭代器
        for it in Icon_Res:
            pic_Num = pic_Num + 1  # 总图片数量+1
            svg_code = ("<" + it.group("svg_Code") + r"svg>").replace('\\', '')  # 拼接，形成完整svgCode
            folder_path = "SVG_and_PNG_Icon\\SVG"
            os.makedirs(folder_path, exist_ok=True)  # 创建文件夹（如果不存在）
            with open(f"{folder_path}\\{pic_Num}.svg", 'w', encoding='utf-8') as f:
                f.write(svg_code)
            print(f"Svg目前爬取数量：{pic_Num}")
        if pic_Num >= pic_TotalNum:  # 终止条件
            break
    if pic_Num >= pic_TotalNum:  # 终止条件
        break
print(f"爬取的总数据量为：{pic_Num}")
print("over!!")