import os
from PIL import Image
from utils import compress_pil_image

# 定义要遍历的文件夹路径
folder_path = 'output_data/image'

# 遍历文件夹及其子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件是否为图片文件
        if file.lower().endswith('.png'):
            img_info = os.path.join(root, file)
            try:
                image = Image.open(img_info)
            except FileNotFoundError:
                print(f"未找到文件: {img_info}，跳过此文件。")
                continue

            # 压缩图片
            compressed_image = compress_pil_image(image, 0.4)

            save_path = os.path.join('output_data/image_compressed', file)

            # 保存压缩后的图片
            compressed_image.save(save_path)