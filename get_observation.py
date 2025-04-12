import os
import json
from PIL import Image
from utils import yolo_predict, ocr_predict, merge_boxes, draw_boxes, postprocess
from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR
from tqdm import tqdm  # 导入 tqdm 库

yolo_weight_path = "model_weights/yolo_weights.pt"

# 定义输入输出文件夹路径
input_folder = 'input_data'
output_image_folder = 'output_data/image'
output_json_folder = 'output_data/json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
screen_size = (2880, 1800)

def observe(image, yolo_model, paddle_ocr_model):
    yolo_boxes = yolo_predict(yolo_model, image, box_threshold=0.15, imgsz=screen_size,
                              iou_threshold=0.05)
    ocr_boxes, textes = ocr_predict(paddle_ocr_model, image, 0.6, device)
    boxes, boxes_area = merge_boxes(yolo_boxes, ocr_boxes, textes)
    screenshot_with_boxes = draw_boxes(image, boxes, boxes_area)
    boxes = postprocess(boxes)
    return screenshot_with_boxes, boxes

def main():
    # 加载模型
    yolo_model = YOLO(yolo_weight_path).to(device)
    paddle_ocr_model = PaddleOCR(
        lang='ch',
        use_angle_cls=False,
        use_gpu=True,
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,
        det_db_score_mode='slow',
        rec_batch_num=1024
    )

    file_list = os.listdir(input_folder)
    for filename in tqdm(file_list, desc="Processing images", unit="image"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        screenshot_with_boxes, boxes = observe(image, yolo_model, paddle_ocr_model)
        screenshot_with_boxes.save(os.path.join(output_image_folder, filename))

        # 替换文件扩展名到 .json
        base_name = os.path.splitext(filename)[0]
        json_filename = base_name + '.json'
        with open(os.path.join(output_json_folder, json_filename), 'w') as f:
            json.dump(boxes, f)

if __name__ == "__main__":
    main()