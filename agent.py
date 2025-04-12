from openai import OpenAI
from utils import *
from paddleocr import PaddleOCR
import torch
import pyautogui as pg
from ultralytics import YOLO

MAX_STEP_NUM = 10

class Agent:
    def __init__(self, api_key, vlm_name, base_url, yolo_weight_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.vlm_name = vlm_name
        self.yolo_model = YOLO(yolo_weight_path).to(self.device)
        self.screen_size = pg.size()
        use_gpu = self.device.type == 'cuda'
        self.paddle_ocr_model = PaddleOCR(
            lang='ch',
            use_angle_cls=False,
            use_gpu=use_gpu,
            show_log=False,
            max_batch_size=1024,
            use_dilation=True,
            det_db_score_mode='slow',
            rec_batch_num=1024
        )

    def run(self, instruction):
        step_cnt = 0
        history = ""
        while step_cnt < MAX_STEP_NUM:
            screenshot_with_boxes, boxes = self.observe()
            screenshot_with_boxes.show()
            action = self.plan(instruction, screenshot_with_boxes, boxes, history)
            print(action)
            if mission_complete(action):
                print("任务完成")
                break
            history += add_history(step_cnt, action)
            execute(action)
            step_cnt += 1

    def observe(self):
        screenshot = pg.screenshot()
        yolo_boxes = yolo_predict(self.yolo_model, screenshot, box_threshold=0.15, imgsz=self.screen_size, iou_threshold=0.05)
        ocr_boxes, textes = ocr_predict(self.paddle_ocr_model, screenshot, 0.6, self.device)
        boxes, boxes_area = merge_boxes(yolo_boxes, ocr_boxes, textes)
        screenshot_with_boxes = draw_boxes(screenshot, boxes, boxes_area)
        boxes = postprocess(boxes)
        return screenshot_with_boxes, boxes

    def plan(self, instruction, screenshot_with_boxes, boxes, history):
        message = create_message(instruction, screenshot_with_boxes, boxes, history)
        print(message)
        completion = self.client.chat.completions.create(
            model=self.vlm_name,
            messages=message,
        )
        return completion.choices[0].message.content