import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import torch
import json
import pyautogui as pg
from supervision.draw.color import ColorPalette

def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode()

def yolo_predict(model, image, box_threshold, imgsz, iou_threshold=0.7):
    result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold,
    )
    yolo_boxes = result[0].boxes.xyxy #坐标
    return yolo_boxes

def ocr_predict(model, image, threshold=0.5, device='cpu'):
    result = [item for item in model.ocr(np.array(image), cls=False)[0] if item[1][1] > threshold]
    boxes = [[item[0][0][0], item[0][0][1], item[0][2][0], item[0][2][1]] for item in result]
    text = [item[1][0] for item in result]
    boxes = torch.tensor(boxes, device=device)
    return boxes, text

def calculate_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def calculate_intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def calculate_iou(box1, box1_area, box2, box2_area):
    intersection = calculate_intersection_area(box1, box2)
    union = box1_area + box2_area - intersection
    ratio1 = intersection / box1_area
    ratio2 = intersection / box2_area
    return max(intersection / union, ratio1, ratio2)

def is_inside(box1, box1_area, box2):
    intersection = calculate_intersection_area(box1, box2)
    ratio1 = intersection / box1_area
    return ratio1 > 0.50

class BoxUnionFind:
    def __init__(self, n, boxes):
        self.parent = list(range(n))
        self.areas = [calculate_box_area(box) for box in boxes]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            area_x = self.areas[root_x]
            area_y = self.areas[root_y]

            if area_x < area_y:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x

def merge_boxes(yolo_boxes, ocr_boxes, textes, threshold=0.9):
    yolo_len = len(yolo_boxes)
    uf = BoxUnionFind(yolo_len, yolo_boxes)

    for i in range(yolo_len):
        for j in range(i + 1, yolo_len):
            if calculate_iou(yolo_boxes[i], uf.areas[i], yolo_boxes[j], uf.areas[j]) > threshold:
                uf.union(i, j)

    ocr_len = len(ocr_boxes)
    ocr_area = []
    for i in range(ocr_len):
        ocr_area.append(calculate_box_area(ocr_boxes[i]))
    result_boxes = []
    result_boxes_area = []
    box_index = 1
    for i in range(yolo_len):
        if uf.find(i) == i:
            content = ""
            for j in range(ocr_len):
                if ocr_area[j] != -1 and is_inside(ocr_boxes[j], ocr_area[j], yolo_boxes[i]):
                    content += textes[j]
                    ocr_area[j] = -1
            result_boxes.append({'index': box_index, 'type': 'icon', 'bbox': yolo_boxes[i], 'interactivity': True, 'content': content})
            result_boxes_area.append(uf.areas[i])
            box_index += 1

    for i in range(ocr_len):
        if ocr_area[i] != -1:
            result_boxes.append({'index': box_index, 'type': 'text', 'bbox': ocr_boxes[i], 'interactivity': False, 'content': textes[i]})
            result_boxes_area.append(ocr_area[i])
            box_index += 1

    return result_boxes, result_boxes_area

system_prompt = """
# 角色定位
你是Windows电脑操作专家，精通Windows系统的各种操作。

# 任务
用户会提供其想要完成的最终任务、电脑当前的截图以及过往的操作历史。截图中的文字和交互组件已经被标记，并附带坐标和语义信息。你的任务是综合以上信息，判断接下来**第一步**要执行的操作，以及这么操作的原因。

# 输入
## 最终任务
用户会提供一个最终任务，为了完成这个任务，可能需要非常多的步骤。

## 电脑截图
### 图片
电脑截图中的所有组件和文字均已用红色框定，并用绿色标注了序号。

### 辅助解读
- 你会收到一份包含所有标记组件的辅助信息，格式如下：
  ```json
  {
    "index": xxx,
    "type": "xxx",
    "coordinate": (xxx, xxx),
    "interactivity": xxx,
    "content": "xxx"
  }
- index：组件序号，从1开始。
- type：组件类型，可为"icon"（图标）或"text"（文本）。
- coordinate：组件的坐标，格式为(x, y)。
- interactivity：是否可交互，可为True或False。
- content：组件内容，文本类型组件显示文本内容，图标类型组件显示图标名称。

## 操作历史
用户会提供过往的操作历史，告知你之前每一步做了什么操作。

# 要求
- 当你想要将鼠标移动到一个组件上时，请先判断这个组件的序号，然后在辅助信息中找到对应序号的coordinate字段，根据coordinate字段提供的坐标，使用鼠标移动操作将鼠标移动到该组件上。
- 禁止自行推测组件坐标，请始终依据提供的coordinate字段进行鼠标操作。
- 确保操作精准无误，按照用户需求和截图信息制定最合理的操作步骤。
- 如果你认为无法完成，请输出“无法完成”
- 当你需要打开一个应用程序时，请使用鼠标**左键双击**

# 可采取的操作
## 鼠标
- 点击：{'action': 'click', 'args': {'clicks': 2, 'interval': xxx, 'button': xxx}}
    - clicks：点击次数，必须为2
    - interval：点击间隔，可选值为0、0.2
    - button：点击的按钮，可选值为'left'、'right'、'middle'
- 移动：{'action': 'moveTo', 'args': {'x': xxx, 'y': xxx}}
    - x：鼠标移动到的x坐标
    - y：鼠标移动到的y坐标

## 键盘
- 键盘输入：{'action': 'typewrite', 'args': {'message': xxx}}
    - message：要输入的文本
    
## 特殊操作
- 完成任务：{'action': 'finish', 'args': {}}

# 输出格式
- 输出应为JSON格式的列表，每个操作均为一个字典，包含：
    - "action"：操作类型（如"click"、"moveTo"、"typewrite"）。
    - "args"：操作参数。
    - "think": 操作的理由
- 禁止输出任何额外内容，仅输出操作列表。

# 示例
用户想要完成的任务是：打开腾讯会议

截图中的信息解读：
{'index': 1, 'type': 'icon', 'coordinate': (150, 280), 'interactivity': True, 'content': '腾讯会议'}
{'index':2, 'type': 'icon', 'coordinate': (301, 487), 'interactivity': True, 'content': '微信'}

过往的操作历史：
第 1 次操作：
{"action": "moveTo", "args": {"x": 150, "y": 280}, "think": "用户想要打开腾讯会议，根据信息解读发现index为1的组件content是“腾讯会议”，截图中1号框的图标样式确实是腾讯会议，所以1号框就是腾讯会议，这个框的坐标是(150, 280)，将鼠标移向它。"}

预期输出：
[
  {"action": "click", "args": {"clicks": 2, "interval": 0.1, "button": "left"}, "think": "双击打开软件。"}
]

"""

def create_message(instruction, screenshot_with_boxes, boxes, history):
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pil_to_base64(screenshot_with_boxes)}"},
                },
                {"type": "text", "text": f"用户想要完成的任务是：{instruction}\n\n截图中的信息解读：\n{boxes}\n\n过往的操作历史：\n{history}"},
            ],
        }
    ]
    return message

def postprocess(boxes):
    processed_boxes = []
    for box in boxes:
        bbox = box['bbox']
        x1, y1, x2, y2 = bbox.cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        new_box = {
            'index': box['index'],
            'type': box['type'],
            'coordinate': (center_x, center_y),
            'interactivity': box['interactivity'],
            'content': box['content']
        }
        processed_boxes.append(new_box)
    return processed_boxes

def add_history(step_cnt, action):
    now_history = f"第 {step_cnt} 次操作：\n{action}\n\n"
    return now_history

def execute(action_json):
    try:
        action_dicts = json.loads(action_json)
        for action_dict in action_dicts:
            action = action_dict['action']
            args = action_dict['args']
            if action == 'click':
                pg.click(clicks=args['clicks'], interval=args['interval'], button=args['button'])
            elif action == 'moveTo':
                pg.moveTo(x=args['x'], y=args['y'])
            elif action == 'typewrite':
                pg.typewrite(message=args['message'])
            else:
                print("未知的操作指令")
    except json.JSONDecodeError:
        print("JSON 解析错误")
    except KeyError:
        print("操作指令格式错误")
    except Exception as e:
        print(f"执行操作时发生错误: {e}")

def mission_complete(action_json):
    try:
        action_dicts = json.loads(action_json)
        for action_dict in action_dicts:
            action = action_dict['action']
            if action == 'finish':
                return True
        return False
    except json.JSONDecodeError:
        print("JSON 解析错误")
        return True
    except KeyError:
        print("操作指令格式错误")
        return True
    except Exception as e:
        print(f"检查是否完成任务时发生错误: {e}")
        return True

def get_optimal_label_pos(label, x1, y1, x2, boxes, boxes_area, label_backgrounds, label_backgrounds_area, image_size):
    text_padding = 10
    text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

    text_x = [x1 + text_padding / 2, x1 - text_padding / 2 - text_width, x2 + text_padding / 2, x2 - text_padding / 2 - text_width]
    text_y = [y1 - text_padding / 2, y1 + text_padding / 2 + text_height, y1 + text_padding / 2 + text_height, y1 - text_padding / 2]
    label_background_x1 = [x1, x1 - text_width - text_padding, x2, x2 - text_width - text_padding]
    label_background_y1 = [y1 - text_height - text_padding, y1, y1, y1 - text_height - text_padding]
    label_background_x2 = [x1 + text_width + text_padding, x1, x2 + text_width + text_padding, x2]
    label_background_y2 = [y1, y1 + text_height + text_padding, y1 + text_height + text_padding, y1]

    cur_label_background_area = calculate_box_area((label_background_x1[0], label_background_y1[0], label_background_x2[0], label_background_y2[0]))

    for i in range(3):
        if label_background_x1[i] < 0 or label_background_y1[i] < 0 or label_background_x2[i] > image_size[0] or label_background_y2[i] > image_size[1]:
            continue
        is_overlap = False
        for j in range(len(boxes)):
            if calculate_iou((label_background_x1[i], label_background_y1[i], label_background_x2[i], label_background_y2[i]), cur_label_background_area, boxes[j]['bbox'], boxes_area[i]) > 0.3:
                is_overlap = True
                break
        if not is_overlap:
            for j in range(len(label_backgrounds)):
                if calculate_iou((label_background_x1[i], label_background_y1[i], label_background_x2[i], label_background_y2[i]), cur_label_background_area, label_backgrounds[j], label_backgrounds_area[j]) > 0.3:
                    is_overlap = True
                    break
        if not is_overlap:
            return int(text_x[i]), int(text_y[i]), label_background_x1[i], label_background_y1[i], label_background_x2[i], label_background_y2[i]

    return int(text_x[3]), int(text_y[3]), label_background_x1[3], label_background_y1[3], label_background_x2[3], label_background_y2[3]


def draw_boxes(image, boxes, boxes_area):
    image_size = image.size
    image_np = np.array(image)

    label_backgrounds = []
    label_backgrounds_area = []

    for i, box in enumerate(boxes):
        bbox = box['bbox'].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox

        color = ColorPalette.DEFAULT.by_idx(i).as_rgb()
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)

        label = f"{i + 1}"
        label_x, label_y, label_background_x1, label_background_y1, label_background_x2, label_background_y2 = get_optimal_label_pos(label, x1, y1, x2, boxes, boxes_area, label_backgrounds, label_backgrounds_area, image_size)

        label_backgrounds.append((label_background_x1, label_background_y1, label_background_x2, label_background_y2))
        label_backgrounds_area.append(calculate_box_area((label_background_x1, label_background_y1, label_background_x2, label_background_y2)))

        cv2.rectangle(image_np, (label_background_x1, label_background_y2), (label_background_x2, label_background_y1), color, cv2.FILLED)

        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        cv2.putText(image_np, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    image_pil = Image.fromarray(image_np)
    return image_pil