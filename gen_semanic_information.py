from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from utils import compress_pil_image
from PIL import Image
import json
import os


def load_tasks(file_path):
    """
    从指定路径加载 tasks.json 文件
    :param file_path: tasks.json 文件的路径
    :return: 加载的任务列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_file_history(task_id, index):
    """
    从 gen_semanic_info_data 文件夹加载历史记录
    :param task_id: 任务 ID
    :param index: 历史记录的索引
    :return: 加载的历史记录列表
    """
    file_path = os.path.join('gen_semanic_info_data', f'{task_id}_{index}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('history', [])
    print("Error !")
    return []


def format_history(history):
    """
    格式化历史记录为字符串
    :param history: 历史记录列表
    :return: 格式化后的字符串
    """
    if not history:
        return "用户此前未执行过任何操作。\n"
    formatted = ""
    for step, hist_item in enumerate(history, start=1):
        formatted += f"第{step}次操作：\n"
        formatted += f"thinking：{hist_item.get('thinking', '')}\n"
        formatted += f"action：{hist_item.get('action', '')}\n\n"
    return formatted


def generate_user_prompt(task_zh, file_history, parser_result, action):
    """
    生成用户提示信息
    :param task_zh: 任务的中文描述
    :param file_history: 历史记录列表
    :param parser_result: 解析结果
    :param action: 当前操作
    :return: 生成的用户提示信息
    """
    formatted_history = format_history(file_history)
    return f'''
用户的最终目标：{task_zh}。
在执行当前操作之前，用户曾执行过{len(file_history)}次操作：
{formatted_history}

当前电脑截图辅助信息为：
{parser_result}

当前用户执行的操作是：{action}
请分析：用户为何执行这一步？他这样做的目的是什么？
'''


model_weights_path = "model_weights/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_weights_path,
    torch_dtype="auto",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=model_weights_path,
    use_fast=False,
)

sys_prompt = '''
# 角色与任务定义
你是一个经验丰富的电脑使用分析专家。用户会向你提供：
- 他们的**最终目标**（想通过电脑完成的任务）。
- 当前的**电脑截图**以及截图中元素的**辅助信息**。
- 他们先前所做的所有**操作历史**。
- 当前正在执行的**操作**。
你的任务是：结合用户的最终目标、电脑截图、辅助信息、操作历史，**推理并解释**用户为什么执行这一步操作？背后的意图是什么？

# 各类信息解读
## 用户可能采取的操作
- click (x, y)：鼠标左键单击(x, y)处。
  - 可能用于点击搜索框、对话框，确保后续打字到正确框中。
  - 可能用于选择某个选项或点击某个按键。
- right click (x, y)：鼠标右键单击(x, y)处。常用于打开菜单功能（复制、粘贴等）。
- double click (x, y)：鼠标左键双击。常用于打开软件。
- drag to (x, y)：鼠标从当前所在位置左键拖动至(x, y)处。
  - 可能用于拖动滚动条。
  - 可能用于拖拽选择区域。
- type text: [content]：输入[content]。
- open task manager：打开任务管理器。
- move to (x, y)：将鼠标移动至(x, y)处。
- enter：回车。
  - 可能用于输入框中换行。
  - 可能用于发送、搜索等。
- finish：用户最终任务已达成，结束。
- select all：全选。在各种文本编辑场景下，如聊天框、word中，选择全部文本。
- copy：复制。在各种文本编辑场景下，如聊天框、word中，复制选中文本。
- paste：粘贴。配合copy使用，在复制完后，使用paste将内容粘贴到文本框中。
- backspace：退格。用于删除文字。
- wait：等待。可能程序、网页还在加载，需要等待一会儿。

## 截图与辅助信息
用户会为你提供当前电脑截图，关键组件会被不同颜色的框选并编号，请着重关注图中的框与**编号**。
用户会给你提供的结构如下的辅助信息：
[
​	{"index": 9, "type": "icon", "coordinate": [524, 74], "interactivity": true, "content": "QQ"},
​	{"index": 45, "type": "text", "coordinate": [2793, 1770], "interactivity": false, "content": " 2025/4/7"}
]
- index：编号，**对应截图中的框**。
- type：元素类型，可为 "icon"（图标）或 "text"（文字）。
- coordinate：中心坐标[x, y]。
- interactivity：是否可交互（true/false）。
- content：该元素显示的文本内容或名称。

## 操作历史
用户会给你提供之前的操作历史，每一次操作都包含两部分内容：
- thinking：这一步操作的原因、推理或目的说明。
- action：执行的具体操作。

# 样例
用户的最终目标：打开chatgpt，并询问"who are you"。
在执行当前操作之前，用户曾执行过6次操作：
第1次操作：
thinking：此时需要打开浏览器，根据截图与辅助信息可知，Microsoft Edge是2号框，coordinate是[225, 495]，因此双击打开它。
action： double click (232, 465)

第2次操作：
thinking：可以看到前一次已经打开浏览器了，此时需要进行搜索，根据截图与辅助信息可知，搜索框在93号框附近，coordinate是[763, 285]，单击右侧，选定搜索框。
action： click (928, 291)

第3次操作：
thinking：可以看到前一次已经点击搜索框了，现在需要可以将需要搜索的内容写入了，用户是想打开chatgpt，所以输入chatgpt。
action： type text: chatgpt

第4次操作：
thinking：可以看到前一次已经输入了需要搜索的文本，现在回车就可以搜索。
action： enter

第5次操作：
thinking：可以看到前一次已经搜索了，现在需要在搜索出的结果中打开ChatGPT的网页，根据截图与辅助信息可知，27号框是ChatGPT的网页标题，coordinate是[446, 1121]，点击即可打开。
action： click (472, 1116)

第6次操作：
thinking：可以看到前一次已经打开chatgpt的界面了，现在要点击对话框方便后续打字了，根据截图与辅助信息可知，搜索框在64号框附近，coordinate是[1086, 792]，单击对话框。
action： click (991, 889)

当前电脑截图辅助信息为：
此处略。

当前用户执行的操作是：type write: who are you
请分析他这么做的原因与目的：可以看到前一次已经点击chatgpt的对话框了，用户是想询问"who are you"，现在需要输入"who are you"。

# 注意事项
- 用户没有任何多余操作，当前这一步的操作目的和原因绝对不会和上一步一样。
- 回答时保持**语言精炼**，禁止使用markdown语法，只用plain text回答。
'''


def main():
    tasks = load_tasks('tasks.json')
    for task in tasks:
        task_id = task.get('task', {}).get('id')
        task_zh = task.get('task', {}).get('zh')
        history = task.get('history', [])

        for index, item in enumerate(history):
            action = item.get('action')
            screenshot = item.get('screenshot')
            parser_result = item.get('parser_result')

            file_history = load_file_history(task_id, index)
            user_prompt = generate_user_prompt(task_zh, file_history, parser_result, action)
            print(user_prompt)

            messages = [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f'output_data/image_compressed/{screenshot}'},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)

            thinking = output_text[0] if output_text else ""

            new_history = file_history + [{"thinking": thinking, "action": action}]

            new_data = {
                "id": task_id,
                "task": task_zh,
                "history_num": len(new_history),
                "history": new_history,
            }

            new_file_path = os.path.join('gen_semanic_info_data', f'{task_id}_{index + 1}.json')
            with open(new_file_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()