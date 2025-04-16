import json
import os

def create_task_dict(task_id, zh_desc, en_desc, history_num, history_list):
    """
    创建一个任务字典，包含任务信息、历史记录数量和历史记录列表。

    :param task_id: 任务的唯一标识符
    :param zh_desc: 任务的中文描述
    :param en_desc: 任务的英文描述
    :param history_num: 历史记录的数量
    :param history_list: 历史记录列表，每个元素是一个包含 action、thinking、screenshot 和 parser result 的字典
    :return: 包含任务信息和历史记录的字典
    """
    task_dict = {
        "task": {
            "id": task_id,
            "zh": zh_desc,
            "en": en_desc,
        },
        "history_num": history_num,
        "history": history_list
    }
    return task_dict

def create_and_save_35_tasks(file_path):
    all_tasks = []
    for i in range(35):
        task_id = f"task_{i + 1}"
        zh_desc = f"示例任务 {i + 1}"
        en_desc = f"Example Task {i + 1}"
        history_num = 2
        history_list = [
            {
                "action": f"action_{i + 1}_1",
                "thinking": f"thinking_{i + 1}_1",
                "screenshot": f"screenshot_{i + 1}_1",
                "parser result": [i * 2, i * 2 + 1]
            },
            {
                "action": f"action_{i + 1}_2",
                "thinking": f"thinking_{i + 1}_2",
                "screenshot": f"screenshot_{i + 1}_2",
                "parser result": [i * 2 + 2, i * 2 + 3]
            }
        ]
        task = create_task_dict(task_id, zh_desc, en_desc, history_num, history_list)
        all_tasks.append(task)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=4)

def update_task_en_values(pc_agent_tasks, tasks):
    try:
        # 读取 pc_agent_tasks.json 文件
        with open(pc_agent_tasks, 'r', encoding='utf-8') as pc_agent_file:
            pc_agent_data = json.load(pc_agent_file)

        # 读取 tasks.json 文件
        with open(tasks, 'r', encoding='utf-8') as tasks_file:
            tasks_data = json.load(tasks_file)

        # 检查两个文件的长度是否一致
        if len(pc_agent_data) != len(tasks_data):
            print("两个文件中的数据数量不一致，请检查文件内容。")
            return

        # 遍历并更新 tasks.json 中的 "en" 字段
        for i in range(len(pc_agent_data)):
            tasks_data[i]["task"]["en"] = pc_agent_data[i]["task"]

        # 将更新后的数据写回 tasks.json 文件
        with open(tasks, 'w', encoding='utf-8') as tasks_file:
            json.dump(tasks_data, tasks_file, ensure_ascii=False, indent=4)

        print("任务的英文名称已成功更新。")

    except FileNotFoundError:
        print("指定的文件未找到，请检查文件路径。")
    except Exception as e:
        print(f"发生错误: {e}")

def update_tasks_with_events(tasks_file_path):
    try:
        # 读取 tasks.json 文件
        with open(tasks_file_path, 'r', encoding='utf-8') as tasks_file:
            tasks_data = json.load(tasks_file)

        for task_dict in tasks_data:
            # 获取 task 中的 id
            task_id = task_dict["task"]["id"]
            # 生成对应的文件名
            file_name = f"{task_id}.jsonl"
            file_path = os.path.join('events', file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                event_lines = []
                # 读取文件内容并提取 action 和 screenshot 字段
                with open(file_path, 'r', encoding='utf-8') as event_file:
                    for line in event_file:
                        try:
                            data = json.loads(line)
                            event_lines.append({
                                "action": data.get("action"),
                                "screenshot": data.get("screenshot")
                            })
                        except json.JSONDecodeError:
                            print(f"解析 {file_path} 中的行 {line} 时出错")

                # 更新 history 和 history_num
                task_dict["history"] = event_lines
                task_dict["history_num"] = len(event_lines)
            else:
                print(f"未找到文件: {file_path}")

        # 将更新后的数据写回 tasks.json 文件
        with open(tasks_file_path, 'w', encoding='utf-8') as tasks_file:
            json.dump(tasks_data, tasks_file, ensure_ascii=False, indent=4)

        print("tasks.json 文件已成功更新。")

    except FileNotFoundError:
        print("指定的文件未找到，请检查文件路径。")
    except Exception as e:
        print(f"发生错误: {e}")

def update_screenshot_paths_and_add_parser_result(tasks_file_path):
    try:
        # 读取 tasks.json 文件
        with open(tasks_file_path, 'r', encoding='utf-8') as tasks_file:
            tasks_data = json.load(tasks_file)

        # 遍历每个任务字典
        for task_dict in tasks_data:
            # 遍历 history 列表中的每个字典
            for history_item in task_dict.get('history', []):
                screenshot_path = history_item.get('screenshot', '')
                if screenshot_path.startswith('events\\screenshot\\'):
                    # 提取文件名部分
                    file_name = os.path.basename(screenshot_path)
                    # 替换 .png 为 .json
                    json_file_name = file_name.replace('.png', '.json')
                    json_file_path = os.path.join('output_data', 'json', json_file_name)

                    # 更新 screenshot 字段
                    history_item['screenshot'] = file_name

                    # 检查 JSON 文件是否存在
                    if os.path.exists(json_file_path):
                        try:
                            # 加载 JSON 文件内容
                            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                                parser_result = json.load(json_file)
                            # 添加 parser_result 字段到 history 项中
                            history_item['parser_result'] = parser_result
                        except json.JSONDecodeError:
                            print(f"解析 {json_file_path} 时出错")
                    else:
                        print(f"未找到文件: {json_file_path}")

        # 将更新后的数据写回 tasks.json 文件
        with open(tasks_file_path, 'w', encoding='utf-8') as tasks_file:
            json.dump(tasks_data, tasks_file, ensure_ascii=False, indent=4)

        print("tasks.json 文件已成功更新。")

    except FileNotFoundError:
        print("指定的文件未找到，请检查文件路径。")
    except Exception as e:
        print(f"发生错误: {e}")


def split_task_history(tasks_file_path):
    # 读取 tasks.json 文件
    with open(tasks_file_path, 'r', encoding='utf-8') as file:
        tasks_data = json.load(file)

    # 提取 "task" 中的 "id" 和 "zh"
    if 'task' in tasks_data:
        task = tasks_data['task']
        task_info = {
            'id': task.get('id'),
            'task': task.get('zh')
        }

    # 处理 history 部分
    if 'history' in tasks_data:
        history = tasks_data['history']
        n = len(history)
        # 创建保存文件的目录
        save_dir = 'gen_semanic_info_data'

        for i in range(n + 1):
            # 截取前 i 个历史记录
            partial_history = history[:i]
            # 构建包含任务信息和部分历史记录的数据
            output_data = {
                'id': task_info['id'],
                'task': task_info['task'],
                'history_num': i,
                'history': partial_history,
            }
            # 生成保存的文件名，使用 id + i 的方式
            file_name = os.path.join(save_dir, f"{task_info['id']}_{i}.json")
            # 保存为单独的JSON文件
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)

def get_task_init_history(tasks_file_path):
    try:
        # 读取 tasks.json 文件
        with open(tasks_file_path, 'r', encoding='utf-8') as file:
            tasks_data = json.load(file)

        # 创建保存文件的目录
        save_dir = 'gen_semanic_info_data'

        # 遍历 tasks_data 中的每个元素
        for task_dict in tasks_data:
            task_info = task_dict.get('task', {})
            task_id = task_info.get('id')
            zh_desc = task_info.get('zh')

            if task_id and zh_desc:
                output_data = {
                    'id': task_id,
                    'task': zh_desc,
                    'history_num': 0,
                    'history': [],
                }

                # 生成保存的文件名，使用 id + _0 的方式
                file_name = os.path.join(save_dir, f"{task_id}_0.json")
                # 保存为单独的JSON文件
                with open(file_name, 'w', encoding='utf-8') as output_file:
                    json.dump(output_data, output_file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print("未找到 tasks.json 文件，请检查文件路径。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    tasks_json_path = 'tasks.json'
    pc_agent_tasks_json_path = 'pc_agent_tasks.json'
    #create_and_save_35_tasks(tasks_json_path)
    #update_task_en_values(pc_agent_tasks_json_path, tasks_json_path)
    #update_tasks_with_events(tasks_json_path)
    #update_screenshot_paths_and_add_parser_result(tasks_json_path)
    #split_task_history(tasks_json_path)
    get_task_init_history(tasks_json_path)