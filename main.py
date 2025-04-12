import agent

api_key = "xxx"
vlm_name = "qwen2.5-vl-7b-instruct"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
yolo_weight_path = "model_weights/yolo_weights.pt"

gui_agent = agent.Agent(api_key, vlm_name, base_url, yolo_weight_path)
gui_agent.run("xxxxxx")
