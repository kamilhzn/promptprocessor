import torch, os
import folder_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型路径配置
MODEL_CATEGORY = "element_get"  # 自定义模型分类名
# 注册模型目录到ComfyUI（让ComfyUI识别该目录）
folder_paths.add_model_folder_path(MODEL_CATEGORY, os.path.join(folder_paths.models_dir, MODEL_CATEGORY))


# 分类标签
kind_labels = ["铜器", "金银器", "漆器", "珐琅器", "玉石器", "雕塑", "陶瓷", "其他"]
dynasty_labels = [
    "夏",
    "商",
    "周",
    "春秋",
    "战国",
    "秦",
    "汉",
    "三国",
    "晋",
    "南北朝",
    "隋",
    "唐",
    "五代十国",
    "辽",
    "宋",
    "金",
    "元",
    "明",
    "清",
    "近现代",
]
# 用途标签
purpose = ["背景", "手持物", "手捧物", "饰品", "展示物"]
