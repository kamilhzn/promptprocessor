import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 把当前目录加入Python路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import model as local_model

sys.modules["model"] = local_model  # 让pickle找到 model 模块

import joblib as jl
import folder_paths
import os
import json
import pandas as pd

MODEL_CATEGORY = "prompt_decoder"  # 自定义模型分类名
# 注册模型目录到ComfyUI（让ComfyUI识别该目录）
folder_paths.add_model_folder_path(MODEL_CATEGORY, os.path.join(folder_paths.models_dir, MODEL_CATEGORY))


class ElementTransform:
    def __init__(self):
        """初始化模型"""
        self.model = None
        self.current_model_path = None  # 记录当前加载的模型路径，避免重复加载

    def load_model(self, model_path):
        """加载模型"""
        # 获取模型的完整路径
        model_path = folder_paths.get_full_path(MODEL_CATEGORY, model_path)

        # 如果模型路径未变，无需重新加载
        if self.current_model_path == model_path and self.model is not None:
            return

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")

        # 加载模型并移至对应设备
        self.model = jl.load(model_path)
        # self.model.eval()

        # 更新当前模型路径
        self.current_model_path = model_path

    @classmethod
    def INPUT_TYPES(s):
        model_list = folder_paths.get_filename_list(MODEL_CATEGORY)
        return {
            "required": {
                "input_str": ("STRING", {"default": "", "tooltip": "输入要转换的元素描述"}),
                "model_path": (model_list if model_list else ["请放入模型到prompt_decoder目录"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("提示词", "面部特征描述", "手部特征描述", "风格描述")
    FUNCTION = "transform_element"
    CATEGORY = "提示词处理/element2prompt"
    DESCRIPTION = "Transform an element description."

    def transform_element(self, input_str, model_path):
        # 解析字符串 - 现在 input_str 是标准 JSON 对象，值是字符串
        data = json.loads(input_str)

        # 创建 DataFrame（单行）
        df = pd.DataFrame([data])

        self.load_model(model_path)  # 确保模型已加载

        if self.model is not None:
            # 使用 predict_comfyui 的简化版本，或者直接用 predict
            # 注意：模型期望的列顺序可能与 data 不完全一致
            # 确保 DataFrame 的列顺序与训练时一致
            result = self.model.predict(df, return_description=True)

        face_desc = ""
        hand_desc = ""
        style_desc = ""

        return (result, face_desc, hand_desc, style_desc)
