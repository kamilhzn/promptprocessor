import torch, os
from .model_and_lables import device, MODEL_CATEGORY, purpose
import folder_paths
import json


def lisan_output(kind_predict, kind_names):
    # 将输出转化成概率
    kind_predict = torch.softmax(kind_predict, dim=1)

    # 找到最大的概率
    max_kind, max_index = torch.max(kind_predict, dim=1)

    # 将张量移至cpu中处理并转化成numpy数组
    max_kind = max_kind.detach().cpu().numpy()
    max_index = max_index.detach().cpu().numpy()

    # 确定最大概率的名字
    max_classes = [kind_names[i] for i in max_index]

    return max_classes


class ImageElementGet_2:
    def __init__(self):
        """初始化：加载模型并设置为推理模式"""
        self.model = None
        self.current_model_path = None  # 记录当前加载的模型路径，避免重复加载

    def load_model(self, model_name):
        """根据选择的模型名加载模型（带缓存，避免重复加载）"""
        # 获取模型的完整路径
        model_path = folder_paths.get_full_path(MODEL_CATEGORY, model_name)

        # 如果模型路径未变，无需重新加载
        if self.current_model_path == model_path and self.model is not None:
            return

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到：{model_path}")

        # 加载模型并移至对应设备
        self.model = torch.jit.load(model_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()

        # 更新当前模型路径
        self.current_model_path = model_path

    @classmethod
    def INPUT_TYPES(s):
        model_list = folder_paths.get_filename_list(MODEL_CATEGORY)
        return {
            "required": {
                "pic": ("IMAGE", {"tooltip": "输入要提取元素的图像"}),
                "model_path": (model_list if model_list else ["请放入用途预测模型purpose.pt到element_get目录"],),
                "input_str": ("STRING", {"tooltip": "输入元素字符串1"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("文物元素串", "用途")
    FUNCTION = "using_select"
    CATEGORY = "提示词处理/物体元素提取"
    DESCRIPTION = "Get a specific element from an image."

    # 输入为BHWC的tensor数组，输出就是处理过并且模型接受的BCHW
    def preprocess(self, imgs, device=device):
        """
        输入:
            imgs: Tensor (B, H, W, C)
        输出:
            Tensor (B, C, H, W)
        """
        # 1️⃣ BHWC → BCHW
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        imgs = imgs / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)

        imgs = (imgs - mean) / std

        return imgs.to(device)

    def using_select(self, pic, model_path, input_str):
        """
        img: ComfyUI图片张量 [1, H, W, C]
        model_path: 模型文件名
        input_str: 输入的元素字符串
        该函数用于处理图片，获取用途信息，并将输入字符串作为附加信息返回
        """
        process_img = self.preprocess(pic, device)  # 预处理图片
        self.load_model(model_path)  # 加载模型
        output = self.model(process_img)  # 获取模型输出
        kind_result = lisan_output(output["purpose"], purpose)  # 获取用途结果

        input_string = json.loads(input_str) if input_str else {}
        input_string["用途"] = kind_result[0]  # 将预测的用途添加到输入字符串中
        input_string = json.dumps(input_string, ensure_ascii=False)
        return (input_string, kind_result[0])
