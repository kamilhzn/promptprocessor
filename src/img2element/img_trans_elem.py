from .path_and_lables import device, MODEL_CATEGORY, kind_labels, dynasty_labels, purpose, integrity, repair, corrosion
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from .lab import primary_secondary_richness_lab, get_finally_level
import os, folder_paths
import numpy as np
import colorsys
import json

# def lisan_output(kind_predict, kind_names):
#     # 将输出转化成概率
#     kind_predict = torch.softmax(kind_predict, dim=1)

#     # 找到最大的概率
#     max_kind, max_index = torch.max(kind_predict, dim=1)

#     # 将张量移至cpu中处理并转化成numpy数组
#     max_kind = max_kind.detach().cpu().numpy()
#     max_index = max_index.detach().cpu().numpy()

#     # 确定最大概率的名字
#     max_classes = [kind_names[i] for i in max_index]

#     return max_classes


# 输入为BHWC的tensor数组，输出就是处理过并且模型接受的BCHW
def preprocess(imgs):
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


def lisan_output(kind_predict, kind_names):
    # 将输出转化成概率
    kind_predict = torch.softmax(kind_predict, dim=1)

    # 找到最大的概率
    _, max_index = torch.max(kind_predict, dim=1)

    # 将张量移至cpu中处理并转化成numpy数组
    max_index = max_index.detach().cpu().numpy()

    # 确定最大概率的名字
    max_classes = kind_names[max_index[0]]

    return max_classes


class ResizeAndPad:
    def __init__(self, size=256, fill=(128, 128, 128)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

        img = F.pad(img, padding, fill=self.fill)
        return img


class ImageElementGet:
    def __init__(self):
        """初始化：加载模型并设置为推理模式"""
        self.kd_model = None
        self.purpose_model = None
        self.irc_model = None
        self.current_model_path = None  # 记录当前加载的模型路径，避免重复加载

    def load_model(self):
        """加载模型（带缓存，避免重复加载）"""
        kd_model_name = "kind_dynasty.pt"
        purpose_model_name = "purpose.pt"
        irc_model_name = "integrity_repair_corrosion.pt"
        # 获取模型的完整路径
        kd_model_path = folder_paths.get_full_path(MODEL_CATEGORY, kd_model_name)
        purpose_model_path = folder_paths.get_full_path(MODEL_CATEGORY, purpose_model_name)
        irc_model_path = folder_paths.get_full_path(MODEL_CATEGORY, irc_model_name)

        # 如果模型路径未变，无需重新加载
        # if self.current_model_path == model_path and self.model is not None:
        #     return

        # 检查模型文件是否存在
        if not os.path.exists(kd_model_path):
            raise FileNotFoundError(f"模型文件未找到：{kd_model_path}")
        if not os.path.exists(purpose_model_path):
            raise FileNotFoundError(f"模型文件未找到：{purpose_model_path}")
        if not os.path.exists(irc_model_path):
            raise FileNotFoundError(f"模型文件未找到：{irc_model_path}")

        # 加载模型并移至对应设备
        self.kd_model = torch.jit.load(kd_model_path, map_location=device)
        self.kd_model = self.kd_model.to(device)
        self.kd_model.eval()

        self.purpose_model = torch.jit.load(purpose_model_path, map_location=device)
        self.purpose_model = self.purpose_model.to(device)
        self.purpose_model.eval()

        self.irc_model = torch.jit.load(irc_model_path, map_location=device)
        self.irc_model = self.irc_model.to(device)
        self.irc_model.eval()

    def preprocess_image(self, comfy_image, mask=None):
        """处理ComfyUI格式的图片（结合mask裁剪有效区域）"""
        # 转换tensor格式：[1, H, W, C] -> [C, H, W]，并缩放至0-255
        img_tensor = comfy_image.squeeze(0).permute(2, 0, 1) * 255.0
        img_tensor = img_tensor.to(torch.uint8)

        # 转换为PIL Image
        img = F.to_pil_image(img_tensor)
        img = img.convert("RGB")

        # 裁剪mask有效区域
        if mask is not None:
            mask_tensor = mask.squeeze(0) if len(mask.shape) == 3 else mask  # [H, W]
            mask_np = mask_tensor.cpu().numpy()
            # 找到mask有效区域的外接矩形
            ys, xs = np.where(mask_np > 0.0)
            if len(xs) > 0 and len(ys) > 0:
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                # 裁剪图片到mask有效区域
                img = img.crop((xmin, ymin, xmax + 1, ymax + 1))

        transform = transforms.Compose(
            [
                # 等比缩放并填充：短边 = 256,
                ResizeAndPad(size=256, fill=(128, 128, 128)),
                # Tensor & Normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 转化图片至输入张量
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        return img

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pic": ("IMAGE", {"tooltip": "输入要提取元素的图像"}),
                "mask": ("MASK", {"tooltip": "蒙版"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("文物元素串", "主色调", "副色调")
    FUNCTION = "get_element"
    CATEGORY = "提示词处理/物体元素提取"
    DESCRIPTION = "Get a specific element from an image."

    def get_element(self, pic, mask):
        # 加载模型
        self.load_model()
        # 初始化完整的字段字典
        result_dict = {
            "主色相": "",
            # "主饱和度": "",
            # "主亮度": "",
            "副色相": "",
            # "副饱和度": "",
            # "副亮度": "",
            "色彩丰富度": "",
            "类别": "",
            "用途": "",
            "长宽比": "",
            "朝代": "",
            "完整程度": "",
            "修补痕迹": "",
            # "光泽": "",
            # "图案性质": "",
            "锈蚀痕迹": "",
        }

        # 处理颜色与丰富度
        p, s, richness = primary_secondary_richness_lab(pic, mask, de_th=12, primary_ratio=0.1, secondary_ratio=0.01)

        primary = []  # 中文版
        primary_en = []  # 英文版
        secondary = []  # 中文版
        secondary_en = []  # 英文版

        # 转换主色调
        for c in p:
            primary.append(self.map_rgb_to_discrete(c, english=False))
            primary_en.append(self.map_rgb_to_discrete(c, english=True))

        # 转换副色调
        for c in s:
            secondary.append(self.map_rgb_to_discrete(c, english=False))
            secondary_en.append(self.map_rgb_to_discrete(c, english=True))

        # 填充颜色相关字段 - 使用逗号分隔的字符串，而不是列表
        result_dict["主色相"] = primary[0] if primary else ""
        result_dict["副色相"] = secondary[0] if secondary else ""
        result_dict["色彩丰富度"] = richness

        # 处理朝代与类型
        # self.load_model(model_path)
        processed_img = self.preprocess_image(pic, mask)

        with torch.no_grad():
            output = self.kd_model(processed_img)

        kind_result = lisan_output(output["kind"], kind_labels)
        dynasty_result = lisan_output(output["dynasty"], dynasty_labels)

        # 填充类别和朝代
        if kind_result:
            result_dict["类别"] = kind_result
        if dynasty_result:
            result_dict["朝代"] = dynasty_result

        # 处理宽高比
        hw_level = get_finally_level(pic, mask=mask)
        result_dict["长宽比"] = hw_level if hw_level else "近方"

        # 主副色调使用英文版单词（逗号分隔的字符串）
        primary_str = ",".join(primary_en) if primary_en else ""
        secondary_str = ",".join(secondary_en) if secondary_en else ""

        # 用途检测
        pre_pic = preprocess(pic)
        with torch.no_grad():
            purpose_output = self.purpose_model(pre_pic)

        purpose_result = lisan_output(purpose_output["purpose"], purpose)
        if purpose_result:
            result_dict["用途"] = purpose_result

        # 完整度、修补痕迹、锈蚀痕迹检测
        with torch.no_grad():
            output = self.irc_model(pre_pic)

        integrity_result = lisan_output(output["integrity"], integrity)
        repair_result = lisan_output(output["repair"], repair)
        corrosion_result = lisan_output(output["corrosion"], corrosion)
        if integrity_result:
            result_dict["完整程度"] = integrity_result
        if repair_result:
            result_dict["修补痕迹"] = repair_result
        if corrosion_result:
            result_dict["锈蚀痕迹"] = corrosion_result

        # 构建符合 DataFrame 解析格式的 JSON 字符串
        input_string = json.dumps(result_dict, ensure_ascii=False)

        return (input_string, primary_str, secondary_str, str(richness), result_dict["长宽比"], result_dict["类别"], result_dict["朝代"])

    def map_rgb_to_discrete(self, rgb, english=False):
        """
        将 RGB 三元组映射到离散颜色：红 橙 黄 绿 青 蓝 紫 黑 白 灰 棕
        使用 HSV (h,s,v) 分区，并结合饱和度/明度判断黑白灰/棕。
        返回中文或英文名称。
        """
        # rgb: (R,G,B) ints 0-255
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360

        # 先判定黑白灰
        if v <= 0.18:
            name = "黑" if not english else "black"
            return name

        if s <= 0.2:
            if v >= 0.88:
                name = "白" if not english else "white"
            else:
                name = "灰" if not english else "gray"
            return name

        # 棕色判定：色相偏橙/黄且明度偏低
        if 15 <= h_deg < 50 and v < 0.6:
            name = "棕" if not english else "brown"
            return name

        # 色相区间映射
        if h_deg < 15 or h_deg >= 345:
            name = "红" if not english else "red"
        elif 15 <= h_deg < 45:
            name = "橙" if not english else "orange"
        elif 45 <= h_deg < 65:
            name = "黄" if not english else "yellow"
        elif 65 <= h_deg < 170:
            name = "绿" if not english else "green"
        elif 170 <= h_deg < 200:
            name = "青" if not english else "cyan"
        elif 200 <= h_deg < 260:
            name = "蓝" if not english else "blue"
        else:
            name = "紫" if not english else "purple"

        return name
