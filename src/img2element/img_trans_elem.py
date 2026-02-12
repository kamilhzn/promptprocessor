import torch, os
import torchvision.transforms.functional as F
from torchvision import transforms
from .primary_secendary_rgb_and_richness import primary_secondary_richness_lab
import folder_paths
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型路径配置
PATH = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PATH, "/models/kind_dynasty.pt")
MODEL_CATEGORY = "kind_dynasty"  # 自定义模型分类名
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


def get_bbox_from_comfy_image(comfy_image, mask=None, alpha_threshold=1):
    """
    从ComfyUI的IMAGE张量中计算主体外接矩形的宽高比（width / height）
    :param comfy_image: ComfyUI图片节点输出的张量，shape=[1, H, W, C]，取值范围0-1
    :param mask: ComfyUI蒙版张量，shape=[B, H, W] 或 [H,W]，取值范围0-1（None则使用原图alpha）
    :param alpha_threshold: 透明通道阈值（仅当无mask且图片有4通道时生效）
    :return: 宽高比（width/height），无主体时返回None
    """
    # 处理ComfyUI张量格式：移除batch维度 + 转为numpy数组 + 缩放至0-255
    img_tensor = comfy_image.squeeze(0)  # [H, W, C]
    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # 转为0-255的numpy数组

    # 使用mask筛选有效像素，无mask则用alpha通道
    if mask is not None:
        # 处理mask形状：统一转为 [H, W]
        mask_tensor = mask.squeeze(0) if len(mask.shape) == 3 else mask  # 移除batch维度
        mask_np = mask_tensor.cpu().numpy()  # [H, W]，0-1范围
        ys, xs = np.where(mask_np > 0.0)  # 蒙版值>0的区域为有效主体
    else:
        # 无mask：用alpha通道（有则用，无则全选）
        if img_np.shape[-1] == 4:
            alpha = img_np[:, :, 3]
            ys, xs = np.where(alpha > alpha_threshold)
        else:
            h, w = img_np.shape[:2]
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            ys = ys.flatten()
            xs = xs.flatten()

    # 检查是否有主体区域
    if len(xs) == 0:
        return None  # 无主体

    # 计算外接矩形的边界
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # 计算宽高和宽高比
    width = xmax - xmin
    height = ymax - ymin

    if height == 0:
        return None  # 避免除零错误

    ratio = width / height
    return ratio


class ImageElementGet:
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
        model_list = folder_paths.get_filename_list(MODEL_CATEGORY)
        return {
            "required": {
                "pic": ("IMAGE", {"description": "输入要提取元素的图像"}),
                "model_name": (model_list if model_list else ["请放入模型到kind_dynasty目录"],),
                "mask": ("MASK", {"description": "蒙版"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("文物类型", "所属朝代", "主色调", "副色调", "色彩丰富度", "宽高比")
    FUNCTION = "get_element"
    CATEGORY = "Prompt Processor/image2prompt"
    DESCRIPTION = "Get a specific element from an image."

    def get_element(self, pic, model_name, mask):
        # 处理朝代与类型
        self.load_model(model_name)  # 加载选择的模型
        processed_img = self.preprocess_image(pic, mask)  # 图片预处理

        with torch.no_grad():  # 模型推理（关闭梯度计算加速）
            output = self.model(processed_img)

        kind_result = lisan_output(output["kind"], kind_labels)
        dynasty_result = lisan_output(output["dynasty"], dynasty_labels)

        # 处理颜色与丰富度
        p, s, r = primary_secondary_richness_lab(pic, mask=mask)

        # 处理宽高比
        ratio = get_bbox_from_comfy_image(pic, mask=mask, alpha_threshold=1)

        return (kind_result, dynasty_result, str(p), str(s), r, ratio if ratio else 0.0)
