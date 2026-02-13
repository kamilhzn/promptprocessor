from inspect import cleandoc
from .prompts import styles
import random


class PromptCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style": ("STRING", {"default": "cocoballking", "description": "写入画风词汇"}),
                "sex": ("STRING", {"default": "1girl", "description": "写入性别有关词汇"}),
                "clothes": ("STRING", {"default": "school uniform", "multiline": True, "description": "写入衣服和装饰词汇"}),
                "hair": ("STRING", {"default": "long hair", "multiline": True, "description": "写入头发词汇"}),
                "face": ("STRING", {"default": "black eyes", "multiline": True, "description": "写入五官词汇"}),
                "weapon": ("STRING", {"default": "", "description": "写入武器词汇"}),
                "obj": ("STRING", {"default": "a cup", "description": "写入物体/武器词汇"}),
                "action": ("STRING", {"default": "holding a cup", "multiline": True, "description": "写入动作词汇"}),
                "others": ("STRING", {"default": "", "multiline": True, "description": "写入其他修饰词汇"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    DESCRIPTION = "Combine various prompt parts into a single prompt."
    FUNCTION = "combine"
    CATEGORY = "Prompt Processor"

    def combine(self, style, sex, clothes, hair, face, weapon, obj, action, others):
        prompt = "very awa, best quality, masterpiece, highres, absurdres, "
        if style != "":
            if style in styles:
                prompt += cleandoc(random.choice(styles[style])) + ", "
            else:
                prompt += "rurudo, "
        if sex != "":
            prompt += cleandoc(sex) + ", solo, full body, "
        if clothes != "":
            prompt += cleandoc(clothes) + ", "
        if hair != "":
            prompt += cleandoc(hair) + ", "
        if face != "":
            prompt += cleandoc(face) + ", "
        if weapon != "":
            prompt += "weapon, " + cleandoc(weapon) + ", "
        if obj != "":
            prompt += cleandoc(obj) + ", "
        if action != "":
            prompt += cleandoc(action) + ", "
        if others != "":
            prompt += cleandoc(others) + ", "

        prompt += "white background, simple background"
        return (prompt,)


class PromptEdit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["exchange", "remove", "fusion"], {"default": "exchange", "description": "选择编辑模式，交换、删除或融合"}),
                "obj_str": ("STRING", {"default": "", "description": "要编辑的对象字符串"}),
            },
        }

    RETURN_TYPES = ("String",)
    DESCRIPTION = "Edit a prompt string."
    FUNCTION = "edit"
    CATEGORY = "Prompt Processor"

    def edit(self, mode, obj_str):
        if mode == "exchange":
            return (
                f"将图1中{obj_str}替换掉图2中的{obj_str}，保持图1的轮廓形状与纹理细节，通过改变其方向与透视来使图1的{obj_str}完美地融入图2",
            )
        elif mode == "remove":
            return (f"将图2中的{obj_str}移除",)
        elif mode == "fusion":
            return (
                f"将图1中的{obj_str}与图2中的{obj_str}融合，保持图1的轮廓形状与纹理细节,保持图2的整体构图与风格，通过改变其方向与透视来使图1的{obj_str}完美地融入图2",
            )
