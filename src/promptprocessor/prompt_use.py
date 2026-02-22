from inspect import cleandoc
from .prompts import styles, using_prompt
import random


class PromptCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style": ("STRING", {"default": "cocoballking", "tooltip": "写入画风词汇"}),
                "person_prompt": ("STRING", {"default": "1girl", "tooltip": "有关人物的提示词"}),
            },
            "optional": {
                "others": ("STRING", {"default": "", "multiline": True, "tooltip": "写入其他修饰词汇"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    DESCRIPTION = "Combine various prompt parts into a single prompt."
    FUNCTION = "combine"
    CATEGORY = "提示词处理"

    def combine(self, style, person_prompt, others):
        prompt = "very awa, best quality, masterpiece, highres, absurdres, "
        if style != "":
            if style in styles:
                prompt += cleandoc(random.choice(styles[style])) + ", "
            else:
                prompt += "rurudo, "
        if person_prompt != "":
            prompt += cleandoc(person_prompt)
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
                "mode": (["exchange", "remove", "fusion"], {"default": "exchange", "tooltip": "选择编辑模式，交换、删除或融合"}),
                "using_str": ("STRING", {"default": "", "tooltip": "要编辑的对象用途词汇，如背景、手持物、饰品等"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("编辑性语句",)
    DESCRIPTION = "Edit a prompt string."
    FUNCTION = "edit"
    CATEGORY = "提示词处理"

    def edit(self, mode, using_str):
        if mode == "exchange":
            return (
                f"将图1中{using_str}替换掉图2中的{using_str}，保持图1的轮廓形状与纹理细节，通过改变其方向与透视来使图1的{using_str}完美地融入图2",
            )
        elif mode == "remove":
            return (f"将图2中的{using_str}移除",)
        elif mode == "fusion":
            return (using_prompt["fusion"][using_str],)
