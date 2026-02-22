from .promptprocessor.prompt_use import PromptCombine, PromptEdit
from .img2element.img_trans_elem import ImageElementGet
from .elem2prompt.elem_trans_prompt import ElementTransform

NODE_CLASS_MAPPINGS = {
    "PromptCombine": PromptCombine,
    "PromptEdit": PromptEdit,
    "ImageElementGet": ImageElementGet,
    "ElementTransform": ElementTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCombine": "提示词合并",
    "PromptEdit": "提示词编辑",
    "ImageElementGet": "图像元素提取",
    "ElementTransform": "元素转提示词",
}
