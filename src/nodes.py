from .promptprocessor.prompt_use import PromptCombine, PromptEdit, ElementCombine
from .img2element.img_trans_elem import ImageElementGet_1
from .img2element.using_select import ImageElementGet_2
from .elem2prompt.elem_trans_prompt import ElementTransform

NODE_CLASS_MAPPINGS = {
    "PromptCombine": PromptCombine,
    "PromptEdit": PromptEdit,
    "ElementCombine": ElementCombine,
    "ImageElementGet_1": ImageElementGet_1,
    "ImageElementGet_2": ImageElementGet_2,
    "ElementTransform": ElementTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCombine": "提示词合并",
    "PromptEdit": "提示词编辑",
    "ElementCombine": "物体元素合并",
    "ImageElementGet_1": "图像元素提取1",
    "ImageElementGet_2": "图像元素提取2",
    "ElementTransform": "元素转提示词",
}
