from inspect import cleandoc


class PromptCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style": ("STRING", {"default": "cocoballking", "description": "写入画风词汇"}),
                "sev": ("STRING", {"default": "1girl", "description": "写入性别有关词汇"}),
                "clothes": ("STRING", {"default": "school uniform", "multiline": True, "description": "写入衣服和装饰词汇"}),
                "hair": ("STRING", {"default": "long hair", "multiline": True, "description": "写入头发词汇"}),
                "face": ("STRING", {"default": "black eyes", "multiline": True, "description": "写入五官词汇"}),
                "weapon": ("STRING", {"default": "", "description": "写入武器词汇"}),
                "object": ("STRING", {"default": "a cup", "description": "写入物体/武器词汇"}),
                "action": ("STRING", {"default": "holding a cup", "multiline": True, "description": "写入动作词汇"}),
                "others": ("STRING", {"default": "", "multiline": True, "description": "写入其他修饰词汇"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    DESCRIPTION = "Combine various prompt parts into a single prompt."
    FUNCTION = "combine"
    CATEGORY = "Prompt Processor"

    def combine(self, style, sex, clothes, hair, face, weapon, object, action, others):
        prompt = "very awa, best quality, masterpiece, highres, absurdres, "
        if style != "":
            prompt += cleandoc(style) + ", "
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
        if object != "":
            prompt += cleandoc(object) + ", "
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


class ImageElementGet:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pic": ("IMAGE", {"default": None, "description": "输入要提取元素的图像"}),
                "model": ("MODEL", {"default": "yolov5", "description": "选择使用的目标检测模型"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("element_str", "color_str")
    FUNCTION = "get_element"
    CATEGORY = "Prompt Processor/image2prompt"
    DESCRIPTION = "Get a specific element from an image."

    def get_element(self, pic, model):
        # TODO:完善模型的调用，目前先返回占位字符串
        elem_str = ""
        color_str = ""
        return (elem_str, color_str)


class ElementTransform:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "element_str": ("STRING", {"default": "", "description": "输入要转换的元素描述"}),
                "color_str": ("STRING", {"default": "", "description": "输入要转换的颜色描述"}),
                "model": ("MODEL", {"default": "cocoballking", "description": "选择使用的转换模型"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "transformed_element_str",
        "color_str",
        "style_str",
        "sex_str",
        "clothes_str",
        "hair_str",
        "face_str",
        "weapon_str",
        "object_str",
    )
    FUNCTION = "transform_element"
    CATEGORY = "Prompt Processor/element2prompt"
    DESCRIPTION = "Transform an element description."

    def transform_element(self, element_str, color_str):
        # TODO:完善元素转换逻辑，目前先返回占位字符串
        style = ""
        sex = ""
        clothes = ""
        hair = ""
        face = ""
        weapon = ""
        object = ""
        action = ""
        others = ""
        return (style, sex, clothes, hair, face, weapon, object, action, others)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PromptCombine": PromptCombine,
    "PromptEdit": PromptEdit,
    "ImageElementGet": ImageElementGet,
    "ElementTransform": ElementTransform,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCombine": "提示词合并",
    "PromptEdit": "提示词编辑",
    "ImageElementGet": "图像元素提取",
    "ElementTransform": "元素转人物词",
}
