from inspect import cleandoc


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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "颜色",
        "画风",
        "性别",
        "衣物",
        "头发",
        "面部",
        "武器",
        "物体",
        "动作",
        "其他",
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
        return (color_str, style, sex, clothes, hair, face, weapon, object, action, others)
