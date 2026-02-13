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
