# 记录提示词库
styles = {
    "锐利硬朗": [
        "blade (galaxist)",
        "chainsaw man",
        "black rock shooter",
        "helltaker",
        "project moon",
        "akakura",
    ],
    "圆润柔和": ["kemono friends", "yuru yuri", "furball", "daigaijin", "aogami", "naga u", "torino aqua"],
    "细腻精致": [
        "fate/grand order",
        "arknights",
        "honkai (series)",
        "granblue fantasy",
        "princess connect!",
        "dairi",
        "kou hiyoyo",
        "erobos",
        "torino, rella",
    ],
    "简约干净": ["girls und panzer", "touken ranbu", "neptune (series)", "danganronpa (series)", "tani takeshi", "hammer (sunset beach)"],
    "传统和风": ["ruu (tksymkw)", "itomugi-kun"],
    "青春日常": ["girls band cry", "persona", "kouji (campus life)", "gomzi"],
}

# 用途：背景、手持物、手捧物、饰品、悬挂物、倚靠物、展示物
using_prompt = {
    "fusion": {
        "background": ["将图2中的物体迁移到图1作为背景物体，占据次要部分，要求自然地融入图1，保持图1的整体构图与风格"],
        "handheld": [
            "将图2中的物体迁移到图1作为手持物，图1角色用手拿着图2物体，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
        "handheld2": [
            "将图2中的物体迁移到图1，图1角色用手捧着图2物体，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
        "accessory": [
            "将图2中的物体迁移到图1作为饰品，图1角色佩戴图2物体或衣服上加上图2物体作为装饰，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
        "hanging": [
            "将图2中的物体迁移到图1作为悬挂物，图1角色头顶上方悬挂着图2物体，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
        "leaning": [
            "将图2中的物体迁移到图1作为倚靠物，图1角色倚靠在图2物体上，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
        "display": [
            "将图2中的物体迁移到图1作为展示物，图2物体放置于角色旁，保持图2物体形状轮廓与纹理不变，自然地融入图1，保持图1的整体构图与风格"
        ],
    }
}

if __name__ == "__main__":
    import random

    prompt = random.choice(styles["细腻精致"])
    print(prompt)
