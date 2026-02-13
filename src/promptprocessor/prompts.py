# 记录提示词库
styles = {
    "锐利硬朗": ["blade (galaxist)", "chainsaw man", "black rock shooter", "helltaker", "project moon"],
    "圆润柔和": ["kemono friends", "yuru yuri", "furball", "daigaijin", "aogami", "naga u"],
    "细腻精致": [
        "fate/grand order",
        "arknights",
        "honkai (series)",
        "granblue fantasy",
        "princess connect!",
        "dairi",
        "kou hiyoyo",
        "erobos",
    ],
    "简约干净": ["girls und panzer", "touken ranbu", "neptune (series)", "danganronpa (series)", "tani takeshi", "hammer (sunset beach)"],
    "传统和风": ["ruu (tksymkw)", "itomugi-kun"],
    "青春日常": ["girls band cry", "persona", "kouji (campus life)"],
}


if __name__ == "__main__":
    import random

    prompt = random.choice(styles["细腻精致"])
    print(prompt)
