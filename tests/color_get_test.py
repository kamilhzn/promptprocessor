import torch
import numpy as np
from img2element.color_trans import primary_secondary_richness_lab


def test_main():
    """
    测试 primary_secondary_richness_lab 函数，输出主色调、副色调和丰富度。
    生成一个 100x100 的图像，包含：
      - 红色区域 (50x50)
      - 绿色区域 (50x50)
      - 蓝色区域 (50x50)
      - 白色区域 (50x50)
      - 黑色背景 (50x50)
      - 灰色区域 (50x50)
    """
    # 图像尺寸
    H, W = 300, 300
    # 创建 RGB 图像 (H, W, 3)，值范围 0-1
    img = np.zeros((H, W, 3), dtype=np.float32)

    # 填充颜色区域（每个区域 100x100）
    # 红色 (1,0,0) 左上角
    img[0:100, 0:100, :] = (1.0, 0.0, 0.0)
    # 绿色 (0,1,0) 右上角
    img[0:100, 100:200, :] = (0.0, 1.0, 0.0)
    # 蓝色 (0,0,1) 左中
    img[100:200, 0:100, :] = (0.0, 0.0, 1.0)
    # 白色 (1,1,1) 右中
    img[100:200, 100:200, :] = (1.0, 1.0, 1.0)
    # 黑色 (0,0,0) 左下
    img[200:300, 0:100, :] = (0.0, 0.0, 0.0)
    # 灰色 (0.5,0.5,0.5) 右下
    img[200:300, 100:200, :] = (0.5, 0.5, 0.5)
    # 剩余区域默认黑色（已为0）

    # 转换为 ComfyUI 张量格式 [1, H, W, C]
    tensor_img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W, C]

    print("===== 无 mask 测试 =====")
    primary, secondary, richness = primary_secondary_richness_lab(tensor_img, mask=None, de_th=12, primary_ratio=0.1, secondary_ratio=0.01)
    print("主色调:", primary)
    print("副色调:", secondary)
    print("丰富度:", richness)
    print()

    # 创建 mask：只保留红色和绿色区域
    mask = np.zeros((H, W), dtype=np.float32)
    mask[0:100, 0:100] = 1.0  # 红色区域
    mask[0:100, 100:200] = 1.0  # 绿色区域
    tensor_mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

    print("===== 有 mask 测试（仅红+绿） =====")
    primary, secondary, richness = primary_secondary_richness_lab(
        tensor_img, mask=tensor_mask, de_th=12, primary_ratio=0.1, secondary_ratio=0.01
    )
    print("主色调:", primary)
    print("副色调:", secondary)
    print("丰富度:", richness)


if __name__ == "__main__":
    # 注意：确保已导入 torch 和 numpy
    # 如果尚未导入，可以在此处添加
    test_main()
