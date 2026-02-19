import torch


def get_one_ratio(img: torch.Tensor, mask: torch.Tensor = None):
    """
    使用 PCA 计算最大高宽比

    参数:
        img:  (B, H, W, C)
        mask: (B, H, W) 或 None

    返回:
        ratios: list[float]
    """

    B, H, W, C = img.shape

    # 自动处理 mask
    if mask is None:
        if C == 4:
            mask = img[..., 3]
        else:
            return [0.0] * B

    ratios = []

    for b in range(B):
        coords = torch.nonzero(mask[b] > 0, as_tuple=False).float()

        # 前景太少
        if coords.shape[0] < 2:
            ratios.append(0.0)
            continue

        # 中心化
        mean = coords.mean(dim=0)
        coords -= mean

        # 协方差矩阵 (2x2)
        cov = coords.T @ coords / coords.shape[0]

        # 特征值分解
        eigvals, _ = torch.linalg.eigh(cov)

        # 防止除0
        if eigvals[0] <= 1e-8:
            ratios.append(0.0)
            continue

        # 主轴比例
        ratio = torch.sqrt(eigvals[-1] / eigvals[0]).item()

        ratios.append(float(ratio))

    return ratios


def get_level(ratios):
    levels = ""

    for ratio in ratios:
        if ratio < 1:
            raise ValueError("高宽比不会小于1，请检查")
        elif ratio <= 1.2:
            levels = "近方"
        elif ratio <= 1.6:
            levels = "轻微纵向"
        elif ratio <= 2.2:
            levels = "纵向"
        elif ratio <= 3.2:
            levels = "修长"
        else:
            levels = "极端纵向"

    return levels


def get_finally_level(img: torch.Tensor, mask: torch.Tensor = None):
    ratios = get_one_ratio(img, mask)
    return get_level(ratios)
