import numpy as np
from skimage.color import rgb2lab, lab2rgb


def split_bw_gray(lab_pixels):
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]

    chroma = np.sqrt(a * a + b * b)

    # 黑白判定（你的原逻辑保留）
    black_mask = (L < 45) & (chroma < 8)
    white_mask = (L > 80) & (chroma < 8)

    # —— 改这里 ——
    # 真正的“灰色”应当满足：低饱和 + 无明显色向 + 中亮度
    gray_mask = (chroma < 4) & (np.abs(a) < 4) & (np.abs(b) < 4) & (20 < L) & (L < 90) & (~black_mask) & (~white_mask)

    color_mask = ~(black_mask | white_mask | gray_mask)

    blacks = lab_pixels[black_mask]
    whites = lab_pixels[white_mask]
    grays = lab_pixels[gray_mask]
    colors = lab_pixels[color_mask]

    return blacks, whites, grays, colors


def bucket_colors(pixels, de_th=12):
    if len(pixels) == 0:
        return np.empty((0, 3)), np.empty((0,), int)

    centers = [pixels[0].copy()]
    counts = [1]

    th2 = de_th * de_th

    for px in pixels[1:]:
        c = np.array(centers)
        dist = np.sum((c - px) ** 2, axis=1)
        k = np.argmin(dist)

        if dist[k] < th2:
            # 加权平均
            new_center = (centers[k] * counts[k] + px) / (counts[k] + 1)
            centers[k] = new_center
            counts[k] += 1
        else:
            centers.append(px.copy())
            counts.append(1)

    return np.array(centers), np.array(counts)


def merge_color_clusters(
    buckets,
    counts,
    light_th=40,
    hue_th=40,
    chroma_th=30,
    weak_chroma_th=18,  # 弱彩色阈值（a/b小→角度不稳）
    weak_ab_th=20,
):  # 弱彩色 a,b 欧氏距离阈值
    """
    第二阶段合并逻辑（最终版）：
    - 真彩色：亮度差 + 色相角差 + 彩度差 需要同时满足
    - 弱彩色：跳过色相角，用 a,b 欧氏距离代替 hue 判断
    - 灰色簇（低饱和）：强制吸附到最近彩色簇
    """

    if len(buckets) == 0:
        return buckets, counts

    # ------- 基本数据 -------
    Ls = buckets[:, 0]
    As = buckets[:, 1]
    Bs = buckets[:, 2]
    chromas = np.sqrt(As * As + Bs * Bs)

    # ------- 第二阶段灰色判定（你要求的版本） -------
    gray_idx = np.where((chromas < 10) & (np.abs(As) < 6) & (np.abs(Bs) < 6) & (Ls > 20) & (Ls < 90))[0]

    color_idx = np.setdiff1d(np.arange(len(buckets)), gray_idx)

    # 彩色簇
    color_b = buckets[color_idx]
    color_c = counts[color_idx]

    # 灰色簇
    gray_b = buckets[gray_idx]
    gray_c = counts[gray_idx]

    # 若全为灰色，直接返回（极少）
    if len(color_b) == 0:
        return gray_b, gray_c

    # ------- 准备合并彩色簇 -------
    M = len(color_b)
    used = np.zeros(M, bool)
    merged_centers = []
    merged_counts = []

    Lc = color_b[:, 0]
    Ac = color_b[:, 1]
    Bc = color_b[:, 2]
    chroma_c = np.sqrt(Ac * Ac + Bc * Bc)
    angles = np.degrees(np.arctan2(Bc, Ac))

    # -----------------------------
    # ✦ 主要循环：弱彩色 + 真彩色合并
    # -----------------------------
    for i in range(M):
        if used[i]:
            continue

        group = [i]

        for j in range(i + 1, M):
            if used[j]:
                continue

            dL = abs(Lc[i] - Lc[j])
            dC = abs(chroma_c[i] - chroma_c[j])

            # ------- 弱彩模式（跳过 hue，使用 a,b 欧氏距离） -------
            if chroma_c[i] < weak_chroma_th and chroma_c[j] < weak_chroma_th:
                d_ab = np.sqrt((Ac[i] - Ac[j]) ** 2 + (Bc[i] - Bc[j]) ** 2)

                if dL < light_th and d_ab < weak_ab_th and dC < chroma_th:
                    group.append(j)
                continue

            # ------- 真彩模式：使用 hue 差 -------
            dA = abs(angles[i] - angles[j])
            if dA > 180:
                dA = 360 - dA

            if dL < light_th and dA < hue_th and dC < chroma_th:
                group.append(j)

        used[group] = True

        # 加权合并
        w = color_c[group] / np.sum(color_c[group])
        center = np.sum(color_b[group] * w[:, None], axis=0)
        merged_centers.append(center)
        merged_counts.append(np.sum(color_c[group]))

    merged_centers = np.array(merged_centers)
    merged_counts = np.array(merged_counts)

    # -----------------------------
    # ✦ 第二阶段灰色：强制吸附（你原本要求）
    # -----------------------------
    for g_center, g_count in zip(gray_b, gray_c):
        d = np.linalg.norm(merged_centers - g_center, axis=1)
        k = np.argmin(d)
        merged_counts[k] += g_count

    return merged_centers, merged_counts


def final_grayline_merge(centers, counts):
    """
    对最终的颜色簇做一次“灰轴小合并”：
    - 若颜色接近灰轴 (a≈0, b≈0)，统一收拢到黑或白
    - 靠近黑的 → 合并到黑
    - 靠近白的 → 合并到白
    - 只动灰轴类，不动彩色类
    """

    if len(centers) == 0:
        return centers, counts

    Ls = centers[:, 0]
    As = centers[:, 1]
    Bs = centers[:, 2]

    # ------- 1. 判断接近灰轴 -------
    gray_axis_idx = np.where((np.abs(As) < 8) & (np.abs(Bs) < 8))[0]

    # 没有灰轴色，直接返回
    if len(gray_axis_idx) <= 1:
        return centers, counts

    # ------- 2. 找出黑 / 白 基准点 -------
    # 黑 = L最小的灰轴色
    black_i = gray_axis_idx[np.argmin(Ls[gray_axis_idx])]
    # black_center = centers[black_i]

    # 白 = L最大的灰轴色
    white_i = gray_axis_idx[np.argmax(Ls[gray_axis_idx])]
    # white_center = centers[white_i]

    # ------- 3. 合并小碎灰（强制吸附）-------
    for i in gray_axis_idx:
        if i == black_i or i == white_i:
            continue

        L = Ls[i]

        if L < 50:
            # 合并到黑
            counts[black_i] += counts[i]
        elif L > 70:
            # 合并到白
            counts[white_i] += counts[i]
        else:
            # 中间灰，合并到最近端
            d_black = abs(L - Ls[black_i])
            d_white = abs(L - Ls[white_i])
            if d_black < d_white:
                counts[black_i] += counts[i]
            else:
                counts[white_i] += counts[i]

    # ------- 4. 删除那些被合并的小灰簇 -------
    keep = np.ones(len(centers), bool)
    for i in gray_axis_idx:
        if i != black_i and i != white_i:
            keep[i] = False

    return centers[keep], counts[keep]


def np_to_tuples(arr):
    return [tuple(int(v) for v in x) for x in arr]


def get_richness(richness):
    if richness > 2:
        return "丰富"
    elif richness == 2:
        return "适中"
    else:
        return "纯色"


def primary_secondary_richness_lab(comfy_image_tensor, de_th=12, primary_ratio=0.1, secondary_ratio=0.01):
    # 1. 转换ComfyUI张量为RGB数组
    # - 移除batch维度 → 调整通道顺序 → 缩放至0-255 → 转为numpy数组
    img_tensor = comfy_image_tensor.squeeze(0)  # [H, W, C]
    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # 0-255

    # 2. 处理透明通道（如果有）
    if img_np.shape[-1] == 4:
        alpha = img_np[:, :, 3]
        rgb = img_np[:, :, :3][alpha > 0] / 255.0  # 只保留不透明区域
    else:
        rgb = img_np / 255.0  # 无透明通道直接使用

    if rgb.size == 0:
        return [], [], "纯色"

    lab_pixels = rgb2lab(rgb.reshape(-1, 3))

    # 灰/黑/白分桶
    blacks, whites, grays, colors = split_bw_gray(lab_pixels)

    # 彩色初分桶
    cb, cc = bucket_colors(colors, de_th=de_th)

    # 彩色二次合并
    cb, cc = merge_color_clusters(cb, cc)

    centers = []
    counts = []

    if len(blacks) > 0:
        centers.append(np.mean(blacks, axis=0))
        counts.append(len(blacks))

    if len(whites) > 0:
        centers.append(np.mean(whites, axis=0))
        counts.append(len(whites))

    if len(grays) > 0:
        centers.append(np.mean(grays, axis=0))
        counts.append(len(grays))

    for c, k in zip(cb, cc):
        centers.append(c)
        counts.append(k)

    centers = np.array(centers)
    counts = np.array(counts)

    centers, counts = final_grayline_merge(centers, counts)

    order = np.argsort(counts)[::-1]
    centers = centers[order]
    counts = counts[order]

    # lab → rgb
    rgb_batch = lab2rgb(centers[:, None, None, :]).reshape(-1, 3)
    rgb_batch = (np.clip(rgb_batch * 255, 0, 255)).astype(int)
    colors_rgb = [tuple(x) for x in rgb_batch]

    total = np.sum(counts)
    ratios = counts / total

    primary = []
    secondary = []

    for c, r in zip(colors_rgb, ratios):
        if r >= primary_ratio:
            primary.append(c)
        elif r >= secondary_ratio:
            secondary.append(c)

    return np_to_tuples(primary), np_to_tuples(secondary), get_richness(len(primary))


if __name__ == "__main__":
    p, s, r = primary_secondary_richness_lab()
    print("主色调：", p)
    print("副色调：", s)
    print("丰富度：", r)
