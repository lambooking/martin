"""
A4_auto — 自动化配准亚像素检测
============================================================
目的：利用算法自动量化 S1 (SAR) 与 S2 (MNDWI) 之间的空间对齐误差。
    无需人工选点，全自动寻找海岸线特征并计算位移。

算法逻辑：
    1. 提取 MNDWI (来自 B1) 和 S1 VH 波段 (来自 B2) 的梯度/边缘。
    2. 在图像中滑动窗口 (Tiles)，找到具有强边缘特征的区域。
    3. 使用相位互相关 (Phase Cross-Correlation) 计算窗口间的亚像素位移。
    4. 对所有分块的偏移结果求中值，得出全局 dx, dy。

输出：
    output/coregistration/auto_report.txt — 配准量化报告
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from skimage import filters, registration
from skimage.measure import regionprops

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, COREG_DIR, PIXEL_SIZE, PERIODS, S1_BAND_ORDER
)

# ------------------------------------------------------------------
# 配置参数
# ------------------------------------------------------------------
TILE_SIZE = 512          # 分块计算的大小
UPSAMPLE_FACTOR = 10     # 亚像素提升倍数 (10代表精度到0.1像元)
EDGE_THRESHOLD = 0.005   # 判定为“活跃块”的边缘强度阈值 (调低以适配稀疏边缘)


def compute_auto_offset(period: str):
    """
    计算某一期的自动配准偏移量。
    """
    mndwi_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")
    s1db_path  = os.path.join(S1_DB_DIR,  f"{period}_s1_db.tif")

    if not os.path.exists(mndwi_path) or not os.path.exists(s1db_path):
        print(f"  ⚠️  期次 {period} 的预处理数据不全，跳过自动配准检查。")
        return None

    # 1. 加载数据
    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1)
        nodata = src.nodata
    
    with rasterio.open(s1db_path) as src:
        # 使用 VH 波段，因为它对海水边界通常比 VV 更敏感
        s1_vh = src.read(S1_BAND_ORDER["vh"] + 1)

    # 2. 预处理：标准化并提取边缘
    # 处理掩膜
    mask = (~np.isnan(mndwi)) & (~np.isnan(s1_vh)) & (mndwi != nodata)
    
    # 简单的归一化使两者尺度接近
    # MNDWI [-1, 1] 映射到 [0, 1]
    img_opt = np.clip((mndwi + 1) / 2, 0, 1)
    
    # S1 VH [-30, -5] 映射到 [1, 0] (注意反转，使水体都变黑)
    img_sar = np.clip((s1_vh + 30) / 25, 0, 1)
    
    # 应用边缘检测 (Sobel)
    edge_opt = filters.sobel(img_opt)
    edge_sar = filters.sobel(img_sar)

    # 3. 分块滑动匹配
    h, w = mndwi.shape
    offsets = []
    
    for r in range(0, h - TILE_SIZE, TILE_SIZE):
        for c in range(0, w - TILE_SIZE, TILE_SIZE):
            tile_opt = edge_opt[r:r+TILE_SIZE, c:c+TILE_SIZE]
            tile_sar = edge_sar[r:r+TILE_SIZE, c:c+TILE_SIZE]
            tile_mask = mask[r:r+TILE_SIZE, c:c+TILE_SIZE]
            
            # 只分析陆海交界（边缘强度高）且无无效值的块
            if np.mean(tile_opt) > EDGE_THRESHOLD and np.all(tile_mask):
                # 计算相位互相关
                # shift 格式: (row_offset, col_offset)
                shift, error, diffphase = registration.phase_cross_correlation(
                    tile_opt, tile_sar, upsample_factor=UPSAMPLE_FACTOR
                )
                offsets.append(shift)

    if not offsets:
        print(f"  ⚠️  {period}: 未能找到足够的海岸线强特征块进行自动对齐。")
        return None

    # 4. 统计结果
    offsets = np.array(offsets)
    median_shift = np.median(offsets, axis=0)
    std_shift = np.std(offsets, axis=0)
    
    # 计算标量位移 (Pixel Offset)
    total_pixel_offset = np.sqrt(np.sum(median_shift**2))
    
    res = {
        "period": period,
        "dx_pix": median_shift[1], # col
        "dy_pix": median_shift[0], # row
        "pixel_offset": total_pixel_offset,
        "meter_offset": total_pixel_offset * PIXEL_SIZE,
        "n_tiles": len(offsets),
        "std_pix": np.sqrt(np.sum(std_shift**2))
    }
    
    return res


def main():
    print("=" * 60)
    print("  A4_auto — 自动化配准精度分检 (基于互相关算法)")
    print("=" * 60)
    
    # 我们默认检查 2019_Q1 或 第一个可用的期次
    target_period = PERIODS[0]
    for p in PERIODS:
        if os.path.exists(os.path.join(MNDWI_DIR, f"{p}_mndwi.tif")):
            target_period = p
            break

    print(f"\n[1/2] 正在对期次 {target_period} 执行扫描对比...")
    os.makedirs(COREG_DIR, exist_ok=True)
    
    result = compute_auto_offset(target_period)
    
    if result:
        print(f"\n[2/2] 自动配准测算结果：")
        print(f"  测算分块数: {result['n_tiles']} 个有效特征块")
        print(f"  估算位移 dx: {result['dx_pix']:.3f} 像素")
        print(f"  估算位移 dy: {result['dy_pix']:.3f} 像素")
        print(f"  总偏移量 (Total Offset): {result['pixel_offset']:.3f} 像元")
        print(f"  地面距离误差: {result['meter_offset']:.2f} 米")
        
        # 保存报告
        report_path = os.path.join(COREG_DIR, "auto_registration_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("黄河三角洲岸线研究 - 自动化配准误差报告\n")
            f.write("="*40 + "\n")
            f.write(f"测试期次: {result['period']}\n")
            f.write(f"中值像素位移: {result['pixel_offset']:.4f} pixels\n")
            f.write(f"转换地面距离: {result['meter_offset']:.2f} meters\n")
            f.write(f"测算置信度 (离散度 Std): {result['std_pix']:.4f}\n")
            f.write("-" * 40 + "\n")
            if result['pixel_offset'] <= 1.0:
                f.write("结论评分: [PASS] 对齐精良，误差在 1 像素以内，无需进行手动重采样。\n")
            else:
                f.write("结论评分: [WARNING] 存在显著偏移，建议执行阶段 A4 手动精细校准。\n")

        print(f"\n✅ 自动化报告已生成：{report_path}")
        print("\n" + "*"*60)
        if result['pixel_offset'] <= 1.0:
            print(f" 🏆 最终判定：对齐程度极佳 ({result['pixel_offset']:.2f} px)！")
            print(" 请直接放心大胆地进行后续 B/C/D 阶段的数据融合计算。")
        else:
            print(f" ⚠️ 最终判定：检测到跨像元位移 ({result['pixel_offset']:.2f} px)。")
            print(" 建议在论文发布前，参照 A4 手动流程进一步确认。")
        print("*"*60 + "\n")
    else:
        print("\n❌ 自动检查失败：请确保 B1 和 B2 模块已经成功运行产生了 TIF 文件。")


if __name__ == "__main__":
    main()
