"""
B6 — 外海连通性约束
============================================================
输入：output/water_clean/YYYY_QN_water_clean.tif
输出：output/sea_mask/YYYY_QN_sea.tif

逻辑：
    1. 对水体二值图做连通域标记（8-连通）
    2. 将外海种子点（经纬度）转为像素坐标
    3. 仅保留包含种子点的连通域（外海区域）
    4. 目的：去除内陆水体、养殖池、河渠等非海洋水体

验证：输出处理前后像元统计，供目视检查。
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import rowcol
from skimage.measure import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WATER_CLEAN_DIR, SEA_MASK_DIR, PERIODS,
    SEA_SEED_COORDS, PIXEL_SIZE
)

WATER_VAL   = 1
NONWATER_VAL = 0
NODATA_VAL  = 255


def lonlat_to_rowcol(transform, lon: float, lat: float):
    """将地理坐标（经纬度）转换为像素行列号（row, col）。"""
    # rasterio rowcol: (transform, xs, ys)
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


def apply_sea_connectivity_for_period(period: str) -> dict:
    """
    对单期水体掩膜执行外海连通性约束。

    Returns
    -------
    dict: period, pixels_before, pixels_after
    """
    in_path  = os.path.join(WATER_CLEAN_DIR, f"{period}_water_clean.tif")
    out_path = os.path.join(SEA_MASK_DIR,    f"{period}_sea.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：{in_path} 不存在")
        return None

    with rasterio.open(in_path) as src:
        data      = src.read(1)
        meta      = src.meta.copy()
        transform = src.transform
        height, width = src.height, src.width

    water_binary = (data == WATER_VAL)
    nodata_mask  = (data == NODATA_VAL)
    n_before     = int(water_binary.sum())

    # --- 连通域标记（8-连通）---
    labeled   = label(water_binary, connectivity=2)
    n_labels  = labeled.max()

    # --- 定位外海种子点 ---
    sea_labels = set()
    for lon, lat in SEA_SEED_COORDS:
        seed_row, seed_col = lonlat_to_rowcol(transform, lon, lat)
        # 边界保护
        seed_row = max(0, min(seed_row, height - 1))
        seed_col = max(0, min(seed_col, width - 1))
        lbl = labeled[seed_row, seed_col]
        if lbl > 0:
            sea_labels.add(lbl)
        else:
            print(f"    ⚠️  种子点 ({lon}, {lat}) 不在水体区域内 "
                  f"→ 像素 ({seed_row}, {seed_col}) = {lbl}")

    if not sea_labels:
        print(f"  ⚠️  {period}: 未找到外海连通域，请检查 SEA_SEED_COORDS！")
        # 退化：保留所有水体（不过滤）
        sea_binary = water_binary.copy()
    else:
        # 仅保留含种子点的连通域
        sea_binary = np.zeros_like(water_binary, dtype=bool)
        for lbl in sea_labels:
            sea_binary |= (labeled == lbl)

    n_after   = int(sea_binary.sum())
    removed   = n_before - n_after
    removed_pct = removed / max(n_before, 1) * 100

    # --- 输出 ---
    result = np.full(data.shape, NODATA_VAL, dtype=np.uint8)
    result[~nodata_mask] = NONWATER_VAL
    result[sea_binary]   = WATER_VAL
    result[nodata_mask]  = NODATA_VAL

    os.makedirs(SEA_MASK_DIR, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(result, 1)

    print(f"  ✅ {period}: "
          f"外海连通域={len(sea_labels)} 个, "
          f"海水={n_after:,} px（含 {n_before:,} px 去除 {removed:,} px={removed_pct:.1f}%）")

    return {
        "period"          : period,
        "pixels_before"   : n_before,
        "pixels_after "   : n_after,
        "pixels_removed"  : removed,
        "removed_pct"     : round(removed_pct, 2),
        "n_sea_labels"    : len(sea_labels),
    }


def main():
    print("=" * 55)
    print("  B6 — 外海连通性约束（共 24 期）")
    print(f"       外海种子点：{SEA_SEED_COORDS}")
    print("=" * 55)
    print("  ⚠️  请在运行前在 config.py 中确认 SEA_SEED_COORDS 坐标是否正确！")
    print()

    os.makedirs(SEA_MASK_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        result = apply_sea_connectivity_for_period(period)
        if result:
            records.append(result)

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print(f"\n  外海约束统计摘要：")
        print(f"  平均移除比例：{df['removed_pct'].mean():.1f}%")

    print(f"\n✅ B6 完成：成功处理 {len(records)} 期")
    print(f"   输出目录：{SEA_MASK_DIR}")
    print("   ⚠️  请目视检查：养殖池/内河/水库是否已被移除？")


if __name__ == "__main__":
    main()
