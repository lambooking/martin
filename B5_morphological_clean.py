"""
B5 — 形态学后处理
============================================================
输入：output/water_mask/YYYY_QN_water.tif（B4 二值掩膜）
输出：output/water_clean/YYYY_QN_water_clean.tif

处理步骤：
    1. 开运算（binary_opening）去除小斑块噪声
    2. 闭运算（binary_closing）填补水体内部小孔
    3. 面积阈值过滤：移除连通域面积 < MIN_WATER_AREA_PX 的碎斑
"""

import os
import sys
import numpy as np
import rasterio
from scipy import ndimage
from skimage.measure import label, regionprops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WATER_MASK_DIR, WATER_CLEAN_DIR, PERIODS,
    MORPH_OPEN_SIZE, MORPH_CLOSE_SIZE, MIN_WATER_AREA_PX
)

WATER_VAL   = 255  # 水体像元值（白色）
NONWATER_VAL = 0   # 非水体（黑色）
NODATA_VAL  = 128  # nodata（灰色）


def morphological_clean_for_period(period: str) -> dict:
    """
    对单期水体掩膜执行形态学后处理。

    Returns
    -------
    dict: period, n_pixels_before, n_pixels_after, n_removed_blobs
    """
    in_path  = os.path.join(WATER_MASK_DIR,  f"{period}_water.tif")
    out_path = os.path.join(WATER_CLEAN_DIR, f"{period}_water_clean.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：{in_path} 不存在")
        return None

    with rasterio.open(in_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    water_binary = (data == WATER_VAL)
    nodata_mask  = (data == NODATA_VAL)
    n_before     = int(water_binary.sum())

    # --- 步骤 1：开运算（去除小噪声斑块）---
    struct_open  = ndimage.generate_binary_structure(2, 1)  # 4-连通
    struct_open  = ndimage.iterate_structure(struct_open, MORPH_OPEN_SIZE // 2 + 1)
    water_opened = ndimage.binary_opening(water_binary, structure=struct_open)

    # --- 步骤 2：闭运算（填补内部小孔）---
    struct_close  = ndimage.generate_binary_structure(2, 2)  # 8-连通
    struct_close  = ndimage.iterate_structure(struct_close, MORPH_CLOSE_SIZE // 2 + 1)
    water_closed  = ndimage.binary_closing(water_opened, structure=struct_close)

    # --- 步骤 3：面积阈值过滤（移除碎斑）---
    labeled = label(water_closed, connectivity=2)
    props   = regionprops(labeled)
    n_removed = 0
    for region in props:
        if region.area < MIN_WATER_AREA_PX:
            water_closed[labeled == region.label] = False
            n_removed += 1

    n_after = int(water_closed.sum())

    # --- 输出 ---
    result = np.full(data.shape, NODATA_VAL, dtype=np.uint8)
    result[~nodata_mask] = NONWATER_VAL
    result[water_closed] = WATER_VAL
    result[nodata_mask]  = NODATA_VAL   # nodata 区域优先

    os.makedirs(WATER_CLEAN_DIR, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(result, 1)

    print(f"  ✅ {period}: "
          f"处理前={n_before:,} px → 处理后={n_after:,} px, "
          f"移除碎斑 {n_removed} 个 "
          f"(减少 {(n_before-n_after)/max(n_before,1)*100:.1f}%)")

    return {
        "period"          : period,
        "pixels_before"   : n_before,
        "pixels_after"    : n_after,
        "blobs_removed"   : n_removed,
        "pixel_change_pct": round((n_before - n_after) / max(n_before, 1) * 100, 2),
    }


def main():
    print("=" * 55)
    print("  B5 — 形态学后处理（共 24 期）")
    print(f"       开运算大小={MORPH_OPEN_SIZE}×{MORPH_OPEN_SIZE}, "
          f"闭运算大小={MORPH_CLOSE_SIZE}×{MORPH_CLOSE_SIZE}, "
          f"最小面积={MIN_WATER_AREA_PX} px")
    print("=" * 55)
    os.makedirs(WATER_CLEAN_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        result = morphological_clean_for_period(period)
        if result:
            records.append(result)

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print(f"\n  形态学处理统计摘要：")
        print(f"  平均减少比例：{df['pixel_change_pct'].mean():.1f}%")
        print(f"  平均移除碎斑：{df['blobs_removed'].mean():.0f} 个")

    print(f"\n✅ B5 完成：成功处理 {len(records)} 期")
    print(f"   输出目录：{WATER_CLEAN_DIR}")


if __name__ == "__main__":
    main()
