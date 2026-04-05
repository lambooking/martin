"""
B1 — MNDWI 计算
============================================================
输入：data/s2/YYYY_QN_s2.tif（Band1=Green, Band2=SWIR）
输出：output/mndwi/YYYY_QN_mndwi.tif（单波段浮点，值域约 -1~1）

公式：MNDWI = (Green - SWIR) / (Green + SWIR)
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    S2_DIR, MNDWI_DIR, PERIODS, S2_BAND_ORDER
)


def compute_mndwi_for_period(period: str) -> str:
    """
    计算单期 MNDWI 并保存为 GeoTIFF。

    Parameters
    ----------
    period : 期次名，如 "2019_Q1"

    Returns
    -------
    out_path : 输出文件路径
    """
    in_path  = os.path.join(S2_DIR, f"{period}_s2.tif")
    out_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：输入文件不存在（{in_path}）")
        return None

    with rasterio.open(in_path) as src:
        # 读取 Green 和 SWIR 波段（转为 float32）
        green = src.read(S2_BAND_ORDER["green"] + 1).astype(np.float32)
        swir  = src.read(S2_BAND_ORDER["swir"]  + 1).astype(np.float32)

        # 获取 nodata 掩膜（任一波段为 nodata 则屏蔽）
        nodata_val = src.nodata if src.nodata is not None else 0
        invalid_mask = (green == nodata_val) | (swir == nodata_val)

        # 计算 MNDWI，分母为 0 时置 NaN
        denom = green + swir
        with np.errstate(invalid="ignore", divide="ignore"):
            mndwi = np.where(denom != 0, (green - swir) / denom, np.nan)

        # 将 nodata 区域置 NaN
        mndwi[invalid_mask] = np.nan

        # 更新元数据：单波段 float32，nodata=NaN
        meta = src.meta.copy()
        meta.update({
            "count"   : 1,
            "dtype"   : "float32",
            "nodata"  : np.nan,
        })

        # 保存
        os.makedirs(MNDWI_DIR, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mndwi, 1)

    # 验证：输出统计信息
    valid = mndwi[~np.isnan(mndwi)]
    if valid.size > 0:
        print(f"  ✅ {period}: min={valid.min():.3f}, "
              f"max={valid.max():.3f}, "
              f"mean={valid.mean():.3f}, "
              f"水体正值占比={(valid > 0).mean()*100:.1f}%")
    else:
        print(f"  ⚠️  {period}: 无有效像元")

    return out_path


def main():
    print("=" * 55)
    print("  B1 — MNDWI 计算（共 24 期）")
    print("=" * 55)
    os.makedirs(MNDWI_DIR, exist_ok=True)

    success, skipped = 0, 0
    for period in PERIODS:
        result = compute_mndwi_for_period(period)
        if result:
            success += 1
        else:
            skipped += 1

    print(f"\n✅ B1 完成：成功 {success} 期，跳过 {skipped} 期（文件缺失）")
    print(f"   输出目录：{MNDWI_DIR}")


if __name__ == "__main__":
    main()
