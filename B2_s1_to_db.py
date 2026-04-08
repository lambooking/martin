"""
B2 — S1 线性值转 dB（或直通已是 dB 的数据）
============================================================
输入：data/s1/S1_YYYY_QN.tif（Band1=VV, Band2=VH）
输出：output/s1_db/YYYY_QN_s1_db.tif（Band1=VV_dB, Band2=VH_dB）

自动检测数据格式：
  - 若输入包含大量负值（最小值 < -1），判定为已是 dB，直接保存。
  - 若输入全为正值（线性功率值），则执行：
      vv_db = 10 * log10(vv_linear + 1e-10)
      vh_db = 10 * log10(vh_linear + 1e-10)

GEE 导出 S1 GRD 有时为线性功率值（约 0.001~0.1），有时已转 dB（约 -45~+5）。
水体区域 VH_dB 约 -30 ~ -15 dB，VV_dB 约 -25 ~ -10 dB。
"""

import os
import sys
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    S1_DIR, S1_DB_DIR, PERIODS, S1_BAND_ORDER
)

EPSILON = 1e-10   # 防 log(0)


def convert_to_db_for_period(period: str) -> str:
    """
    将单期 S1 线性值转换为 dB 并保存为 GeoTIFF。

    Parameters
    ----------
    period : 期次名，如 "2019_Q1"

    Returns
    -------
    out_path : 输出文件路径，或 None（跳过）
    """
    in_path  = os.path.join(S1_DIR, f"S1_{period}.tif")
    out_path = os.path.join(S1_DB_DIR, f"{period}_s1_db.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：输入文件不存在（{in_path}）")
        return None

    with rasterio.open(in_path) as src:
        vv_raw = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_raw = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)
        nodata_val = src.nodata

        # 构建 nodata 掩膜：nodata 值 或 NaN
        if nodata_val is not None:
            nodata_mask = (vv_raw == nodata_val) | (vh_raw == nodata_val)
        else:
            nodata_mask = np.zeros(vv_raw.shape, dtype=bool)
        nodata_mask |= np.isnan(vv_raw) | np.isnan(vh_raw)

        # 自动检测格式：有大量负值 → 已是 dB，否则视为线性
        valid_sample = vv_raw[~nodata_mask]
        is_already_db = (valid_sample.size > 0 and float(valid_sample.min()) < -1.0)

        if is_already_db:
            vv_db = vv_raw.copy()
            vh_db = vh_raw.copy()
            print(f"  ℹ️  {period}: 检测到数据已为 dB 格式，直接保存（跳过 log10 转换）")
        else:
            # 线性转 dB（同时将线性值 ≤ 0 的像元视为无效）
            nodata_mask |= (vv_raw <= 0) | (vh_raw <= 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                vv_db = 10.0 * np.log10(vv_raw + EPSILON)
                vh_db = 10.0 * np.log10(vh_raw + EPSILON)

        # 无效像元置 NaN
        vv_db[nodata_mask] = np.nan
        vh_db[nodata_mask] = np.nan

        # 更新元数据
        meta = src.meta.copy()
        meta.update({
            "count"  : 2,
            "dtype"  : "float32",
            "nodata" : np.nan,
        })

        os.makedirs(S1_DB_DIR, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(vv_db, 1)
            dst.write(vh_db, 2)

    # 验证：输出水体典型范围
    for arr, name in [(vv_db, "VV"), (vh_db, "VH")]:
        valid = arr[~np.isnan(arr)]
        if valid.size > 0:
            # 水体像元大致为最低 25% 分位以下
            water_proxy = valid[valid < np.percentile(valid, 25)]
            print(f"  ✅ {period} {name}: "
                  f"全域 [{valid.min():.1f}, {valid.max():.1f}] dB, "
                  f"低值区（水体代理） [{water_proxy.min():.1f}, {water_proxy.max():.1f}] dB")

    return out_path


def main():
    print("=" * 55)
    print("  B2 — S1 线性值转 dB（共 24 期）")
    print("=" * 55)
    os.makedirs(S1_DB_DIR, exist_ok=True)

    success, skipped = 0, 0
    for period in PERIODS:
        result = convert_to_db_for_period(period)
        if result:
            success += 1
        else:
            skipped += 1

    print(f"\n✅ B2 完成：成功 {success} 期，跳过 {skipped} 期（文件缺失）")
    print(f"   输出目录：{S1_DB_DIR}")
    print("   说明：水体区域 VH 约 -30 ~ -15 dB，VV 约 -25 ~ -10 dB")


if __name__ == "__main__":
    main()
