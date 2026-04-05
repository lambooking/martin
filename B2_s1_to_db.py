"""
B2 — S1 线性值转 dB
============================================================
输入：data/s1/YYYY_QN_s1.tif（Band1=VV_linear, Band2=VH_linear）
输出：output/s1_db/YYYY_QN_s1_db.tif（Band1=VV_dB, Band2=VH_dB）

公式：
    vv_db = 10 * log10(vv_linear + 1e-10)
    vh_db = 10 * log10(vh_linear + 1e-10)

GEE 导出的 S1 GRD 数据为线性功率值，水体区域 dB 约 -30 ~ -10 dB。
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
    in_path  = os.path.join(S1_DIR, f"{period}_s1.tif")
    out_path = os.path.join(S1_DB_DIR, f"{period}_s1_db.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：输入文件不存在（{in_path}）")
        return None

    with rasterio.open(in_path) as src:
        vv_linear = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_linear = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)

        nodata_val = src.nodata if src.nodata is not None else 0
        invalid_mask = (vv_linear <= 0) | (vh_linear <= 0) | \
                       (vv_linear == nodata_val) | (vh_linear == nodata_val)

        # 线性转 dB
        with np.errstate(invalid="ignore", divide="ignore"):
            vv_db = 10.0 * np.log10(vv_linear + EPSILON)
            vh_db = 10.0 * np.log10(vh_linear + EPSILON)

        # 无效像元置 NaN
        vv_db[invalid_mask] = np.nan
        vh_db[invalid_mask] = np.nan

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
