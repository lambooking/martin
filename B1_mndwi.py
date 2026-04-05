"""
B1 — MNDWI 计算 / 读取
============================================================
输入：data/s2/YYYY_QN_s2.tif（该项目数据已在GEE计算成了单波段 MNDWI）
输出：output/mndwi/YYYY_QN_mndwi.tif（单波段浮点，值域约 -1~1）

逻辑：因为用户下载的S2直接就是MNDWI波段，本脚本将直接读取Band 1，处理好 NoData 后格式化输出。
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    S2_DIR, MNDWI_DIR, PERIODS
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
    in_path  = os.path.join(S2_DIR, f"S2_MNDWI_{period}.tif")
    out_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：输入文件不存在（{in_path}）")
        return None

    with rasterio.open(in_path) as src:
        # 用户数据已是单波段 MNDWI，直接读 Band 1
        mndwi = src.read(1).astype(np.float32)

        nodata_val = src.nodata if src.nodata is not None else 0
        invalid_mask = (mndwi == nodata_val)

        # 确保值域被限制在 -1 到 1 中间（去除异常值）而且处理 nodata 
        mndwi[invalid_mask] = np.nan
        mndwi = np.clip(mndwi, -1.0, 1.0)

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
