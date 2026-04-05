"""
B4 — 联合阈值水体提取
============================================================
输入：
    output/mndwi/YYYY_QN_mndwi.tif
    output/s1_db/YYYY_QN_s1_db.tif
    output/thresholds/thresholds.csv

输出：
    output/water_mask/YYYY_QN_water.tif
        （二值栅格：1=水体，0=非水体，nodata=255）

逻辑：
    water = (MNDWI > T1) AND (VH_dB < T2) AND (VV_dB < T3)

验证：输出每期水体面积（km²），验证时序是否无异常突变。
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, THRESH_DIR, WATER_MASK_DIR,
    FIGURES_DIR, PERIODS, S1_BAND_ORDER, PIXEL_SIZE
)

WATER_VAL   = 1    # 水体像元值
NONWATER_VAL = 0   # 非水体
NODATA_VAL  = 255  # nodata


def extract_water_for_period(period: str, row: pd.Series) -> dict:
    """
    对单期执行联合阈值水体提取。

    Parameters
    ----------
    period : 期次名
    row    : thresholds.csv 中该期的阈值行（含 T1, T2, T3）

    Returns
    -------
    dict: period, water_area_km2, water_pixel_count
    """
    mndwi_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")
    s1db_path  = os.path.join(S1_DB_DIR,  f"{period}_s1_db.tif")
    out_path   = os.path.join(WATER_MASK_DIR, f"{period}_water.tif")

    for p in [mndwi_path, s1db_path]:
        if not os.path.exists(p):
            print(f"  ⚠️  跳过 {period}：{p} 不存在")
            return None

    T1, T2, T3 = float(row["T1"]), float(row["T2"]), float(row["T3"])

    # 读取数据
    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1).astype(np.float32)
        meta  = src.meta.copy()

    with rasterio.open(s1db_path) as src:
        vv_db = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_db = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)

    # 有效像元掩膜（任一波段为 NaN 则无效）
    valid = (~np.isnan(mndwi)) & (~np.isnan(vv_db)) & (~np.isnan(vh_db))

    # 联合阈值判别
    water = (
        (mndwi > T1) &
        (vh_db < T2) &
        (vv_db < T3) &
        valid
    )

    # 输出二值图（uint8，nodata=255）
    result = np.full(mndwi.shape, NODATA_VAL, dtype=np.uint8)
    result[valid]  = NONWATER_VAL
    result[water]  = WATER_VAL

    meta.update({
        "count"  : 1,
        "dtype"  : "uint8",
        "nodata" : NODATA_VAL,
    })
    os.makedirs(WATER_MASK_DIR, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(result, 1)

    # 统计
    n_water   = int(water.sum())
    pixel_m2  = PIXEL_SIZE ** 2
    area_km2  = n_water * pixel_m2 / 1e6
    n_valid   = int(valid.sum())
    water_pct = n_water / n_valid * 100 if n_valid > 0 else 0

    print(f"  ✅ {period}: 水体面积={area_km2:.2f} km²  "
          f"({n_water:,} px, {water_pct:.1f}% of valid)  "
          f"T1={T1:.3f}, T2={T2:.1f}, T3={T3:.1f}")

    return {
        "period"          : period,
        "water_area_km2"  : round(area_km2, 4),
        "water_pixel_count": n_water,
        "valid_pixel_count": n_valid,
        "water_pct"       : round(water_pct, 2),
    }


def plot_water_area_timeseries(records: list, out_dir: str) -> str:
    """绘制水体面积时序折线图（B4 验证检查点）。"""
    import datetime
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "water_area_timeseries.png")

    q_month = {"Q1": 2, "Q2": 5, "Q3": 8, "Q4": 11}
    dates, areas = [], []
    for r in records:
        y, q = r["period"].split("_")
        dates.append(datetime.date(int(y), q_month[q], 1))
        areas.append(r["water_area_km2"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, areas, marker="o", color="steelblue", linewidth=1.5,
            markersize=5, label="水体面积（含外海约束前）")
    ax.fill_between(dates, areas, alpha=0.15, color="steelblue")

    # 添加年均值虚线
    years_data = {}
    for d, a in zip(dates, areas):
        years_data.setdefault(d.year, []).append(a)
    for yr, vals in years_data.items():
        mean_v = np.mean(vals)
        ax.axhline(mean_v, xmin=(yr-2019)/6, xmax=(yr-2019+1)/6,
                   color="orange", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("时间")
    ax.set_ylabel("水体面积 (km²)")
    ax.set_title("B4 验证：联合阈值水体面积时序（2019Q1–2024Q4）\n"
                 "期望：面积波动合理，无异常突变", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 水体面积时序图已保存：{out_path}")
    return out_path


def main():
    print("=" * 55)
    print("  B4 — 联合阈值水体提取（共 24 期）")
    print("=" * 55)

    # 读取阈值表
    thresh_csv = os.path.join(THRESH_DIR, "thresholds.csv")
    if not os.path.exists(thresh_csv):
        print(f"❌ 阈值文件不存在，请先运行 B3！（{thresh_csv}）")
        sys.exit(1)

    thresh_df = pd.read_csv(thresh_csv).set_index("period")
    os.makedirs(WATER_MASK_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        if period not in thresh_df.index:
            print(f"  ⚠️  {period} 无阈值记录，跳过")
            continue
        row = thresh_df.loc[period]
        result = extract_water_for_period(period, row)
        if result:
            records.append(result)

    if records:
        summary_df = pd.DataFrame(records)
        summary_path = os.path.join(WATER_MASK_DIR, "water_area_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n  面积统计摘要（km²）：")
        print(f"  均值={summary_df['water_area_km2'].mean():.2f}, "
              f"最小={summary_df['water_area_km2'].min():.2f}, "
              f"最大={summary_df['water_area_km2'].max():.2f}")

        plot_water_area_timeseries(records, FIGURES_DIR)

    print(f"\n✅ B4 完成：成功处理 {len(records)} 期")
    print("   验证：请查看 water_area_timeseries.png，确认无异常突变期")


if __name__ == "__main__":
    main()
