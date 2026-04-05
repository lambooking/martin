"""
B3 — 自适应阈值计算（T1/T2/T3）
============================================================
输入：
    output/mndwi/YYYY_QN_mndwi.tif
    output/s1_db/YYYY_QN_s1_db.tif

输出：
    output/thresholds/thresholds.csv（每期一行：period, T1, T2, T3）
    output/figures/thresholds_timeseries.png（阈值时序折线图，验证季节规律）

逻辑：
    T1: Otsu 算法作用于研究区内有效 MNDWI 像元
    高置信水体掩膜: MNDWI > T1
    T2: 高置信水体像元上 VH_dB 的 P95 分位（用作上界约束：VH < T2）
    T3: 同理，VV_dB 的 P95 分位（VV < T3）
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, THRESH_DIR, FIGURES_DIR,
    PERIODS, S1_BAND_ORDER
)


# ------------------------------------------------------------------
# 辅助：将期次字符串解析为近似日期（用于时序图横轴）
# ------------------------------------------------------------------
def period_to_date(period: str):
    """将 '2019_Q1' 解析为该季第二月的第一天（近似）。"""
    import datetime
    year, q = period.split("_")
    q_month = {"Q1": 2, "Q2": 5, "Q3": 8, "Q4": 11}
    return datetime.date(int(year), q_month[q], 1)


# ------------------------------------------------------------------
# 核心：计算单期阈值
# ------------------------------------------------------------------
def compute_thresholds_for_period(period: str) -> dict:
    """
    计算单期 T1/T2/T3 自适应阈值。

    Returns
    -------
    dict with keys: period, T1, T2, T3
        或 None（文件缺失时）
    """
    mndwi_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")
    s1db_path  = os.path.join(S1_DB_DIR,  f"{period}_s1_db.tif")

    for p in [mndwi_path, s1db_path]:
        if not os.path.exists(p):
            print(f"  ⚠️  跳过 {period}：文件不存在（{p}）")
            return None

    # --- 读取 MNDWI ---
    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1).astype(np.float32)

    # --- 读取 VH & VV dB ---
    with rasterio.open(s1db_path) as src:
        vv_db = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_db = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)

    # --- T1：Otsu 阈值 ---
    mndwi_valid = mndwi[~np.isnan(mndwi)]
    if mndwi_valid.size < 100:
        warnings.warn(f"{period}: 有效 MNDWI 像元过少（{mndwi_valid.size}），跳过")
        return None
    T1 = float(threshold_otsu(mndwi_valid))

    # --- 高置信水体掩膜：MNDWI > T1 ---
    water_mask = (mndwi > T1) & (~np.isnan(mndwi)) & \
                 (~np.isnan(vh_db)) & (~np.isnan(vv_db))

    n_water = water_mask.sum()
    if n_water < 10:
        warnings.warn(f"{period}: 高置信水体像元过少（{n_water}），T2/T3 可能不可靠")

    # --- T2/T3：水体像元上的 P95 ---
    T2 = float(np.nanpercentile(vh_db[water_mask], 95)) if n_water >= 10 else np.nan
    T3 = float(np.nanpercentile(vv_db[water_mask], 95)) if n_water >= 10 else np.nan

    print(f"  {period}: T1(MNDWI Otsu)={T1:.4f}, "
          f"T2(VH P95)={T2:.2f} dB, "
          f"T3(VV P95)={T3:.2f} dB  "
          f"[水体像元 {n_water:,} 个]")

    return {"period": period, "T1": T1, "T2": T2, "T3": T3}


# ------------------------------------------------------------------
# 可视化：阈值时序折线图
# ------------------------------------------------------------------
def plot_thresholds_timeseries(df: pd.DataFrame, out_dir: str) -> str:
    """绘制 T1/T2/T3 时序折线图，验证季节规律。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "thresholds_timeseries.png")

    dates = [period_to_date(p) for p in df["period"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("自适应阈值时序（2019Q1–2024Q4）\n"
                 "黄河三角洲岸线变化研究",
                 fontsize=13, fontweight="bold")

    configs = [
        ("T1", "MNDWI Otsu 阈值", "steelblue",
         "期望：夏季（Q3）偏低（浑浊水体），季节规律可见"),
        ("T2", "VH_dB P95 阈值（VH < T2 为水体）", "darkorange",
         "期望：约 -15 ~ -10 dB，稳定波动"),
        ("T3", "VV_dB P95 阈值（VV < T3 为水体）", "seagreen",
         "期望：约 -10 ~ -5 dB，稳定波动"),
    ]
    for ax, (col, ylabel, color, note) in zip(axes, configs):
        vals = df[col].values
        ax.plot(dates, vals, marker="o", color=color, linewidth=1.5,
                markersize=4, zorder=3)
        ax.fill_between(dates, vals, alpha=0.15, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(0.01, 0.97, note, transform=ax.transAxes,
                va="top", fontsize=8, color="gray")
        ax.grid(True, alpha=0.3)
        # 标注季度
        for i, (d, v) in enumerate(zip(dates, vals)):
            if not np.isnan(v):
                q = df["period"].iloc[i].split("_")[1]
                ax.annotate(
                    q, (d, v), textcoords="offset points",
                    xytext=(0, 8), fontsize=6, ha="center", color=color
                )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].set_xlabel("年份")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 阈值时序图已保存：{out_path}")
    return out_path


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  B3 — 自适应阈值计算（共 24 期）")
    print("=" * 55)
    os.makedirs(THRESH_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        result = compute_thresholds_for_period(period)
        if result:
            records.append(result)

    if not records:
        print("❌ 没有找到任何有效期次，请先运行 B1 和 B2！")
        return

    df = pd.DataFrame(records)
    out_csv = os.path.join(THRESH_DIR, "thresholds.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 阈值表已保存：{out_csv}（共 {len(df)} 期）")

    # 统计摘要
    print("\n  阈值统计摘要（有效期）：")
    print(df[["T1", "T2", "T3"]].describe().round(4).to_string())

    # 绘制时序图（验证 B3 检查点）
    plot_thresholds_timeseries(df, FIGURES_DIR)

    print("\n✅ B3 完成！")
    print("   验证：T1 时序是否显示夏季（Q3）偏低的季节规律？")


if __name__ == "__main__":
    main()
