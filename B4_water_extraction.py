"""
B4 — 决策级融合水体提取（投票制）
============================================================
指南依据（guide.md §1.2 / 模块B3–B4）：

    Step 1：单通道判别
        W_M   = I(MNDWI > T1)
        W_VH  = I(VH_dB  < T2)
        W_VV  = I(VV_dB  < T3)

    Step 2：决策级融合（投票 ≥ 2）
        vote  = W_M + W_VH + W_VV        （0–3 整数）
        W_F   = I(vote ≥ 2)

    规则 3：S1 已为 dB，不允许再 log10

输入：
    output/mndwi/YYYY_QN_mndwi.tif
    output/s1_db/YYYY_QN_s1_db.tif
    output/thresholds/thresholds.csv

输出：
    output/water_single/YYYY_QN_WM.tif     单通道：MNDWI
    output/water_single/YYYY_QN_WVH.tif    单通道：VH
    output/water_single/YYYY_QN_WVV.tif    单通道：VV
    output/water_mask/YYYY_QN_vote.tif     投票数（0–3）
    output/water_mask/YYYY_QN_water.tif    最终水体掩膜 WF（下游 B5/B6 输入）

    output/figures/water_area_timeseries.png
    output/water_mask/water_area_summary.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, THRESH_DIR, WATER_MASK_DIR,
    SINGLE_MASK_DIR, FIGURES_DIR, PERIODS, S1_BAND_ORDER, PIXEL_SIZE,
)

# 二值栅格编码
WATER_VAL    = 255   # 水体（白色）
NONWATER_VAL = 0     # 非水体（黑色）
NODATA_VAL   = 128   # NoData（灰色）


# ──────────────────────────────────────────────────────────
# 核心处理函数
# ──────────────────────────────────────────────────────────

def _write_binary(arr: np.ndarray, valid: np.ndarray,
                  meta: dict, path: str) -> None:
    """将布尔掩膜写成 uint8 二值 GeoTIFF（255=水, 0=非水, 128=nodata）。"""
    out = np.full(arr.shape, NODATA_VAL, dtype=np.uint8)
    out[valid]          = NONWATER_VAL
    out[valid & arr]    = WATER_VAL
    m = meta.copy()
    m.update({"count": 1, "dtype": "uint8", "nodata": NODATA_VAL})
    with rasterio.open(path, "w", **m) as dst:
        dst.write(out, 1)


def extract_water_for_period(period: str, row: pd.Series) -> dict | None:
    """
    对单期执行投票制融合水体提取，保存所有中间与最终产品。

    Parameters
    ----------
    period : 期次名，如 "2019_Q1"
    row    : thresholds.csv 中该期的阈值行（含 T1, T2, T3）

    Returns
    -------
    dict or None
    """
    mndwi_path = os.path.join(MNDWI_DIR,  f"{period}_mndwi.tif")
    s1db_path  = os.path.join(S1_DB_DIR,  f"{period}_s1_db.tif")

    for p in [mndwi_path, s1db_path]:
        if not os.path.exists(p):
            print(f"  ⚠️  跳过 {period}：{p} 不存在")
            return None

    T1 = float(row["T1"])
    T2 = float(row["T2"])
    T3 = float(row["T3"])

    # ── 读取输入 ──────────────────────────────────
    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1).astype(np.float32)
        meta  = src.meta.copy()

    with rasterio.open(s1db_path) as src:
        vv_db = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_db = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)

    # ── 有效像元掩膜（三个波段均需有效）─────────────
    valid = (~np.isnan(mndwi)) & (~np.isnan(vv_db)) & (~np.isnan(vh_db))

    # ── Step 1：单通道判别 ─────────────────────────
    W_M   = valid & (mndwi > T1)           # MNDWI 水体
    W_VH  = valid & (vh_db < T2)           # VH 水体（dB 越低越可能是水）
    W_VV  = valid & (vv_db < T3)           # VV 水体

    # ── Step 2：投票融合（≥2 即判为水）─────────────
    vote  = W_M.astype(np.uint8) + W_VH.astype(np.uint8) + W_VV.astype(np.uint8)
    W_F   = valid & (vote >= 2)

    # ── 保存单通道结果 ─────────────────────────────
    os.makedirs(SINGLE_MASK_DIR, exist_ok=True)
    _write_binary(W_M,  valid, meta, os.path.join(SINGLE_MASK_DIR, f"{period}_WM.tif"))
    _write_binary(W_VH, valid, meta, os.path.join(SINGLE_MASK_DIR, f"{period}_WVH.tif"))
    _write_binary(W_VV, valid, meta, os.path.join(SINGLE_MASK_DIR, f"{period}_WVV.tif"))

    # ── 保存投票图（0–3）──────────────────────────
    vote_out = np.full(mndwi.shape, 255, dtype=np.uint8)   # 255=nodata
    vote_out[valid] = vote[valid]
    vm = meta.copy()
    vm.update({"count": 1, "dtype": "uint8", "nodata": 255})
    with rasterio.open(os.path.join(WATER_MASK_DIR, f"{period}_vote.tif"), "w", **vm) as dst:
        dst.write(vote_out, 1)

    # ── 保存最终融合结果 WF ────────────────────────
    os.makedirs(WATER_MASK_DIR, exist_ok=True)
    _write_binary(W_F, valid, meta, os.path.join(WATER_MASK_DIR, f"{period}_water.tif"))

    # ── 统计 ───────────────────────────────────────
    n_valid  = int(valid.sum())
    n_wm     = int(W_M.sum())
    n_wvh    = int(W_VH.sum())
    n_wvv    = int(W_VV.sum())
    n_wf     = int(W_F.sum())
    area_km2 = n_wf * PIXEL_SIZE ** 2 / 1e6
    wf_pct   = n_wf / n_valid * 100 if n_valid > 0 else 0

    print(
        f"  ✅ {period}: "
        f"WM={n_wm:,}  WVH={n_wvh:,}  WVV={n_wvv:,}  "
        f"WF={n_wf:,}px ({wf_pct:.1f}%)  "
        f"= {area_km2:.2f} km²  "
        f"[T1={T1:.3f}, T2={T2:.1f}dB, T3={T3:.1f}dB]"
    )

    return {
        "period"           : period,
        "water_area_km2"   : round(area_km2, 4),
        "water_pixel_count": n_wf,
        "valid_pixel_count": n_valid,
        "water_pct"        : round(wf_pct, 2),
        "wm_px"            : n_wm,
        "wvh_px"           : n_wvh,
        "wvv_px"           : n_wvv,
        "T1": T1, "T2": T2, "T3": T3,
    }


# ──────────────────────────────────────────────────────────
# 验证图：水体面积时序
# ──────────────────────────────────────────────────────────

def plot_water_area_timeseries(records: list, out_dir: str) -> str:
    """绘制 WF 水体面积时序折线图，用于 B4 验证。"""
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
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(dates, areas, marker="o", color="#1565C0", linewidth=1.8,
            markersize=5, label="WF 水体面积（融合，≥2票）", zorder=3)
    ax.fill_between(dates, areas, alpha=0.12, color="#1565C0", zorder=2)

    # 年均值参考线
    years_data: dict = {}
    for d, a in zip(dates, areas):
        years_data.setdefault(d.year, []).append(a)
    for yr, vals in years_data.items():
        ax.axhline(np.mean(vals),
                   xmin=(yr - 2019) / 6, xmax=(yr - 2019 + 1) / 6,
                   color="#E65100", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("水体面积（km²）", fontsize=12)
    ax.set_title(
        "B4 验证：决策级融合（≥2 票）水体面积时序（2019Q1–2024Q4）\n"
        "期望：面积波动合理，无异常突变",
        fontsize=11
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3, color="#BDBDBD")
    ax.legend(fontsize=10)
    for sp in ax.spines.values():
        sp.set_color("#9E9E9E")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 水体面积时序图已保存：{out_path}")
    return out_path


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  B4 — 决策级融合水体提取（投票制 ≥ 2，共 24 期）")
    print("  规则：W_F = I(W_M + W_VH + W_VV ≥ 2)")
    print("=" * 60)

    thresh_csv = os.path.join(THRESH_DIR, "thresholds.csv")
    if not os.path.exists(thresh_csv):
        print(f"❌ 阈值文件不存在，请先运行 B3！（{thresh_csv}）")
        sys.exit(1)

    thresh_df = pd.read_csv(thresh_csv).set_index("period")
    os.makedirs(WATER_MASK_DIR, exist_ok=True)
    os.makedirs(SINGLE_MASK_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        if period not in thresh_df.index:
            print(f"  ⚠️  {period} 无阈值记录，跳过")
            continue
        result = extract_water_for_period(period, thresh_df.loc[period])
        if result:
            records.append(result)

    if records:
        summary_df = pd.DataFrame(records)
        summary_path = os.path.join(WATER_MASK_DIR, "water_area_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n  ── WF 面积统计摘要 ──")
        print(f"  均值  = {summary_df['water_area_km2'].mean():.2f} km²")
        print(f"  最小  = {summary_df['water_area_km2'].min():.2f} km²  "
              f"({summary_df.loc[summary_df['water_area_km2'].idxmin(), 'period']})")
        print(f"  最大  = {summary_df['water_area_km2'].max():.2f} km²  "
              f"({summary_df.loc[summary_df['water_area_km2'].idxmax(), 'period']})")

        plot_water_area_timeseries(records, FIGURES_DIR)

    print(f"\n✅ B4 完成：成功处理 {len(records)} 期")
    print("   输出目录：")
    print(f"     单通道掩膜（WM/WVH/WVV）→ {SINGLE_MASK_DIR}")
    print(f"     投票图（vote）+ WF → {WATER_MASK_DIR}")
    print("   验证：请查看 water_area_timeseries.png，确认无异常突变期")


if __name__ == "__main__":
    main()
