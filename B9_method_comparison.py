"""
B9 — 水体提取方法对比实验
============================================================
目标：在数据质量可接受（T1 > 0.05）的典型期次上，定量比较三种方法：
    ① MNDWI-only : water = (MNDWI > T1) & valid
    ② S1-only    : water = (VH < T2) & (VV < T3) & valid
    ③ 融合(现行) : water = ① & ②

统计指标：
    - 水体面积（km²）
    - IoU(方法①, 方法③) 和 IoU(方法②, 方法③)
    - 边界像元差异率（边界膨胀后交集/并集）

输出：
    output/figures/B9_method_comparison.png   （5行×4列：三方法掩膜 + 差异图）
    output/figures/B9_area_chart.png          （分组条形图）
    output/accuracy/B9_comparison_table.csv
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
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, THRESH_DIR, ACCURACY_DIR, FIGURES_DIR,
    PIXEL_SIZE, S1_BAND_ORDER,
)

# ── 对比期次（T1 > 0.05，排除严重云污染）──
# 覆盖：清晰冬季 / 浑浊夏季 / 春季中等 / 低对比度冬季 / 冬季对照
B9_COMPARISON_PERIODS = [
    "2019_Q1",   # T1=0.273，光学清晰冬季（基准）
    "2020_Q3",   # T1=0.191，夏季浑浊水体
    "2021_Q2",   # T1=0.143，春季悬浮泥沙
    "2023_Q4",   # T1=0.052，冬季低光学对比度
    "2024_Q4",   # T1=0.076，冬季对照
]

# T1 ≤ 0.05 的期次在此处列出，不参与统计
EXCLUDED_PERIODS = {
    "2022_Q2": "T1=-0.008，光学严重云污染",
    "2023_Q2": "T1=-0.005，光学严重云污染",
    "2024_Q2": "T1=0.005，光学严重云污染",
    "2022_Q3": "T1=0.011，近零阈值",
    "2023_Q3": "T1=0.008，近零阈值",
    "2024_Q3": "T1=0.021，低质量光学",
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else float("nan")


def boundary_diff_rate(a: np.ndarray, b: np.ndarray, dilation_px: int = 3) -> float:
    """
    计算两个掩膜边界的差异率：
    先对 a 做形态膨胀取边界，然后计算两个边界的不重合率。
    """
    edge_a = binary_dilation(a, iterations=dilation_px) & ~a
    edge_b = binary_dilation(b, iterations=dilation_px) & ~b
    union = (edge_a | edge_b).sum()
    if union == 0:
        return 0.0
    sym_diff = (edge_a ^ edge_b).sum()
    return float(sym_diff / union)


def compare_period(period: str, thresh_df: pd.DataFrame) -> dict | None:
    """对单期运行三种方法，返回统计结果字典。"""
    if period not in thresh_df.index:
        print(f"  ⚠️  {period} 无阈值记录，跳过")
        return None

    mndwi_path = os.path.join(MNDWI_DIR, f"{period}_mndwi.tif")
    s1db_path  = os.path.join(S1_DB_DIR,  f"{period}_s1_db.tif")
    for p in [mndwi_path, s1db_path]:
        if not os.path.exists(p):
            print(f"  ⚠️  {period} 缺少文件：{p}")
            return None

    row = thresh_df.loc[period]
    T1, T2, T3 = float(row["T1"]), float(row["T2"]), float(row["T3"])

    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1).astype(np.float32)
        res   = src.res
    with rasterio.open(s1db_path) as src:
        vv_db = src.read(S1_BAND_ORDER["vv"] + 1).astype(np.float32)
        vh_db = src.read(S1_BAND_ORDER["vh"] + 1).astype(np.float32)

    valid = (~np.isnan(mndwi)) & (~np.isnan(vv_db)) & (~np.isnan(vh_db))
    px_km2 = (PIXEL_SIZE ** 2) / 1e6  # 每像素面积（km²）

    # 三方法掩膜
    m1 = (mndwi > T1) & valid                  # MNDWI-only
    m2 = (vh_db < T2) & (vv_db < T3) & valid   # S1-only
    m3 = m1 & m2                                # 融合

    area1 = float(m1.sum()) * px_km2
    area2 = float(m2.sum()) * px_km2
    area3 = float(m3.sum()) * px_km2

    return {
        "period"       : period,
        "T1"           : round(T1, 4),
        "T2_VH"        : round(T2, 2),
        "T3_VV"        : round(T3, 2),
        "area_mndwi"   : round(area1, 1),
        "area_s1"      : round(area2, 1),
        "area_fusion"  : round(area3, 1),
        "iou_m1_m3"    : round(iou(m1, m3), 4),
        "iou_m2_m3"    : round(iou(m2, m3), 4),
        "boundary_diff_m1_m3": round(boundary_diff_rate(m1, m3), 4),
        "boundary_diff_m2_m3": round(boundary_diff_rate(m2, m3), 4),
        # 保存掩膜数组供可视化（仅前 5 期有）
        "_m1": m1, "_m2": m2, "_m3": m3,
    }


def plot_comparison_grid(results: list, out_path: str):
    """5行 × 4列 对比图（白色底图版）"""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    fig.patch.set_facecolor("white")
    fig.suptitle("图3-3  三种水体提取方法对比",
                 fontsize=13, fontweight="bold")
    col_titles = ["① MNDWI-only", "② S1-only", "③ 融合", "差异图（蓝=仅①；红=仅②；绿=共有）"]

    for row_i, res in enumerate(results):
        period = res["period"]
        m1, m2, m3 = res["_m1"], res["_m2"], res["_m3"]

        step = max(1, m1.shape[0] // 500)
        m1d, m2d, m3d = m1[::step, ::step], m2[::step, ::step], m3[::step, ::step]

        # 差异图（白底）
        diff = np.ones((*m3d.shape, 3), dtype=np.float32)  # 白色背景
        diff[m3d & m1d & m2d]  = [0.13, 0.55, 0.13]  # 三者一致（绿）
        diff[m1d & ~m3d]        = [0.12, 0.47, 0.71]  # 仅 MNDWI 多出（蓝）
        diff[m2d & ~m3d]        = [0.84, 0.15, 0.16]  # 仅 S1 多出（红）

        masks = [m1d, m2d, m3d]
        cm_bin = mcolors.ListedColormap(["#F5F5F5", "#1565C0"])
        for col_i, (mask, title) in enumerate(zip(masks, col_titles[:3])):
            ax = axes[row_i, col_i]
            ax.imshow(mask.astype(np.uint8), cmap=cm_bin, vmin=0, vmax=1,
                      interpolation="nearest")
            if row_i == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            area = res[["area_mndwi", "area_s1", "area_fusion"][col_i]]
            ax.set_xlabel(f"{area:.0f} km²", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.5); sp.set_color("#BDBDBD")
            if col_i == 0:
                ax.set_ylabel(f"{period}\nT1={res['T1']}", fontsize=8,
                              rotation=0, labelpad=55, va="center")

        ax4 = axes[row_i, 3]
        ax4.imshow(diff, interpolation="nearest")
        if row_i == 0:
            ax4.set_title(col_titles[3], fontsize=9, fontweight="bold")
        ax4.set_xlabel(
            f"IoU(①③)={res['iou_m1_m3']:.3f}  IoU(②③)={res['iou_m2_m3']:.3f}",
            fontsize=7)
        ax4.set_xticks([]); ax4.set_yticks([])
        for sp in ax4.spines.values(): sp.set_linewidth(0.5); sp.set_color("#BDBDBD")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 对比图 → {out_path}")


def plot_area_chart(results: list, out_path: str):
    """分组条形图：5期 × 3方法水体面积"""
    periods = [r["period"] for r in results]
    a1 = [r["area_mndwi"]  for r in results]
    a2 = [r["area_s1"]     for r in results]
    a3 = [r["area_fusion"] for r in results]

    x  = np.arange(len(periods))
    w  = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    for _ax in axes:
        _ax.set_facecolor("white")
        for sp in _ax.spines.values(): sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
        _ax.tick_params(colors="black", labelsize=9)

    # 左：面积条形图
    ax = axes[0]
    ax.bar(x - w, a1, w, label="① MNDWI-only", color="#1565C0", alpha=0.85)
    ax.bar(x,     a2, w, label="② S1-only",     color="#2E7D32", alpha=0.85)
    ax.bar(x + w, a3, w, label="③ 融合",         color="#E65100", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(periods, rotation=20, ha="right")
    ax.set_ylabel("水体面积（km²）", fontsize=11)
    ax.set_title("三种方法水体面积对比", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#BDBDBD")
    ax.grid(axis="y", alpha=0.3, color="#BDBDBD")

    # 右：IoU 折线图
    ax2 = axes[1]
    iou13 = [r["iou_m1_m3"] for r in results]
    iou23 = [r["iou_m2_m3"] for r in results]
    ax2.plot(x, iou13, "o-", color="#1565C0", lw=2, label="IoU（MNDWI vs 融合）")
    ax2.plot(x, iou23, "s-", color="#2E7D32", lw=2, label="IoU（S1 vs 融合）")
    ax2.axhline(1.0, ls="--", color="#9E9E9E", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(periods, rotation=20, ha="right")
    ax2.set_ylim(0.85, 1.02)
    ax2.set_ylabel("IoU（与融合方法）", fontsize=11)
    ax2.set_title("单源方法与融合方法的空间一致性", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#BDBDBD")
    ax2.grid(alpha=0.3, color="#BDBDBD")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 面积/IoU 图 → {out_path}")


def main():
    print("=" * 60)
    print("  B9 — 水体提取方法对比实验")
    print("=" * 60)
    print(f"  对比期次（T1 > 0.05）：{B9_COMPARISON_PERIODS}")
    print(f"  已排除极端云污染期次：{list(EXCLUDED_PERIODS.keys())}")

    thresh_csv = os.path.join(THRESH_DIR, "thresholds.csv")
    if not os.path.exists(thresh_csv):
        print("❌ 未找到阈值文件，请先运行 B3！")
        sys.exit(1)
    thresh_df = pd.read_csv(thresh_csv).set_index("period")

    os.makedirs(ACCURACY_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = []
    for period in B9_COMPARISON_PERIODS:
        print(f"\n  处理 {period}...")
        res = compare_period(period, thresh_df)
        if res:
            results.append(res)
            print(f"    面积: MNDWI={res['area_mndwi']:.1f} | "
                  f"S1={res['area_s1']:.1f} | 融合={res['area_fusion']:.1f} km²")
            print(f"    IoU(①③)={res['iou_m1_m3']:.4f}  IoU(②③)={res['iou_m2_m3']:.4f}")

    if not results:
        print("❌ 无有效结果")
        return

    # 输出 CSV（去掉掩膜数组）
    csv_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df = pd.DataFrame(csv_results)
    csv_path = os.path.join(ACCURACY_DIR, "B9_comparison_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 统计表 → {csv_path}")

    print("\n  均值统计：")
    print(f"    MNDWI-only 均值面积: {df.area_mndwi.mean():.1f} km²")
    print(f"    S1-only    均值面积: {df.area_s1.mean():.1f} km²")
    print(f"    融合       均值面积: {df.area_fusion.mean():.1f} km²")
    print(f"    IoU(MNDWI, 融合) 均值: {df.iou_m1_m3.mean():.4f}")
    print(f"    IoU(S1,    融合) 均值: {df.iou_m2_m3.mean():.4f}")

    # 生成图件
    comp_path  = os.path.join(FIGURES_DIR, "B9_method_comparison.png")
    chart_path = os.path.join(FIGURES_DIR, "B9_area_chart.png")
    plot_comparison_grid(results, comp_path)
    plot_area_chart(results, chart_path)

    print(f"\n✅ B9 完成！")
    print(f"   对比图：{comp_path}")
    print(f"   统计图：{chart_path}")
    print(f"   数据表：{csv_path}")

    if EXCLUDED_PERIODS:
        print("\n  已排除期次（不参与统计）：")
        for p, reason in EXCLUDED_PERIODS.items():
            print(f"    {p}: {reason}")


if __name__ == "__main__":
    main()
