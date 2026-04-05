"""
A4 — S1/S2 配准质量检验与误差分析
============================================================
目的：定量验证 S1 与 S2 数据之间的空间配准精度。
    论文 §2.3.3 需报告"中位配准误差（像元）"。

前置条件（人工完成）：
    用 QGIS 同时打开 S2 真彩色合成和 S1 VV 波段，
    人工拾取 ≥ 20 个稳定地物控制点，
    保存到 data/boundary/control_points_manual.csv
    CSV 格式：point_id, lon, lat, s2_row, s2_col, s1_row, s1_col

输出：
    output/coregistration/control_points.csv      — 含偏差列的控制点表
    output/coregistration/coregistration_report.txt — 统计摘要
    output/coregistration/error_scatter.png        — 误差空间分布图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONTROL_PTS_CSV, COREG_DIR, FIGURES_DIR,
    PIXEL_SIZE, COREG_CHECK_PERIODS
)


# ------------------------------------------------------------------
# 1. 读取人工控制点
# ------------------------------------------------------------------
def load_control_points(csv_path: str) -> pd.DataFrame:
    """
    读取人工采集的控制点 CSV。

    期望列：point_id, lon, lat, s2_row, s2_col, s1_row, s1_col
    支持每个控制点有多期数据（列名带后缀，如 s2_row_2020Q1）。
    若 CSV 为宽表（每期一组列），会自动转换为长表。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到控制点文件：{csv_path}\n"
            "请先在 QGIS 中人工采集控制点并保存为上述路径。"
        )
    df = pd.read_csv(csv_path)
    required = {"point_id", "lon", "lat", "s2_row", "s2_col", "s1_row", "s1_col"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"控制点 CSV 缺少必要列。期望包含：{required}\n"
            f"实际列：{set(df.columns)}"
        )
    return df


# ------------------------------------------------------------------
# 2. 计算像元偏差
# ------------------------------------------------------------------
def compute_offsets(df: pd.DataFrame, pixel_size: float = 20.0) -> pd.DataFrame:
    """
    计算每个控制点的像元偏差（欧氏距离）和地面距离（米）。

    Parameters
    ----------
    df          : 含 s2_row, s2_col, s1_row, s1_col 列的 DataFrame
    pixel_size  : 像元地面分辨率（米）

    Returns
    -------
    df          : 添加了 row_offset, col_offset, pixel_offset, meter_offset 列
    """
    df = df.copy()
    df["row_offset"] = df["s1_row"] - df["s2_row"]
    df["col_offset"] = df["s1_col"] - df["s2_col"]
    df["pixel_offset"] = np.sqrt(df["row_offset"] ** 2 + df["col_offset"] ** 2)
    df["meter_offset"] = df["pixel_offset"] * pixel_size
    return df


# ------------------------------------------------------------------
# 3. 统计汇总
# ------------------------------------------------------------------
def compute_statistics(df: pd.DataFrame) -> dict:
    """计算配准误差统计量。"""
    offs = df["pixel_offset"]
    stats = {
        "n_points"          : len(df),
        "median_pixel"      : float(np.median(offs)),
        "mean_pixel"        : float(np.mean(offs)),
        "max_pixel"         : float(np.max(offs)),
        "std_pixel"         : float(np.std(offs, ddof=1)),
        "median_meter"      : float(np.median(offs)) * PIXEL_SIZE,
        "mean_meter"        : float(np.mean(offs)) * PIXEL_SIZE,
        "max_meter"         : float(np.max(offs)) * PIXEL_SIZE,
    }
    # 品质等级
    if stats["median_pixel"] < 0.5:
        stats["quality"] = "优（对融合结果影响可忽略）"
    elif stats["median_pixel"] < 1.0:
        stats["quality"] = "合格（年际变化分析不受影响）"
    else:
        stats["quality"] = "⚠️ 不合格，请检查 A3 预处理是否正确"
    return stats


# ------------------------------------------------------------------
# 4. 保存结果文件
# ------------------------------------------------------------------
def save_control_points(df: pd.DataFrame, out_dir: str) -> str:
    """保存含偏差列的控制点 CSV。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "control_points.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ 控制点文件已保存：{out_path}")
    return out_path


def save_report(stats: dict, out_dir: str) -> str:
    """保存纯文本统计摘要报告。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coregistration_report.txt")
    lines = [
        "=" * 60,
        "  S1/S2 配准质量检验报告",
        "  黄河三角洲岸线变化研究",
        "=" * 60,
        "",
        f"  检验控制点数量   : {stats['n_points']} 个",
        "",
        "  误差统计（单位：像元 / 米）",
        f"  中位误差 (Median) : {stats['median_pixel']:.3f} 像元"
          f"  /  {stats['median_meter']:.1f} m   ← 论文报告此值",
        f"  均值    (Mean)    : {stats['mean_pixel']:.3f} 像元"
          f"  /  {stats['mean_meter']:.1f} m",
        f"  最大值  (Max)     : {stats['max_pixel']:.3f} 像元"
          f"  /  {stats['max_meter']:.1f} m",
        f"  标准差  (Std)     : {stats['std_pixel']:.3f} 像元",
        "",
        "  品质评价",
        f"  → {stats['quality']}",
        "",
        "  判断标准",
        "  < 0.5 像元 (< 10m) : 优（融合影响可忽略）",
        "  0.5–1.0 像元 (10–20m) : 合格",
        "  > 1.0 像元 (> 20m) : 需检查 A3 预处理",
        "",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"  ✅ 报告已保存：{out_path}")
    return out_path


# ------------------------------------------------------------------
# 5. 误差空间分布散点图
# ------------------------------------------------------------------
def plot_error_scatter(df: pd.DataFrame, stats: dict, out_dir: str) -> str:
    """
    绘制控制点误差空间分布散点图。
    颜色表示像元偏差大小，箭头表示 (col_offset, row_offset) 方向和幅度。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "error_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("S1/S2 配准误差空间分布\n黄河三角洲（2019–2024）",
                 fontsize=14, fontweight="bold", y=1.01)

    # --- 子图 1：误差量级空间分布 ---
    ax = axes[0]
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["pixel_offset"],
        cmap="RdYlGn_r",
        vmin=0, vmax=max(1.5, df["pixel_offset"].max()),
        s=80, edgecolors="k", linewidths=0.4, zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("像元偏差 (pixel)", fontsize=10)
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title("像元偏差空间分布（颜色深度=误差大小）")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # 添加阈值虚线标注（中位误差）
    med = stats["median_pixel"]
    ax.text(
        0.03, 0.97,
        f"中位误差 = {med:.3f} px\n= {med*PIXEL_SIZE:.1f} m",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    ax.grid(True, alpha=0.3)

    # --- 子图 2：行列偏差矢量图 ---
    ax2 = axes[1]
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    sc2 = ax2.scatter(
        df["col_offset"], df["row_offset"],
        c=df["pixel_offset"],
        cmap="RdYlGn_r",
        vmin=0, vmax=max(1.5, df["pixel_offset"].max()),
        s=80, edgecolors="k", linewidths=0.4, zorder=3
    )
    # 单位圆（0.5 和 1.0 像元阈值圆）
    theta = np.linspace(0, 2 * np.pi, 300)
    for r, ls, lbl in [(0.5, "--", "0.5 px"), (1.0, "-", "1.0 px")]:
        ax2.plot(r * np.cos(theta), r * np.sin(theta),
                 color="steelblue", linestyle=ls, linewidth=1.2,
                 label=lbl, zorder=2)
    fig.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02).set_label(
        "像元偏差 (pixel)", fontsize=10)
    ax2.set_xlabel("列方向偏差 (像元)")
    ax2.set_ylabel("行方向偏差 (像元)")
    ax2.set_title("行列方向偏差分布（蓝圈=0.5/1.0 px 阈值）")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 误差散点图已保存：{out_path}")
    return out_path


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  A4 — S1/S2 配准质量检验")
    print("=" * 55)

    # 1. 读取控制点
    print(f"\n[1/4] 读取控制点：{CONTROL_PTS_CSV}")
    df = load_control_points(CONTROL_PTS_CSV)
    print(f"      共读取 {len(df)} 个控制点")

    # 2. 计算像元偏差
    print("\n[2/4] 计算像元偏差...")
    df = compute_offsets(df, pixel_size=PIXEL_SIZE)

    # 3. 统计
    print("\n[3/4] 统计误差...")
    stats = compute_statistics(df)

    # 4. 保存输出
    print("\n[4/4] 保存输出文件...")
    save_control_points(df, COREG_DIR)
    save_report(stats, COREG_DIR)
    plot_error_scatter(df, stats, COREG_DIR)

    print("\n✅ A4 模块运行完成！")
    print(f"   论文报告值：中位配准误差 = {stats['median_pixel']:.3f} 像元"
          f" = {stats['median_meter']:.1f} m")


if __name__ == "__main__":
    main()
