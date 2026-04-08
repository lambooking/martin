"""
B8 — 水体提取质量抽检（精度评估）
============================================================
输入：
    抽取 3–5 期的 sea_mask + 对应期的 S2 影像（用于人工判读参考）
输出：
    output/accuracy/validation_points_YYYY_QN.gpkg   — 随机验证点
    output/accuracy/reference_map_YYYY_QN.png         — S2 叠加参考图
    output/accuracy/accuracy_report.csv               — 精度报告（需人工填写，代码计算）

工作流：
    步骤 1（代码）：在研究区海岸线缓冲区 ±500m 内随机生成 200 个验证点
    步骤 2（代码）：提取每点的算法分类结果，输出叠加参考图
    步骤 3（人工）：目视检查参考图，填写 reference_labels_YYYY_QN.csv
    步骤 4（代码）：读取人工标注，计算混淆矩阵 / OA / UA / PA
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol as rasterio_rowcol
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEA_MASK_DIR, WATERLINE_DIR, S2_DIR, ACCURACY_DIR, FIGURES_DIR,
    ACCURACY_N_POINTS, ACCURACY_BUFFER_M, ACCURACY_SAMPLE_PERIODS,
    PIXEL_SIZE
)

WATER_VAL   = 1
NODATA_VAL  = 255
RANDOM_SEED = 42


# ------------------------------------------------------------------
# 步骤 1–2：生成验证点 + 提取算法分类结果
# ------------------------------------------------------------------
def generate_validation_points_for_period(period: str) -> bool:
    """
    在岸线缓冲区内随机采样验证点，提取算法分类值，输出参考图。
    """
    sea_path      = os.path.join(SEA_MASK_DIR, f"{period}_sea.tif")
    waterline_gpkg = os.path.join(WATERLINE_DIR, f"{period}_waterline.gpkg")
    s2_path       = os.path.join(S2_DIR, f"S2_MNDWI_{period}.tif")
    out_gpkg      = os.path.join(ACCURACY_DIR, f"validation_points_{period}.gpkg")
    out_ref_csv   = os.path.join(ACCURACY_DIR, f"validation_extracted_{period}.csv")
    out_png       = os.path.join(ACCURACY_DIR, f"reference_map_{period}.png")

    # 检查必要文件是否存在
    if not os.path.exists(sea_path):
        print(f"  ⚠️  {period}: sea_mask 不存在，跳过")
        return False
    if not os.path.exists(waterline_gpkg):
        print(f"  ⚠️  {period}: waterline 不存在（需先运行 B7），跳过")
        return False

    # --- 读取海水掩膜 ---
    with rasterio.open(sea_path) as src:
        sea_data  = src.read(1)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    # --- 从水边线生成缓冲区 ---
    waterline_gdf = gpd.read_file(waterline_gpkg)
    waterline_gdf = waterline_gdf.to_crs(crs)

    # 使用水边线的缓冲区作为采样范围
    buffer_geom = waterline_gdf.geometry.buffer(ACCURACY_BUFFER_M).unary_union

    # --- 在缓冲区内随机生成点 ---
    rng = np.random.default_rng(RANDOM_SEED)
    xmin, ymin, xmax, ymax = buffer_geom.bounds

    points, labels_algo = [], []
    max_attempts = ACCURACY_N_POINTS * 20
    attempt = 0
    while len(points) < ACCURACY_N_POINTS and attempt < max_attempts:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        pt = Point(x, y)
        if buffer_geom.contains(pt):
            # 提取对应像素值
            row, col = rasterio_rowcol(transform, x, y)
            row, col = int(row), int(col)
            if 0 <= row < sea_data.shape[0] and 0 <= col < sea_data.shape[1]:
                val = sea_data[row, col]
                if val != NODATA_VAL:
                    algo_label = 1 if val == WATER_VAL else 0  # 1=水 0=非水
                    points.append((x, y, row, col, algo_label))
        attempt += 1

    if len(points) == 0:
        print(f"  ⚠️  {period}: 缓冲区内无有效像元，跳过")
        return False

    # 超出目标时截断
    points = points[:ACCURACY_N_POINTS]
    n_actual = len(points)

    # --- 保存验证点 GeoPackage ---
    xs, ys, rows, cols, algo_labels = zip(*points)
    gdf_pts = gpd.GeoDataFrame(
        {
            "point_id"  : range(n_actual),
            "algo_label": algo_labels,   # 0=非水, 1=水
            "ref_label" : [-1] * n_actual,  # -1=待人工填写
            "note"      : [""] * n_actual,
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )
    os.makedirs(ACCURACY_DIR, exist_ok=True)
    gdf_pts.to_file(out_gpkg, driver="GPKG")

    # 也保存为 CSV 供人工填写
    ref_csv_path = os.path.join(ACCURACY_DIR, f"reference_labels_{period}.csv")
    ref_csv = pd.DataFrame({
        "point_id"  : range(n_actual),
        "x"         : xs,
        "y"         : ys,
        "algo_label": algo_labels,
        "ref_label" : [-1] * n_actual,   # ← 人工填写：0=非水, 1=水
        "note"      : [""] * n_actual,
    })
    ref_csv.to_csv(ref_csv_path, index=False, encoding="utf-8-sig")

    # --- 保存提取结果 CSV（不含人工标注，仅算法结果）---
    pd.DataFrame({
        "point_id"  : range(n_actual),
        "x"         : xs,
        "y"         : ys,
        "pixel_row" : rows,
        "pixel_col" : cols,
        "algo_label": algo_labels,
    }).to_csv(out_ref_csv, index=False, encoding="utf-8-sig")

    # --- 绘制参考图 ---
    _plot_reference_map(period, sea_data, transform, crs, gdf_pts, s2_path, out_png)

    n_water = sum(algo_labels)
    print(f"  ✅ {period}: 生成 {n_actual} 个验证点 "
          f"（水={n_water}, 非水={n_actual-n_water}）")
    print(f"     → 请打开 {ref_csv_path} 填写 ref_label 列（0=非水, 1=水）")
    return True


def _plot_reference_map(period, sea_data, transform, crs,
                        gdf_pts, s2_path, out_png):
    """输出海水掩膜 + 验证点叠加图（供人工判读参考）。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：海水掩膜
    ax = axes[0]
    disp = np.where(sea_data == NODATA_VAL, np.nan, sea_data.astype(float))
    ax.imshow(disp, cmap="Blues_r", interpolation="none",
              extent=[transform.c, transform.c + transform.a * sea_data.shape[1],
                      transform.f + transform.e * sea_data.shape[0], transform.f])
    # 叠加验证点
    wpts = gdf_pts[gdf_pts["algo_label"] == 1]
    lpts = gdf_pts[gdf_pts["algo_label"] == 0]
    ax.scatter(
        [p.x for p in wpts.geometry], [p.y for p in wpts.geometry],
        c="cyan", s=10, label="算法=水体", zorder=5, edgecolors="navy", linewidths=0.3
    )
    ax.scatter(
        [p.x for p in lpts.geometry], [p.y for p in lpts.geometry],
        c="red", s=10, label="算法=非水体", zorder=5, edgecolors="darkred", linewidths=0.3
    )
    ax.set_title(f"{period} — 海水掩膜 + 验证点\n（请结合右图人工判读）")
    ax.legend(fontsize=8, loc="upper right")

    # 右图：S2 Green 波段（如有）
    ax2 = axes[1]
    if os.path.exists(s2_path):
        with rasterio.open(s2_path) as src:
            green = src.read(1).astype(np.float32)
            ext   = [src.bounds.left, src.bounds.right,
                     src.bounds.bottom, src.bounds.top]
        green_norm = np.clip((green - np.nanpercentile(green, 2)) /
                              (np.nanpercentile(green, 98) - np.nanpercentile(green, 2) + 1e-6),
                             0, 1)
        ax2.imshow(green_norm, cmap="gray", extent=ext, interpolation="bilinear")
        ax2.scatter(
            [p.x for p in wpts.geometry], [p.y for p in wpts.geometry],
            c="cyan", s=10, label="算法=水体", zorder=5, edgecolors="navy", linewidths=0.3
        )
        ax2.scatter(
            [p.x for p in lpts.geometry], [p.y for p in lpts.geometry],
            c="red", s=10, label="算法=非水体", zorder=5, edgecolors="darkred", linewidths=0.3
        )
        ax2.set_title(f"{period} — S2 Green 波段参考\n（用于人工判读真实类别）")
        ax2.legend(fontsize=8, loc="upper right")
    else:
        ax2.text(0.5, 0.5, f"S2 文件不存在：\n{s2_path}",
                 ha="center", va="center", transform=ax2.transAxes, color="red")
        ax2.set_title("S2 参考图（文件缺失）")

    fig.suptitle(
        f"{period} — 精度验证参考图\n"
        "请在 reference_labels_YYYY_QN.csv 中填写 ref_label 列（0=非水, 1=水）",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 步骤 4：计算精度（需人工完成 ref_label 后运行）
# ------------------------------------------------------------------
def compute_accuracy_for_period(period: str) -> dict:
    """
    读取人工标注，计算混淆矩阵 / OA / UA / PA。
    """
    ref_csv_path = os.path.join(ACCURACY_DIR, f"reference_labels_{period}.csv")
    if not os.path.exists(ref_csv_path):
        return None

    df = pd.read_csv(ref_csv_path)
    # 过滤未标注的点
    labeled = df[df["ref_label"].isin([0, 1])].copy()
    if len(labeled) == 0:
        print(f"  ℹ️  {period}: ref_label 尚未填写，跳过精度计算")
        return None

    y_pred = labeled["algo_label"].values
    y_true = labeled["ref_label"].values

    # 混淆矩阵（水=1 正类，非水=0 负类）
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    N  = len(labeled)

    OA = (TP + TN) / N if N > 0 else np.nan
    UA = TP / (TP + FP) if (TP + FP) > 0 else np.nan  # 水体用户精度
    PA = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # 水体生产者精度
    F1 = 2 * UA * PA / (UA + PA) if (UA + PA) > 0 else np.nan

    result = {
        "period": period,
        "n_points": N,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "OA": round(OA, 4),
        "Water_UA": round(UA, 4),
        "Water_PA": round(PA, 4),
        "F1": round(F1, 4),
    }
    print(f"  ✅ {period}: OA={OA:.3f}, UA(水)={UA:.3f}, PA(水)={PA:.3f}, F1={F1:.3f}")
    return result


def main():
    print("=" * 60)
    print("  B8 — 水体提取质量抽检")
    print("=" * 60)
    os.makedirs(ACCURACY_DIR, exist_ok=True)

    # --- 阶段 1：生成验证点 ---
    print("\n[阶段1] 生成验证点（供人工判读）...")
    for period in ACCURACY_SAMPLE_PERIODS:
        generate_validation_points_for_period(period)

    # --- 阶段 2：计算精度（需人工完成后运行）---
    print("\n[阶段2] 计算精度（若 ref_label 已填写）...")
    acc_records = []
    for period in ACCURACY_SAMPLE_PERIODS:
        result = compute_accuracy_for_period(period)
        if result:
            acc_records.append(result)

    if acc_records:
        acc_df = pd.DataFrame(acc_records)
        out_path = os.path.join(ACCURACY_DIR, "accuracy_report.csv")
        acc_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  精度汇总：")
        print(acc_df[["period", "n_points", "OA", "Water_UA", "Water_PA", "F1"]].to_string(index=False))
        print(f"\n  ✅ 精度报告已保存：{out_path}")
    else:
        print("\n  ℹ️  尚未有完成标注的期次。")
        print("  ⚠️  工作流说明：")
        print("  1. 运行本脚本，在 output/accuracy/ 中查找 reference_labels_YYYY_QN.csv")
        print("  2. 结合 reference_map_YYYY_QN.png 参考图，人工填写 ref_label 列")
        print("  3. 再次运行本脚本，自动计算精度指标")

    print(f"\n✅ B8 完成！验证点和参考图已保存至：{ACCURACY_DIR}")


if __name__ == "__main__":
    main()
