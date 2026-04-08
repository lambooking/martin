"""
E 阶段：可视化输出
============================================================
输出高质量的可视化图表，用于论文：
E1 — 水边线时序叠加图
E2 — NSM / EPR 沿岸分布图
E3 — 热点岸段专题图
E4 — 典型断面位置时序图

地图类图件（E1/E3）统一规范：
    - 白色底图（无外部底图依赖）
    - 边框经纬度刻度
    - 比例尺（单位 km）
    - 指北针
    - 图例标题统一为"图例"
    - 图名不含括号补充说明
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIGURES_DIR, WATERLINE_DIR, CHANGE_DIR, TRANSECT_DIR,
    DISTANCE_DIR, ANNUAL_SL_DIR, YEARS, PERIODS
)


# ── 地图辅助函数 ──────────────────────────────────────────────────────────────

def _fmt_lon(x, pos):
    """经度刻度格式：119.00°E"""
    return f"{x:.2f}°E"


def _fmt_lat(x, pos):
    """纬度刻度格式：38.00°N"""
    return f"{x:.2f}°N"


def _setup_geo_axes(ax, crs_is_geographic: bool):
    """
    为地图坐标轴设置白色背景与经纬度刻度格式。
    必须在 gdf.plot() 之后调用，以确保刻度范围正确。
    """
    fig = ax.get_figure()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if crs_is_geographic:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
        ax.set_xlabel("经度", fontsize=11)
        ax.set_ylabel("纬度", fontsize=11)
    else:
        ax.set_xlabel("东向 (m)", fontsize=11)
        ax.set_ylabel("北向 (m)", fontsize=11)

    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC")
    for sp in ax.spines.values():
        sp.set_color("#333333")
        sp.set_linewidth(0.8)


def _add_scale_bar(ax, crs_is_geographic: bool, bar_km: float = 5.0):
    """
    在地图左下角绘制比例尺，单位标注为 km。

    Parameters
    ----------
    crs_is_geographic : True 表示地理坐标（度），False 表示投影坐标（米）
    bar_km            : 比例尺代表的实际距离（km）
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # 将 km 换算为坐标系数据单位
    if crs_is_geographic:
        bar_data = bar_km / 111.0          # 度（1° ≈ 111 km）
    else:
        bar_data = bar_km * 1000.0         # 米

    # 比例尺起止位置（数据坐标）
    x_start = xlim[0] + x_range * 0.05
    x_end   = x_start + bar_data
    y_pos   = ylim[0] + y_range * 0.055
    tick_h  = y_range * 0.007

    # 主线
    ax.plot([x_start, x_end], [y_pos, y_pos],
            color="black", lw=2.5, solid_capstyle="butt", zorder=6)
    # 端点刻度
    for xp in [x_start, x_end]:
        ax.plot([xp, xp], [y_pos - tick_h, y_pos + tick_h],
                color="black", lw=1.5, zorder=6)
    # 标注文字（英文 km）
    ax.text(
        (x_start + x_end) / 2,
        y_pos - tick_h * 3.5,
        f"{bar_km:.0f} km",
        ha="center", va="top",
        fontsize=9, fontweight="bold", zorder=6,
    )


def _add_north_arrow(ax, x=0.93, y=0.87, size=0.07):
    """
    在地图右上角绘制指北针（箭头 + N 字）。

    Parameters
    ----------
    x, y  : 箭尾在坐标轴归一化坐标中的位置
    size  : 箭头在归一化坐标中的长度
    """
    ax.annotate(
        "",
        xy=(x, y + size), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.8,
            mutation_scale=18,
        ),
        zorder=7,
    )
    ax.text(
        x, y + size + 0.025,
        "N",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        transform=ax.transAxes,
        zorder=7,
    )


# ── E1：水边线时序叠加图 ──────────────────────────────────────────────────────

def plot_e1_waterline_timeseries():
    """E1 — 24期瞬时水边线叠加图，颜色随时间渐变。"""
    print("\n[E1] 绘制水边线时序图...")
    out_path = os.path.join(FIGURES_DIR, "E1_waterline_timeseries.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=len(PERIODS) - 1)

    plotted  = 0
    gdf_last = None
    for i, period in enumerate(PERIODS):
        wl_path = os.path.join(WATERLINE_DIR, f"{period}_waterline.gpkg")
        if not os.path.exists(wl_path):
            continue
        try:
            gdf = gpd.read_file(wl_path)
            gdf.plot(ax=ax, color=cmap(norm(i)), linewidth=0.6, alpha=0.75)
            gdf_last = gdf
            plotted += 1
        except Exception as e:
            print(f"  读取 {period} 失败: {e}")

    if plotted == 0:
        print("  ⚠️  没有可用的水边线数据。")
        plt.close(fig)
        return

    crs_is_geo = gdf_last.crs.is_geographic if gdf_last is not None else True

    # 白色背景 + 经纬度刻度
    _setup_geo_axes(ax, crs_is_geo)

    # 比例尺 & 指北针
    _add_scale_bar(ax, crs_is_geo, bar_km=5)
    _add_north_arrow(ax)

    # 色标（图例）
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.set_title("图例", fontsize=10, pad=6)
    cbar.set_ticks([0, len(PERIODS) // 2, len(PERIODS) - 1])
    cbar.set_ticklabels([PERIODS[0], PERIODS[len(PERIODS) // 2], PERIODS[-1]])
    cbar.set_label("观测期次", fontsize=10)

    ax.set_title("黄河三角洲水边线演变动态", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ E1 保存: {out_path}")


# ── E2：沿岸 NSM / EPR 分布图 ─────────────────────────────────────────────────

def plot_e2_nsm_epr_distribution():
    """E2 — 沿岸 NSM 和 EPR 分布图（统计图，无地图要素）"""
    print("\n[E2] 绘制沿岸变化分布图...")
    nsm_path = os.path.join(CHANGE_DIR, "NSM.csv")
    epr_path = os.path.join(CHANGE_DIR, "EPR.csv")

    if not (os.path.exists(nsm_path) and os.path.exists(epr_path)):
        print("  ⚠️  缺少 NSM/EPR 文件，依赖 D 模块的结果。")
        return

    nsm_df = pd.read_csv(nsm_path)
    epr_df = pd.read_csv(epr_path)
    df = pd.merge(nsm_df, epr_df, on="transect_id")

    if len(df) == 0:
        return

    out_path = os.path.join(FIGURES_DIR, "E2_NSM_EPR_along_coast.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("white")

    t_ids = df["transect_id"]

    colors1 = ["#d73027" if v > 0 else "#4575b4" for v in df["NSM_m"]]
    ax1.set_facecolor("white")
    ax1.bar(t_ids, df["NSM_m"], color=colors1, width=1.0)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_ylabel("净位移 NSM (m)", fontsize=11)
    ax1.set_title("黄河三角洲沿岸 NSM 空间分布特征", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    colors2 = ["#d73027" if v > 0 else "#4575b4" for v in df["EPR_m_yr"]]
    ax2.set_facecolor("white")
    ax2.bar(t_ids, df["EPR_m_yr"], color=colors2, width=1.0)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("沿岸断面编号", fontsize=12)
    ax2.set_ylabel("端点速率 EPR (m/yr)", fontsize=11)
    ax2.set_title("黄河三角洲沿岸 EPR 空间分布特征", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    custom_lines = [
        Line2D([0], [0], color="#d73027", lw=4),
        Line2D([0], [0], color="#4575b4", lw=4),
    ]
    ax1.legend(
        custom_lines, ["向海淤积", "向陆侵蚀"],
        loc="upper right",
        title="图例", title_fontsize=10,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ E2 保存: {out_path}")


# ── E3：热点岸段专题图 ─────────────────────────────────────────────────────────

def plot_e3_hotspots_map():
    """E3 — 热点岸段专题图（地图，含经纬度/比例尺/指北针/图例）"""
    print("\n[E3] 绘制热点岸段地图...")
    hotspots_path  = os.path.join(CHANGE_DIR,   "hotspots.gpkg")
    transects_path = os.path.join(TRANSECT_DIR, "transects.gpkg")

    if not (os.path.exists(hotspots_path) and os.path.exists(transects_path)):
        print("  ⚠️  缺少所需几何文件。")
        return

    out_path = os.path.join(FIGURES_DIR, "E3_hotspots_map.png")
    fig, ax = plt.subplots(figsize=(10, 10))

    try:
        t_gdf  = gpd.read_file(transects_path)
        hs_gdf = gpd.read_file(hotspots_path)

        # 白色背景 + 经纬度刻度（先绘制数据确定范围）
        t_gdf.plot(ax=ax, color="lightgray", linewidth=0.5, alpha=0.5, label="所有断面")

        acc = hs_gdf[hs_gdf["type"] == "Accretion"]
        ero = hs_gdf[hs_gdf["type"] == "Erosion"]
        if not acc.empty:
            acc.plot(ax=ax, color="#d73027", linewidth=4, alpha=0.9, label="强淤积热点")
        if not ero.empty:
            ero.plot(ax=ax, color="#4575b4", linewidth=4, alpha=0.9, label="强侵蚀热点")

        crs_is_geo = hs_gdf.crs.is_geographic
        _setup_geo_axes(ax, crs_is_geo)

        # 图例、比例尺、指北针
        ax.legend(
            fontsize=10, loc="lower left",
            title="图例", title_fontsize=10,
        )
        _add_scale_bar(ax, crs_is_geo, bar_km=5)
        _add_north_arrow(ax)

        ax.set_title("岸线演变关键热点区段分布图", fontsize=15, fontweight="bold", pad=12)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✅ E3 保存: {out_path}")

    except Exception as e:
        print(f"  ❌ 绘制失败: {e}")
        plt.close(fig)


# ── E4：典型断面位置时序图 ──────────────────────────────────────────────────────

def plot_e4_transect_timeseries():
    """E4 — 典型断面位置时序图（统计图，无地图要素）"""
    print("\n[E4] 绘制典型断面时序波动图...")
    dist_path = os.path.join(DISTANCE_DIR,  "distance_matrix.csv")
    mhw_path  = os.path.join(ANNUAL_SL_DIR, "MHW_proxy_distances.csv")

    if not (os.path.exists(dist_path) and os.path.exists(mhw_path)):
        print("  ⚠️  缺少距离文件。")
        return

    dist_df = pd.read_csv(dist_path, index_col="transect_id")
    mhw_df  = pd.read_csv(mhw_path,  index_col="transect_id")

    if len(dist_df) < 3:
        return

    # 选取有至少一期有效数据的断面，等距抽取 3 条作为典型代表
    valid_tids = dist_df.dropna(how="all", axis=0).index.tolist()
    if len(valid_tids) < 3:
        print("  ⚠️  有效断面过少。")
        return

    step       = len(valid_tids) // 3
    sample_ids = [valid_tids[0], valid_tids[step], valid_tids[2 * step]]

    out_path = os.path.join(FIGURES_DIR, "E4_transect_timeseries.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.patch.set_facecolor("white")

    x_positions = np.arange(len(PERIODS))

    for idx, (ax, t_id) in enumerate(zip(axes, sample_ids)):
        ax.set_facecolor("white")
        t_row = dist_df.loc[t_id]

        y_vals, x_plot = [], []
        for i, p in enumerate(PERIODS):
            if p in t_row.index and not np.isnan(t_row[p]):
                y_vals.append(t_row[p])
                x_plot.append(x_positions[i])

        if not y_vals:
            continue

        ax.scatter(x_plot, y_vals, color="gray", alpha=0.6, s=40,
                   label="单期瞬时水边线位置", zorder=2)
        ax.plot(x_plot, y_vals, color="lightgray", alpha=0.4, linewidth=1, zorder=1)

        # 年度 MHW proxy 参考线
        mhw_row = mhw_df.loc[t_id]
        for yi, year in enumerate(YEARS):
            col_name = str(year)
            if col_name in mhw_row.index and not np.isnan(mhw_row[col_name]):
                val     = mhw_row[col_name]
                start_x = yi * 4
                end_x   = min(start_x + 3.8, len(PERIODS) - 1)
                ax.hlines(
                    val, start_x, end_x,
                    color="teal", linewidth=3, alpha=0.8,
                    label="MHW proxy 年度参考线" if yi == 0 else "",
                )

        ax.set_title(f"断面 T{t_id} 临岸距离时序", fontsize=12, fontweight="bold", pad=8, loc="left")
        ax.set_ylabel("距基线距离 (m)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)

        if idx == 0:
            ax.legend(loc="upper right", title="图例", title_fontsize=10, fontsize=10)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(PERIODS, rotation=45, ha="right")
    axes[-1].set_xlabel("合成期次", fontsize=12)

    fig.suptitle("年内波动与年际变化趋势对比", fontsize=15, fontweight="bold", y=0.94)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ E4 保存: {out_path}")


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  E 阶段 — 图形报告渲染")
    print("=" * 55)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_e1_waterline_timeseries()
    plot_e2_nsm_epr_distribution()
    plot_e3_hotspots_map()
    plot_e4_transect_timeseries()

    print("\n✅ E 模块图表全部生成完成！")


if __name__ == "__main__":
    main()
