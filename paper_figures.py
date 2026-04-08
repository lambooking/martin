"""
paper_figures.py — 论文图件生成脚本（图3-1 至 图4-11）
============================================================
统一规范（v2）：
  - 白色背景，深色文字
  - 投影：EPSG:32650
  - 矢量图剪裁至研究区内部（去除数据边框线伪影）
  - 图名不含括号说明，不含最大/最小值注释
  - DPI=300，bbox_inches='tight'
"""

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from shapely.geometry import box as sbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRANSECT_DIR, ANNUAL_SL_DIR, CHANGE_DIR, WATERLINE_DIR, FIGURES_DIR

OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── 颜色方案 ─────────────────────────────────────────────
YEAR_COLORS = {
    "2019": "#1565C0",
    "2020": "#2E7D32",
    "2021": "#F57F17",
    "2022": "#E65100",
    "2023": "#B71C1C",
    "2024": "#6A1B9A",
}
QUARTER_COLORS = {
    "Q1": "#1976D2",
    "Q2": "#388E3C",
    "Q3": "#F57C00",
    "Q4": "#7B1FA2",
}
FONT_TITLE  = 15
FONT_LABEL  = 13
FONT_TICK   = 11
FONT_LEGEND = 11

# ── 研究区裁剪边界（EPSG:32650，去除边框伪岸线）───────────
# 左/右/底 inset 2000m，顶 inset 600m（真实海岸距顶 ~1300m）
_BBOX = dict(left=604956, right=693408, bottom=4196813, top=4241394)
CLIP_BOX = sbox(
    _BBOX["left"]   + 2000,
    _BBOX["bottom"] + 2000,
    _BBOX["right"]  - 2000,
    _BBOX["top"]    - 600,
)
CLIP_CRS = "EPSG:32650"


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {name}")
    return path


def _read_gpkg(path, layer=None):
    if not os.path.exists(path):
        return None
    try:
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    except Exception:
        return None


def _to32650(gdf):
    """统一投影到 EPSG:32650。"""
    if gdf is None or len(gdf) == 0:
        return gdf
    if gdf.crs and str(gdf.crs).upper() != "EPSG:32650":
        return gdf.to_crs("EPSG:32650")
    return gdf


def _clip_interior(gdf):
    """裁剪至研究区内部，去除数据边框伪岸线。"""
    if gdf is None or len(gdf) == 0:
        return gdf
    clip_gdf = gpd.GeoDataFrame(geometry=[CLIP_BOX], crs=CLIP_CRS)
    try:
        clipped = gpd.clip(gdf, clip_gdf)
        return clipped if len(clipped) > 0 else gdf
    except Exception:
        return gdf


def _fit_bounds(ax, gdfs, pad_ratio=0.03):
    """依据矢量数据设置等比例坐标轴范围（5% 边距）。"""
    combined = [g for g in gdfs if g is not None and len(g) > 0]
    if not combined:
        return
    all_b = np.array([g.total_bounds for g in combined])
    minx, miny = all_b[:, 0].min(), all_b[:, 1].min()
    maxx, maxy = all_b[:, 2].max(), all_b[:, 3].max()
    pad = max(maxx - minx, maxy - miny) * pad_ratio
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="datalim")


def _style_map_ax(ax):
    """矢量地图子图统一样式（白底）。"""
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#9E9E9E")


def _style_chart_ax(ax):
    """统计图子图统一样式（白底）。"""
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=FONT_TICK, length=4)
    for sp in ax.spines.values():
        sp.set_color("#9E9E9E")
        sp.set_linewidth(0.8)
    ax.grid(axis="y", alpha=0.35, color="#BDBDBD", linewidth=0.6)


# ══════════════════════════════════════════════════════════
# 图3-1  断面体系构建示意图
# ══════════════════════════════════════════════════════════
def plot_fig3_1():
    transects = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "transects.gpkg"))))
    baseline  = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))
    waterline = _clip_interior(_to32650(_read_gpkg(os.path.join(WATERLINE_DIR, "2019_Q1_waterline.gpkg"))))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    if transects is not None:
        transects.iloc[::20].plot(ax=ax, color="#5C6BC0", lw=0.6, alpha=0.75, zorder=2)
    if baseline is not None:
        baseline.plot(ax=ax, color="#E65100", lw=2.2, zorder=4)
    if waterline is not None:
        waterline.plot(ax=ax, color="#1565C0", lw=1.2, zorder=3, linestyle="--")

    legend = [
        Line2D([0], [0], color="#5C6BC0", lw=1.5, label="法向断面（每20条采样）"),
        Line2D([0], [0], color="#E65100", lw=2.5, label="基线"),
        Line2D([0], [0], color="#1565C0", lw=1.5, ls="--", label="水边线示例（2019_Q1）"),
    ]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, [transects, baseline])
    ax.set_title("图3-1  断面与基线构建示意图", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图3-1_transect_framework.png")


# ══════════════════════════════════════════════════════════
# 图3-2  年度岸线构建示意图（2021年）
# ══════════════════════════════════════════════════════════
def plot_fig3_2():
    target = "EPSG:32650"
    baseline = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))
    mhw   = _clip_interior(_to32650(_read_gpkg(os.path.join(ANNUAL_SL_DIR, "MHW_proxy_2021.gpkg"))))
    outer = _clip_interior(_to32650(_read_gpkg(os.path.join(ANNUAL_SL_DIR, "Outer_P95_2021.gpkg"))))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    all_gdfs = []
    for q, clr in QUARTER_COLORS.items():
        path = os.path.join(WATERLINE_DIR, f"2021_{q}_waterline.gpkg")
        gdf = _clip_interior(_to32650(_read_gpkg(path)))
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax, color=clr, lw=0.8, alpha=0.55, zorder=2)
            all_gdfs.append(gdf)

    if baseline is not None:
        baseline.plot(ax=ax, color="#9E9E9E", lw=0.7, alpha=0.5, zorder=2)
        all_gdfs.append(baseline)
    if mhw is not None:
        mhw.plot(ax=ax, color="#B71C1C", lw=2.0, zorder=5)
        all_gdfs.append(mhw)
    if outer is not None:
        outer.plot(ax=ax, color="#1565C0", lw=2.0, zorder=4, linestyle="--")
        all_gdfs.append(outer)

    legend = [
        Line2D([0], [0], color=c, lw=1.5, label=f"2021_{q} 水边线")
        for q, c in QUARTER_COLORS.items()
    ] + [
        Line2D([0], [0], color="#B71C1C", lw=2.5, label="MHW_proxy（年度代表岸线）"),
        Line2D([0], [0], color="#1565C0", lw=2.5, ls="--", label="Outer_P95（外包络线）"),
        Line2D([0], [0], color="#9E9E9E", lw=1.0, label="基线"),
    ]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, all_gdfs)
    ax.set_title("图3-2  年度代表岸线与外包络线构建示意图", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图3-2_annual_construction.png")


# ══════════════════════════════════════════════════════════
# 图4-1  季度岸线叠加（2021年）
# ══════════════════════════════════════════════════════════
def plot_fig4_1():
    baseline = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    all_gdfs = [baseline] if baseline is not None else []
    if baseline is not None:
        baseline.plot(ax=ax, color="#BDBDBD", lw=0.6, alpha=0.6, zorder=2)

    for q, clr in QUARTER_COLORS.items():
        path = os.path.join(WATERLINE_DIR, f"2021_{q}_waterline.gpkg")
        gdf = _clip_interior(_to32650(_read_gpkg(path)))
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax, color=clr, lw=1.5, zorder=3)
            all_gdfs.append(gdf)

    legend = [Line2D([0], [0], color=c, lw=2, label=f"2021_{q} 水边线")
              for q, c in QUARTER_COLORS.items()]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, all_gdfs)
    ax.set_title("图4-1  2021年季度岸线空间分布", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图4-1_seasonal_shorelines.png")


# ══════════════════════════════════════════════════════════
# 图4-2  各断面季度岸线波动（标准差）
# ══════════════════════════════════════════════════════════
def plot_fig4_2():
    dm_path = os.path.join(os.path.dirname(CHANGE_DIR), "distances/distance_matrix.csv")
    dm = pd.read_csv(dm_path, index_col=0)
    std_per = dm.std(axis=1)
    # 排除边界断面（std=0 的边界断面）
    valid_mask = std_per > 0
    x_all = std_per.index.values
    y_all = std_per.values

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    _style_chart_ax(ax)

    # 竖向色块条（每断面一条竖线），用颜色代表波动强弱
    cmap = plt.cm.get_cmap("RdYlBu_r")
    y_max = float(np.nanpercentile(y_all[valid_mask], 98))
    norm = mcolors.Normalize(vmin=0, vmax=y_max)
    colors = cmap(norm(np.clip(y_all, 0, y_max)))
    ax.vlines(x_all[valid_mask], 0, y_all[valid_mask],
              colors=colors[valid_mask], linewidth=0.6, alpha=0.9)

    # 移动平均平滑线（窗口200断面），突出整体趋势
    win = 200
    y_ser = pd.Series(y_all, index=x_all)
    y_smooth = y_ser.rolling(win, center=True, min_periods=10).mean()
    ax.plot(x_all, y_smooth.values, color="#B71C1C", lw=2.0,
            label=f"滑动均值（窗口={win}条断面）", zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, label="季度位置标准差（m）", shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=FONT_TICK)

    ax.set_xlabel("断面编号", fontsize=FONT_LABEL)
    ax.set_ylabel("标准差（m）", fontsize=FONT_LABEL)
    ax.set_xlim(x_all[0], x_all[-1])
    ax.set_ylim(0, y_max * 1.05)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right",
              framealpha=0.9, edgecolor="#BDBDBD")
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title("图4-2  各断面季度岸线位置波动", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图4-2_seasonal_nsm.png")


# ══════════════════════════════════════════════════════════
# 图4-3  时空变化热力图（相对异常）
# ══════════════════════════════════════════════════════════
def plot_fig4_3():
    dm_path = os.path.join(os.path.dirname(CHANGE_DIR), "distances/distance_matrix.csv")
    dm = pd.read_csv(dm_path, index_col=0)

    # 去除全NaN行（边界断面）
    dm = dm.dropna(how="all")
    # 只保留有效数据行（std > 0）
    dm = dm[dm.std(axis=1) > 0]

    # 计算相对异常（每断面减去其时间均值），突出时间变化趋势
    dm_mean = dm.mean(axis=1)
    dm_anom = dm.subtract(dm_mean, axis=0)

    # 等间距采样 ~400 行
    step = max(1, len(dm_anom) // 400)
    dm_plot = dm_anom.iloc[::step]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    # 左图：时空热力图（异常值）
    ax = axes[0]
    vmax = float(np.nanpercentile(np.abs(dm_anom.values), 90))
    im = ax.imshow(dm_plot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, label="岸线位置异常（m，相对于断面时间均值）", shrink=0.85)
    cb.ax.tick_params(labelsize=FONT_TICK)
    ax.set_xlabel("时间（季度）", fontsize=FONT_LABEL)
    ax.set_ylabel(f"断面编号（间隔{step}采样）", fontsize=FONT_LABEL)
    ax.set_title("图4-3  季度岸线位置时空变化热力图（相对距离异常）",
                 fontsize=FONT_TITLE, fontweight="bold")
    # 时间轴标签
    n_cols = dm_plot.shape[1]
    tick_step = max(1, n_cols // 8)
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_xticklabels(dm_plot.columns[::tick_step], rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_facecolor("white")

    # 右图：各期次的岸线位置中位数趋势
    ax2 = axes[1]
    ax2.set_facecolor("white")
    period_medians = dm.median(axis=0).values
    y_pos = range(len(period_medians))
    ax2.barh(list(y_pos), period_medians - period_medians.mean(),
             color=["#EF5350" if v > 0 else "#1E88E5" for v in period_medians - period_medians.mean()],
             height=0.7)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(list(y_pos)[::tick_step])
    ax2.set_yticklabels(dm.columns[::tick_step], fontsize=7)
    ax2.set_xlabel("中位距离偏差（m）", fontsize=FONT_LABEL)
    ax2.set_title("各期次中位距离", fontsize=10, fontweight="bold")
    for sp in ax2.spines.values():
        sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
    ax2.grid(axis="x", alpha=0.3, color="#BDBDBD")
    ax2.tick_params(colors="black", labelsize=FONT_TICK)

    fig.tight_layout()
    return _save(fig, "图4-3_distance_heatmap.png")


# ══════════════════════════════════════════════════════════
# 图4-4  6年年度岸线叠加（MHW_proxy）
# ══════════════════════════════════════════════════════════
def plot_fig4_4():
    baseline = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    all_gdfs = [baseline] if baseline is not None else []
    if baseline is not None:
        baseline.plot(ax=ax, color="#E0E0E0", lw=0.7, alpha=0.7, zorder=2)

    for yr, clr in YEAR_COLORS.items():
        gdf = _clip_interior(_to32650(_read_gpkg(os.path.join(ANNUAL_SL_DIR, f"MHW_proxy_{yr}.gpkg"))))
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax, color=clr, lw=1.3, zorder=3)
            all_gdfs.append(gdf)

    legend = [Line2D([0], [0], color=c, lw=2, label=f"{yr}")
              for yr, c in YEAR_COLORS.items()]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              title="年份", title_fontsize=FONT_LEGEND,
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, all_gdfs)
    ax.set_title("图4-4  2019—2024年年度代表岸线变化对比", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图4-4_annual_shorelines.png")


# ══════════════════════════════════════════════════════════
# 图4-5  NSM 空间分布
# ══════════════════════════════════════════════════════════
def plot_fig4_5():
    nsm_df = pd.read_csv(os.path.join(CHANGE_DIR, "NSM.csv"))
    x_col  = "transect_id" if "transect_id" in nsm_df.columns else nsm_df.columns[0]
    v_col  = "NSM_m" if "NSM_m" in nsm_df.columns else "NSM"
    x, nsm = nsm_df[x_col].values, nsm_df[v_col].values
    # 排除边界断面（NSM=0 且相邻也是0，说明是全NaN断面）
    valid = np.abs(nsm) > 0.01

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    _style_chart_ax(ax)

    pos = nsm >= 0
    ax.bar(x[ pos & valid],  nsm[ pos & valid],  width=1, color="#EF5350", alpha=0.85, label="淤积（NSM > 0）")
    ax.bar(x[~pos & valid],  nsm[~pos & valid],  width=1, color="#1E88E5", alpha=0.85, label="侵蚀（NSM < 0）")
    ax.axhline(0, color="black", lw=0.6)

    ax.set_xlabel("断面编号", fontsize=FONT_LABEL)
    ax.set_ylabel("净迁移距离（m）", fontsize=FONT_LABEL)
    ax.set_title("图4-5  年际尺度岸线净迁移量 NSM 空间分布", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#BDBDBD")
    fig.tight_layout()
    return _save(fig, "图4-5_nsm_distribution.png")


# ══════════════════════════════════════════════════════════
# 图4-6  EPR 变化速率
# ══════════════════════════════════════════════════════════
def plot_fig4_6():
    epr_df = pd.read_csv(os.path.join(CHANGE_DIR, "EPR.csv"))
    x_col  = "transect_id" if "transect_id" in epr_df.columns else epr_df.columns[0]
    v_col  = "EPR_m_yr" if "EPR_m_yr" in epr_df.columns else "EPR"
    x, epr = epr_df[x_col].values, epr_df[v_col].values
    valid  = np.abs(epr) > 0.01

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    _style_chart_ax(ax)

    pos = epr >= 0
    ax.bar(x[ pos & valid],  epr[ pos & valid],  width=1, color="#EF5350", alpha=0.85, label="淤积（EPR > 0）")
    ax.bar(x[~pos & valid],  epr[~pos & valid],  width=1, color="#1E88E5", alpha=0.85, label="侵蚀（EPR < 0）")
    ax.axhline(0, color="black", lw=0.6)

    ax.set_xlabel("断面编号", fontsize=FONT_LABEL)
    ax.set_ylabel("变化速率（m/yr）", fontsize=FONT_LABEL)
    ax.set_title("图4-6  岸线年变化速率 EPR 空间分布", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#BDBDBD")
    fig.tight_layout()
    return _save(fig, "图4-6_epr_distribution.png")


# ══════════════════════════════════════════════════════════
# 图4-7  Outer_P95 外包络线叠加
# ══════════════════════════════════════════════════════════
def plot_fig4_7():
    baseline = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    all_gdfs = [baseline] if baseline is not None else []
    if baseline is not None:
        baseline.plot(ax=ax, color="#E0E0E0", lw=0.7, alpha=0.6, zorder=2)

    for yr, clr in YEAR_COLORS.items():
        gdf = _clip_interior(_to32650(_read_gpkg(os.path.join(ANNUAL_SL_DIR, f"Outer_P95_{yr}.gpkg"))))
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax, color=clr, lw=1.2, zorder=3)
            all_gdfs.append(gdf)

    legend = [Line2D([0], [0], color=c, lw=2, label=f"{yr}")
              for yr, c in YEAR_COLORS.items()]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              title="年份", title_fontsize=FONT_LEGEND,
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, all_gdfs)
    ax.set_title("图4-7  年度外包络线 Outer_P95 变化特征", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图4-7_outer_envelope.png")


# ══════════════════════════════════════════════════════════
# 图4-8  波动范围（Outer_P95 − MHW_proxy）
# ══════════════════════════════════════════════════════════
def plot_fig4_8():
    dist_dir = os.path.join(os.path.dirname(CHANGE_DIR), "distances")
    dm = pd.read_csv(os.path.join(dist_dir, "distance_matrix.csv"), index_col=0)

    years = [yr for yr in YEAR_COLORS]
    volatility = {}
    for yr in years:
        yr_cols = [c for c in dm.columns if c.startswith(yr)]
        if len(yr_cols) < 2:
            continue
        sub = dm[yr_cols]
        outer = sub.quantile(0.95, axis=1)
        mhw_p = sub.median(axis=1)
        vol = (outer - mhw_p).fillna(0)
        volatility[yr] = vol.values

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # 左：各年折线
    ax = axes[0]
    ax.set_facecolor("white")
    for sp in ax.spines.values(): sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
    for yr, vol in volatility.items():
        x = np.where(vol > 0)[0]
        y = vol[vol > 0]
        ax.plot(x, y, color=YEAR_COLORS[yr], lw=0.9, alpha=0.85, label=yr)
    ax.set_xlabel("断面编号", fontsize=FONT_LABEL)
    ax.set_ylabel("波动范围（m）", fontsize=FONT_LABEL)
    ax.set_title("图4-8a  各年岸线动态波动范围", fontsize=11, fontweight="bold")
    ax.tick_params(colors="black", labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#BDBDBD")
    ax.grid(alpha=0.3, color="#BDBDBD")

    # 右：均值热力图（年份 × 断面，采样）
    ax2 = axes[1]
    vol_matrix = np.column_stack([volatility[yr] for yr in years if yr in volatility])
    step = max(1, vol_matrix.shape[0] // 500)
    im = ax2.imshow(vol_matrix[::step].T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=float(np.nanpercentile(vol_matrix, 90)))
    cb = fig.colorbar(im, ax=ax2, label="波动范围（m）", shrink=0.85)
    cb.ax.tick_params(labelsize=FONT_TICK)
    ax2.set_yticks(range(len(years)))
    ax2.set_yticklabels(list(years), fontsize=FONT_TICK)
    ax2.set_xlabel("断面编号（采样）", fontsize=FONT_LABEL)
    ax2.set_title("图4-8b  波动范围热力图", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="black", labelsize=FONT_TICK)
    ax2.set_facecolor("white")

    fig.tight_layout()
    return _save(fig, "图4-8_volatility.png")


# ══════════════════════════════════════════════════════════
# 图4-9  NSM 分段统计柱状图
# ══════════════════════════════════════════════════════════
def plot_fig4_9():
    nsm_df = pd.read_csv(os.path.join(CHANGE_DIR, "NSM.csv"))
    x_col = "transect_id" if "transect_id" in nsm_df.columns else nsm_df.columns[0]
    v_col = "NSM_m" if "NSM_m" in nsm_df.columns else "NSM"
    nsm_df = nsm_df.sort_values(x_col).reset_index(drop=True)
    # 排除边界断面
    nsm_df = nsm_df[np.abs(nsm_df[v_col]) > 0.01]

    seg_size = 200
    n = len(nsm_df)
    segs = range(0, n, seg_size)
    seg_means  = [nsm_df[v_col].iloc[i:i+seg_size].mean() for i in segs]
    seg_starts = [nsm_df[x_col].iloc[i] for i in segs]

    colors = ["#EF5350" if v >= 0 else "#1E88E5" for v in seg_means]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    _style_chart_ax(ax)
    ax.bar(range(len(seg_means)), seg_means, color=colors, width=0.7, edgecolor="none")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(0, len(seg_means), max(1, len(seg_means)//12)))
    ax.set_xticklabels(
        [f"{seg_starts[i]}" for i in range(0, len(seg_means), max(1, len(seg_means)//12))],
        rotation=45, ha="right", fontsize=6.5
    )
    ax.set_xlabel("断面编号（分段起始）", fontsize=FONT_LABEL)
    ax.set_ylabel("NSM 均值（m）", fontsize=FONT_LABEL)
    ax.set_title("图4-9  各断面区段岸线净迁移量 NSM 统计", fontsize=FONT_TITLE, fontweight="bold")
    legend = [mpatches.Patch(color="#EF5350", label="淤积段"),
              mpatches.Patch(color="#1E88E5", label="侵蚀段")]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="#BDBDBD")
    fig.tight_layout()
    return _save(fig, "图4-9_nsm_bar.png")


# ══════════════════════════════════════════════════════════
# 图4-10  热点岸段地图
# ══════════════════════════════════════════════════════════
def plot_fig4_10():
    baseline = _clip_interior(_to32650(_read_gpkg(os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg"))))
    hotspots = _to32650(_read_gpkg(os.path.join(CHANGE_DIR, "hotspots.gpkg")))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _style_map_ax(ax)

    all_gdfs = [baseline] if baseline is not None else []
    if baseline is not None:
        baseline.plot(ax=ax, color="#9E9E9E", lw=0.8, alpha=0.6, zorder=2)

    if hotspots is not None:
        type_col = "type" if "type" in hotspots.columns else None
        if type_col:
            acc = hotspots[hotspots[type_col] == "Accretion"]
            ero = hotspots[hotspots[type_col] == "Erosion"]
            if len(acc) > 0:
                acc.plot(ax=ax, color="#E65100", lw=3.0, zorder=4)
            if len(ero) > 0:
                ero.plot(ax=ax, color="#6A1B9A", lw=3.0, zorder=4)
        else:
            hotspots.plot(ax=ax, color="#E65100", lw=2.0, zorder=4)
        all_gdfs.append(hotspots)

    legend = [
        Line2D([0], [0], color="#9E9E9E", lw=1.5, label="基线"),
        Line2D([0], [0], color="#E65100", lw=3, label="淤积热点"),
        Line2D([0], [0], color="#6A1B9A", lw=3, label="侵蚀热点"),
    ]
    ax.legend(handles=legend, fontsize=FONT_LEGEND, loc="best",
              framealpha=0.9, edgecolor="#BDBDBD")
    _fit_bounds(ax, all_gdfs)
    ax.set_title("图4-10  2019—2024年岸线变化热点区空间分布",
                 fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "图4-10_hotspot_map.png")


# ══════════════════════════════════════════════════════════
# 图4-11  多尺度对比（重新设计）
# ══════════════════════════════════════════════════════════
def plot_fig4_11():
    ms_path = os.path.join(CHANGE_DIR, "multiscale_comparison.csv")
    dm_path = os.path.join(os.path.dirname(CHANGE_DIR), "distances/distance_matrix.csv")
    if not os.path.exists(ms_path):
        print("  ⚠️  multiscale_comparison.csv 不存在")
        return None

    ms = pd.read_csv(ms_path)
    dm = pd.read_csv(dm_path, index_col=0)

    # 计算实际的年内季度波动 = 各年 4 个季度的标准差均值
    years = [yr for yr in YEAR_COLORS]
    intra_std_list = []
    for yr in years:
        yr_cols = [c for c in dm.columns if c.startswith(yr)]
        if len(yr_cols) < 2:
            continue
        intra_std_list.append(dm[yr_cols].std(axis=1))
    intra_std = pd.concat(intra_std_list, axis=1).mean(axis=1)  # 各断面各年均值

    # NSM（绝对值）
    nsm_col = "NSM_m" if "NSM_m" in ms.columns else "NSM"
    nsm_abs = np.abs(ms[nsm_col].values)
    x = ms["transect_id"].values if "transect_id" in ms.columns else np.arange(len(ms))

    # 对齐 intra_std 到 x 的断面
    intra_vals = intra_std.reindex(x).fillna(0).values

    # 信噪比
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(intra_vals > 0, nsm_abs / intra_vals, 0)

    # 有效断面过滤（去除边界零值断面）
    valid = (nsm_abs > 1) | (intra_vals > 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.patch.set_facecolor("white")

    # 上：年际净变化 |NSM|
    ax0 = axes[0]
    ax0.set_facecolor("white")
    for sp in ax0.spines.values(): sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
    pos_v = nsm_abs.copy(); pos_v[~valid] = np.nan
    ax0.fill_between(x, 0, pos_v, alpha=0.7, color="#EF5350", label="|NSM|（年际净变化）")
    ax0.set_ylabel("|NSM|（m）", fontsize=FONT_LABEL)
    ax0.legend(fontsize=FONT_LEGEND, loc="upper right", framealpha=0.9, edgecolor="#BDBDBD")
    ax0.tick_params(colors="black", labelsize=FONT_TICK)
    ax0.grid(alpha=0.3, color="#BDBDBD", linewidth=0.5)
    ax0.set_title("图4-11  季度尺度波动与年际净变化多尺度对比",
                  fontsize=FONT_TITLE, fontweight="bold")

    # 中：年内季度波动
    ax1 = axes[1]
    ax1.set_facecolor("white")
    for sp in ax1.spines.values(): sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
    intra_plot = intra_vals.copy(); intra_plot[~valid] = np.nan
    ax1.fill_between(x, 0, intra_plot, alpha=0.7, color="#1E88E5", label="年内季度波动（std）")
    ax1.set_ylabel("季度波动 std（m）", fontsize=FONT_LABEL)
    ax1.legend(fontsize=FONT_LEGEND, loc="upper right", framealpha=0.9, edgecolor="#BDBDBD")
    ax1.tick_params(colors="black", labelsize=FONT_TICK)
    ax1.grid(alpha=0.3, color="#BDBDBD", linewidth=0.5)

    # 下：信噪比
    ax2 = axes[2]
    ax2.set_facecolor("white")
    for sp in ax2.spines.values(): sp.set_color("#BDBDBD"); sp.set_linewidth(0.7)
    snr_plot = snr.copy(); snr_plot[~valid] = np.nan
    # 截断到合理范围（去掉极端离群值）
    snr_cap = float(np.nanpercentile(snr_plot[valid], 95))
    snr_plot = np.clip(snr_plot, 0, snr_cap)
    sig_mask = (snr_plot >= 1) & valid
    ax2.bar(x[sig_mask],  snr_plot[sig_mask],  width=1, color="#43A047", alpha=0.8,
            label="信噪比 ≥ 1（年际信号显著）")
    ax2.bar(x[~sig_mask & valid], snr_plot[~sig_mask & valid], width=1, color="#BDBDBD", alpha=0.6,
            label="信噪比 < 1（季度噪声主导）")
    ax2.axhline(1.0, color="#E53935", lw=1.2, ls="--")
    pct = float(sig_mask.sum()) / max(valid.sum(), 1) * 100
    ax2.set_xlabel("断面编号", fontsize=FONT_LABEL)
    ax2.set_ylabel("信噪比 |NSM|/std", fontsize=FONT_LABEL)
    ax2.legend(fontsize=FONT_LEGEND, loc="upper right", framealpha=0.9, edgecolor="#BDBDBD")
    ax2.tick_params(colors="black", labelsize=FONT_TICK)
    ax2.grid(alpha=0.3, color="#BDBDBD", linewidth=0.5)
    ax2.text(0.01, 0.93, f"信噪比 ≥ 1 断面占比：{pct:.1f}%",
             transform=ax2.transAxes, va="top", fontsize=8.5, color="#333")

    fig.tight_layout()
    return _save(fig, "图4-11_multiscale.png")


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════
PLOT_FUNCS = [
    ("图3-1",  plot_fig3_1),
    ("图3-2",  plot_fig3_2),
    ("图4-1",  plot_fig4_1),
    ("图4-2",  plot_fig4_2),
    ("图4-3",  plot_fig4_3),
    ("图4-4",  plot_fig4_4),
    ("图4-5",  plot_fig4_5),
    ("图4-6",  plot_fig4_6),
    ("图4-7",  plot_fig4_7),
    ("图4-8",  plot_fig4_8),
    ("图4-9",  plot_fig4_9),
    ("图4-10", plot_fig4_10),
    ("图4-11", plot_fig4_11),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成论文图件（白色底图版）")
    parser.add_argument("--fig", "-f", default=None, help="指定图号，如 3-1 / 4-10")
    args = parser.parse_args()

    print("=" * 60)
    print("  paper_figures.py v2 — 论文图件生成（白色底图）")
    print(f"  输出目录：{OUT_DIR}")
    print("=" * 60)

    for name, func in PLOT_FUNCS:
        if args.fig and not name.endswith(args.fig):
            continue
        print(f"\n  生成 {name}...")
        try:
            func()
        except Exception as e:
            print(f"  ❌ {name} 失败：{e}")
            import traceback; traceback.print_exc()

    print(f"\n✅ 完成！输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
