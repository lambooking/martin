"""
quick_preview.py — 快速预览各期次的完整处理流程结果
============================================================
用法：
    python quick_preview.py                     # 预览第一个可用期次（栅格+矢量）
    python quick_preview.py --period 2019_Q1    # 预览指定期次
    python quick_preview.py --all               # 批量生成全部 24 期
    python quick_preview.py --vector            # 仅输出矢量总览图（全时序）

可视化内容（2 行布局）：
  第一行（栅格，与期次相关）：
    (1) B1  MNDWI
    (2) B2  S1 VH dB
    (3) B4  水体掩膜（阈值后，形态前）
    (4) B5  水体掩膜（形态清洁后）
    (5) B6  外海掩膜（连通约束后）

  第二行（矢量叠加）：
    (6) B7  当期水边线（叠加在外海掩膜上）
    (7) C1  断面网格（叠加在外海掩膜上）
    (8) C3  该年 MHW_proxy 岸线（如存在）
    (9) D   热点分布（淤积/侵蚀）

  另外 --vector 模式输出独立的"矢量总览"PNG，不依赖期次，
  展示基线、断面、全部年度岸线（6 年）和热点。

输出：
    output/figures/preview_{period}.png
    output/figures/preview_vectors.png   (--vector 模式)

Linux 查看 TIF 的其他方式：
    QGIS（推荐）:
        qgis output/sea_mask/2019_Q1_sea.tif
        → 图层右键 Properties → Symbology → Min/Max 设为 0/255
    GDAL 命令行:
        gdalinfo output/sea_mask/2019_Q1_sea.tif
        gdal_translate -of PNG output/sea_mask/2019_Q1_sea.tif /tmp/out.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MNDWI_DIR, S1_DB_DIR, WATER_MASK_DIR, WATER_CLEAN_DIR, SEA_MASK_DIR,
    WATERLINE_DIR, TRANSECT_DIR, ANNUAL_SL_DIR, CHANGE_DIR,
    FIGURES_DIR, PERIODS, S1_BAND_ORDER,
)

WATER_VAL  = 255
NODATA_VAL = 128

# 年度岸线颜色（2019→2024 蓝→红渐变）
YEAR_COLORS = {
    "2019": "#2196F3",
    "2020": "#4CAF50",
    "2021": "#FFC107",
    "2022": "#FF9800",
    "2023": "#F44336",
    "2024": "#9C27B0",
}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _read_band(path: str, band: int = 1) -> tuple[np.ndarray | None, dict]:
    """读取单波段，返回 (data, meta)；不存在返回 (None, {})。"""
    if not os.path.exists(path):
        return None, {}
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nd = src.nodata
        meta = {"bounds": src.bounds, "crs": src.crs}
    if nd is not None:
        data[data == nd] = np.nan
    return data, meta


def _read_gpkg(path: str, layer: str | None = None) -> gpd.GeoDataFrame | None:
    """读取 GeoPackage，失败返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    except Exception:
        return None


def _show_raster(ax, data, title, cmap, vmin, vmax, note, *, three_class=False):
    """在 ax 上绘制栅格数据。"""
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(note, fontsize=7, color="gray")
    ax.set_xticks([]); ax.set_yticks([])
    if data is None:
        ax.text(0.5, 0.5, "文件不存在", ha="center", va="center",
                transform=ax.transAxes, color="#f44336", fontsize=9)
        return
    if three_class:
        cm = mcolors.ListedColormap(["black", "gray", "white"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 128.5, 255.5], cm.N)
        ax.imshow(data, cmap=cm, norm=norm, interpolation="nearest")
    else:
        kw = {}
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", **kw)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)


def _add_vector_to_raster_ax(ax, gdf, color, lw=0.8, zorder=3, label=None, linestyle="-"):
    """将矢量 GeoDataFrame 叠加到已显示栅格的 axes 上（处理坐标变换）。"""
    if gdf is None or len(gdf) == 0:
        return
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type in ("LineString", "MultiLineString"):
            xs, ys = [], []
            coords = list(geom.coords) if geom.geom_type == "LineString" \
                else [c for g in geom.geoms for c in g.coords]
            if not coords:
                continue
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, linestyle=linestyle,
                    label=label)
            label = None  # 只设一次 label
        elif geom.geom_type in ("Point", "MultiPoint"):
            pts = [geom] if geom.geom_type == "Point" else list(geom.geoms)
            for p in pts:
                ax.plot(p.x, p.y, "o", color=color, ms=2, zorder=zorder)


def _get_raster_extent(meta: dict):
    """从 meta['bounds'] 取 imshow extent。"""
    if not meta:
        return None
    b = meta["bounds"]
    return [b.left, b.right, b.bottom, b.top]


# ─────────────────────────────────────────────
# 主可视化函数
# ─────────────────────────────────────────────

def preview_period(period: str) -> str | None:
    """生成单期 2×5 对比图（栅格行 + 矢量叠加行）。"""
    # ── 路径 ──
    mndwi_path  = os.path.join(MNDWI_DIR,      f"{period}_mndwi.tif")
    s1db_path   = os.path.join(S1_DB_DIR,       f"{period}_s1_db.tif")
    wmask_path  = os.path.join(WATER_MASK_DIR,  f"{period}_water.tif")
    wclean_path = os.path.join(WATER_CLEAN_DIR, f"{period}_water_clean.tif")
    sea_path    = os.path.join(SEA_MASK_DIR,    f"{period}_sea.tif")
    wline_path  = os.path.join(WATERLINE_DIR,   f"{period}_waterline.gpkg")
    transect_path = os.path.join(TRANSECT_DIR,  "transects.gpkg")
    baseline_path = os.path.join(TRANSECT_DIR,  "auto_smoothed_baseline.gpkg")
    year = period.split("_")[0]
    mhw_path   = os.path.join(ANNUAL_SL_DIR, f"MHW_proxy_{year}.gpkg")
    hot_path   = os.path.join(CHANGE_DIR,     "hotspots.gpkg")

    # ── 读取栅格 ──
    mndwi,  meta = _read_band(mndwi_path,  1)
    vh_db,  _    = _read_band(s1db_path,   S1_BAND_ORDER["vh"] + 1)
    wmask,  _    = _read_band(wmask_path,  1)
    wclean, _    = _read_band(wclean_path, 1)
    sea,    sea_meta = _read_band(sea_path, 1)

    if all(v is None for v in [mndwi, vh_db, wmask, wclean, sea]):
        print(f"  跳过 {period}：所有栅格文件均不存在")
        return None

    # ── 读取矢量 ──
    wline     = _read_gpkg(wline_path, layer="waterline")
    transects = _read_gpkg(transect_path)
    baseline  = _read_gpkg(baseline_path)
    mhw_sl    = _read_gpkg(mhw_path)
    hotspots  = _read_gpkg(hot_path)

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    fig.suptitle(f"黄河三角洲 — {period} 完整处理流程可视化", fontsize=13, fontweight="bold")

    # —— 第一行：栅格 ——
    panels_r = [
        (mndwi,  "B1  MNDWI",           "RdYlGn", -1, 1,   "绿=水(>0) 红=陆(<0)"),
        (vh_db,  "B2  S1 VH dB",        "gray_r",  None, None, "深=海水 浅=陆地"),
        (wmask,  "B4  水体掩膜（阈值后）", None, 0, 255, "白=水(255) 灰=nodata(128) 黑=陆(0)"),
        (wclean, "B5  水体掩膜（形态后）", None, 0, 255, "开运算/闭运算去碎斑后"),
        (sea,    "B6  外海掩膜（连通后）", None, 0, 255, "仅保留与外海相连的水域"),
    ]
    for ax, (data, title, cmap, vmin, vmax, note) in zip(axes[0], panels_r):
        three = (cmap is None)
        _show_raster(ax, data, title, cmap, vmin, vmax, note, three_class=three)

    # —— 第二行：矢量叠加 ——
    raster_crs = sea_meta.get("crs") if sea_meta else None

    def _prep_vec_ax(ax, title, note, facecolor="#1a1a2e"):
        """初始化纯矢量面板：隐藏 spine、设背景色、设标题。"""
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(note, fontsize=7, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(facecolor)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _fit_vec_bounds(ax, gdfs, pad_ratio=0.05):
        """根据矢量数据自适应比例尺，aspect='equal'，5% 边距。"""
        combined = [g for g in gdfs if g is not None and len(g) > 0]
        if not combined:
            return
        all_bounds = np.array([g.total_bounds for g in combined])
        minx, miny = all_bounds[:, 0].min(), all_bounds[:, 1].min()
        maxx, maxy = all_bounds[:, 2].max(), all_bounds[:, 3].max()
        pad = max(maxx - minx, maxy - miny) * pad_ratio
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal", adjustable="datalim")

    def _reproj(gdf, target_crs):
        if gdf is None or len(gdf) == 0:
            return gdf
        try:
            return gdf.to_crs(target_crs) if (target_crs and gdf.crs and gdf.crs != target_crs) else gdf
        except Exception:
            return gdf

    # Panel 6: B7 水边线（叠加在栅格底图，保持栅格坐标系）
    ax6 = axes[1, 0]
    ax6.set_title("B7  水边线（矢量）", fontsize=9, fontweight="bold")
    ax6.set_xlabel("已提取" if wline is not None else "文件不存在", fontsize=7, color="gray")
    ax6.set_xticks([]); ax6.set_yticks([])
    if sea is not None:
        cm_bg = mcolors.ListedColormap(["#1a1a2e", "#6c757d", "#e9ecef"])
        norm_bg = mcolors.BoundaryNorm([-0.5, 0.5, 128.5, 255.5], cm_bg.N)
        ax6.imshow(sea, cmap=cm_bg, norm=norm_bg, interpolation="nearest")
    if wline is not None and sea_meta:
        b = sea_meta["bounds"]
        try:
            _reproj(wline, raster_crs).plot(ax=ax6, color="#00e5ff", lw=0.8, zorder=3)
            ax6.set_xlim(b.left, b.right)
            ax6.set_ylim(b.bottom, b.top)
        except Exception:
            pass

    # Panel 7: C1 断面 + 基线（叠加在栅格底图）
    ax7 = axes[1, 1]
    ax7.set_title("C1  断面 & 基线", fontsize=9, fontweight="bold")
    ax7.set_xlabel("蓝=断面 黄=基线", fontsize=7, color="gray")
    ax7.set_xticks([]); ax7.set_yticks([])
    if sea is not None:
        ax7.imshow(sea, cmap=cm_bg, norm=norm_bg, interpolation="nearest")
    if sea_meta:
        b = sea_meta["bounds"]
        if transects is not None and len(transects) > 0:
            try:
                _reproj(transects, raster_crs).plot(ax=ax7, color="#42A5F5", lw=0.4, alpha=0.7, zorder=3)
                ax7.set_xlim(b.left, b.right); ax7.set_ylim(b.bottom, b.top)
            except Exception:
                pass
        if baseline is not None and len(baseline) > 0:
            try:
                _reproj(baseline, raster_crs).plot(ax=ax7, color="#FFD700", lw=1.5, zorder=4)
                ax7.set_xlim(b.left, b.right); ax7.set_ylim(b.bottom, b.top)
            except Exception:
                pass

    # Panel 8: C3 年度岸线（6 年全叠加）——纯矢量，自适应比例尺
    ax8 = axes[1, 2]
    _prep_vec_ax(ax8, "C3  MHW_proxy 年度岸线（6年）", "每年最低水位代理岸线")
    legend_patches = []
    all_mhw = []
    for yr, clr in YEAR_COLORS.items():
        gdf = _read_gpkg(os.path.join(ANNUAL_SL_DIR, f"MHW_proxy_{yr}.gpkg"))
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax8, color=clr, lw=0.9, zorder=3)
            legend_patches.append(mpatches.Patch(color=clr, label=yr))
            all_mhw.append(gdf)
    _fit_vec_bounds(ax8, all_mhw)
    if legend_patches:
        ax8.legend(handles=legend_patches, fontsize=6, loc="best", framealpha=0.7)

    # Panel 9: D 热点分布——纯矢量，自适应比例尺（以基线为参考范围）
    ax9 = axes[1, 3]
    _prep_vec_ax(ax9, "D   热点岸段（淤积/侵蚀）", "橙=淤积热点  紫=侵蚀热点")
    if baseline is not None and len(baseline) > 0:
        baseline.plot(ax=ax9, color="#546e7a", lw=0.6, alpha=0.5, zorder=2)
    if hotspots is not None and len(hotspots) > 0:
        type_col = "type" if "type" in hotspots.columns else None
        if type_col:
            acc = hotspots[hotspots[type_col] == "Accretion"]
            ero = hotspots[hotspots[type_col] == "Erosion"]
            if len(acc) > 0:
                acc.plot(ax=ax9, color="#FF6F00", lw=2.0, zorder=4, label=f"淤积 ({len(acc)}段)")
            if len(ero) > 0:
                ero.plot(ax=ax9, color="#CE93D8", lw=2.0, zorder=4, label=f"侵蚀 ({len(ero)}段)")
            ax9.legend(fontsize=7, loc="best", framealpha=0.7)
        else:
            hotspots.plot(ax=ax9, color="#FF6F00", lw=1.5, zorder=4)
    # 用基线范围定视野（热点仅为基线上的片段）
    _fit_vec_bounds(ax9, [baseline])

    # Panel 10: C3/C4 该年岸线包络——纯矢量，自适应比例尺
    ax10 = axes[1, 4]
    _prep_vec_ax(ax10, f"C3/C4  {year} 年岸线包络", "蓝=MHW_proxy  绿=Outer_P95")
    mhw_gdf   = _read_gpkg(os.path.join(ANNUAL_SL_DIR, f"MHW_proxy_{year}.gpkg"))
    outer_gdf = _read_gpkg(os.path.join(ANNUAL_SL_DIR, f"Outer_P95_{year}.gpkg"))
    if mhw_gdf is not None and len(mhw_gdf) > 0:
        mhw_gdf.plot(ax=ax10, color="#2196F3", lw=1.2, zorder=3, label="MHW_proxy")
    if outer_gdf is not None and len(outer_gdf) > 0:
        outer_gdf.plot(ax=ax10, color="#4CAF50", lw=1.0, zorder=3, linestyle="--", label="Outer_P95")
    _fit_vec_bounds(ax10, [mhw_gdf, outer_gdf])
    ax10.legend(fontsize=7, loc="best", framealpha=0.7)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"preview_{period}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {period} → {out_path}")
    return out_path


def preview_vectors() -> str:
    """
    与期次无关的矢量总览图：
      基线 + 全部断面 + 6年 MHW_proxy + 6年 Outer_P95 + 热点
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle("黄河三角洲 — 矢量结果总览（2019–2024）", fontsize=13, fontweight="bold")

    transect_path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    baseline_path = os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg")
    hot_path      = os.path.join(CHANGE_DIR,   "hotspots.gpkg")

    transects = _read_gpkg(transect_path)
    baseline  = _read_gpkg(baseline_path)
    hotspots  = _read_gpkg(hot_path)

    def _vax(ax, title, note, n_transects=None):
        """初始化矢量总览子图。"""
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_facecolor("#1a1a2e")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        label = note if n_transects is None else f"{note}（{n_transects} 条）"
        ax.set_xlabel(label, fontsize=8, color="gray")

    def _fit_bounds(ax, gdfs, pad_ratio=0.05):
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

    # ── 子图1：断面网格 ──
    ax = axes[0]
    n_t = len(transects) if transects is not None else 0
    _vax(ax, "C1  断面网格 & 基线", "断面", n_t)
    if transects is not None and len(transects) > 0:
        transects.plot(ax=ax, color="#42A5F5", lw=0.3, alpha=0.5, zorder=2)
    if baseline is not None and len(baseline) > 0:
        baseline.plot(ax=ax, color="#FFD700", lw=2.0, zorder=3)
    _fit_bounds(ax, [transects, baseline])

    # ── 子图2：年度岸线演变 ──
    ax = axes[1]
    _vax(ax, "C3  MHW_proxy 年度岸线演变（6年）", "各年 MHW_proxy")
    legend_patches = []
    all_mhw = []
    for yr, clr in YEAR_COLORS.items():
        p = os.path.join(ANNUAL_SL_DIR, f"MHW_proxy_{yr}.gpkg")
        gdf = _read_gpkg(p)
        if gdf is not None and len(gdf) > 0:
            gdf.plot(ax=ax, color=clr, lw=0.9, zorder=3)
            legend_patches.append(mpatches.Patch(color=clr, label=yr))
            all_mhw.append(gdf)
    _fit_bounds(ax, all_mhw)
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=8, loc="best", framealpha=0.7)

    # ── 子图3：热点 ──
    ax = axes[2]
    _vax(ax, "D  热点岸段分布（淤积/侵蚀）", "橙=淤积  紫=侵蚀")
    if baseline is not None and len(baseline) > 0:
        baseline.plot(ax=ax, color="#546e7a", lw=0.8, alpha=0.6, zorder=2, label="基线")
    if hotspots is not None and len(hotspots) > 0:
        type_col = "type" if "type" in hotspots.columns else None
        if type_col:
            acc = hotspots[hotspots[type_col] == "Accretion"]
            ero = hotspots[hotspots[type_col] == "Erosion"]
            if len(acc) > 0:
                acc.plot(ax=ax, color="#FF6F00", lw=2.5, zorder=4, label=f"淤积 ({len(acc)}段)")
            if len(ero) > 0:
                ero.plot(ax=ax, color="#CE93D8", lw=2.5, zorder=4, label=f"侵蚀 ({len(ero)}段)")
        ax.legend(fontsize=8, loc="best", framealpha=0.7)
    _fit_bounds(ax, [baseline])

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "preview_vectors.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 矢量总览 → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="快速预览黄河三角洲各期次栅格+矢量处理结果"
    )
    parser.add_argument("--period", "-p", default=None,
                        help="指定期次，如 2019_Q1（默认：第一个有数据的期次）")
    parser.add_argument("--all",    "-a", action="store_true",
                        help="批量生成全部 24 期预览图")
    parser.add_argument("--vector", "-v", action="store_true",
                        help="仅输出矢量总览图（基线/断面/年度岸线/热点）")
    args = parser.parse_args()

    print("=" * 60)
    print("  quick_preview — 黄河三角洲处理流程可视化")
    print("=" * 60)

    if args.vector:
        preview_vectors()
        return

    if args.all:
        targets = PERIODS
        print(f"  批量模式：共 {len(targets)} 期")
    elif args.period:
        if args.period not in PERIODS:
            print(f"  ❌ 期次 {args.period!r} 不在配置列表中")
            print(f"     合法值示例：{PERIODS[:4]} ...")
            sys.exit(1)
        targets = [args.period]
    else:
        targets = None
        for p in PERIODS:
            if os.path.exists(os.path.join(MNDWI_DIR, f"{p}_mndwi.tif")):
                targets = [p]
                break
        if targets is None:
            print("  ❌ 未找到任何 MNDWI 文件，请先运行 B1")
            sys.exit(1)
        print(f"  自动选择期次：{targets[0]}")

    success = 0
    for period in targets:
        result = preview_period(period)
        if result:
            success += 1

    print(f"\n✅ 完成：共生成 {success} 张预览图")
    print(f"   输出目录：{FIGURES_DIR}")
    print("\n─── Linux 查看 TIF 参考 ─────────────────────────────")
    print("  QGIS（推荐 GUI）：")
    print("    qgis output/sea_mask/2019_Q1_sea.tif")
    print("    → 图层右键 Properties → Symbology → Min/Max 设为 0/255")
    print("  GDAL 命令行：")
    print("    gdalinfo output/sea_mask/2019_Q1_sea.tif          # 查看元数据")
    print("    gdal_translate -of PNG \\")
    print("        output/sea_mask/2019_Q1_sea.tif /tmp/out.png  # 转 PNG")


if __name__ == "__main__":
    main()
