"""
C2 — 水边线与断面求交，获取距离序列
============================================================
输入：
    output/waterlines/ 下 24 期水边线 GPKG
    output/transects/transects.gpkg
输出：
    output/distances/distance_matrix.csv
    行 = 断面 ID，列 = 各期（2019_Q1 … 2024_Q4），值 = 沿断面距离（米）

逻辑：
    对每期水边线 L(t) 与每条断面 T(k) 求交点
    交点沿断面的距离 = 交点到断面起点（陆侧端）的距离（shapely project）
    若某断面与某期水边线无交点，记为 NaN

验证：NaN 比例应 < 5%，否则检查水边线质量
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union
from shapely.errors import TopologicalError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WATERLINE_DIR, TRANSECT_DIR, DISTANCE_DIR, PERIODS
)


def load_transects() -> gpd.GeoDataFrame:
    """加载断面矢量。"""
    transect_path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    if not os.path.exists(transect_path):
        raise FileNotFoundError(
            f"断面文件不存在：{transect_path}\n请先运行 C1！"
        )
    gdf = gpd.read_file(transect_path)
    print(f"  载入断面：{len(gdf)} 条，CRS={gdf.crs}")
    return gdf


def load_waterline(period: str, target_crs) -> gpd.GeoDataFrame:
    """加载单期水边线并转换 CRS。"""
    gpkg_path = os.path.join(WATERLINE_DIR, f"{period}_waterline.gpkg")
    if not os.path.exists(gpkg_path):
        return None
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def compute_distance_for_transect(transect_geom, waterline_geom):
    """
    计算断面与水边线交点沿断面的距离（从陆侧起点量起）。

    Parameters
    ----------
    transect_geom  : shapely LineString（从陆→海）
    waterline_geom : 当期所有水边线的 unary_union

    Returns
    -------
    float or np.nan
    """
    try:
        inter = transect_geom.intersection(waterline_geom)
    except (TopologicalError, Exception):
        return np.nan

    if inter.is_empty:
        return np.nan

    # 若有多个交点（水边线多次穿越断面），取最靠海侧（project 最大值）
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type == "MultiPoint":
        pts = list(inter.geoms)
    elif inter.geom_type in ("LineString", "MultiLineString",
                              "GeometryCollection"):
        # 取代表点
        pts = []
        geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
        for g in geoms:
            if g.geom_type == "Point":
                pts.append(g)
            elif hasattr(g, "centroid"):
                pts.append(g.centroid)
    else:
        pts = [inter.centroid]

    if not pts:
        return np.nan

    # project 返回沿断面从起点（陆侧）的距离
    distances = [transect_geom.project(p) for p in pts]
    # 取最大值（最靠海侧，即最外侧瞬时水边线）
    return float(max(distances))


def compute_distance_matrix(transects_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    计算所有断面 × 所有期次的距离矩阵。

    Returns
    -------
    DataFrame: index=transect_id, columns=period
    """
    n_transects = len(transects_gdf)
    target_crs  = transects_gdf.crs

    # 初始化为 NaN
    dist_matrix = pd.DataFrame(
        np.full((n_transects, len(PERIODS)), np.nan),
        index=transects_gdf["transect_id"].values,
        columns=PERIODS,
    )
    dist_matrix.index.name = "transect_id"

    for t_idx, period in enumerate(PERIODS):
        wl_gdf = load_waterline(period, target_crs)
        if wl_gdf is None or wl_gdf.empty:
            print(f"  ⚠️  {period}: 水边线文件不存在，跳过")
            continue

        # 合并当期所有水边线为一个几何体
        wl_union = unary_union(wl_gdf.geometry)

        n_intersect = 0
        for _, row in transects_gdf.iterrows():
            t_id   = row["transect_id"]
            t_geom = row.geometry
            d = compute_distance_for_transect(t_geom, wl_union)
            if not np.isnan(d):
                dist_matrix.at[t_id, period] = d
                n_intersect += 1

        nan_pct = (1 - n_intersect / n_transects) * 100
        print(f"  ✅ {period}: 有交点断面={n_intersect}/{n_transects} "
              f"(NaN={nan_pct:.1f}%)")

    return dist_matrix


def validate_distance_matrix(dist_matrix: pd.DataFrame):
    """验证距离矩阵质量：NaN 比例应 < 5%。"""
    total = dist_matrix.size
    n_nan = dist_matrix.isna().sum().sum()
    nan_pct = n_nan / total * 100
    print(f"\n  距离矩阵验证：")
    print(f"  总元素 {total:,}，NaN {n_nan:,}（{nan_pct:.2f}%）")
    if nan_pct < 5:
        print("  ✅ NaN 比例 < 5%，质量合格")
    elif nan_pct < 15:
        print("  ⚠️  NaN 比例 5–15%，请检查部分期次水边线是否完整")
    else:
        print("  ❌ NaN 比例 > 15%，强烈建议检查 B7 水边线提取质量！")

    # 每期 NaN 统计
    per_period_nan = dist_matrix.isna().mean() * 100
    worst = per_period_nan.nlargest(5)
    print(f"  NaN 比例最高的 5 期：")
    for p, v in worst.items():
        print(f"    {p}: {v:.1f}%")


def main():
    print("=" * 60)
    print("  C2 — 水边线与断面求交（距离矩阵计算）")
    print("=" * 60)

    # 载入断面
    print("\n[1/3] 载入断面...")
    transects_gdf = load_transects()

    # 计算距离矩阵
    print(f"\n[2/3] 计算距离矩阵（{len(transects_gdf)} 条断面 × {len(PERIODS)} 期）...")
    dist_matrix = compute_distance_matrix(transects_gdf)

    # 保存
    os.makedirs(DISTANCE_DIR, exist_ok=True)
    out_path = os.path.join(DISTANCE_DIR, "distance_matrix.csv")
    dist_matrix.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n[3/3] 距离矩阵已保存：{out_path}")
    print(f"      形状：{dist_matrix.shape[0]} 行（断面）× {dist_matrix.shape[1]} 列（期次）")

    # 验证
    validate_distance_matrix(dist_matrix)

    print(f"\n✅ C2 完成！")


if __name__ == "__main__":
    main()
