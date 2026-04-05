"""
C3 — MHW_proxy 年度高潮线代理构建
C4 — Outer/P95 年度包络线构建
============================================================
输入：output/distances/distance_matrix.csv
输出：
    output/annual_shorelines/MHW_proxy_YYYY.gpkg   — 6条年度岸线（2019–2024）
    output/annual_shorelines/Outer_P95_YYYY.gpkg   — 6条年度包络线

逻辑：
    MHW_proxy[k, y] = mean(sort(quarterly_distance[:2]))  # 最小两值均值（靠陆侧）
    Outer_P95[k, y] = percentile(quarterly_distance, 95)  # 向海 P95

NaN 处理：
    - 某断面某期无交点 → 该期跳过
    - 某断面有效期数 < 2 → 该断面整体标记为无效（NaN）

将每年所有断面的位置点连线，生成年度岸线矢量（GeoPackage）
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DISTANCE_DIR, TRANSECT_DIR, ANNUAL_SL_DIR, YEARS, PERIODS
)

MIN_VALID_QUARTERS = 2   # 每个断面每年至少需要此数量有效期次


def load_distance_matrix() -> pd.DataFrame:
    """加载距离矩阵 CSV。"""
    csv_path = os.path.join(DISTANCE_DIR, "distance_matrix.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"距离矩阵不存在：{csv_path}\n请先运行 C2！"
        )
    df = pd.read_csv(csv_path, index_col="transect_id")
    print(f"  距离矩阵形状：{df.shape}  (断面 × 期次)")
    return df


def load_transects() -> gpd.GeoDataFrame:
    """加载断面矢量（用于将距离转换回空间坐标）。"""
    path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    if not os.path.exists(path):
        raise FileNotFoundError(f"断面文件不存在：{path}\n请先运行 C1！")
    gdf = gpd.read_file(path).set_index("transect_id")
    return gdf


def distance_to_point(transect_geom, distance_m: float) -> Point:
    """将沿断面的距离（米）转换为空间坐标点。"""
    return transect_geom.interpolate(distance_m)


def compute_annual_positions(dist_matrix: pd.DataFrame) -> dict:
    """
    计算每个断面每年的 MHW_proxy 和 Outer_P95 位置距离。

    Returns
    -------
    dict: {
        'MHW_proxy': DataFrame (transect_id × year),
        'Outer_P95': DataFrame (transect_id × year),
    }
    """
    mhw_dict   = {}
    outer_dict = {}

    for year in YEARS:
        # 选出该年的 4 个季度列
        year_cols = [f"{year}_Q{q}" for q in range(1, 5)]
        available = [c for c in year_cols if c in dist_matrix.columns]
        if not available:
            continue

        year_data = dist_matrix[available]  # shape: (n_transects, ≤4)

        mhw_vals   = []
        outer_vals = []
        t_ids      = dist_matrix.index

        for t_id in t_ids:
            row = year_data.loc[t_id].dropna().values
            if len(row) < MIN_VALID_QUARTERS:
                mhw_vals.append(np.nan)
                outer_vals.append(np.nan)
            else:
                sorted_vals = np.sort(row)
                mhw  = float(np.mean(sorted_vals[:2]))     # 最小两值均值
                out  = float(np.percentile(row, 95))        # P95
                mhw_vals.append(mhw)
                outer_vals.append(out)

        mhw_dict[year]   = pd.Series(mhw_vals, index=t_ids, name=year)
        outer_dict[year] = pd.Series(outer_vals, index=t_ids, name=year)

    mhw_df   = pd.DataFrame(mhw_dict)
    outer_df = pd.DataFrame(outer_dict)
    return {"MHW_proxy": mhw_df, "Outer_P95": outer_df}


def positions_to_shoreline(
    transects_gdf: gpd.GeoDataFrame,
    positions_series: pd.Series,
    year: int,
    label: str,
) -> gpd.GeoDataFrame:
    """
    将一年的断面位置距离序列转换为年度岸线矢量。

    逻辑：每个有效断面产生一个空间点，点序列按断面 ID 顺序连为折线。
    """
    pts = []
    valid_ids = []
    for t_id, dist in positions_series.items():
        if np.isnan(dist):
            continue
        if t_id not in transects_gdf.index:
            continue
        t_geom = transects_gdf.loc[t_id, "geometry"]
        pt = distance_to_point(t_geom, dist)
        pts.append(pt)
        valid_ids.append(t_id)

    if len(pts) < 2:
        warnings.warn(f"{label} {year}: 有效断面点 < 2，无法生成岸线")
        return None

    # 按断面 ID 顺序连线（断面 ID 已经是沿岸有序的）
    valid_ids_sorted = sorted(zip(valid_ids, pts), key=lambda x: x[0])
    pts_sorted = [p for _, p in valid_ids_sorted]

    coastline = LineString([(p.x, p.y) for p in pts_sorted])

    gdf = gpd.GeoDataFrame(
        {
            "year"        : [year],
            "label"       : [label],
            "n_transects" : [len(pts)],
        },
        geometry=[coastline],
        crs=transects_gdf.crs,
    )
    return gdf


def build_annual_shorelines(
    annual_positions: dict,
    transects_gdf: gpd.GeoDataFrame,
    label: str,   # "MHW_proxy" or "Outer_P95"
):
    """生成并保存 6 条年度岸线 GeoPackage。"""
    os.makedirs(ANNUAL_SL_DIR, exist_ok=True)
    df_positions = annual_positions[label]

    for year in YEARS:
        if year not in df_positions.columns:
            print(f"  ⚠️  {label} {year}: 无数据，跳过")
            continue

        gdf = positions_to_shoreline(
            transects_gdf, df_positions[year], year, label
        )
        if gdf is None:
            continue

        out_path = os.path.join(ANNUAL_SL_DIR, f"{label}_{year}.gpkg")
        gdf.to_file(out_path, driver="GPKG")

        n_valid   = df_positions[year].notna().sum()
        n_total   = len(df_positions)
        nan_pct   = (1 - n_valid / n_total) * 100
        line_len  = gdf.geometry.length.sum() / 1000

        print(f"  ✅ {label} {year}: "
              f"有效断面={n_valid}/{n_total} (NaN={nan_pct:.1f}%), "
              f"岸线长={line_len:.1f} km → {out_path}")


def main():
    print("=" * 55)
    print("  C3/C4 — 年度代表岸线构建（MHW_proxy + Outer_P95）")
    print("=" * 55)

    # 加载数据
    print("\n[1/3] 加载距离矩阵...")
    dist_matrix = load_distance_matrix()

    print("\n[2/3] 加载断面矢量...")
    transects_gdf = load_transects()

    # 计算年度位置
    print("\n[3/3] 计算年度位置并生成岸线...")
    annual_positions = compute_annual_positions(dist_matrix)

    # 保存断面位置均值/P95 表（供调试）
    annual_positions["MHW_proxy"].to_csv(
        os.path.join(ANNUAL_SL_DIR, "MHW_proxy_distances.csv"), encoding="utf-8-sig"
    )
    annual_positions["Outer_P95"].to_csv(
        os.path.join(ANNUAL_SL_DIR, "Outer_P95_distances.csv"), encoding="utf-8-sig"
    )

    print("\n  --- MHW_proxy 年度岸线 ---")
    build_annual_shorelines(annual_positions, transects_gdf, "MHW_proxy")

    print("\n  --- Outer_P95 年度包络线 ---")
    build_annual_shorelines(annual_positions, transects_gdf, "Outer_P95")

    print(f"\n✅ C3/C4 完成！")
    print(f"   输出目录：{ANNUAL_SL_DIR}")
    print("   验证：6 条 MHW_proxy 年度线应空间连续，无大量断裂")


if __name__ == "__main__":
    main()
