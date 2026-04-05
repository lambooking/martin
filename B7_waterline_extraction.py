"""
B7 — 栅格转矢量（瞬时水边线提取）
============================================================
输入：output/sea_mask/YYYY_QN_sea.tif
输出：output/waterlines/YYYY_QN_waterline.gpkg（线矢量 GeoPackage）

处理步骤：
    1. 多边形化（rasterio.features.shapes）
    2. 仅保留值=1（海水）的多边形
    3. 合并所有多边形后取外边界（差集研究区矩形外边框）
    4. 多边形转线（boundary）
    5. 碎线过滤：移除长度 < MIN_WATERLINE_LENGTH 的短线段
    6. 简化：shapely simplify(tolerance=WATERLINE_SIMPLIFY)
    7. 输出为 GeoPackage，CRS 与输入一致
"""

import os
import sys
import warnings
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon, MultiLineString, LineString
from shapely.ops import unary_union, linemerge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEA_MASK_DIR, WATERLINE_DIR, PERIODS,
    MIN_WATERLINE_LENGTH, WATERLINE_SIMPLIFY
)

WATER_VAL  = 1
NODATA_VAL = 255


def extract_waterline_for_period(period: str) -> dict:
    """
    从海水掩膜中提取瞬时水边线矢量。

    Returns
    -------
    dict: period, n_lines, total_length_km
    """
    in_path  = os.path.join(SEA_MASK_DIR,  f"{period}_sea.tif")
    out_path = os.path.join(WATERLINE_DIR, f"{period}_waterline.gpkg")

    if not os.path.exists(in_path):
        print(f"  ⚠️  跳过 {period}：{in_path} 不存在")
        return None

    with rasterio.open(in_path) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    # --- 多边形化：提取值=1 的区域 ---
    water_uint8 = (data == WATER_VAL).astype(np.uint8)
    polys = []
    for geom_dict, val in shapes(water_uint8, mask=water_uint8, transform=transform):
        if val == 1:
            polys.append(shape(geom_dict))

    if not polys:
        print(f"  ⚠️  {period}: 无海水多边形，跳过")
        return None

    # --- 合并所有海水多边形 ---
    sea_union = unary_union(polys)

    # --- 提取外边界（多边形 → 线）---
    # 注意：仅取 exterior，不取内部孔洞边界（内部孔洞为岛屿边界）
    lines = []
    if sea_union.geom_type == "Polygon":
        lines.append(sea_union.exterior)
    elif sea_union.geom_type == "MultiPolygon":
        for poly in sea_union.geoms:
            lines.append(poly.exterior)
    else:
        # GeometryCollection 等情形
        for geom in sea_union.geoms if hasattr(sea_union, "geoms") else [sea_union]:
            if geom.geom_type == "Polygon":
                lines.append(geom.exterior)
            elif geom.geom_type == "MultiPolygon":
                for p in geom.geoms:
                    lines.append(p.exterior)

    if not lines:
        print(f"  ⚠️  {period}: 边界提取失败，跳过")
        return None

    # --- 碎线过滤：移除长度 < MIN_WATERLINE_LENGTH 的短线段 ---
    filtered_lines = []
    n_removed = 0
    for line in lines:
        # 自动处理 LinearRing → LineString
        ls = LineString(line.coords) if not isinstance(line, LineString) else line
        segs = [ls]
        # 若为 MultiLineString，拆分
        if isinstance(ls, MultiLineString):
            segs = list(ls.geoms)

        for seg in segs:
            if seg.length >= MIN_WATERLINE_LENGTH:
                filtered_lines.append(seg)
            else:
                n_removed += 1

    if not filtered_lines:
        print(f"  ⚠️  {period}: 过滤后无满足长度要求的线，跳过（碎线过多？）")
        return None

    # --- 简化处理 ---
    simplified = [
        line.simplify(WATERLINE_SIMPLIFY, preserve_topology=True)
        for line in filtered_lines
    ]

    total_length_m  = sum(l.length for l in simplified)
    total_length_km = total_length_m / 1000.0

    # --- 保存为 GeoPackage ---
    gdf = gpd.GeoDataFrame(
        {
            "period"    : [period] * len(simplified),
            "length_m"  : [round(l.length, 1) for l in simplified],
        },
        geometry=simplified,
        crs=crs,
    )
    os.makedirs(WATERLINE_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer=period)

    print(f"  ✅ {period}: "
          f"{len(simplified)} 条线, "
          f"总长={total_length_km:.1f} km, "
          f"短线过滤={n_removed} 条")

    return {
        "period"           : period,
        "n_lines"          : len(simplified),
        "total_length_km"  : round(total_length_km, 2),
        "n_removed_short"  : n_removed,
    }


def main():
    print("=" * 55)
    print("  B7 — 栅格转矢量水边线提取（共 24 期）")
    print(f"       最小保留长度={MIN_WATERLINE_LENGTH} m, "
          f"简化容差={WATERLINE_SIMPLIFY} m")
    print("=" * 55)
    os.makedirs(WATERLINE_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        result = extract_waterline_for_period(period)
        if result:
            records.append(result)

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print(f"\n  水边线统计摘要：")
        print(f"  平均线条数：{df['n_lines'].mean():.1f} 条/期")
        print(f"  平均总长度：{df['total_length_km'].mean():.1f} km/期")
        # 检查：每期线条数是否大多为 1（期望主线1条）
        single_line = (df["n_lines"] == 1).sum()
        print(f"  恰好 1 条主线的期数：{single_line}/{len(df)}")

    print(f"\n✅ B7 完成：成功处理 {len(records)} 期")
    print(f"   输出目录：{WATERLINE_DIR}")
    print("   验证：每期应有 1 条主水边线，无大量碎线")


if __name__ == "__main__":
    main()
