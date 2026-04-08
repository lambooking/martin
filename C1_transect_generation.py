"""
C1 — 基线构建与断面生成
============================================================
输入：output/waterlines/2019_Q1_waterline.gpkg（用第一期的水边线自动平滑生成）
输出：output/transects/transects.gpkg（垂直断面矢量，间距50m）

说明：
    - 基线为固定参考线，所有 24 期水边线共用同一套断面
    - 断面的生成需要一条相对平滑稳定的底线，所以我们把 2019第一期的海线使用高强度平滑处理充当底线。
    - 断面方向：垂直于基线局部切线，向海3km，向陆1km
"""

import os
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WATERLINE_DIR, TRANSECT_DIR,
    TRANSECT_SPACING, TRANSECT_SEA_LENGTH, TRANSECT_LAND_LENGTH
)


def generate_transects(
    baseline_geom,
    baseline_crs,
    spacing: float = 50,
    length_sea: float = 3000,
    length_land: float = 1000,
) -> gpd.GeoDataFrame:
    """
    沿基线每隔 spacing 米生成一条垂直断面。

    断面定义：
        - 起点（陆侧）：基线足点向陆方向 length_land 米
        - 终点（海侧）：基线足点向海方向 length_sea 米
        - 法向量由基线局部切线旋转 90° 得到
        - 正方向（即 shapely.project 值增大方向）= 从陆到海

    Parameters
    ----------
    baseline_geom : shapely LineString（已投影坐标系，单位：米）
    baseline_crs  : pyproj CRS 对象
    spacing       : 断面间距（米）
    length_sea    : 向海延伸长度（米）
    length_land   : 向陆延伸长度（米）

    Returns
    -------
    GeoDataFrame: transect_id, foot_lon, foot_lat, geometry
    """
    total_len = baseline_geom.length
    distances = np.arange(0, total_len, spacing)

    transects       = []
    transect_ids    = []
    foot_xs, foot_ys = [], []

    for i, d in enumerate(distances):
        # 足点坐标
        pt = baseline_geom.interpolate(d)

        # 局部切线（前后各取 0.5m）
        d_ahead  = min(d + 0.5, total_len)
        d_behind = max(d - 0.5, 0)
        pt_ahead  = baseline_geom.interpolate(d_ahead)
        pt_behind = baseline_geom.interpolate(d_behind)

        dx = pt_ahead.x - pt_behind.x
        dy = pt_ahead.y - pt_behind.y
        tangent_len = np.hypot(dx, dy)
        if tangent_len < 1e-9:
            continue  # 退化点，跳过

        # 法向量（垂直于切线，逆时针旋转 90°）
        # 结果方向需要根据研究区判断哪侧是海（目视验证）
        nx = -dy / tangent_len
        ny =  dx / tangent_len

        # 断面端点
        pt_land = Point(pt.x - nx * length_land, pt.y - ny * length_land)
        pt_sea  = Point(pt.x + nx * length_sea,  pt.y + ny * length_sea)

        # 断面：从陆到海（正方向向海，project 值增大）
        transect = LineString([pt_land, pt_sea])

        transects.append(transect)
        transect_ids.append(i)
        foot_xs.append(pt.x)
        foot_ys.append(pt.y)

    gdf = gpd.GeoDataFrame(
        {
            "transect_id": transect_ids,
            "foot_x"     : foot_xs,
            "foot_y"     : foot_ys,
            "length_m"   : [length_sea + length_land] * len(transects),
        },
        geometry=transects,
        crs=baseline_crs,
    )
    return gdf


def main():
    print("=" * 55)
    print("  C1 — 基于2019原初线的自动化基线断面生成")
    print(f"       间距={TRANSECT_SPACING} m, "
          f"向海={TRANSECT_SEA_LENGTH} m, "
          f"向陆={TRANSECT_LAND_LENGTH} m")
    print("=" * 55)

    base_waterline_path = os.path.join(WATERLINE_DIR, "2019_Q1_waterline.gpkg")

    if not os.path.exists(base_waterline_path):
        print(f"❌ 找不到初始参照水边线：{base_waterline_path}")
        print("   请确保 B 阶段已经成功运行并输出了 2019_Q1 期的海岸线！")
        return

    # 读取起始图层生成作为基线使用
    print(f"\n  正在加载 {base_waterline_path} ...")
    baseline_gdf = gpd.read_file(base_waterline_path, layer="waterline")
    
    # 确保是投影坐标系（单位米），否则无法按米计算间距
    if baseline_gdf.crs is not None and baseline_gdf.crs.is_geographic:
        print("  ⚠️  检测到地理坐标系，由于求法向量需要米作为单位，将转为 EPSG:32650...")
        baseline_gdf = baseline_gdf.to_crs("EPSG:32650")
        
    # 合并并执行强平滑，从而洗掉曲折的海滩锯齿，做成完美的参考平滑线
    from shapely.ops import linemerge, unary_union
    all_lines = unary_union(baseline_gdf.geometry)
    if all_lines.geom_type == "MultiLineString":
        all_lines = linemerge(all_lines)
        
    print("  正在执行海岸线强平滑操作以构建断面基准...")
    # simplify容差给100米来去除一切小锯齿
    smooth_baseline = all_lines.simplify(100, preserve_topology=True)
    print(f"  基线总长度：{smooth_baseline.length/1000:.2f} km")

    # 生成断面
    print(f"\n  生成断面中...")
    transects_gdf = generate_transects(
        baseline_geom=smooth_baseline,
        baseline_crs=baseline_gdf.crs,
        spacing=TRANSECT_SPACING,
        length_sea=TRANSECT_SEA_LENGTH,
        length_land=TRANSECT_LAND_LENGTH,
    )
    n_transects = len(transects_gdf)
    print(f"  共生成 {n_transects} 条断面")

    # 保存
    os.makedirs(TRANSECT_DIR, exist_ok=True)
    out_path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    transects_gdf.to_file(out_path, driver="GPKG")
    
    # 也把生成的平滑基线存一份用于给用户对比参阅
    baseline_out_path = os.path.join(TRANSECT_DIR, "auto_smoothed_baseline.gpkg")
    gpd.GeoDataFrame(geometry=[smooth_baseline], crs=baseline_gdf.crs).to_file(baseline_out_path, driver="GPKG")
    
    print(f"\n  ✅ 断面体系已保存：{out_path}")
    print(f"  ✅ 自动生成的包络基线已保存：{baseline_out_path}")

    # 验证提示
    print("\n  ⚠️  验证步骤（建议）：")
    print("  1. 在 QGIS 中打开 transects.gpkg / auto_smoothed_baseline.gpkg 检查")
    print("  2. 目视检查：断面是否从陆地方感指向海洋")
    print("  3. 若所有断面切面方向正好全反过来了，在代码生成法向量处修改一下即可")
    print(f"\n✅ C1 完美完成！生成 {n_transects} 条断面")


if __name__ == "__main__":
    main()
