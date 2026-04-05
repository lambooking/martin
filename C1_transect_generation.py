"""
C1 — 基线构建与断面生成
============================================================
输入：data/boundary/baseline.shp（手工数字化的沿岸参考基线）
输出：output/transects/transects.gpkg（垂直断面矢量，间距50m，总长4km）

说明：
    - 基线为固定参考线，所有 24 期水边线共用同一套断面
    - 断面方向：垂直于基线局部切线，向海3km，向陆1km
    - ⚠️ 生成后务必目视验证法向量方向（正方向=向海）
"""

import os
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASELINE_SHP, TRANSECT_DIR,
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
    print("  C1 — 基线断面生成")
    print(f"       间距={TRANSECT_SPACING} m, "
          f"向海={TRANSECT_SEA_LENGTH} m, "
          f"向陆={TRANSECT_LAND_LENGTH} m")
    print("=" * 55)

    if not os.path.exists(BASELINE_SHP):
        print(f"❌ 基线文件不存在：{BASELINE_SHP}")
        print("   请先在 QGIS 中手工数字化研究区沿岸基线，保存为上述路径。")
        print("   基线要求：")
        print("   - 使用投影坐标系（如 CGCS2000 / EPSG:4490 等效投影）")
        print("   - 走向大致平行于岸线")
        print("   - 保存为 ESRI Shapefile 格式")
        return

    # 读取基线
    baseline_gdf = gpd.read_file(BASELINE_SHP)
    print(f"  基线 CRS：{baseline_gdf.crs}")
    print(f"  基线要素数：{len(baseline_gdf)}")

    # 确保是投影坐标系（单位米），否则无法按米计算间距
    if baseline_gdf.crs is not None and baseline_gdf.crs.is_geographic:
        print("  ⚠️  检测到地理坐标系（度），建议先转为投影坐标系（米）！")
        print("       尝试转换为 EPSG:32650 (UTM 50N) ...")
        baseline_gdf = baseline_gdf.to_crs("EPSG:32650")
        print(f"       转换后 CRS：{baseline_gdf.crs}")

    # 合并所有基线段为一条
    from shapely.ops import linemerge, unary_union
    all_lines = unary_union(baseline_gdf.geometry)
    if all_lines.geom_type == "MultiLineString":
        all_lines = linemerge(all_lines)
    print(f"  基线总长度：{all_lines.length/1000:.2f} km")

    # 生成断面
    print(f"\n  生成断面中...")
    transects_gdf = generate_transects(
        baseline_geom=all_lines,
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
    print(f"\n  ✅ 断面已保存：{out_path}")

    # 验证提示
    print("\n  ⚠️  验证步骤（必须！）：")
    print("  1. 在 QGIS 中打开 transects.gpkg 和研究区底图")
    print("  2. 目视检查：断面是否从陆地方向指向海洋方向")
    print("  3. 若方向相反（断面从海指向陆），请在代码中将 nx/ny 取反后重新生成")
    print("  4. 检查断面是否均匀分布，无明显错误")
    print(f"\n✅ C1 完成！生成 {n_transects} 条断面")


if __name__ == "__main__":
    main()
