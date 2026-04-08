"""
B7 — 栅格转矢量（瞬时水边线提取）
============================================================
输入：output/sea_mask/YYYY_QN_sea.tif
输出：output/waterlines/YYYY_QN_waterline.gpkg（线矢量 GeoPackage）

处理步骤：
    1. 多边形化（rasterio.features.shapes）
    2. 仅保留值=1（海水）的多边形
    3. 合并所有多边形后取外边界
    4. 多边形转线（boundary）
    5. 碎线过滤：移除长度 < MIN_WATERLINE_LENGTH 的短线段
    6. 简化：shapely simplify(tolerance=WATERLINE_SIMPLIFY)
    7. 人工边界剔除（B5 Step 4）：
           ① 长直线   — sinuosity < ARTIF_SINUOSITY_THRESH 且段长 ≥ ARTIF_STRAIGHT_MIN_M
           ② 低曲率   — 窗口均值转角 < ARTIF_CURVE_THRESH_DEG
           ③ 规则直角 — 窗口内 ≥ ARTIF_RA_MIN_COUNT 个转角在 90°±ARTIF_RA_TOL_DEG 内
    8. 输出为 GeoPackage，CRS 与输入一致
"""

import os
import sys
import warnings
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, LineString
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEA_MASK_DIR, WATERLINE_DIR, PERIODS,
    MIN_WATERLINE_LENGTH, WATERLINE_SIMPLIFY,
    ARTIF_RESAMPLE_M, ARTIF_STRAIGHT_MIN_M, ARTIF_SINUOSITY_THRESH,
    ARTIF_CURVE_THRESH_DEG, ARTIF_RA_TOL_DEG, ARTIF_RA_MIN_COUNT,
    ARTIF_RA_WINDOW_M, ARTIF_MIN_KEEP_M,
)

WATER_VAL  = 255   # 水体像元值（白色）
NODATA_VAL = 128   # nodata（灰色）


# ── 人工边界剔除（B5 Step 4） ──────────────────────────────────────────────────

def _resample_line(line: LineString, spacing_crs: float) -> np.ndarray:
    """
    将线段重采样为等间距点序列，返回形状 (N, 2) 的坐标数组。
    spacing_crs 单位与 CRS 一致（地理坐标系为度，投影坐标系为米）。
    """
    length = line.length
    if length <= 0 or spacing_crs <= 0:
        return np.array(list(line.coords))
    n_pts = max(int(length / spacing_crs) + 1, 3)
    dists = np.linspace(0, length, n_pts)
    return np.array(
        [[line.interpolate(d).x, line.interpolate(d).y] for d in dists]
    )


def _turning_angles_deg(pts: np.ndarray) -> np.ndarray:
    """
    计算等间距点序列中每个中间点的转向角（度）。

    角度定义：
        0°  → 方向不变（笔直延伸）
        90° → 直角转弯
        180°→ 完全折回（U 形）

    返回形状 (N-2,) 的数组；angles[k] 对应 pts[k+1] 处的转向角。
    """
    if len(pts) < 3:
        return np.zeros(0)

    v_in  = pts[1:-1] - pts[:-2]    # 入射方向向量
    v_out = pts[2:]   - pts[1:-1]   # 出射方向向量

    n_in  = np.linalg.norm(v_in,  axis=1)
    n_out = np.linalg.norm(v_out, axis=1)
    valid = (n_in > 1e-12) & (n_out > 1e-12)

    # 有效像元用点积计算夹角，无效像元默认 0°（笔直）
    dot = np.einsum("ij,ij->i", v_in, v_out)
    denom = np.where(valid, n_in * n_out, 1.0)
    cos_a = np.where(valid, np.clip(dot / denom, -1.0, 1.0), 1.0)
    return np.degrees(np.arccos(cos_a))


def _build_natural_lines(pts: np.ndarray,
                          keep_mask: np.ndarray,
                          min_length_crs: float) -> list:
    """
    根据点保留掩膜将点序列切分为自然段列表。

    连续的 keep_mask=True 点组成一条自然段；
    长度 < min_length_crs 的段直接丢弃。
    """
    segs, buf = [], []
    for pt, keep in zip(pts, keep_mask):
        if keep:
            buf.append(tuple(pt))
        else:
            if len(buf) >= 2:
                seg = LineString(buf)
                if seg.length >= min_length_crs:
                    segs.append(seg)
            buf = []
    # 末尾剩余
    if len(buf) >= 2:
        seg = LineString(buf)
        if seg.length >= min_length_crs:
            segs.append(seg)
    return segs


def remove_artificial_segments(
    lines: list,
    crs_is_geographic: bool,
    resample_m:        float = ARTIF_RESAMPLE_M,
    straight_min_m:    float = ARTIF_STRAIGHT_MIN_M,
    sinuosity_thresh:  float = ARTIF_SINUOSITY_THRESH,
    curve_thresh_deg:  float = ARTIF_CURVE_THRESH_DEG,
    ra_tol_deg:        float = ARTIF_RA_TOL_DEG,
    ra_min_count:      int   = ARTIF_RA_MIN_COUNT,
    ra_window_m:       float = ARTIF_RA_WINDOW_M,
    min_keep_m:        float = ARTIF_MIN_KEEP_M,
) -> list:
    """
    从水边线列表中剔除人工边界线段（B5 Step 4 — 形态删线）。

    对每条线重采样后用滑动窗口逐段扫描，窗口内满足以下任一标准则将
    该窗口对应的顶点标记为"人工段"并从最终结果中移除：

        ① 长直线   ：窗口 sinuosity < sinuosity_thresh
                      （路径长 / 直线距 < 阈值 ≈ 直线，典型防波堤 <1.005）
        ② 低曲率   ：窗口内均值转角 < curve_thresh_deg
                      （平均几乎不转弯，补充捕捉极端直线段）
        ③ 规则直角 ：在 ra_window_m 长度的窗口内
                      转角在 90°±ra_tol_deg 内的数量 ≥ ra_min_count
                      （港池、围堰等规则矩形结构）

    剔除后剩余的自然段若长度 < min_keep_m 则同样丢弃。
    若某条线整体被判为人工段（无自然段留存），出于保守原则保留原线。

    Parameters
    ----------
    lines              : 输入 LineString 列表
    crs_is_geographic  : True=地理坐标（度），False=投影坐标（米）
    其余参数           : 对应 config.py 中 ARTIF_* 系列参数

    Returns
    -------
    list of LineString
    """
    # 单位换算：将米制参数转换为 CRS 单位
    scale     = 1.0 / 111_000.0 if crs_is_geographic else 1.0
    spc_crs   = resample_m  * scale       # 重采样间距（CRS 单位）
    keep_crs  = min_keep_m  * scale       # 最小保留长度（CRS 单位）

    # 窗口大小（点数），保证下限避免窗口过小
    w_s = max(int(straight_min_m / resample_m), 3)       # 直线/曲率检测窗口
    w_r = max(int(ra_window_m    / resample_m), ra_min_count + 1)  # 直角检测窗口

    result, n_modified = [], 0

    for line in lines:
        pts = _resample_line(line, spc_crs)
        n   = len(pts)

        # 点数太少无法分析，直接保留
        if n < 5:
            result.append(line)
            continue

        # 转向角数组，长度 n-2，angles[k] 对应 pts[k+1] 处的转角
        angles = _turning_angles_deg(pts)

        # 预计算累积弧长（用于高效计算窗口路径长度）
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # shape (n-1,)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])   # shape (n,)

        keep_mask = np.ones(n, dtype=bool)   # True = 保留

        # ── 标准①②：长直线 / 低曲率（滑动窗口，步长 1 点）──────────────
        for i in range(n - w_s + 1):
            j = i + w_s - 1            # 窗口末尾点索引

            # 窗口路径长度与直线距离
            path_len   = cum_lens[j] - cum_lens[i]
            straight_d = np.linalg.norm(pts[j] - pts[i])

            if straight_d > 1e-12:
                # 标准①：sinuosity 接近 1 → 直线
                sinuosity = path_len / straight_d
                if sinuosity < sinuosity_thresh:
                    keep_mask[i: j + 1] = False
                    continue            # 已标记，跳过标准②

            # 标准②：窗口内均值转角极低 → 低曲率
            # angles[k] 对应 pts[k+1]，故窗口 [i,j] 内部角为 angles[i : j-1]
            ai = i
            aj = min(j - 1, len(angles))   # 不含窗口端点的角
            if aj > ai:
                mean_angle = np.mean(angles[ai:aj])
                if mean_angle < curve_thresh_deg:
                    keep_mask[i: j + 1] = False

        # ── 标准③：规则直角结构（滑动窗口，向量化卷积）─────────────────
        if len(angles) >= w_r:
            is_ra = (np.abs(angles - 90.0) <= ra_tol_deg).astype(np.int8)
            # ra_counts[i] = 窗口 angles[i : i+w_r] 内的直角数
            ra_counts = np.convolve(is_ra, np.ones(w_r, dtype=np.int8), mode="valid")
            for i in np.where(ra_counts >= ra_min_count)[0]:
                # angles[i..i+w_r-1] 对应 pts[i+1..i+w_r]
                keep_mask[i + 1: i + w_r + 1] = False

        # ── 切分自然段 ────────────────────────────────────────────────────
        if not keep_mask.all():
            n_modified += 1
            natural = _build_natural_lines(pts, keep_mask, keep_crs)
            if natural:
                result.extend(natural)
            else:
                # 全段被判为人工时，保守保留原线（避免整条岸线丢失）
                warnings.warn(
                    f"人工边界剔除：某线整体被判为人工段，已保守保留原线 "
                    f"(length={line.length * (111.0 if crs_is_geographic else 0.001):.1f} km)"
                )
                result.append(line)
        else:
            result.append(line)

    if n_modified:
        print(
            f"    [人工边界剔除] {n_modified}/{len(lines)} 条线含人工段 → "
            f"剔除后共 {len(result)} 条自然线段"
        )
    else:
        print(f"    [人工边界剔除] 未检测到人工边界段（全部保留）")

    return result


# ── 水边线提取主函数 ───────────────────────────────────────────────────────────

def extract_waterline_for_period(period: str) -> dict:
    """
    从海水掩膜中提取瞬时水边线矢量（含人工边界剔除）。

    Returns
    -------
    dict: period, n_lines, total_length_km, n_removed_short, n_removed_artif
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

    # 根据 CRS 换算长度/简化阈值（地理坐标系单位为度，1度 ≈ 111000m）
    if crs.is_geographic:
        length_threshold   = MIN_WATERLINE_LENGTH / 111_000.0
        simplify_tolerance = WATERLINE_SIMPLIFY   / 111_000.0
        min_poly_area      = (500.0 / 111_000.0) ** 2
        deg_to_km          = 111.0
    else:
        length_threshold   = MIN_WATERLINE_LENGTH
        simplify_tolerance = WATERLINE_SIMPLIFY
        min_poly_area      = 500.0 ** 2
        deg_to_km          = 0.001   # m → km

    # ── Step 1–2：多边形化，过滤小多边形 ──────────────────────────────────
    water_uint8 = (data == WATER_VAL).astype(np.uint8)
    polys = []
    for geom_dict, val in shapes(water_uint8, mask=water_uint8, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            if geom.area >= min_poly_area:
                polys.append(geom)

    if not polys:
        print(f"  ⚠️  {period}: 无有效海水多边形，跳过")
        return None

    # 按面积降序，取最大的 50 个（避免大量小多边形拖慢 unary_union）
    polys.sort(key=lambda p: p.area, reverse=True)
    polys = polys[:50]

    # ── Step 3–4：合并 + 外边界提取 ───────────────────────────────────────
    sea_union = unary_union(polys)

    def _get_exteriors(geom):
        """递归提取所有 Polygon 的 exterior LinearRing。"""
        if geom.geom_type == "Polygon":
            return [geom.exterior]
        elif geom.geom_type == "MultiPolygon":
            return [p.exterior for p in geom.geoms]
        elif hasattr(geom, "geoms"):
            result = []
            for g in geom.geoms:
                result.extend(_get_exteriors(g))
            return result
        return []

    raw_lines = _get_exteriors(sea_union)

    if not raw_lines:
        print(f"  ⚠️  {period}: 边界提取失败，跳过")
        return None

    # ── Step 5–6：碎线过滤 + 简化 ─────────────────────────────────────────
    filtered_lines = []
    n_removed_short = 0
    for ring in raw_lines:
        ls = LineString(ring.coords)
        if ls.length >= length_threshold:
            simplified = ls.simplify(simplify_tolerance, preserve_topology=True)
            filtered_lines.append(simplified)
        else:
            n_removed_short += 1

    if not filtered_lines:
        print(f"  ⚠️  {period}: 碎线过滤后无线保留，跳过")
        return None

    # ── Step 7：人工边界剔除（B5 Step 4）─────────────────────────────────
    n_before_artif = len(filtered_lines)
    filtered_lines = remove_artificial_segments(
        filtered_lines,
        crs_is_geographic=crs.is_geographic,
    )

    # 人工段剔除后可能产生新的短线，再次过滤
    filtered_lines = [l for l in filtered_lines if l.length >= length_threshold]
    n_removed_artif = n_before_artif - len(filtered_lines)   # 净减少条数（≥0）

    if not filtered_lines:
        print(f"  ⚠️  {period}: 人工边界剔除后无线保留，跳过")
        return None

    # ── Step 8：保存 ──────────────────────────────────────────────────────
    total_length_km = sum(l.length * deg_to_km for l in filtered_lines)

    gdf = gpd.GeoDataFrame(
        {
            "period"    : [period] * len(filtered_lines),
            "length_km" : [round(l.length * deg_to_km, 2) for l in filtered_lines],
        },
        geometry=filtered_lines,
        crs=crs,
    )
    os.makedirs(WATERLINE_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="waterline")

    print(
        f"  ✅ {period}: {len(filtered_lines)} 条线, "
        f"总长={total_length_km:.1f} km "
        f"[短线过滤={n_removed_short}, 人工段剔除净减={n_removed_artif}]"
    )

    return {
        "period"            : period,
        "n_lines"           : len(filtered_lines),
        "total_length_km"   : round(total_length_km, 2),
        "n_removed_short"   : n_removed_short,
        "n_removed_artif"   : n_removed_artif,
    }


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  B7 — 栅格转矢量水边线提取（含人工边界剔除，共 24 期）")
    print(f"       最小保留长度={MIN_WATERLINE_LENGTH} m, "
          f"简化容差={WATERLINE_SIMPLIFY} m")
    print(f"       人工剔除：直线窗口={ARTIF_STRAIGHT_MIN_M}m, "
          f"sinuosity<{ARTIF_SINUOSITY_THRESH}, "
          f"曲率<{ARTIF_CURVE_THRESH_DEG}°, "
          f"直角({ARTIF_RA_MIN_COUNT}个/{ARTIF_RA_WINDOW_M}m)")
    print("=" * 60)
    os.makedirs(WATERLINE_DIR, exist_ok=True)

    records = []
    for period in PERIODS:
        result = extract_waterline_for_period(period)
        if result:
            records.append(result)

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print(f"\n  水边线统计摘要（{len(df)} 期）：")
        print(f"  平均线条数    ：{df['n_lines'].mean():.1f} 条/期")
        print(f"  平均总长度    ：{df['total_length_km'].mean():.1f} km/期")
        print(f"  人工段净剔除  ：{df['n_removed_artif'].sum()} 条（合计）")
        single_line = (df["n_lines"] == 1).sum()
        print(f"  恰好 1 条主线 ：{single_line}/{len(df)} 期")

    print(f"\n✅ B7 完成：成功处理 {len(records)} 期")
    print(f"   输出目录：{WATERLINE_DIR}")
    print("   验证：每期应有 1 条主水边线；人工段剔除数量应与已知人工岸线数量吻合")


if __name__ == "__main__":
    main()
