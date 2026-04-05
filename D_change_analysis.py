"""
D1 — NSM 净位移计算
D2 — EPR 端点速率计算
D3 — 热点岸段识别
============================================================
输入：
    output/annual_shorelines/MHW_proxy_distances.csv
    output/transects/transects.gpkg
输出：
    output/change/NSM.csv
    output/change/EPR.csv
    output/change/hotspots.gpkg

逻辑：
    NSM_m = MHW_proxy_2024 - MHW_proxy_2019
        （正值=岸线距离起点变远=向海推淤，负值=向陆侵蚀）
    EPR_m_per_year = NSM_m / (2024 - 2019)
    识别热点：取 EPR 绝对值的 Top 10% 断面，并尝试将空间连续的热点断面连接为热点区段。
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ANNUAL_SL_DIR, TRANSECT_DIR, CHANGE_DIR, YEARS
)


def compute_nsm_epr(mhw_distances: pd.DataFrame) -> pd.DataFrame:
    """计算每个断面的 NSM 和 EPR。"""
    year_start = str(YEARS[0])
    year_end   = str(YEARS[-1])
    dt_years   = YEARS[-1] - YEARS[0]

    if year_start not in mhw_distances.columns or year_end not in mhw_distances.columns:
        raise ValueError(f"缺失起始年 {year_start} 或终止年 {year_end} 的数据无法计算全周期 NSM/EPR。")

    df = mhw_distances[[year_start, year_end]].copy()
    
    # NSM = EndDistance - StartDistance
    # 距离定义为从基线陆侧向海，所以变大=淤积(正)，变小=侵蚀(负)
    df["NSM_m"] = df[year_end] - df[year_start]
    df["EPR_m_yr"] = df["NSM_m"] / dt_years

    # 统计信息
    df_valid = df.dropna(subset=["NSM_m"])
    print(f"  有效计算断面: {len(df_valid)} / {len(df)}")
    print(f"  NSM 范围: {df_valid['NSM_m'].min():.1f}m ~ {df_valid['NSM_m'].max():.1f}m")
    print(f"  EPR 范围: {df_valid['EPR_m_yr'].min():.1f}m/yr ~ {df_valid['EPR_m_yr'].max():.1f}m/yr")
    print(f"  淤积断面比例: {(df_valid['NSM_m'] > 0).mean()*100:.1f}%")
    print(f"  侵蚀断面比例: {(df_valid['NSM_m'] < 0).mean()*100:.1f}%")

    return df


def extract_hotspots(epr_df: pd.DataFrame, transects_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    提取侵蚀或淤积热点（Top 10%）。
    为便于可视化，我们将热点提取为“沿岸区的岸段线”。
    逻辑：在基线上找到这些断面的足点，把相邻的足点连成线。
    """
    df_valid = epr_df.dropna(subset=["EPR_m_yr"]).copy()
    if len(df_valid) == 0:
        return None

    # 计算 90% 分位数（取绝对值排名的 Top 10%）
    threshold = np.percentile(np.abs(df_valid["EPR_m_yr"]), 90)
    print(f"\n  热点 EPR 阈值 (Top 10%): 绝对值 >= {threshold:.1f} m/yr")

    hotspot_idx = df_valid[np.abs(df_valid["EPR_m_yr"]) >= threshold].index
    print(f"  识别到 {len(hotspot_idx)} 个热点断面")

    if len(hotspot_idx) == 0:
        return None

    # 将热点按侵蚀/淤积分类
    hotspot_df = df_valid.loc[hotspot_idx].copy()
    hotspot_df["type"] = np.where(hotspot_df["EPR_m_yr"] > 0, "Accretion", "Erosion")

    # 构建空间连续的热点区段
    # 由于断面ID是沿基线顺序的，连续的ID即代表空间上的连续
    # 我们用transects的foot坐标（基线足点）来连线
    transects_gdf_indexed = transects_gdf.set_index("transect_id")
    
    segments = []
    types = []
    mean_eprs = []

    # 按断面ID排序
    sorted_hs = hotspot_df.sort_index()
    
    # 寻找连续片段
    current_segment_ids = []
    current_type = None

    for t_id, row in sorted_hs.iterrows():
        t_type = row["type"]
        
        if not current_segment_ids:
            current_segment_ids = [t_id]
            current_type = t_type
        else:
            # 判断是否连续（ID差1）且类型相同
            if t_id == current_segment_ids[-1] + 1 and t_type == current_type:
                current_segment_ids.append(t_id)
            else:
                # 结束上一个片段
                if len(current_segment_ids) > 1:
                    # 提取足点坐标连成线
                    pts = []
                    for cid in current_segment_ids:
                        if cid in transects_gdf_indexed.index:
                            row_t = transects_gdf_indexed.loc[cid]
                            pts.append((row_t["foot_x"], row_t["foot_y"]))
                    if len(pts) >= 2:
                        segments.append(LineString(pts))
                        types.append(current_type)
                        mean_eprs.append(hotspot_df.loc[current_segment_ids, "EPR_m_yr"].mean())
                
                # 开始新片段
                current_segment_ids = [t_id]
                current_type = t_type

    # 处理最后一个片段
    if len(current_segment_ids) > 1:
        pts = []
        for cid in current_segment_ids:
            if cid in transects_gdf_indexed.index:
                row_t = transects_gdf_indexed.loc[cid]
                pts.append((row_t["foot_x"], row_t["foot_y"]))
        if len(pts) >= 2:
            segments.append(LineString(pts))
            types.append(current_type)
            mean_eprs.append(hotspot_df.loc[current_segment_ids, "EPR_m_yr"].mean())

    if not segments:
        print("  ⚠️ 没有找到包含2个及以上连续断面的热点区段")
        return None

    hotspot_segments_gdf = gpd.GeoDataFrame(
        {
            "segment_id": range(len(segments)),
            "type": types,
            "mean_EPR_m_yr": mean_eprs,
            "length_m": [s.length for s in segments]
        },
        geometry=segments,
        crs=transects_gdf.crs
    )

    print(f"  生成 {len(hotspot_segments_gdf)} 个连续热点岸段，总长度 {hotspot_segments_gdf['length_m'].sum()/1000:.1f} km")
    return hotspot_segments_gdf


def main():
    print("=" * 55)
    print("  D1/D2/D3 — 岸线变化定量分析 (NSM, EPR, Hotspots)")
    print("=" * 55)

    os.makedirs(CHANGE_DIR, exist_ok=True)

    # 1. 载入年度 MHW_proxy 距离
    mhw_path = os.path.join(ANNUAL_SL_DIR, "MHW_proxy_distances.csv")
    if not os.path.exists(mhw_path):
        print(f"❌ MHW距离文件不存在：{mhw_path}\n请先提取 MHW")
        sys.exit(1)
    
    mhw_distances = pd.read_csv(mhw_path, index_col="transect_id")
    
    # 2. 载入断面数据
    transect_path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    if not os.path.exists(transect_path):
        print(f"❌ 断面文件不存在：{transect_path}\n请先生成断面")
        sys.exit(1)
    transects_gdf = gpd.read_file(transect_path)

    # D1 & D2 计算
    print("\n[1/3] 计算 NSM 和 EPR...")
    try:
        change_df = compute_nsm_epr(mhw_distances)
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        return

    out_nsm = os.path.join(CHANGE_DIR, "NSM.csv")
    out_epr = os.path.join(CHANGE_DIR, "EPR.csv")
    change_df[["NSM_m"]].to_csv(out_nsm, encoding="utf-8-sig")
    change_df[["EPR_m_yr"]].to_csv(out_epr, encoding="utf-8-sig")
    print(f"  ✅ NSM 已保存: {out_nsm}")
    print(f"  ✅ EPR 已保存: {out_epr}")

    # D3 热点识别
    print("\n[2/3] 识别拓扑热点岸段...")
    hotspot_gdf = extract_hotspots(change_df, transects_gdf)
    
    if hotspot_gdf is not None:
        out_hotspots = os.path.join(CHANGE_DIR, "hotspots.gpkg")
        hotspot_gdf.to_file(out_hotspots, driver="GPKG")
        print(f"  ✅ 热点岸段已保存: {out_hotspots}")
    
    # D4 多尺度比对准备 (这里只生成带有必要数据的CSV，E阶段可视化)
    # 计算年内变化带宽均值 (Outer_P95 - MHW_proxy)
    print("\n[3/3] 准备多尺度对比数据...")
    outer_path = os.path.join(ANNUAL_SL_DIR, "Outer_P95_distances.csv")
    if os.path.exists(outer_path):
        outer_distances = pd.read_csv(outer_path, index_col="transect_id")
        
        # 计算每一年的带宽(P95 - MHW)，然后求所有年份的均值
        bandwidths = []
        for year in YEARS:
            y_str = str(year)
            if y_str in outer_distances.columns and y_str in mhw_distances.columns:
                bw = outer_distances[y_str] - mhw_distances[y_str]
                bandwidths.append(bw)
        
        if bandwidths:
            mean_bandwidth = pd.concat(bandwidths, axis=1).mean(axis=1)
            multiscale_df = pd.DataFrame({
                "NSM_m": change_df["NSM_m"],
                "Mean_IntraAnnual_Bandwidth_m": mean_bandwidth
            })
            # 信噪比 = 年际净变化 / 年内波动幅度
            multiscale_df["Signal_Noise_Ratio"] = np.abs(multiscale_df["NSM_m"]) / multiscale_df["Mean_IntraAnnual_Bandwidth_m"].clip(lower=1)
            
            out_ms = os.path.join(CHANGE_DIR, "multiscale_comparison.csv")
            multiscale_df.to_csv(out_ms, encoding="utf-8-sig")
            print(f"  ✅ 多尺度验证数据已保存: {out_ms}")
            print(f"     信噪比 > 1 的断面占比: {(multiscale_df['Signal_Noise_Ratio'] > 1).mean()*100:.1f}%")
            print(f"     (表示年际趋势信号强于年内季度波动噪声)")
    else:
        print(f"  ⚠️ Outer_P95 文件不存在，跳过多尺度数据合并")

    print("\n✅ D1-D3 模块运行完成！")


if __name__ == "__main__":
    main()
