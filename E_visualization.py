"""
E 阶段：可视化输出
============================================================
输出高质量的可视化图表，用于论文：
E1 — 水边线时序叠加图
E2 — NSM / EPR 沿岸分布图
E3 — 热点岸段专题图
E4 — 典型断面位置时序图
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import contextily as cx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIGURES_DIR, WATERLINE_DIR, CHANGE_DIR, TRANSECT_DIR,
    DISTANCE_DIR, ANNUAL_SL_DIR, YEARS, PERIODS
)

# 设置中文字体（Mac的常用中文字体）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_e1_waterline_timeseries():
    """E1 — 24期瞬时水边线叠加图，颜色随时间渐变（冷色到暖色）"""
    print("\n[E1] 绘制水边线时序图...")
    out_path = os.path.join(FIGURES_DIR, "E1_waterline_timeseries.png")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=len(PERIODS)-1)
    
    plotted = 0
    for i, period in enumerate(PERIODS):
        wl_path = os.path.join(WATERLINE_DIR, f"{period}_waterline.gpkg")
        if not os.path.exists(wl_path):
            continue
            
        try:
            gdf = gpd.read_file(wl_path)
            # 为了更好的可视化，如果 CRS 是地理坐标系（度），不做改变，如果在 cx 中遇到问题再处理
            # 简化绘制
            color = cmap(norm(i))
            gdf.plot(ax=ax, color=color, linewidth=0.5, alpha=0.7)
            plotted += 1
        except Exception as e:
            print(f"  读取 {period} 失败: {e}")

    if plotted == 0:
        print("  ⚠️ 没有可用的水边线数据。")
        plt.close(fig)
        return

    # 尝试添加底图（若有网络则添加 contextily，也可以跳过）
    try:
        # ctx 需要将坐标系转为 Web Mercator (EPSG:3857) 才能对齐地图
        # 但这里为了普适性，只画线框
        ax.set_facecolor('#f0f4f8') # 浅蓝色海洋底色
        cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron, attribution_size=6)
    except Exception as e:
        print(f"  ⚠️ 底图加载跳过: {e}")

    # 色标
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, len(PERIODS)//2, len(PERIODS)-1])
    cbar.set_ticklabels([PERIODS[0], PERIODS[len(PERIODS)//2], PERIODS[-1]])
    cbar.set_label("观测期次")

    ax.set_title("黄河三角洲水边线演变动态 (2019Q1 - 2024Q4)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("经度 / X坐标", fontsize=11)
    ax.set_ylabel("纬度 / Y坐标", fontsize=11)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ E1 保存: {out_path}")


def plot_e2_nsm_epr_distribution():
    """E2 — 沿岸 NSM 和 EPR 分布图"""
    print("\n[E2] 绘制沿岸变化分布图...")
    nsm_path = os.path.join(CHANGE_DIR, "NSM.csv")
    epr_path = os.path.join(CHANGE_DIR, "EPR.csv")
    
    if not (os.path.exists(nsm_path) and os.path.exists(epr_path)):
        print("  ⚠️ 缺少 NSM/EPR 文件，依赖 D 模块的结果。")
        return

    nsm_df = pd.read_csv(nsm_path)
    epr_df = pd.read_csv(epr_path)
    df = pd.merge(nsm_df, epr_df, on="transect_id")
    
    if len(df) == 0:
        return

    out_path = os.path.join(FIGURES_DIR, "E2_NSM_EPR_along_coast.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    t_ids = df["transect_id"]
    
    # 颜色：大于0为淤积(红色系)，小于0为侵蚀(蓝色系)
    colors1 = ['#d73027' if v > 0 else '#4575b4' for v in df["NSM_m"]]
    ax1.bar(t_ids, df["NSM_m"], color=colors1, width=1.0)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel("净位移 NSM (m)", fontsize=11)
    ax1.set_title("黄河三角洲沿岸 NSM 空间分布特征 (2019 - 2024)", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    colors2 = ['#d73027' if v > 0 else '#4575b4' for v in df["EPR_m_yr"]]
    ax2.bar(t_ids, df["EPR_m_yr"], color=colors2, width=1.0)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel("沿岸断面编号 (从南到北跨越半岛)", fontsize=12)
    ax2.set_ylabel("端点速率 EPR (m/yr)", fontsize=11)
    ax2.set_title("黄河三角洲沿岸 EPR 空间分布特征", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 图例
    custom_lines = [Line2D([0], [0], color='#d73027', lw=4),
                    Line2D([0], [0], color='#4575b4', lw=4)]
    ax1.legend(custom_lines, ['向海淤积', '向陆侵蚀'], loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ E2 保存: {out_path}")


def plot_e3_hotspots_map():
    """E3 — 热点岸段专题图"""
    print("\n[E3] 绘制热点岸段地图...")
    hotspots_path = os.path.join(CHANGE_DIR, "hotspots.gpkg")
    transects_path = os.path.join(TRANSECT_DIR, "transects.gpkg")
    
    if not (os.path.exists(hotspots_path) and os.path.exists(transects_path)):
        print("  ⚠️ 缺少所需几何文件。")
        return

    out_path = os.path.join(FIGURES_DIR, "E3_hotspots_map.png")
    fig, ax = plt.subplots(figsize=(10, 10))

    try:
        t_gdf = gpd.read_file(transects_path)
        hs_gdf = gpd.read_file(hotspots_path)
        
        # 背景：画出所有断面的足点组成的基线轮廓作为参考底线
        t_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, label='所有提取断面')
        
        # 获取单独的类别进行独立图例控制
        acc = hs_gdf[hs_gdf['type'] == 'Accretion']
        ero = hs_gdf[hs_gdf['type'] == 'Erosion']
        
        if not acc.empty:
            acc.plot(ax=ax, color='#d73027', linewidth=4, alpha=0.9, label='强淤积热点 (Top 10%)')
        if not ero.empty:
            ero.plot(ax=ax, color='#4575b4', linewidth=4, alpha=0.9, label='强侵蚀热点 (Top 10%)')

        try:
            cx.add_basemap(ax, crs=hs_gdf.crs, source=cx.providers.CartoDB.Positron, attribution_size=6)
        except:
            ax.set_facecolor('#fafafa')

        ax.set_title("岸线演变关键热点区段分布图", fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel("坐标X", fontsize=11)
        ax.set_ylabel("坐标Y", fontsize=11)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ E3 保存: {out_path}")
    except Exception as e:
        print(f"  ❌ 绘制失败: {e}")


def plot_e4_transect_timeseries():
    """E4 — 典型断面位置时序图"""
    print("\n[E4] 绘制典型断面时序波动图...")
    dist_path = os.path.join(DISTANCE_DIR, "distance_matrix.csv")
    mhw_path = os.path.join(ANNUAL_SL_DIR, "MHW_proxy_distances.csv")
    
    if not (os.path.exists(dist_path) and os.path.exists(mhw_path)):
        print("  ⚠️ 缺少距离文件。")
        return

    dist_df = pd.read_csv(dist_path, index_col="transect_id")
    mhw_df = pd.read_csv(mhw_path, index_col="transect_id")
    
    if len(dist_df) < 3:
        return

    # 从存在数据的断面中等距选择3个断面作为典型代表（避免随机导致展示效果差）
    valid_tids = dist_df.dropna(how='all', axis=1).index.tolist()
    if len(valid_tids) < 3:
        print("  ⚠️ 有效断面过少。")
        return
        
    step = len(valid_tids) // 3
    sample_ids = [valid_tids[0], valid_tids[step], valid_tids[2*step]]
    
    out_path = os.path.join(FIGURES_DIR, "E4_transect_timeseries.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    x_ticks_labels = PERIODS
    x_positions = np.arange(len(PERIODS))

    for idx, (ax, t_id) in enumerate(zip(axes, sample_ids)):
        t_row = dist_df.loc[t_id]
        
        # 散点图绘制各期的瞬时距离
        y_vals = []
        x_plot = []
        for i, p in enumerate(PERIODS):
            if p in t_row.index and not np.isnan(t_row[p]):
                y_vals.append(t_row[p])
                x_plot.append(x_positions[i])
                
        if len(y_vals) == 0:
            continue

        ax.scatter(x_plot, y_vals, color='gray', alpha=0.6, s=40, label='单期瞬时水边线位置 (原始散点)', zorder=2)
        ax.plot(x_plot, y_vals, color='lightgray', alpha=0.4, linewidth=1, zorder=1)

        # 叠加年度 MHW_proxy 线 (跨越每年的范围)
        mhw_row = mhw_df.loc[t_id]
        mhw_x = []
        mhw_y = []
        for yi, year in enumerate(YEARS):
            col_name = str(year)
            if col_name in mhw_row.index and not np.isnan(mhw_row[col_name]):
                # 画出覆盖当年4个季度的阶梯线或者中间线段
                val = mhw_row[col_name]
                q_center = yi * 4 + 1.5 # 年度的中心横坐标近似
                mhw_x.append(q_center)
                mhw_y.append(val)
                # 画一条覆盖整年的横线段
                start_x = yi * 4
                end_x = min(start_x + 3.8, len(PERIODS)-1)
                ax.hlines(val, start_x, end_x, color='teal', linewidth=3, alpha=0.8,
                          label='MHW proxy 年度参考线' if yi==0 else "")
                
        if len(mhw_x) > 1:
            ax.plot(mhw_x, mhw_y, '--', color='teal', linewidth=1.5, alpha=0.5)

        # 调整视觉：刻度，图例
        ax.set_title(f"序列断面临岸趋势: T{t_id}", fontsize=12, fontweight='bold', pad=10, loc='left')
        ax.set_ylabel("距离基线距离 (m)", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        if idx == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_ticks_labels, rotation=45, ha='right')
    axes[-1].set_xlabel("合成期次 (Time Period)", fontsize=12)

    fig.suptitle("多尺度时间特征：年内剧烈波动 vs 年际长程趋势", fontsize=15, fontweight='bold', y=0.94)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ E4 保存: {out_path}")


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
