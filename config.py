# config.py
# 黄河三角洲岸线变化研究 — 全局配置文件
# 所有路径、参数集中在此处管理，不要在脚本中硬编码

import os

# ------------------------------------------------------------------
# 基础路径
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

S2_DIR       = os.path.join(DATA_DIR, "s2")
S1_DIR       = os.path.join(DATA_DIR, "s1")
BOUNDARY_DIR = os.path.join(DATA_DIR, "boundary")

# ------------------------------------------------------------------
# 输入文件
# ------------------------------------------------------------------
STUDY_AREA_SHP  = os.path.join(BOUNDARY_DIR, "study_area.shp")
BASELINE_SHP    = os.path.join(BOUNDARY_DIR, "baseline.shp")
CONTROL_PTS_CSV = os.path.join(BOUNDARY_DIR, "control_points_manual.csv")

# ------------------------------------------------------------------
# 输出子目录
# ------------------------------------------------------------------
MNDWI_DIR        = os.path.join(OUTPUT_DIR, "mndwi")
S1_DB_DIR        = os.path.join(OUTPUT_DIR, "s1_db")
THRESH_DIR       = os.path.join(OUTPUT_DIR, "thresholds")
WATER_MASK_DIR   = os.path.join(OUTPUT_DIR, "water_mask")
WATER_CLEAN_DIR  = os.path.join(OUTPUT_DIR, "water_clean")
SEA_MASK_DIR     = os.path.join(OUTPUT_DIR, "sea_mask")
WATERLINE_DIR    = os.path.join(OUTPUT_DIR, "waterlines")
TRANSECT_DIR     = os.path.join(OUTPUT_DIR, "transects")
DISTANCE_DIR     = os.path.join(OUTPUT_DIR, "distances")
ANNUAL_SL_DIR    = os.path.join(OUTPUT_DIR, "annual_shorelines")
CHANGE_DIR       = os.path.join(OUTPUT_DIR, "change")
ACCURACY_DIR     = os.path.join(OUTPUT_DIR, "accuracy")
COREG_DIR        = os.path.join(OUTPUT_DIR, "coregistration")
FIGURES_DIR      = os.path.join(OUTPUT_DIR, "figures")

# ------------------------------------------------------------------
# 时序配置
# ------------------------------------------------------------------
YEARS   = list(range(2019, 2025))         # 2019–2024
PERIODS = [f"{y}_Q{q}" for y in YEARS for q in range(1, 5)]  # 24 期

# ------------------------------------------------------------------
# 波段顺序（0-indexed）
# ------------------------------------------------------------------
S2_BAND_ORDER = {"green": 0, "swir": 1}   # Band1=Green(B3), Band2=SWIR(B11)
S1_BAND_ORDER = {"vv": 0, "vh": 1}        # Band1=VV, Band2=VH（线性值）

# 空间分辨率（米）
PIXEL_SIZE = 20

# ------------------------------------------------------------------
# B 阶段：水体提取参数
# ------------------------------------------------------------------
MORPH_OPEN_SIZE      = 3    # 形态学开运算结构元素尺寸（像元）
MORPH_CLOSE_SIZE     = 3    # 形态学闭运算结构元素尺寸（像元）
MIN_WATER_AREA_PX    = 50   # 最小水体连通域面积（像元）
MIN_WATERLINE_LENGTH = 500  # 最小水边线保留长度（米）
WATERLINE_SIMPLIFY   = 20   # shapely simplify 容差（米）

# 外海种子点（研究区海洋一侧，需实际确认后修改）
# 格式：(lon, lat)，选取研究区东侧/北侧已知海洋像元
SEA_SEED_COORDS = [(119.25, 38.10)]   # ⚠️ 待实际确认后修改

# ------------------------------------------------------------------
# C 阶段：断面参数
# ------------------------------------------------------------------
TRANSECT_SPACING     = 50    # 断面间距（米）
TRANSECT_SEA_LENGTH  = 3000  # 断面向海延伸长度（米）
TRANSECT_LAND_LENGTH = 1000  # 断面向陆延伸长度（米）

# ------------------------------------------------------------------
# B8 精度评估
# ------------------------------------------------------------------
ACCURACY_N_POINTS     = 200   # 随机验证点数量
ACCURACY_BUFFER_M     = 500   # 海岸线缓冲区宽度（米）
ACCURACY_SAMPLE_PERIODS = ["2019_Q1", "2020_Q3", "2022_Q1", "2023_Q2", "2024_Q3"]  # 抽检期次

# ------------------------------------------------------------------
# A4 配准检验
# ------------------------------------------------------------------
COREG_CHECK_PERIODS = ["2020_Q1", "2021_Q3", "2022_Q4"]  # 用于检验的期次

# ------------------------------------------------------------------
# 工具函数：确保输出目录存在
# ------------------------------------------------------------------
def ensure_dirs():
    """创建所有输入与输出子目录（若已存在则跳过）。"""
    dirs = [
        # 输入数据存放目录
        DATA_DIR, S2_DIR, S1_DIR, BOUNDARY_DIR,
        # 各阶段产生缓存与输出目录
        OUTPUT_DIR,
        MNDWI_DIR, S1_DB_DIR, THRESH_DIR, WATER_MASK_DIR,
        WATER_CLEAN_DIR, SEA_MASK_DIR, WATERLINE_DIR, TRANSECT_DIR,
        DISTANCE_DIR, ANNUAL_SL_DIR, CHANGE_DIR, ACCURACY_DIR,
        COREG_DIR, FIGURES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print("✅ 所有输出目录已就绪。")
    print(f"   BASE_DIR   = {BASE_DIR}")
    print(f"   DATA_DIR   = {DATA_DIR}")
    print(f"   OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"   共 {len(PERIODS)} 期：{PERIODS[0]} → {PERIODS[-1]}")
