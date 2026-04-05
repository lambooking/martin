# 黄河三角洲岸线变化研究系统 — 使用指南

这份指南将帮助你在本地正确配置环境、准备数据并顺利运行所有提取与分析的代码模块。

---

## 步骤一：环境配置

强烈建议使用 Python 的虚拟环境（如 Anaconda / Miniconda 等），以避免包版本冲突问题。

1. **打开终端 (Terminal)**。
2. **创建一个新的 conda 虚拟环境**（比如叫 `coastline_env`）（可选但推荐）：
   ```bash
   conda create -n coastline_env python=3.10
   conda activate coastline_env
   ```
3. **安装必须的第三方依赖包**：
   在 `code` 目录下输入以下命令安装 `requirements.txt` 中指定的包：
   ```bash
   cd /Users/dengxianchi/Library/CloudStorage/OneDrive-个人/马婷/code
   pip install -r requirements.txt
   ```
   > ⚠️ 如果地图出图有问题，可能会提示缺失 `geopandas` 或者 `contextily` 的一些底层C依赖。Mac上如果是使用 conda 也可以直接通过 `conda install -c conda-forge geopandas contextily` 来安装最稳定的版本。

---

## 步骤二：准备数据

所有的 Python 脚本都**极其依赖于标准化的文件路径和命名**。进入项目目录下的 `/code/data/` 文件夹。你需要将你在 Google Earth Engine (GEE) 或 QGIS 产生的数据严格按照下面的格式放入对应的子文件夹：

```text
code/
└── data/
    ├── s2/
    │   ├── 2019_Q1_s2.tif     (⚠️ 请确保波段顺序: Band1=Green, Band2=SWIR)
    │   ├── 2019_Q2_s2.tif
    │   └── ... (共 24 期)
    ├── s1/
    │   ├── 2019_Q1_s1.tif     (⚠️ 波段顺序: Band1=VV_linear, Band2=VH_linear)
    │   ├── 2019_Q2_s1.tif
    │   └── ... (共 24 期)
    └── boundary/
        ├── baseline.shp       (⚠️ 手工绘制的基线文件，沿海岸走向)
        ├── study_area.shp     (研究区大致边缘)
        └── control_points_manual.csv (⚠️ 控制点文件，A4 模块用。需包含: point_id, lon, lat, s2_row, s2_col, s1_row, s1_col)
```

---

## 步骤三：检查初始参数 (非常重要！)

在开始运行前，必须打开 `code/config.py` 文件确认几个核心物理参数：

- `SEA_SEED_COORDS`：外海种子点的经纬度。**极其重要！** B6在剔除内陆渔场时，会向该坐标灌水只保留相连的海域。打开看该坐标是否点错了成了陆地。
- `TRANSECT_SPACING`：生成等距断面的间隔（默认 50米），如果想做更精细研究可改小。
- `S2_BAND_ORDER` / `S1_BAND_ORDER`：确认你在导出GeoTiff时的波段顺序是否符合这里字典配置的索引（索引0代表第一个波段）。

---

## 步骤四：运行系统

你有两种运行方式：

### 方式一：流水线全自动执行 (强烈建议数据配齐后使用)

只要你将上面的数据完全准备好了，你可以直接运行主程序脚本完成 A 阶段到 E 阶段所有的操作。

```bash
python main_pipeline.py
```
这会自动按顺序调动所有模块：从配准检验 $\rightarrow$ 水体像元识别 $\rightarrow$ 二值过滤 $\rightarrow$ 高潮代理基线推算 $\rightarrow$ 断面切割 $\rightarrow$ 最后生成用于论文的热点与统计图表。

### 方式二：模块步进执行试错法 (方便查找数据Bug)

你可以像搭积木一样，逐个执行 Python 脚本，以确定某一步的影像长什么样（特别是B1 - B6）：
1. 测试 Mndwi: `python B1_mndwi.py`
2. 等待结束后，可以在 `code/output/mndwi/` 里查看生成的影像，以此类推。

> **关于 B8（精度评估模块）使用提示**：
> 这个模块包含**人机交互反馈**。首次运行 `B8_accuracy_assessment.py` 会生成验证散点 `csv`（缺 ref_label）。
> 你需找到并打开 `output/accuracy/reference_labels_YYYY_QN.csv`，通过观察 `output/accuracy/reference_map...png` 图片人工填写每一行真实是水(1)还是非水(0)。保存后再运行一次该模块，系统将计算混淆矩阵和 F1 等指标并输出！

---

## 步骤五：查看并获取科研产品 (论文绘图)

运行完成后，你会得到一个完全成长的 `output/` 掩膜工厂：
- 如果你需要**矢量海岸线文件**拿去 ArcGIS 叠图，去 `output/waterlines`（24个期次）与 `output/annual_shorelines`（年度代理线）。
- 如果你需要**论文插图**，直接去 `output/figures` 目录下拷贝，里面有 E1到E4 的标准出图。
- 如果你需要**实验数据表格（NSM, EPR变化率等）**，去 `output/change` 下面拷取 CSV 数据自己汇总进论文表格。

祝科研顺利！
