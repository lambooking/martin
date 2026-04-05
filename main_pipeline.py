"""
黄河三角洲岸线变化研究 — 主管道入口
============================================================
按序执行所有阶段的代码处理脚本。
执行前请确认：
1. data 目录中的源文件齐备（s1, s2, boundary）
2. config.py 中的外海种子点已根据研究区确认：SEA_SEED_COORDS
3. 控制点验证文件：control_points_manual.csv 是否存在
"""

import os
import sys

def run_script(script_name):
    cmd = f"python {script_name}"
    print(f"\n>>>> 正在执行：{script_name}...")
    res = os.system(cmd)
    if res != 0:
        print(f"\n❌ 执行 {script_name} 失败！终止管道。")
        sys.exit(1)

def main():
    print("="*60)
    print("  黄河三角洲岸线提取分析全自动处理管道")
    print("="*60)

    scripts = [
        "config.py",
        "A4_coregistration_check.py", # 手动验证脚本（可选）
        "B1_mndwi.py",               # 预处理数据
        "B2_s1_to_db.py",            # 预处理数据
        "A4_auto_coregistration_check.py", # 自动化配准亚像素检测 (依赖 B1/B2)
        "B3_thresholds.py",
        "B4_water_extraction.py",
        "B5_morphological_clean.py",
        "B6_sea_connectivity.py",
        "B7_waterline_extraction.py",
        "B8_accuracy_assessment.py",
        "C1_transect_generation.py",
        "C2_distance_matrix.py",
        "C3_C4_annual_shorelines.py",
        "D_change_analysis.py",
        "E_visualization.py"
    ]
    
    for script in scripts:
        if script == "config.py":
            run_script(script)
            continue
            
        print(f"\n--- 准备运行模块: {script} ---")
        
        # 对于 A 阶段这种可选或需要人工的文件，如果报错不强制中断
        if "A4" in script:
            try:
                run_script(script)
            except Exception as e:
                print(f"  ⚠️  {script} 跳过或执行受阻（正常现象，若已手动校准请忽略）")
            continue
            
        run_script(script)
        
    print("\n✅ 所有阶段均已成功执行完成！请查阅 output 目录。")

if __name__ == "__main__":
    main()
