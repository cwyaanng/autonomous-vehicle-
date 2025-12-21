# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 경로 설정 =====
COLLIDED_DIR = "result/Town04_학습_결과데이터/done_collided"
REACHED_DIR  = "result/Town04_학습_결과데이터/done_reached"
WAYPOINT_DIR = "result/Town04_학습_결과데이터/waypoint_ahead" 

OUTDIR = "result/Town04_학습_결과데이터/plots"
os.makedirs(OUTDIR, exist_ok=True)

def collect_files(base_dir):
    files = glob.glob(os.path.join(base_dir, "*.csv"))
    groups = {"Proposed Method": [], "CQL": [], "AWAC": [], "SAC": []}
    
    for f in files:
        filename = os.path.basename(f).lower()
        if "cql" in filename:
            groups["CQL"].append(f)
        elif "awac" in filename:
            groups["AWAC"].append(f)
        elif "sac" in filename:
            groups["SAC"].append(f)
        elif "proposed" in filename:
            groups["Proposed Method"].append(f)
            
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups

collided_groups = collect_files(COLLIDED_DIR)
reached_groups  = collect_files(REACHED_DIR)
waypoint_groups = collect_files(WAYPOINT_DIR)


def file_rate(csv_path):
    df = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        print(f"Warning: 파일을 읽을 수 없습니다 -> {csv_path}")
        return 0.0

    col = None
    if "Value" in df.columns:
        col = "Value"
    else:
        matches = [c for c in df.columns if str(c).strip().lower() == "value"]
        if matches:
            col = matches[0]
    
    if col is None:
        return 0.0

    v = df[col].replace({"True":1, "False":0, "true":1, "false":0})
    v = pd.to_numeric(v, errors="coerce").fillna(0)
    v = (v > 0.5).astype(int)

    total = int(len(v))
    ones  = int((v == 1).sum())
    return (ones / total) if total > 0 else 0.0


def file_numeric_mean(csv_path):
    df = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return np.nan

    col = None
    if "Value" in df.columns:
        col = "Value"
    else:
        matches = [c for c in df.columns if str(c).strip().lower() == "value"]
        if matches:
            col = matches[0]
            
    if col is None:
        return np.nan
    
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(v) == 0:
        return 0.0
    return v.mean()


def group_macro_mean(file_list):
    if not file_list:
        return np.nan, 0
    rates = [file_rate(f) for f in file_list]
    return float(np.mean(rates)), len(file_list)


def group_numeric_macro_mean(file_list):
    if not file_list:
        return np.nan, 0
    means = [file_numeric_mean(f) for f in file_list]
    means = [m for m in means if not np.isnan(m)]
    
    if not means:
        return 0.0, 0
    return float(np.mean(means)), len(file_list)


order = ["Proposed Method", "CQL", "AWAC", "SAC"]

collision_means = {}
reached_means   = {}
waypoint_means  = {} 

nfiles_col = {}
nfiles_rea = {}
nfiles_way = {}      

for g in order:
    collision_means[g], nfiles_col[g] = group_macro_mean(collided_groups.get(g, []))
    reached_means[g],   nfiles_rea[g] = group_macro_mean(reached_groups.get(g, []))
    waypoint_means[g],  nfiles_way[g] = group_numeric_macro_mean(waypoint_groups.get(g, []))

summary = pd.DataFrame({
    "Technique": order,
    "Collision Rate (%)": [collision_means[g]*100 for g in order],
    "Reached Rate (%)":   [reached_means[g]*100   for g in order],
    "Route Completion":   [waypoint_means[g]      for g in order], 
    "#files(col)":        [nfiles_col[g]          for g in order],
    "#files(rea)":        [nfiles_rea[g]          for g in order],
    "#files(way)":        [nfiles_way[g]          for g in order], 
})

print("===== Summary Table =====")
print(summary)
summary_path = os.path.join(OUTDIR, "summary_rate_by_group_macro.csv")
summary.to_csv(summary_path, index=False)


# ===== [수정됨] 시각화 준비 =====
colors = {
  "Proposed Method": "#FFA500",  # 주황색
  "CQL": "#B0B0B0",              # 회색
  "AWAC": "#B0B0B0",
  "SAC": "#B0B0B0"
}

# 1. 색상 리스트는 '줄바꿈 없는' 원래 이름으로 먼저 매칭해서 만듭니다.
bar_colors = [colors[g] for g in summary["Technique"]]

# 2. 그래프 그리기용 DataFrame을 따로 만들어서 이름을 바꿉니다. (CSV는 원본 유지)
plot_summary = summary.copy()
plot_summary["Technique"] = plot_summary["Technique"].replace("Proposed Method", "Proposed\nMethod")


# 1. Collision Rate
plt.figure(figsize=(7,7))
plt.bar(plot_summary["Technique"], plot_summary["Collision Rate (%)"], color=bar_colors)
plt.title("Collision Rate", fontsize=30)
plt.ylabel("% Percent", fontsize=30)
plt.ylim(0, 100)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "collision_rate_bar_grouped_macro.png"), dpi=300, bbox_inches="tight")
plt.show()

# 2. Reached Rate
plt.figure(figsize=(7,7))
plt.bar(plot_summary["Technique"], plot_summary["Reached Rate (%)"], color=bar_colors)
plt.title("Reached Rate", fontsize=30)
plt.ylabel("% Percent", fontsize=30)
plt.ylim(0, 100)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "reached_rate_bar_grouped_macro.png"), dpi=300, bbox_inches="tight")
plt.show()

# 3. Route Completion
plt.figure(figsize=(7,7))
plt.bar(plot_summary["Technique"], plot_summary["Route Completion"], color=bar_colors)
plt.title("Route Completion", fontsize=30)
plt.ylabel("Average Value", fontsize=30) 
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "route_completion_bar_grouped_macro.png"), dpi=300, bbox_inches="tight")
plt.show()

print(f" - CSV 저장 경로: {summary_path}")
print(f" - 그래프 저장: {OUTDIR} 폴더 확인")