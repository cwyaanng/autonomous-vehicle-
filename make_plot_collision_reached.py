# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 경로 =====
COLLIDED_DIR = "result/final_result_town4/done_collided"
REACHED_DIR  = "result/final_result_town4/done_reached"
OUTDIR = "result/final_result_town4/plots"
os.makedirs(OUTDIR, exist_ok=True)

# ===== 파일 수집 (정확한 접미어/키워드 매칭) =====
def collect_files(base_dir):
    files = glob.glob(os.path.join(base_dir, "*.csv"))
    groups = {"Proposed Method": [], "CQL": [], "AWAC": [], "SAC": []}
    for f in files:
        b = os.path.basename(f)
        bl = b.lower()

        # Proposed Method: 끝이 _1차실험.csv ~ _6차실험.csv
        if re.search(r"_([1-6])차실험\.csv$", b):
            groups["Proposed Method"].append(f)
            continue

        # CQL: cql 또는 cql2를 포함 (둘 다 CQL로 묶음)
        if re.search(r"_cql\d*\.csv$", bl):
            groups["CQL"].append(f)
            continue

        # AWAC/SAC: 끝이 _awac.csv / _sac.csv (변형도 포착)
        if re.search(r"_awac\d*\.csv$", bl):
            groups["AWAC"].append(f)
            continue
        if re.search(r"_sac\d*\.csv$", bl):
            groups["SAC"].append(f)
            continue

    # 정렬(보기 좋게)
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups

collided_groups = collect_files(COLLIDED_DIR)
reached_groups  = collect_files(REACHED_DIR)

# ===== 파일 하나 → (Value==1)/행수 (파일별 비율) =====
def file_rate(csv_path):
    # 인코딩 유연 처리
    df = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("CSV 읽기 실패: {}".format(csv_path))

    # 'Value' 컬럼 찾기 (대소문자 무시)
    if "Value" in df.columns:
        col = "Value"
    else:
        matches = [c for c in df.columns if str(c).strip().lower() == "value"]
        if not matches:
            raise KeyError("'Value' 컬럼이 없습니다: {} (columns={})".format(csv_path, list(df.columns)))
        col = matches[0]

    v = df[col].replace({"True":1, "False":0, "true":1, "false":0})
    v = pd.to_numeric(v, errors="coerce").fillna(0)
    v = (v > 0.5).astype(int)

    total = int(len(v))
    ones  = int((v == 1).sum())
    rate  = (ones / total) if total > 0 else 0.0
    return rate

# ===== 그룹(여러 파일) → “파일별 비율들의 평균”(macro average) =====
def group_macro_mean(file_list):
    if not file_list:
        return np.nan, 0
    rates = [file_rate(f) for f in file_list]
    return float(np.mean(rates)), len(file_list)

# ===== 충돌/도달 각각 그룹 평균 =====
order = ["Proposed Method", "CQL", "AWAC", "SAC"]

collision_means = {}
reached_means   = {}
nfiles_col = {}
nfiles_rea = {}

for g in order:
    collision_means[g], nfiles_col[g] = group_macro_mean(collided_groups.get(g, []))
    reached_means[g],   nfiles_rea[g] = group_macro_mean(reached_groups.get(g, []))

summary = pd.DataFrame({
    "Technique": order,
    "Collision Rate (%)": [collision_means[g]*100 for g in order],
    "Reached Rate (%)":   [reached_means[g]*100   for g in order],
    "#files(collided)":   [nfiles_col[g]          for g in order],
    "#files(reached)":    [nfiles_rea[g]          for g in order],
})

print(summary)
summary_path = os.path.join(OUTDIR, "summary_rate_by_group_macro.csv")
summary.to_csv(summary_path, index=False)
# ===== 색상 지정 =====
colors = {
  "Proposed Method": "#FFA500",
  "CQL": "#B0B0B0",
  "AWAC": "#B0B0B0",
  "SAC": "#B0B0B0"
}

# ===== 플롯: 충돌률 =====
plt.figure(figsize=(7,7))
bar_colors = [colors[g] for g in summary["Technique"]]  # 🔥 각 막대에 색상 적용
plt.bar(summary["Technique"], summary["Collision Rate (%)"], color=bar_colors)
plt.title("Collision Rate", fontsize=25)
plt.ylabel("% Percent", fontsize=25)
plt.ylim(0, 100)
plt.xticks(rotation=15, fontsize=18)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "collision_rate_bar_grouped_macro.png"), dpi=300, bbox_inches="tight")
plt.show()

# ===== 플롯: 도달률 =====
plt.figure(figsize=(7,7))
bar_colors = [colors[g] for g in summary["Technique"]]  # 🔥 같은 색상 매핑
plt.bar(summary["Technique"], summary["Reached Rate (%)"], color=bar_colors)
plt.title("Reached Rate", fontsize=25)
plt.ylabel("% Percent", fontsize=25)
plt.ylim(0, 100)
plt.xticks(rotation=15, fontsize=18)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "reached_rate_bar_grouped_macro.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ 저장 완료:")
print(" -", summary_path)
print(" -", os.path.join(OUTDIR, "collision_rate_bar_grouped_macro.png"))
print(" -", os.path.join(OUTDIR, "reached_rate_bar_grouped_macro.png"))
