# -*- coding: utf-8 -*-
# make_plot_progress_banded_separate.py (FINAL CLEAN VERSION)
# - Target Algos: Proposed, CQL, AWAC, SAC (MCAC Removed)
# - Metrics: Waypoint (Plot), Collision/Reached (Stats Only)
# - Folder Structure: result/TownXX/metric_folder/*.csv

import re, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
ROOT_DIR   = "result/final_result_town3"       
OUTPUT_DIR = os.path.join(ROOT_DIR, "plots")

# 그래프 설정
SMOOTH_WINDOW = 100
X_GAMMA = 1.5
X_GRID = np.linspace(0.0, 100.0, 201)

# Waypoint 정규화 범위 (Town03: ~930, Town04: 맵 길이에 맞춰 수정 필요)
WAYPOINT_MIN = 0.0
WAYPOINT_MAX = 930.0 

# 알고리즘별 색상 (MCAC 제거됨)
COLOR_MAP = {
    "Proposed": "#d95f02", # Orange
    "CQL":      "#36128b", # Purple
    "AWAC":     "#17becf", # Cyan
    "SAC":      "#3ccd10", # Green
}

LINEWIDTH_MEAN = 4.0
ALPHA_BAND     = 0.10

# 결과 저장 폴더 생성
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -------------------- DATA PROCESSING HELPERS --------------------
def moving_average(y, window=100):
    """데이터 스무딩 (이동 평균)"""
    s = pd.Series(y, dtype=float)
    return s.rolling(window=window, min_periods=max(1, window//5)).mean().to_numpy()

def to_progress_percent(step):
    """Step을 0~100% 진행률로 변환"""
    x = np.asarray(step, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 100.0

def warp_percent(pct, gamma=2.0):
    """X축(진행률) 왜곡 보정 (학습 후반부를 더 넓게 보기 위함)"""
    p = np.clip(np.asarray(pct, dtype=float), 0.0, 100.0)
    return (p / 100.0) ** gamma * 100.0

def bin_to_grid(progress, values, xgrid):
    """서로 다른 길이의 실험 데이터를 공통 X축 Grid에 매핑"""
    step = xgrid[1] - xgrid[0]
    bins = np.concatenate([[xgrid[0]-step/2], (xgrid[:-1]+xgrid[1:])/2, [xgrid[-1]+step/2]])
    idx = np.digitize(progress, bins) - 1
    out = np.full_like(xgrid, np.nan, dtype=float)
    for i in range(len(xgrid)):
        sel = values[idx == i]
        if sel.size:
            out[i] = np.nanmean(sel)
    return out

def load_tensorboard_csv(path):
    """CSV 파일 로드 및 전처리"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except:
        try:
            df = pd.read_csv(path, encoding="cp949")
        except:
            return pd.DataFrame() # 로드 실패 시 빈 DF 반환

    cols = {c.lower(): c for c in df.columns}
    if "step" in cols and "value" in cols:
        df = df[[cols["step"], cols["value"]]].rename(columns={cols["step"]: "Step", cols["value"]: "Value"})
    else:
        return pd.DataFrame()

    df = df.dropna().copy()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna().sort_values("Step").drop_duplicates(subset="Step", keep="last").reset_index(drop=True)
    return df

def normalize_waypoint_fixed(y, lo=WAYPOINT_MIN, hi=WAYPOINT_MAX):
    """Waypoint Index -> 0.0~1.0 정규화"""
    y = np.asarray(y, dtype=float)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)

# -------------------- RUN PREPARATION --------------------
def prepare_run(path, algo_label, value_type):
    """단일 실험 파일 처리"""
    df = load_tensorboard_csv(path)
    if df.empty: return None

    sm = moving_average(df["Value"].to_numpy(), window=SMOOTH_WINDOW)
    pct  = to_progress_percent(df["Step"].to_numpy())
    pctw = warp_percent(pct, gamma=X_GAMMA)
    
    if value_type == "waypoint":
        sm = normalize_waypoint_fixed(sm)
        
    y_grid = bin_to_grid(pctw, sm, X_GRID)
    return (algo_label, value_type, y_grid, pctw, sm)

def prepare_scalar_run(path, algo_label):
    """단일 스칼라 값(충돌/완주) 처리"""
    df = load_tensorboard_csv(path)
    if df.empty: return None
    
    vals = df["Value"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return None

    mean_val = float(np.mean(vals)) # 전체 평균 사용
    return (algo_label, mean_val)

def aggregate_scalar_runs(runs):
    """스칼라 값 통계 (Mean ± Std)"""
    by_algo = {}
    for algo, v in runs:
        by_algo.setdefault(algo, []).append(v)
    return {algo: (float(np.mean(vs)), float(np.std(vs))) for algo, vs in by_algo.items()}

def compute_final_waypoint_stats(runs, algos):
    """Waypoint 그래프의 마지막 지점 통계"""
    results = {}
    for algo in algos:
        agg = aggregate_runs(runs, algo)
        if agg:
            mean, std = agg
            valid = np.isfinite(mean)
            if np.any(valid):
                last = np.where(valid)[0][-1]
                results[algo] = (float(mean[last]), float(std[last]))
    return results

def aggregate_runs(runs, target_algo):
    """같은 알고리즘의 여러 Seed 평균 계산"""
    Ys = [y for a, t, y, *_ in runs if a == target_algo]
    if not Ys: return None
    Y = np.vstack(Ys)
    return np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)


# -------------------- FILE LOADING --------------------
root = Path(ROOT_DIR)
exp_re = re.compile(r"(\d+)\s*차") # '{n}차' 패턴 확인용

def get_algo_name(filename):
    """파일명에서 알고리즘 이름 추출 (MCAC 제외)"""
    name = filename.lower()
    if "proposed" in name: return "Proposed"
    if "cql" in name:      return "CQL"
    if "awac" in name:     return "AWAC"
    if "sac" in name:      return "SAC"
    return None 

def is_valid_run(path):
    """파일명에 '{n}차'가 포함되어 있는지 확인"""
    return exp_re.search(str(path)) is not None

# 파일 경로 수집
waypoint_files  = sorted(glob.glob(str(root / "waypoint_ahead" / "*.csv")))
collision_files = sorted(glob.glob(str(root / "done_collided" / "*.csv")))
reached_files   = sorted(glob.glob(str(root / "done_reached" / "*.csv")))

runs_waypt = []
scalar_collision_runs = []
scalar_reached_runs = []

print(f"Loading data from: {ROOT_DIR} ...")

# 1. Waypoint Data Load
for f in waypoint_files:
    if not is_valid_run(f): continue
    label = get_algo_name(Path(f).name)
    if label:
        res = prepare_run(f, label, "waypoint")
        if res: runs_waypt.append(res)

# 2. Collision Data Load
for f in collision_files:
    if not is_valid_run(f): continue
    label = get_algo_name(Path(f).name)
    if label:
        res = prepare_scalar_run(f, label)
        if res: scalar_collision_runs.append(res)

# 3. Reached Data Load
for f in reached_files:
    if not is_valid_run(f): continue
    label = get_algo_name(Path(f).name)
    if label:
        res = prepare_scalar_run(f, label)
        if res: scalar_reached_runs.append(res)

algos_waypt = sorted(set(a for a, *_ in runs_waypt))

print(f" - Waypoint files loaded: {len(runs_waypt)}")
print(f" - Collision files loaded: {len(scalar_collision_runs)}")
print(f" - Reached files loaded: {len(scalar_reached_runs)}")
print(f" - Algorithms found: {algos_waypt}")


# -------------------- PLOT GENERATION --------------------
def plot_and_save_single(runs, algos, ylabel, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for algo in algos:
        agg = aggregate_runs(runs, algo)
        if agg:
            mean, std = agg
            valid = np.isfinite(mean)
            if np.any(valid):
                last = np.where(valid)[0][-1]
                # 범례에 최종 값 표기 (예: Proposed(0.95 ± 0.02))
                label = f"{algo}({mean[last]:.2f} ± {std[last]:.2f})"
            else:
                label = algo

            color = COLOR_MAP.get(algo, "gray")
            
            # 평균선
            ax.plot(X_GRID, mean, linewidth=LINEWIDTH_MEAN,
                    label=label, color=color)
            # 표준편차 밴드
            ax.fill_between(X_GRID, mean-std, mean+std,
                            alpha=ALPHA_BAND, linewidth=0, color=color)

    
    ax.set_title(title, fontsize=35, pad=10)
    ax.set_xlabel("Progress percent", fontsize=35, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=35, labelpad=10)
 
    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.set_xticks(warp_percent([0, 25, 50, 75, 100], X_GAMMA))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=30)
    ax.set_xlim(0, X_GRID[-1])

  
    if ylim:
        ax.set_ylim(*ylim)
        yticks = np.arange(ylim[0], ylim[1] + 1e-9, 0.2)
        yticks = yticks[yticks != 0.0]
        ax.set_yticks(yticks)


    if "waypoint" in fname.lower():
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5)

    handles, labels = ax.get_legend_handles_labels()
    
   
    ax.legend(handles, labels, 
              loc="upper center", 
              ncol=2, 
              frameon=False, 
              bbox_to_anchor=(0.5, 1.25), 
              fontsize=24)

    save_path = Path(OUTPUT_DIR) / fname
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def print_overtake_stats(runs, algos, ref_algo="Proposed"):
    ref_agg = aggregate_runs(runs, ref_algo)
    if ref_agg is None: return
    ref_mean, _ = ref_agg

    print(f"\n===== OVERTAKE STATS (Reference: {ref_algo}) =====")
    for algo in algos:
        if algo == ref_algo: continue
        agg = aggregate_runs(runs, algo)
        if agg is None: continue
        mean, _ = agg
        valid = np.isfinite(ref_mean) & np.isfinite(mean)
        if not np.any(valid): continue

        ref_better = ref_mean[valid] > mean[valid]
        print(f" vs {algo:10s} : {ref_algo} Higher {np.mean(ref_better)*100:.1f}% of time")


# -------------------- EXECUTE --------------------

# 1. Waypoint Plot 그리기
if runs_waypt:
    plot_and_save_single(
        runs=runs_waypt,
        algos=algos_waypt,
        ylabel="Waypoint progress",
        title="Final Waypoint Reached",
        fname="rl_results_waypoint.png",
        ylim=(0.0, 1.1)
    )
    print_overtake_stats(runs_waypt, algos_waypt)
else:
    print("No waypoint data found!")

# 2. 통계 출력 (Waypoint, Collision, Reached)
print("\n===== FINAL WAYPOINT MEAN / STD =====")
for algo, (m, s) in compute_final_waypoint_stats(runs_waypt, algos_waypt).items():
    print(f"{algo:10s} : {m:.3f} ± {s:.3f}")

