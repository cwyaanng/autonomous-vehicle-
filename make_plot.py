# -*- coding: utf-8 -*-
# make_plot_progress_banded_separate.py (FINAL MERGED VERSION)
# - Waypoint/Return plot + Overtake stats
# - Final metrics (waypoint-last, collision rate, reached rate) mean/std 출력 포함

import re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
ROOT_DIR   = "result/final_result_town3"
OUTPUT_DIR = "result/final_result_town3/plots"
SMOOTH_WINDOW = 100
X_GAMMA = 1.5

WAYPOINT_MIN = 0.0
WAYPOINT_MAX = 930.0

X_GRID = np.linspace(0.0, 100.0, 201)

COLOR_MAP = {
    "Proposed": "#d95f02",
    "CQL": "#36128b",
    "AWAC": "#17becf",
    "SAC": "#3ccd10",
}

LINEWIDTH_MEAN = 4.0
LINEWIDTH_RUN  = 1.0
ALPHA_RUN      = 0.12
ALPHA_BAND     = 0.10

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -------------------- HELPERS --------------------
def moving_average(y, window=100):
    s = pd.Series(y, dtype=float)
    return s.rolling(window=window, min_periods=max(1, window//5)).mean().to_numpy()

def to_progress_percent(step):
    x = np.asarray(step, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 100.0

def warp_percent(pct, gamma=2.0):
    p = np.clip(np.asarray(pct, dtype=float), 0.0, 100.0)
    return (p / 100.0) ** gamma * 100.0

def bin_to_grid(progress, values, xgrid):
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
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    if "step" in cols and "value" in cols:
        df = df[[cols["step"], cols["value"]]].rename(columns={cols["step"]: "Step", cols["value"]: "Value"})
    else:
        raise ValueError(f"Unrecognized columns in {path}: {list(df.columns)}")

    df = df.dropna().copy()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna().sort_values("Step").drop_duplicates(subset="Step", keep="last").reset_index(drop=True)
    return df

def normalize_waypoint_fixed(y, lo=WAYPOINT_MIN, hi=WAYPOINT_MAX):
    y = np.asarray(y, dtype=float)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)

def prepare_run(path, algo_label, value_type):
    df = load_tensorboard_csv(path)
    sm = moving_average(df["Value"].to_numpy(), window=SMOOTH_WINDOW)
    pct  = to_progress_percent(df["Step"].to_numpy())
    pctw = warp_percent(pct, gamma=X_GAMMA)
    if value_type == "waypoint":
        sm = normalize_waypoint_fixed(sm)
    y_grid = bin_to_grid(pctw, sm, X_GRID)
    return (algo_label, value_type, y_grid, pctw, sm)

# -------------------- SCALAR (collision/reached) --------------------
def prepare_scalar_run(path, algo_label):
    df = load_tensorboard_csv(path)
    vals = df["Value"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    # 🔥 변경: 마지막 값이 아니라 전체 평균을 사용
    mean_val = float(np.mean(vals))
    return (algo_label, mean_val)


def aggregate_scalar_runs(runs):
    by_algo = {}
    for algo, v in runs:
        by_algo.setdefault(algo, []).append(v)
    return {algo: (float(np.mean(vs)), float(np.std(vs))) for algo, vs in by_algo.items()}

def compute_final_waypoint_stats(runs, algos):
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


# -------------------- FILE DISCOVERY --------------------
root = Path(ROOT_DIR)

# Proposed (Exp1~6)
reward_exp   = sorted(glob.glob(str(root / "episode_reward_*차실험*.csv")))
waypoint_exp = sorted(glob.glob(str(root / "waypoint_ahead/waypoint*_*차실험*.csv")))

# Baselines
baseline_reward = sorted(set(
    glob.glob(str(root / "episode_return_*.csv")) +
    glob.glob(str(root / "episode_reward_*sac*.csv"))
))
baseline_waypoint = sorted(set(
    glob.glob(str(root / "*waypoint_ahead/waypoint*_*cql*.csv"))  +
    glob.glob(str(root / "*waypoint_ahead/waypoint*_*awac*.csv")) +
    glob.glob(str(root / "*waypoint_ahead/waypoint*_*mcac*.csv")) +
    glob.glob(str(root / "*waypoint_ahead/waypoint*_*sac*.csv"))
))

# Scalar metrics
collision_exp = sorted(glob.glob(str(root / "done_collided/done_collided_*차실험*.csv")))
reached_exp   = sorted(glob.glob(str(root / "done_reached/done_reached_*차실험*.csv")))

baseline_collision = sorted(set(
    glob.glob(str(root / "done_collided/done_collided_*cql*.csv")) +
    glob.glob(str(root / "done_collided/done_collided_*awac*.csv")) +
    glob.glob(str(root / "done_collided/done_collided_*sac*.csv"))
))
baseline_reached = sorted(set(
    glob.glob(str(root / "done_reached/done_reached_*cql*.csv")) +
    glob.glob(str(root / "done_reached/done_reached_*awac*.csv")) +
    glob.glob(str(root / "done_reached/done_reached_*sac*.csv"))
))

exp_re = re.compile(r"(\d)\s*차\s*실\s*험")
def is_exp_run(path): return exp_re.search(str(path)) is not None

def infer_baseline_label(fpath):
    n = Path(fpath).stem.lower()
    if "cql" in n:  return "CQL"
    if "awac" in n: return "AWAC"
    if "sac" in n:  return "SAC"
    return None


# -------------------- LOAD & PREP --------------------
runs_return, runs_waypt = [], []
scalar_collision_runs, scalar_reached_runs = [], []

# Proposed
for f in reward_exp:
    if is_exp_run(f):
        runs_return.append(prepare_run(f, "Proposed", "return"))
for f in waypoint_exp:
    if is_exp_run(f):
        runs_waypt.append(prepare_run(f, "Proposed", "waypoint"))

for f in collision_exp:
    if is_exp_run(f):
        r = prepare_scalar_run(f, "Proposed")
        if r: scalar_collision_runs.append(r)
for f in reached_exp:
    if is_exp_run(f):
        r = prepare_scalar_run(f, "Proposed")
        if r: scalar_reached_runs.append(r)

# Baselines
for f in baseline_reward:
    label = infer_baseline_label(f)
    if label:
        runs_return.append(prepare_run(f, label, "return"))
for f in baseline_waypoint:
    label = infer_baseline_label(f)
    if label:
        runs_waypt.append(prepare_run(f, label, "waypoint"))

for f in baseline_collision:
    label = infer_baseline_label(f)
    if label:
        r = prepare_scalar_run(f, label)
        if r: scalar_collision_runs.append(r)

for f in baseline_reached:
    label = infer_baseline_label(f)
    if label:
        r = prepare_scalar_run(f, label)
        if r: scalar_reached_runs.append(r)


# -------------------- AGGREGATION --------------------
def aggregate_runs(runs, target_algo):
    Ys = [y for a, t, y, *_ in runs if a == target_algo]
    if not Ys:
        return None
    Y = np.vstack(Ys)
    return np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)

algos_return = sorted(set(a for a, *_ in runs_return))
algos_waypt  = sorted(set(a for a, *_ in runs_waypt))


# -------------------- PLOT --------------------
def plot_and_save_single(runs, algos, ylabel, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for algo in algos:
        agg = aggregate_runs(runs, algo)
        if agg:
            mean, std = agg
            valid = np.isfinite(mean)
            if np.any(valid):
                last = np.where(valid)[0][-1]
                label = f"{algo}({mean[last]:.2f} ± {std[last]:.2f})"
            else:
                label = algo

            ax.plot(X_GRID, mean, linewidth=LINEWIDTH_MEAN,
                    label=label, color=COLOR_MAP.get(algo, None))
            ax.fill_between(X_GRID, mean-std, mean+std,
                            alpha=ALPHA_BAND, linewidth=0,
                            color=COLOR_MAP.get(algo, None))

    ax.set_title(title, fontsize=30)
    ax.set_xlabel("Progress percent", fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_xticks(warp_percent([0, 25, 50, 75, 100], X_GAMMA))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
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
              loc="upper center", ncol=2,
              frameon=False, bbox_to_anchor=(0.5, 1.20))

    fig.savefig(Path(OUTPUT_DIR)/fname, dpi=300, bbox_inches="tight")
    fig.savefig(Path(OUTPUT_DIR)/fname.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def print_overtake_stats(runs, algos, ref_algo="Proposed"):
    ref_agg = aggregate_runs(runs, ref_algo)
    if ref_agg is None:
        return
    ref_mean, _ = ref_agg

    for algo in algos:
        if algo == ref_algo:
            continue

        agg = aggregate_runs(runs, algo)
        if agg is None: continue
        mean, _ = agg

        valid = np.isfinite(ref_mean) & np.isfinite(mean)
        if not np.any(valid): continue

        ref_better = ref_mean[valid] > mean[valid]
        other_better = mean[valid] > ref_mean[valid]

        print(f"[overtake] {ref_algo} vs {algo}:")
        print(f"  - {ref_algo} better: {np.mean(ref_better)*100:.1f}%")
        print(f"  - {algo} better: {np.mean(other_better)*100:.1f}%")


# -------------------- RUN --------------------
plot_and_save_single(
    runs=runs_waypt,
    algos=algos_waypt,
    ylabel="Waypoint progress (0: start, 1: goal)",
    title="Final Waypoint Reached(Town04)",
    fname="rl_results_waypoint.png",
    ylim=(0.0, 1.1)
)

print_overtake_stats(runs_waypt, algos_waypt)

plot_and_save_single(
    runs=runs_return,
    algos=algos_return,
    ylabel="Return",
    title="Episode Return",
    fname="rl_results_return.png"
)

# -------------------- SUMMARY STATS --------------------
print("\n===== FINAL WAYPOINT MEAN / STD =====")
for algo, (m, s) in compute_final_waypoint_stats(runs_waypt, algos_waypt).items():
    print(f"{algo:10s} : {m:.3f} ± {s:.3f}")

print("\n===== COLLISION RATE MEAN / STD =====")
for algo, (m, s) in aggregate_scalar_runs(scalar_collision_runs).items():
    print(f"{algo:10s} : {m:.3f} ± {s:.3f}")

print("\n===== REACHED RATE MEAN / STD =====")
for algo, (m, s) in aggregate_scalar_runs(scalar_reached_runs).items():
    print(f"{algo:10s} : {m:.3f} ± {s:.3f}")
