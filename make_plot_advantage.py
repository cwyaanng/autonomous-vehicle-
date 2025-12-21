
"""
make_plot_advantage.py

각각의 기법이 학습 중에 어디서 얼마나 역전했는지 분석하기 위한 코드입니다. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


BASE_DIR = "result/final_result_town3/waypoint_ahead"

"""!!! 각 Town의 waypoint 개수로 꼭 변경해주세요!!!"""
MAX_WAYPOINT = 930


FIG_SIZE = (10, 10)
LINE_WIDTH = 6.0         
INDIV_LINE_WIDTH = 5.0    
FONT_SIZE_TITLE = 30
FONT_SIZE_LABEL = 35
FONT_SIZE_TICKS = 30
FONT_SIZE_LEGEND = 20
ALPHA_AREA = 0.2          


X_SCALE_POWER = 1.5

COLOR_MAP = {
    "CQL": "#36128b",    
    "AWAC": "#17becf",   
    "SAC": "#3ccd10",    
    "Proposed": "black"  
}

plt.rcParams['axes.unicode_minus'] = False

COMMON_PROGRESS = np.linspace(0, 100, 201)
WINDOW_SIZE = 100


file_patterns = {
    "Proposed": os.path.join(BASE_DIR, "*waypoint_ahead*proposed*.csv"),
    "AWAC":     os.path.join(BASE_DIR, "*waypoint_ahead*awac*.csv"),
    "CQL":      os.path.join(BASE_DIR, "*waypoint_ahead*cql*.csv"),
    "SAC":      os.path.join(BASE_DIR, "*waypoint_ahead*sac*.csv")
}


def load_tensorboard_csv(path):
    """CSV 로드 및 전처리 """
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except:
            df = pd.read_csv(path, encoding="cp949")

        cols = {c.lower(): c for c in df.columns}
        if "step" in cols and "value" in cols:
            df = df[[cols["step"], cols["value"]]].rename(
                columns={cols["step"]: "Step", cols["value"]: "Value"}
            )
        else:
            return None

        df = df.dropna().copy()
        df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        
        df = (
            df.dropna()
              .sort_values("Step")
              .drop_duplicates(subset="Step", keep="last")
              .reset_index(drop=True)
        )
        return df
    except Exception:
        return None

def analyze_comprehensive_report(x_axis, y_proposed, y_baseline, name):
    """
   
    """

    prop_abs_pct = (y_proposed / MAX_WAYPOINT) * 100
    base_abs_pct = (y_baseline / MAX_WAYPOINT) * 100

    p_mean, p_std = np.mean(prop_abs_pct), np.std(prop_abs_pct)
    b_mean, b_std = np.mean(base_abs_pct), np.std(base_abs_pct)
    
    diff = y_proposed - y_baseline
    winning_mask = diff > 0


    if np.all(winning_mask):
        lead_status = "Dominant (전 구간 성능 우위)"
    elif not np.any(winning_mask):
        max_gap = np.max(diff)
        lead_status = f"Inferior (최대 격차: {max_gap:.1f} WP)"
    else:
        first_win_idx = np.argmax(winning_mask)
        first_win_x = x_axis[first_win_idx]

        if first_win_idx == 0:
            lead_status = "Initial Lead (초기부터 우위 유지)"
        else:
            prev_gap = diff[first_win_idx - 1]
            curr_gap = diff[first_win_idx]
            lead_status = (
                f"Crossover Point: {first_win_x:.1f}% "
                f"(Gap shift: {prev_gap:.1f} -> +{curr_gap:.1f})"
            )

    report = [
        f">>> Comparative Analysis: Proposed vs {name}",
        f"1. Statistics (Normalized by {MAX_WAYPOINT} WPs)",
        f"   - Proposed : {p_mean:.2f}% (std: {p_std:.2f})",
        f"   - {name:<8} : {b_mean:.2f}% (std: {b_std:.2f})",
        f"2. Performance Lead",
        f"   - Status: {lead_status}",
        f"3. Deficit Intervals (Proposed < {name})"
    ]

    # 열세 구간 분석
    diff_percent = ((y_proposed - y_baseline) /
                    (np.abs(y_baseline) + 1e-6)) * 100
    neg_indices = np.where(diff_percent < 0)[0]

    if len(neg_indices) == 0:
        report.append("   - None (No deficit observed)")
        return "\n".join(report)

    # 구간 그룹화
    groups = np.split(
        neg_indices, np.where(np.diff(neg_indices) != 1)[0] + 1
    )

    for i, g in enumerate(groups):
        if len(g) == 0: continue
        start_x = x_axis[g[0]]
        end_x = x_axis[g[-1]]
        mean_loss = np.mean(diff_percent[g])

        if start_x <= 0.5:
            timing_str = f"~{end_x:.1f}% (Initial)"
        else:
            timing_str = f"{start_x:.1f}% ~ {end_x:.1f}%"

        report.append(
            f"   - Case {i+1}: {timing_str} / Avg Diff: {mean_loss:.1f}%"
        )

    return "\n".join(report)



method_means = {}
print(f"Loading data from: {BASE_DIR}...")

for method, pattern in file_patterns.items():

    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.basename(pattern))
    
    if not files:
        print(f"Warning: No files found for [{method}]")
        continue

    interpolated_runs = []
    for file in files:
        df = load_tensorboard_csv(file)
        if df is None: continue

        max_step = df['Step'].max()
        if max_step == 0: continue

        df['Progress'] = (df['Step'] / max_step) * 100

        df['Smoothed'] = df['Value'].rolling(
            window=WINDOW_SIZE, min_periods=1
        ).mean()
        
      
        interp_val = np.interp(
            COMMON_PROGRESS, df['Progress'], df['Smoothed']
        )
        interpolated_runs.append(interp_val)

    if interpolated_runs:
        method_means[method] = np.mean(
            np.vstack(interpolated_runs), axis=0
        )
        print(f" - [{method}] Loaded {len(files)} runs.")

if "Proposed" in method_means:
    proposed_curve = method_means["Proposed"]
    

    baseline_data = {
        m: curve for m, curve in method_means.items() if m != "Proposed"
    }
    baseline_curves = list(baseline_data.values())

    if baseline_curves:

        avg_baseline_curve = np.mean(
            np.vstack(baseline_curves), axis=0
        )


        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        

        print(analyze_comprehensive_report(
            COMMON_PROGRESS, proposed_curve,
            avg_baseline_curve, "Average"
        ))
        print("-" * 60)

        for base_name, base_curve in baseline_data.items():
            print(analyze_comprehensive_report(
                COMMON_PROGRESS, proposed_curve,
                base_curve, base_name
            ))
            print("-" * 20)
        print("=" * 60 + "\n")


        fig, ax = plt.subplots(figsize=FIG_SIZE)


        ax.set_xscale(
            'function',
            functions=(
                lambda x: x**X_SCALE_POWER,
                lambda x: x**(1 / X_SCALE_POWER)
            )
        )


        ax.axhline(0, color='gray', linewidth=2, linestyle='--')
        
        for base_name, base_curve in baseline_data.items():
 
            indiv_adv_pct = (
                proposed_curve - base_curve
            ) / MAX_WAYPOINT * 100
            
            color = COLOR_MAP.get(base_name, 'gray')

      
            ax.plot(
                COMMON_PROGRESS, indiv_adv_pct,
                color=color,
                linewidth=INDIV_LINE_WIDTH,
                alpha=0.9,
                label=f'Proposed vs {base_name}'
            )

 
            ax.fill_between(
                COMMON_PROGRESS, 0, indiv_adv_pct,
                where=(indiv_adv_pct >= 0),
                color=color, alpha=0.05, interpolate=True
            )
            ax.fill_between(
                COMMON_PROGRESS, 0, indiv_adv_pct,
                where=(indiv_adv_pct < 0),
                color=color, alpha=0.05, interpolate=True
            )

  
        ax.set_title(
            "Performance Advantage over Baselines",
            fontsize=FONT_SIZE_TITLE,
            pad=10
        )
        ax.set_xlabel(
            "Progress (%)",
            fontsize=FONT_SIZE_LABEL,
            labelpad=15
        )
        ax.set_ylabel(
            "Advantage (% of Total Track)",
            fontsize=FONT_SIZE_LABEL,
            labelpad=20
        )

 
        ticks = [0, 25, 50, 75, 100]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{t}%" for t in ticks],
            fontsize=FONT_SIZE_TICKS
        )

        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
        ax.set_xlim(0, 100)

  
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="upper center", ncol=2,
            frameon=False, fontsize=FONT_SIZE_LEGEND,
            bbox_to_anchor=(0.5, 1.20)
        )

    
        output_filename = 'rl_results_advantage_no_average_town3.png'
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Graph saved to: {output_filename}")
        plt.show()
else:
    print("[Error] 'Proposed' data not found. Cannot calculate advantage.")