import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 1. CONFIG & STYLE
# ==========================================
BASE_DIR = "result/final_result_town3/waypoint_ahead"

# 스타일 상수
FIG_SIZE = (10, 10)
LINE_WIDTH = 6.0         
INDIV_LINE_WIDTH = 5.0    
FONT_SIZE_TITLE = 30
FONT_SIZE_LABEL = 35
FONT_SIZE_TICKS = 30
FONT_SIZE_LEGEND = 20
ALPHA_AREA = 0.2          # (현재 평균 영역 안 쓰지만 남겨둠)

# 최대 Waypoint 기준값 (전체 경로 길이)
MAX_WAYPOINT = 930

# X축 왜곡 강도 (1.0=선형, 2.0=제곱 등)
X_SCALE_POWER = 1.5

COLOR_MAP = {
    "CQL": "#36128b",    # 보라
    "AWAC": "#17becf",   # 청록
    "SAC": "#3ccd10",    # 초록
    "Proposed": "black"
}

plt.rcParams['axes.unicode_minus'] = False

# X축 (0~100%)
COMMON_PROGRESS = np.linspace(0, 100, 201)
WINDOW_SIZE = 100

file_patterns = {
    "Proposed": os.path.join(BASE_DIR, "waypoint_ahaed*.csv"),
    "AWAC": os.path.join(BASE_DIR, "waypoint_ahead_awac*.csv"),
    "CQL": os.path.join(BASE_DIR, "waypoint_ahead_cql*.csv"),
    "SAC": os.path.join(BASE_DIR, "waypoint_ahead_sac*.csv")
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_tensorboard_csv(path):
    """TensorBoard CSV 형식 (step, value) 로드 & 정제."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
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


# ==========================================
# 3. [최종] 리포트 분석 함수
# ==========================================
def analyze_comprehensive_report(x_axis, y_proposed, y_baseline, name):
    # 1. 절대 성능 통계 (Max Waypoint 기준)
    prop_abs_pct = (y_proposed / MAX_WAYPOINT) * 100
    base_abs_pct = (y_baseline / MAX_WAYPOINT) * 100

    p_mean, p_std = np.mean(prop_abs_pct), np.std(prop_abs_pct)
    b_mean, b_std = np.mean(base_abs_pct), np.std(base_abs_pct)

    # 2. 추월(Overtake) 시점 분석
    diff = y_proposed - y_baseline
    winning_mask = diff > 0

    if np.all(winning_mask):
        overtake_msg = "👑 시작부터 끝까지 압도적 우위 (Always Win)"
    elif not np.any(winning_mask):
        max_gap = np.max(diff)
        overtake_msg = f"💀 추월 실패 (최선일 때도 {max_gap:.1f} Waypoint 뒤쳐짐)"
    else:
        first_win_idx = np.argmax(winning_mask)
        first_win_x = x_axis[first_win_idx]

        if first_win_idx == 0:
            overtake_msg = "🚀 시작부터 리드 유지 (Initial Lead)"
        else:
            prev_gap = diff[first_win_idx - 1]
            curr_gap = diff[first_win_idx]
            overtake_msg = (
                f"🔥 Progress {first_win_x:.1f}% 에서 역전 성공! "
                f"(Gap: {prev_gap:.1f} ➔ +{curr_gap:.1f})"
            )

    # 3. 리포트 텍스트 구성
    report = [
        f"Proposed vs {name} Analysis Result:",
        f"   📊 [Absolute Stats] (Max {MAX_WAYPOINT} WP = 100%)",
        f"       🔴 Proposed : {p_mean:.2f}% ± {p_std:.2f}% (Avg ± Std)",
        f"       🔵 {name:<8} : {b_mean:.2f}% ± {b_std:.2f}%",
        f"   ⚔️ [Critical Moment]",
        f"       {overtake_msg}",
        f"   📉 [Loss Intervals] 상대적 열세 구간 상세"
    ]

    # 4. 상대적 열세(Loss) 구간 상세 분석
    diff_percent = ((y_proposed - y_baseline) /
                    (np.abs(y_baseline) + 1e-6)) * 100
    neg_indices = np.where(diff_percent < 0)[0]

    if len(neg_indices) == 0:
        report.append("       ✨ No Loss Intervals (무결점)")
        return "\n".join(report)

    groups = np.split(
        neg_indices, np.where(np.diff(neg_indices) != 1)[0] + 1
    )

    for i, g in enumerate(groups):
        if len(g) == 0:
            continue
        start_x = x_axis[g[0]]
        end_x = x_axis[g[-1]]
        mean_loss = np.mean(diff_percent[g])

        if start_x <= 0.5:
            timing_str = f"🏁 초반 열세 (~{end_x:.1f}% 까지)"
        else:
            timing_str = (
                f"⚠️ {start_x:.1f}%에서 역전당함 -> {end_x:.1f}%에서 회복"
            )

        report.append(
            f"       [{i+1}] {timing_str} (평균 차이: {mean_loss:.1f}%)"
        )

    return "\n".join(report)


# ==========================================
# 4. DATA PROCESSING
# ==========================================
method_means = {}
print(f"Loading data from: {BASE_DIR}")

for method, pattern in file_patterns.items():
    files = glob.glob(pattern)
    if not files:
        # 혹시 상대 경로 패턴만 있는 경우 대비
        files = glob.glob(os.path.basename(pattern))
    if not files:
        continue

    interpolated_runs = []
    for file in files:
        df = load_tensorboard_csv(file)
        if df is None:
            continue

        max_step = df['Step'].max()
        if max_step == 0:
            continue

        # Step → Progress (%)
        df['Progress'] = (df['Step'] / max_step) * 100
        # Rolling smoothing
        df['Smoothed'] = df['Value'].rolling(
            window=WINDOW_SIZE, min_periods=1
        ).mean()

        # 공통 X축(COMMON_PROGRESS)에 맞춰 보간
        interp_val = np.interp(
            COMMON_PROGRESS, df['Progress'], df['Smoothed']
        )
        interpolated_runs.append(interp_val)

    if interpolated_runs:
        method_means[method] = np.mean(
            np.vstack(interpolated_runs), axis=0
        )

# ==========================================
# 5. ADVANTAGE 계산 + 플로팅 (평균 곡선은 그래프에서 제거)
# ==========================================
if "Proposed" in method_means:
    proposed_curve = method_means["Proposed"]
    baseline_data = {
        m: curve for m, curve in method_means.items() if m != "Proposed"
    }
    baseline_curves = list(baseline_data.values())

    if baseline_curves:
        # 평균 곡선은 리포트용으로만 사용 (그래프에는 안 그림)
        avg_baseline_curve = np.mean(
            np.vstack(baseline_curves), axis=0
        )

        # --- 리포트 출력 ---
        print("\n" + "=" * 60)
        print("📢 최종 종합 분석 리포트 (Stats + Overtake + Loss)")
        print("=" * 60)
        print(
            analyze_comprehensive_report(
                COMMON_PROGRESS, proposed_curve,
                avg_baseline_curve, "Average"
            )
        )
        print("-" * 60)
        for base_name, base_curve in baseline_data.items():
            print(
                analyze_comprehensive_report(
                    COMMON_PROGRESS, proposed_curve,
                    base_curve, base_name
                )
            )
            print("-" * 20)
        print("=" * 60 + "\n")

        # --- 그래프 그리기 (평균 곡선 없음, 개별만) ---
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        # X축 스케일 변환 (0~25% 구간 조금 넓게 보는 효과)
        ax.set_xscale(
            'function',
            functions=(
                lambda x: x**X_SCALE_POWER,
                lambda x: x**(1 / X_SCALE_POWER)
            )
        )

        ax.axhline(0, color='gray', linewidth=2, linestyle='--')

        # 개별 Baseline 대비 Advantage (실선 + 매우 옅은 영역)
        for base_name, base_curve in baseline_data.items():
            indiv_adv_pct = (
                proposed_curve - base_curve
            ) / MAX_WAYPOINT * 100
            color = COLOR_MAP.get(base_name, 'gray')

            # 실선
            ax.plot(
                COMMON_PROGRESS, indiv_adv_pct,
                color=color,
                linewidth=INDIV_LINE_WIDTH,
                alpha=0.9,
                label=f'Proposed vs {base_name}'
            )

            # 아주 옅은 영역 (서로 겹쳐도 평균을 안 그리니 덜 복잡함)
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

        # 타이틀 & 라벨
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

        # 축 설정
        ticks = [0, 25, 50, 75, 100]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{t}%" for t in ticks],
            fontsize=FONT_SIZE_TICKS
        )

        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
        ax.set_xlim(0, 100)

        # 범례
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="upper center", ncol=2,
            frameon=False, fontsize=FONT_SIZE_LEGEND,
            bbox_to_anchor=(0.5, 1.20)
        )

        output_filename = 'rl_results_advantage_no_average_town3.png'
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Graph generated: {output_filename}")
        plt.show()

    else:
        print("비교할 Baseline 데이터가 없습니다.")
else:
    print("Proposed(제안 기법) 데이터가 없습니다.")
