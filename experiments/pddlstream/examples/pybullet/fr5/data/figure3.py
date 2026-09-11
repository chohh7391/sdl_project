#!/usr/bin/env python3
"""
Figure 3: Planning Time Distribution and Success Rate by Task
CSV 데이터로부터 그래프를 생성합니다.

사용법:
    python3 figure3_plot.py

CSV 파일 경로를 아래 DATA_PATHS에서 수정하세요.
각 CSV는 다음 컬럼을 포함해야 합니다:
    trial, success, planning_time_sec, ...
"""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# CSV 파일 경로 설정 (여기만 수정하면 됩니다)
# ============================================================
DATA_PATHS = {
    # Transfer
    'transfer_cpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/transfer_cpu_6dof.csv',
    'transfer_cpu_7dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/transfer_cpu_7dof.csv',
    'transfer_gpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/transfer_gpu_6dof.csv',
    # Move
    'move_cpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/move_cpu_6dof.csv',
    'move_gpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/move_gpu_6dof.csv',
    # Stir
    'stir_cpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/stir_cpu_6dof.csv',
    'stir_gpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/data/stir_gpu_6dof.csv',
}

# 출력 경로
OUTPUT_PNG = '/home/home/sdl_ws/src/sdl_project/experiments/figures/figure3_planning_time.png'
OUTPUT_PDF = '/home/home/sdl_ws/src/sdl_project/experiments/figures/figure3_planning_time.pdf'

# Timeout 기준 (초)
TIMEOUT_THRESHOLD = 119.0

# 총 trial 수
N_TRIALS = 30


# ============================================================
# 데이터 로드
# ============================================================
def load_csv(filepath):
    """CSV를 로드하여 success_times, fail_fast_times, fail_timeout_times로 분리"""
    success_times = []
    fail_fast_times = []
    fail_timeout_times = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row['planning_time_sec'])
            s = int(row['success'])
            if s == 1:
                success_times.append(t)
            else:
                if t >= TIMEOUT_THRESHOLD:
                    fail_timeout_times.append(t)
                else:
                    fail_fast_times.append(t)

    return success_times, fail_fast_times, fail_timeout_times


# ============================================================
# 그래프 그리기
# ============================================================
def plot_box_strip(ax, pos, success_data, fail_fast_data, color):
    """메인 축에 box + strip plot 그리기 (성공 trial 분포 + 빠른 실패)"""
    if len(success_data) > 1:
        bp = ax.boxplot([success_data], positions=[pos], widths=0.5,
                        patch_artist=True, showfliers=False, zorder=2)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.25)
        bp['boxes'][0].set_edgecolor(color)
        bp['medians'][0].set_color(color)
        bp['medians'][0].set_linewidth(2)
        for w in bp['whiskers']:
            w.set_color(color)
            w.set_linewidth(1.2)
        for c in bp['caps']:
            c.set_color(color)
            c.set_linewidth(1.2)
    elif len(success_data) == 1:
        ax.scatter([pos], success_data, color=color, alpha=0.7, s=40, zorder=3,
                   edgecolors='white', linewidths=0.5)

    # 성공 trial 점
    if success_data:
        jitter = np.random.uniform(-0.15, 0.15, len(success_data))
        ax.scatter([pos + j for j in jitter], success_data,
                   color=color, alpha=0.7, s=28, zorder=3,
                   edgecolors='white', linewidths=0.5)

    # 빠른 실패 점 (X 마커)
    for f in fail_fast_data:
        j = np.random.uniform(-0.15, 0.15)
        ax.scatter(pos + j, f, color=FAIL_COLOR, marker='x', s=40,
                   zorder=4, linewidths=1.5)


def plot_timeout_bar(ax, pos, n_timeout):
    """상단 축에 timeout 횟수 막대 그리기"""
    if n_timeout > 0:
        ax.bar(pos, n_timeout, width=0.5, color=TIMEOUT_COLOR, alpha=0.6,
               edgecolor=TIMEOUT_COLOR, linewidth=0.8, zorder=2)
        ax.text(pos, n_timeout + 0.2, str(n_timeout), ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=TIMEOUT_COLOR)


def add_success_rate(ax, pos, n_success, n_total, color, ylim_top):
    """성공률 텍스트 표시"""
    rate = n_success / n_total * 100
    ax.text(pos, ylim_top * 0.95, f'{rate:.1f}%', ha='center', va='top',
            fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))


# 색상 설정
CPU_COLOR = '#E07B54'
CPU_7DOF_COLOR = '#C9A84C'
GPU_COLOR = '#4A90D9'
TIMEOUT_COLOR = '#D94040'
FAIL_COLOR = '#D94040'


def main():
    np.random.seed(42)

    # 데이터 로드
    data = {}
    for key, path in DATA_PATHS.items():
        try:
            data[key] = load_csv(path)
            print(f"Loaded: {key} ({path})")
        except FileNotFoundError:
            print(f"WARNING: File not found: {path} — using empty data for '{key}'")
            data[key] = ([], [], [])

    # 통계 출력
    print("\n" + "=" * 60)
    print("데이터 요약")
    print("=" * 60)
    for key, (success, fail_fast, fail_timeout) in data.items():
        total = len(success) + len(fail_fast) + len(fail_timeout)
        rate = len(success) / total * 100 if total > 0 else 0
        mean_t = np.mean(success) if success else float('nan')
        std_t = np.std(success) if len(success) > 1 else float('nan')
        print(f"  {key:25s} | success: {len(success):2d}/{total:2d} ({rate:5.1f}%) | "
              f"time: {mean_t:6.2f} ± {std_t:6.2f}s | timeout: {len(fail_timeout)}")
    print("=" * 60)

    # Figure 설정: 상단(timeout) + 하단(planning time)
    fig, (ax_top, ax_main) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.08},
        sharex=True
    )

    # 그룹 위치
    positions = {
        'move_cpu_6dof': 1, 'move_gpu_6dof': 2,
        'transfer_cpu_6dof': 4, 'transfer_cpu_7dof': 5, 'transfer_gpu_6dof': 6,
        'stir_cpu_6dof': 8, 'stir_gpu_6dof': 9,
    }

    colors = {
        'move_cpu_6dof': CPU_COLOR, 'move_gpu_6dof': GPU_COLOR,
        'transfer_cpu_6dof': CPU_COLOR, 'transfer_cpu_7dof': CPU_7DOF_COLOR, 'transfer_gpu_6dof': GPU_COLOR,
        'stir_cpu_6dof': CPU_COLOR, 'stir_gpu_6dof': GPU_COLOR,
    }

    # 메인 축: Box + Strip plot
    for key, pos in positions.items():
        success, fail_fast, fail_timeout = data[key]
        color = colors[key]
        plot_box_strip(ax_main, pos, success, fail_fast, color)

    # y축 범위 결정 (성공 trial + 빠른 실패의 최대값 기준)
    all_visible_times = []
    for key in positions:
        success, fail_fast, _ = data[key]
        all_visible_times.extend(success)
        all_visible_times.extend(fail_fast)
    if all_visible_times:
        y_max = min(max(all_visible_times) * 1.15, 100)
    else:
        y_max = 100
    ax_main.set_ylim(-2, y_max)

    # 성공률 라벨
    for key, pos in positions.items():
        success, fail_fast, fail_timeout = data[key]
        total = len(success) + len(fail_fast) + len(fail_timeout)
        if total > 0:
            add_success_rate(ax_main, pos, len(success), total, colors[key], y_max)

    # 상단 축: Timeout 막대
    max_timeout = 0
    for key, pos in positions.items():
        _, _, fail_timeout = data[key]
        plot_timeout_bar(ax_top, pos, len(fail_timeout))
        max_timeout = max(max_timeout, len(fail_timeout))

    # === 축 서식 ===
    # 메인 축
    ax_main.set_ylabel('Planning Time (s)', fontsize=12)
    ax_main.set_xlim(0, 10)

    # 그룹 라벨
    ax_main.set_xticks([1.5, 5, 8.5])
    ax_main.set_xticklabels(['Move', 'Transfer', 'Stir'], fontsize=12, fontweight='bold')

    # 서브 라벨
    sub_labels = {
        1: 'CPU\n6-DoF', 2: 'GPU\n6-DoF',
        4: 'CPU\n6-DoF', 5: 'CPU\n7-DoF', 6: 'GPU\n6-DoF',
        8: 'CPU\n6-DoF', 9: 'GPU\n6-DoF',
    }
    for pos, label in sub_labels.items():
        ax_main.text(pos, -2 - y_max * 0.06, label, ha='center', va='top', fontsize=7.5, color='gray')

    # 그룹 구분선
    for ax in [ax_main, ax_top]:
        ax.axvline(x=3, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=7, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

    # 상단 축
    ax_top.set_ylabel('Timeouts\n(≥120s)', fontsize=9, color=TIMEOUT_COLOR)
    ax_top.set_ylim(0, max(max_timeout + 1.5, 3))
    ax_top.set_yticks(range(0, max_timeout + 2))
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)

    # 제목
    ax_top.set_title('Planning Time Distribution and Success Rate by Task',
                     fontsize=13, fontweight='bold', pad=10)

    # 범례
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CPU_COLOR, markersize=8,
               label='CPU (PDDLStream) 6-DoF'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CPU_7DOF_COLOR, markersize=8,
               label='CPU (PDDLStream) 7-DoF'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GPU_COLOR, markersize=8,
               label='GPU (cuTAMP) 6-DoF'),
        Line2D([0], [0], marker='x', color=FAIL_COLOR, markersize=8,
               label='Failed (< timeout)', linestyle='None'),
        plt.Rectangle((0, 0), 1, 1, fc=TIMEOUT_COLOR, alpha=0.6, ec=TIMEOUT_COLOR,
                       label='Timeout (≥120s)'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=8.5, framealpha=0.9)

    # 저장
    import os
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()