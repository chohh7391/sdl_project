#!/usr/bin/env python3
"""
Figure 3: Planning Time Distribution and Success Rate by Task
- Timeout budget: 30 s
- 1단 구조: ax_main (0~30s) only
- Times New Roman 폰트
"""

import csv, os
import numpy as np
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# 경로 설정
# ============================================================
DATA_PATHS = {
    'transfer_cpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_fr5.csv',
    'transfer_cpu_7dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_panda.csv',
    'transfer_gpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/cutamp_datat/transfer.csv',
    'move_cpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/move_fr5.csv',
    'move_gpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/cutamp_datat/move.csv',
    'stir_cpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/stir_fr5.csv',
    'stir_gpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/cutamp_datat/stir.csv',
}
OUTPUT_PNG = '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/figure3_journal.png'
OUTPUT_PDF = '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/figure3_journal.pdf'

TIMEOUT_THRESHOLD = 35.0
MAIN_YMAX         = 40.0

CPU_COLOR      = '#D32F2F'
CPU_7DOF_COLOR = '#F57F17'
GPU_COLOR      = '#003D8F'
FAIL_COLOR     = '#7A0000'

positions = {
    'move_cpu_6dof':1,     'move_gpu_6dof':2,
    'transfer_cpu_6dof':4, 'transfer_cpu_7dof':5, 'transfer_gpu_6dof':6,
    'stir_cpu_6dof':8,     'stir_gpu_6dof':9,
}
colors = {
    'move_cpu_6dof':CPU_COLOR,     'move_gpu_6dof':GPU_COLOR,
    'transfer_cpu_6dof':CPU_COLOR, 'transfer_cpu_7dof':CPU_7DOF_COLOR,
    'transfer_gpu_6dof':GPU_COLOR,
    'stir_cpu_6dof':CPU_COLOR,     'stir_gpu_6dof':GPU_COLOR,
}

# ============================================================
# 헬퍼
# ============================================================
def load_csv(path):
    s, ff, ft = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t = float(row['planning_time_sec']); ok = int(row['success'])
            if t >= TIMEOUT_THRESHOLD: ft.append(t)
            elif ok: s.append(t)
            else: ff.append(t)
    return s, ff, ft

def draw_boxstrip(ax, pos, times, color, jitter_w=0.13):
    if not times: return
    rng = np.random.default_rng(42)
    if len(times) > 1:
        bp = ax.boxplot([times], positions=[pos], widths=0.44,
                        patch_artist=True, showfliers=False, zorder=2)
        bp['boxes'][0].set(facecolor=color, alpha=0.40, edgecolor=color, linewidth=1.5)
        bp['medians'][0].set(color=color, linewidth=2.5)
        for el in bp['whiskers'] + bp['caps']:
            el.set(color=color, linewidth=1.2)
    j = rng.uniform(-jitter_w, jitter_w, len(times))
    ax.scatter(pos + j, times, color=color, s=30, alpha=0.8, zorder=3,
               edgecolors='white', linewidths=0.4)

def draw_fail_x(ax, pos, times):
    rng = np.random.default_rng(99)
    for t in times:
        j = rng.uniform(-0.13, 0.13)
        ax.scatter(pos + j, min(t, MAIN_YMAX - 0.5),
                   color=FAIL_COLOR, marker='x', s=50, zorder=4, linewidths=1.6)

# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(42)

    data = {}
    for key, path in DATA_PATHS.items():
        try:
            data[key] = load_csv(path)
            s, ff, ft = data[key]; total = len(s)+len(ff)+len(ft)
            rate = len(s)/total*100 if total else 0
            mu = np.mean(s) if s else float('nan')
            sd = np.std(s, ddof=1) if len(s) > 1 else float('nan')
            print(f"  {key:25s} | {len(s):2d}/{total} ({rate:5.1f}%) "
                  f"| {mu:5.2f}±{sd:5.2f}s | timeout:{len(ft)}")
        except FileNotFoundError:
            print(f"  WARNING: {path} not found — empty data used")
            data[key] = ([], [], [])

    # ── 1단 레이아웃 ─────────────────────────────────────────
    fig, ax_main = plt.subplots(figsize=(16, 7))

    for key, pos in positions.items():
        s, ff, ft = data[key]; col = colors[key]
        draw_boxstrip(ax_main, pos, s,  col)
        draw_fail_x  (ax_main, pos, ff)

    # 30s 텍스트
    # ax_main.text(9.8, TIMEOUT_THRESHOLD + 0.4, '30 s timeout',
    #              ha='right', va='bottom', fontsize=10, color='#111111')

    # ── ax_main 서식 ─────────────────────────────────────────
    ax_main.set_ylim(-0.8, MAIN_YMAX)
    ax_main.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax_main.tick_params(axis='y', labelsize=13)
    ax_main.set_ylabel('Planning Time (s)', fontsize=14)
    ax_main.set_xlim(0, 10)
    ax_main.set_xticks([1.5, 5, 8.5])
    ax_main.set_xticklabels(['Move', 'Transfer', 'Stir'], fontsize=15, fontweight='bold')
    ax_main.tick_params(axis='x', bottom=True, labelbottom=True, pad=55)

    sub_labels = {1:'CPU\n6-DoF', 2:'GPU\n6-DoF', 4:'CPU\n6-DoF', 5:'CPU\n7-DoF',
                  6:'GPU\n6-DoF', 8:'CPU\n6-DoF', 9:'GPU\n6-DoF'}
    for pos, lbl in sub_labels.items():
        ax_main.text(pos, -0.8 - MAIN_YMAX*0.045, lbl,
                     ha='center', va='top', fontsize=11, color='#111111')

    # 성공률 — 상단
    for key, pos in positions.items():
        s, ff, ft = data[key]; total = len(s)+len(ff)+len(ft)
        if total:
            col = colors[key]
            ax_main.text(pos, MAIN_YMAX*0.96, f'{len(s)/total*100:.1f}%',
                         ha='center', va='top', fontsize=11.5,
                         fontweight='bold', color=col)

    # 구분선
    for xv in [3, 7]:
        ax_main.axvline(x=xv, color='#888888', linewidth=0.7)

    # 제목
    ax_main.set_title('Planning Time Distribution and Success Rate by Task',
                      fontsize=15, fontweight='bold', pad=12)

    # 범례
    legend_elements = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor=CPU_COLOR,
               markersize=8, label='CPU (PDDLStream) \u2014 6-DoF'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=CPU_7DOF_COLOR,
               markersize=8, label='CPU (PDDLStream) \u2014 7-DoF'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=GPU_COLOR,
               markersize=8, label='GPU (cuTAMP) \u2014 6-DoF'),
        Line2D([0],[0],marker='x',color=FAIL_COLOR,markersize=8,
               label='Failed (< 30 s)',linestyle='None'),
    ]
    ax_main.legend(handles=legend_elements,
                   bbox_to_anchor=(0.01, 0.93), loc='upper left',
                   fontsize=9.0, framealpha=0.93, edgecolor='#888888',
                   handlelength=1.4, borderpad=0.6, labelspacing=0.4)

    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_PDF,           bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")

if __name__ == '__main__':
    main()