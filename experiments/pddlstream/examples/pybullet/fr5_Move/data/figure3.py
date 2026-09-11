#!/usr/bin/env python3
"""
Figure 3: Planning Time Distribution and Success Rate by Task
- Timeout budget: 30 s
- 2단 구조: ax_top (timeout counts) / ax_main (0~30s)
- Times New Roman 폰트
- 성공률 bbox 없음, 점선 없음
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ============================================================
# 경로 설정
# ============================================================
DATA_PATHS = {
    'transfer_cpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_fr5.csv',
    'transfer_cpu_7dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_panda.csv',
    'transfer_gpu_6dof': '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_fr5_sample_z.csv',
    'move_cpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/move_fr5.csv',
    'move_gpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_fr5_sample_z.csv',
    'stir_cpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/stir_fr5.csv',
    'stir_gpu_6dof':     '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/transfer_fr5_sample_z.csv',
}
OUTPUT_PNG = '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/figure3_journal.png'
OUTPUT_PDF = '/home/home/sdl_ws/src/sdl_project/experiments/pddlstream/examples/pybullet/fr5_Move/data/figure3_journal.pdf'

TIMEOUT_THRESHOLD = 29.0
MAIN_YMAX         = 30

CPU_COLOR      = '#D32F2F'
CPU_7DOF_COLOR = '#F57F17'
GPU_COLOR      = '#003D8F'
TIMEOUT_COLOR  = '#7A0000'
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

def draw_timeout_bar(ax, pos, n):
    if n > 0:
        ax.bar(pos, n, width=0.44, color=TIMEOUT_COLOR, alpha=0.52,
               edgecolor=TIMEOUT_COLOR, linewidth=0.8, zorder=2)
        ax.text(pos, n + 0.12, str(n), ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=TIMEOUT_COLOR)

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

    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.0, figure=fig)
    ax_top  = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    for key, pos in positions.items():
        s, ff, ft = data[key]; col = colors[key]
        draw_boxstrip(ax_main, pos, s,  col)
        draw_fail_x  (ax_main, pos, ff)
        draw_timeout_bar(ax_top, pos, len(ft))

    # 30s 텍스트 (점선 없음, 검은색)
    ax_main.text(9.8, TIMEOUT_THRESHOLD + 0.4, '30 s timeout',
                 ha='right', va='bottom', fontsize=10, color='#111111')

    ax_main.set_ylim(-0.8, MAIN_YMAX)
    ax_main.set_yticks([0, 5, 10, 15, 20, 25, 30])
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

    # 성공률 — bbox 없이 수치만
    for key, pos in positions.items():
        s, ff, ft = data[key]; total = len(s)+len(ff)+len(ft)
        if total:
            col = colors[key]
            ax_main.text(pos, MAIN_YMAX*0.96, f'{len(s)/total*100:.1f}%',
                         ha='center', va='top', fontsize=11.5,
                         fontweight='bold', color=col)

    max_to = max(len(data[k][2]) for k in positions)
    ax_top.set_ylim(0, max(max_to + 1.5, 3))
    ax_top.set_yticks(range(0, max_to + 2))
    ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_top.set_ylabel('Timeouts (\u226530 s)', fontsize=12, color=TIMEOUT_COLOR)
    ax_top.set_xlim(0, 10)
    ax_top.tick_params(axis='y', labelsize=12)

    for ax in [ax_top, ax_main]:
        for xv in [3, 7]:
            ax.axvline(x=xv, color='#888888', linewidth=0.7)

    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_main.spines['top'].set_visible(False)

    ax_top.set_title('Planning Time Distribution and Success Rate by Task',
                     fontsize=15, fontweight='bold', pad=12)

    legend_elements = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor=CPU_COLOR,
               markersize=8, label='CPU (PDDLStream) \u2014 6-DoF'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=CPU_7DOF_COLOR,
               markersize=8, label='CPU (PDDLStream) \u2014 7-DoF'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=GPU_COLOR,
               markersize=8, label='GPU (cuTAMP) \u2014 6-DoF'),
        Line2D([0],[0],marker='x',color=FAIL_COLOR,markersize=8,
               label='Failed (< 30 s)',linestyle='None'),
        plt.Rectangle((0,0),1,1,fc=TIMEOUT_COLOR,alpha=0.52,
                      ec=TIMEOUT_COLOR,label='Timeout (\u226530 s)'),
    ]
    ax_top.legend(handles=legend_elements, loc='upper left',
                  fontsize=10.5, framealpha=0.93, edgecolor='#888888',
                  handlelength=1.4, borderpad=0.6, labelspacing=0.4)

    plt.show()

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_PDF,           bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")

if __name__ == '__main__':
    main()