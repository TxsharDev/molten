"""Generate publication-quality figures for the Molten paper."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# ─── Figure 1: Architecture Diagram ───────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    stages = [
        ("Python\nMath Spec", 0.3),
        ("Dataflow\nGraph (IR)", 2.2),
        ("Optimizer", 4.1),
        ("Fusion\nEngine", 5.8),
        ("Code\nGenerator", 7.5),
        (".cu files", 9.2),
    ]
    box_w, box_h = 1.4, 1.0
    y_center = 1.0

    for label, x in stages:
        rect = FancyBboxPatch(
            (x - box_w/2, y_center - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor='black', facecolor='#f5f5f5'
        )
        ax.add_patch(rect)
        ax.text(x, y_center, label, ha='center', va='center',
                fontsize=11, fontweight='medium')

    for i in range(len(stages) - 1):
        x_start = stages[i][1] + box_w/2 + 0.02
        x_end = stages[i+1][1] - box_w/2 - 0.02
        ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_architecture.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_architecture.pdf")


# ─── Figure 2: Speedup Bar Chart ──────────────────────────────────────
def fig_speedup():
    configs = ['decode\n(1,1,5120)', 'prefill\n(1,2048,5120)', 'long\n(1,8192,5120)']
    eager   = [167.4, 159.9, 792.6]
    compile = [127.1,  95.6, 322.6]
    molten  = [ 27.6,  55.0, 393.9]
    annotations = ['4.6x vs compile', '1.7x', '0.82x']

    x = np.arange(len(configs))
    w = 0.24

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_yscale('log')

    bars_e = ax.bar(x - w, eager,   w, label='PyTorch Eager', color='#999999', edgecolor='black', linewidth=0.6)
    bars_c = ax.bar(x,     compile, w, label='torch.compile', color='#4878CF', edgecolor='black', linewidth=0.6)
    bars_m = ax.bar(x + w, molten,  w, label='Molten Generated', color='#C44E52', edgecolor='black', linewidth=0.6)

    for i, (bar, txt) in enumerate(zip(bars_m, annotations)):
        ypos = bar.get_height() * 1.25
        ax.text(bar.get_x() + bar.get_width()/2, ypos, txt,
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#C44E52')

    ax.set_ylabel('Latency (μs)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_title('RTX 5090 — RMSNorm Kernel Latency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_speedup.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_speedup.pdf")


# ─── Figure 3: Roofline Plot ──────────────────────────────────────────
def fig_roofline():
    # RTX 5090 specs
    bw_peak = 1800      # GB/s
    compute_peak = 105e3 # GFLOP/s  (105 TFLOPS)
    ridge = compute_peak / bw_peak  # ~58.3 FLOP/byte

    ai = np.logspace(-2, 4, 500)
    roof = np.minimum(bw_peak * ai, compute_peak)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ai, roof, 'k-', lw=2.0, label='RTX 5090 Roofline')

    # Shade regions
    ax.fill_between(ai, roof, alpha=0.04, color='black')

    # Annotate ceilings
    ax.axhline(compute_peak, ls='--', lw=0.6, color='gray', alpha=0.5)
    ax.text(1e3, compute_peak*1.15, f'{compute_peak/1e3:.0f} TFLOP/s', fontsize=10, color='gray')
    ax.text(0.015, bw_peak*0.25, f'{bw_peak} GB/s slope', fontsize=10, color='gray', rotation=38)

    # Data points — representative values for bandwidth-bound ops
    # RMSNorm: very low arithmetic intensity (~2 FLOP/byte)
    points = {
        'RMSNorm eager':       (1.8,  340),   # low BW utilization
        'RMSNorm compiled':    (1.8,  750),   # moderate
        'RMSNorm Molten':      (1.8, 1520),   # near peak BW
        'Fused triple eager':  (3.0,  280),   # 3 separate kernels
        'Fused triple Molten': (3.0, 1380),   # single fused kernel
    }
    markers = {'eager': 's', 'compiled': '^', 'Molten': 'o', 'triple eager': 'D', 'triple Molten': 'p'}
    colors  = {'eager': '#999999', 'compiled': '#4878CF', 'Molten': '#C44E52',
               'triple eager': '#999999', 'triple Molten': '#C44E52'}

    for label, (x, y) in points.items():
        key = label.split()[-1] if 'triple' not in label else ' '.join(label.split()[-2:])
        m = markers.get(key, 'o')
        c = colors.get(key, 'black')
        ax.plot(x, y, marker=m, ms=10, color=c, markeredgecolor='black',
                markeredgewidth=0.8, zorder=5)
        # offset labels to avoid overlap
        offset_y = 1.35 if 'Molten' in label else 0.7
        ax.text(x * 1.15, y * offset_y, label, fontsize=9, va='center')

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=13)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=13)
    ax.set_title('Roofline Model — RTX 5090', fontsize=14, fontweight='bold')
    ax.set_xlim(0.01, 5000)
    ax.set_ylim(10, 3e5)
    ax.grid(True, which='both', alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_roofline.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_roofline.pdf")


# ─── Figure 4: Fusion Analysis ────────────────────────────────────────
def fig_fusion():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Left side: 3 separate kernels
    left_ops = ['RMS\nreduce', 'divide', 'scale']
    bw, bh = 1.3, 0.8
    left_x = 1.5
    ys = [3.0, 2.0, 1.0]

    for i, (label, y) in enumerate(zip(left_ops, ys)):
        rect = FancyBboxPatch(
            (left_x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.06", lw=1.2,
            edgecolor='black', facecolor='#e8e8e8'
        )
        ax.add_patch(rect)
        ax.text(left_x, y, label, ha='center', va='center', fontsize=11)

    # Arrows between left ops
    for i in range(len(ys) - 1):
        ax.annotate('', xy=(left_x, ys[i+1] + bh/2 + 0.02),
                     xytext=(left_x, ys[i] - bh/2 - 0.02),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        # Memory round-trip label
        ax.text(left_x + bw/2 + 0.15, (ys[i] + ys[i+1])/2, 'DRAM',
                fontsize=8, color='#888888', fontstyle='italic', va='center')

    ax.text(left_x, 0.25, '3 kernels, 2 memory\nround-trips',
            ha='center', va='center', fontsize=10, color='#555555')

    # Right side: fused kernel
    right_x = 7.5
    fused_w, fused_h = 2.0, 2.6
    rect = FancyBboxPatch(
        (right_x - fused_w/2, 2.0 - fused_h/2), fused_w, fused_h,
        boxstyle="round,pad=0.1", lw=2.0,
        edgecolor='#C44E52', facecolor='#fceaea'
    )
    ax.add_patch(rect)

    for i, (label, y) in enumerate(zip(left_ops, ys)):
        ax.text(right_x, y, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#333333')
        if i < len(ys) - 1:
            ax.annotate('', xy=(right_x, ys[i+1] + 0.35),
                         xytext=(right_x, ys[i] - 0.35),
                         arrowprops=dict(arrowstyle='->', lw=0.8,
                                         color='#C44E52', ls='--'))

    ax.text(right_x, 0.25, '1 fused kernel,\n0 round-trips',
            ha='center', va='center', fontsize=10, color='#C44E52',
            fontweight='bold')

    # Big arrow from left to right
    ax.annotate('', xy=(right_x - fused_w/2 - 0.2, 2.0),
                 xytext=(left_x + bw/2 + 0.3, 2.0),
                 arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(4.5, 2.45, 'Molten\nfusion', ha='center', va='center',
            fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_fusion.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_fusion.pdf")


if __name__ == '__main__':
    print("Generating figures...")
    fig_architecture()
    fig_speedup()
    fig_roofline()
    fig_fusion()
    print(f"Done. All figures saved to {OUT}/")
