import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(here, '..', 'results')
figures = os.path.join(here, '..', 'figures')
os.makedirs(figures, exist_ok=True)

surface = '#fcfcfb'
ink = '#0b0b0b'
ink2 = '#52514e'
grid = '#e4e3df'
blue = '#2a78d6'
orange = '#eb6834'
aqua = '#1baf7a'

plt.rcParams.update({
    'figure.facecolor': surface,
    'axes.facecolor': surface,
    'axes.edgecolor': grid,
    'axes.labelcolor': ink2,
    'axes.titlecolor': ink,
    'xtick.color': ink2,
    'ytick.color': ink2,
    'text.color': ink,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'semibold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': grid,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
})


def read_pca():
    rows = {}
    for line in open(os.path.join(results, 'pca.log')):
        p = line.split()
        if p and p[0] == 'PCA':
            rows[int(p[2])] = float(p[6])
    return rows


def read_ae():
    runs = {}
    for path in glob.glob(os.path.join(results, 'ae_*.log')):
        for line in open(path):
            p = line.split()
            if not p or p[0] != 'RESULT':
                continue
            mode = p[2]
            width = int(p[5])
            code = int(p[7])
            prauc = float(p[13])
            runs.setdefault((mode, width, code), []).append(prauc)
    return runs


def read_diag():
    rows = []
    path = os.path.join(results, 'diagnose.log')
    if not os.path.exists(path):
        return rows
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'DIAG':
            continue
        rows.append({
            'width': int(p[2]), 'code': int(p[4]), 'seed': int(p[6]),
            'prauc': float(p[8]), 'normal': float(p[10]), 'fraud': float(p[12]),
        })
    return rows


pca = read_pca()
ae = read_ae()
collapse = 0.35

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ks = sorted(k for k in pca if k <= 28)
ax.plot(ks, [pca[k] for k in ks], color=blue, linewidth=2, marker='o', markersize=6, label='PCA', zorder=3)
for width, color, label in [(16, orange, 'deep autoencoder, 16 wide'), (28, aqua, 'deep autoencoder, 28 wide')]:
    codes = sorted(c for (m, w, c) in ae if m == 'none' and w == width)
    xs = []
    ys = []
    for c in codes:
        for v in ae[('none', width, c)]:
            xs.append(c)
            ys.append(v)
    jitter = 0.35 if width == 16 else -0.35
    ax.scatter(np.array(xs) + jitter, ys, s=42, color=color, edgecolor=surface, linewidth=1.5, zorder=4, label=label + ' (one dot per seed)')
    means = []
    for c in codes:
        means.append(np.mean(ae[('none', width, c)]))
    ax.plot(np.array(codes) + jitter, means, color=color, linewidth=2, zorder=3)
ax.axhline(collapse, color=ink2, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
ax.text(28.6, collapse + 0.012, 'collapse threshold for deep models (0.35)', color=ink2, fontsize=9, ha='right')
ax.annotate('PCA k=27: ' + format(pca[27], '.2f'), xy=(27, pca[27]), xytext=(20.2, 0.9), color=ink, fontsize=10,
            arrowprops=dict(arrowstyle='-', color=ink2, linewidth=1))
ax.set_xlabel('code size k (numbers kept in the middle)')
ax.set_ylabel('validation PR-AUC')
ax.set_title('PCA vs deep autoencoder, by code size')
ax.set_ylim(0, 1)
ax.set_xticks([2, 4, 8, 12, 16, 20, 24, 27])
ax.legend(loc='upper left', fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(figures, 'prauc_by_code_size.png'), dpi=160)
plt.close(fig)

settings = [(24, 8), (24, 16), (24, 20), (28, 8), (28, 16), (28, 20)]
modes = [('none', 'no fix'), ('denoise', 'denoising'), ('wd', 'weight decay')]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=True)
for ax, (width, code) in zip(axes.ravel(), settings):
    for i, (mode, name) in enumerate(modes):
        vals = ae.get((mode, width, code), [])
        solid = []
        hollow = []
        for v in vals:
            if v < collapse:
                hollow.append(v)
            else:
                solid.append(v)
        ax.scatter([i] * len(solid), solid, s=46, color=aqua, edgecolor=surface, linewidth=1.5, zorder=4)
        ax.scatter([i] * len(hollow), hollow, s=46, facecolor=surface, edgecolor=aqua, linewidth=1.8, zorder=4)
        if vals:
            ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=ink, linewidth=2, zorder=5)
    ax.axhline(collapse, color=ink2, linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax.axhline(pca[27], color=blue, linewidth=1.5, zorder=2)
    ax.set_title(str(width) + ' wide, k=' + str(code), fontsize=11)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([m[1] for m in modes], fontsize=9.5)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1)
for ax in axes[:, 0]:
    ax.set_ylabel('validation PR-AUC')
handles = [
    plt.Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=aqua, markeredgecolor=surface, label='one seed'),
    plt.Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=surface, markeredgecolor=aqua, label='collapsed seed (below 0.35)'),
    plt.Line2D([], [], color=ink, linewidth=2, label='mean of 5 seeds'),
    plt.Line2D([], [], color=blue, linewidth=1.5, label='PCA k=27 (' + format(pca[27], '.2f') + ')'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9.5)
fig.suptitle('Do denoising or weight decay stop the collapses?', fontsize=13, fontweight='semibold', color=ink)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(os.path.join(figures, 'fixes_by_setting.png'), dpi=160)
plt.close(fig)

diag = read_diag()
if diag:
    labels = []
    normals = []
    frauds = []
    for d in diag:
        state = 'collapsed' if d['prauc'] < collapse else 'working'
        labels.append(str(d['width']) + ' wide, k=' + str(d['code']) + ', seed ' + str(d['seed']) + '\n' + state + ', PR-AUC ' + format(d['prauc'], '.2f'))
        normals.append(d['normal'])
        frauds.append(d['fraud'])
    x = np.arange(len(diag))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bw = 0.36
    ax.bar(x - bw / 2 - 0.01, normals, bw, color=blue, label='median score, normal rows', zorder=3)
    ax.bar(x + bw / 2 + 0.01, frauds, bw, color=orange, label='median score, fraud rows', zorder=3)
    for i in range(len(diag)):
        ax.text(x[i] - bw / 2 - 0.01, normals[i] * 1.12, format(normals[i], '.1f'), ha='center', fontsize=9, color=ink2)
        ax.text(x[i] + bw / 2 + 0.01, frauds[i] * 1.12, format(frauds[i], '.0f'), ha='center', fontsize=9, color=ink2)
    ax.set_yscale('log')
    ax.set_ylim(1, 3000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('median reconstruction score (log scale)')
    ax.set_title('Collapsed models rebuild the frauds too well')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, 'collapse_diagnosis.png'), dpi=160)
    plt.close(fig)

print('charts written to', os.path.abspath(figures))
