"""Figures for the results folder and the presentation (static PNGs, light theme)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# Categorical slots 1-3 of the reference palette (CVD-validated as a set).
# Colour follows the augmentation setting everywhere; models are separated by panel/row.
AUG_COLORS = {"none": "#2a78d6", "affine": "#eb6834", "full": "#1baf7a"}
AUG_LABELS = {"none": "No augmentation", "affine": "Affine", "full": "Affine + elastic + erasing"}
MODEL_LABELS = {"mlp": "MLP", "cnn": "CNN", "vit": "Transformer (ViT)"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _line(ax, x, y, aug, **kw):
    style = dict(color=AUG_COLORS[aug], label=AUG_LABELS[aug], marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.plot(x, y, **{**style, **kw})


def _runs_by_model(runs):
    out = {}
    for r in runs:
        out.setdefault(r["model"], []).append(r)
    order = [m for m in MODEL_LABELS if m in out]
    return [(m, sorted(out[m], key=lambda r: list(AUG_COLORS).index(r["aug"]))) for m in order]


def test_error_bars(runs, path, ensemble=None):
    """Horizontal bars: test error (%) per run, grouped by model, coloured by augmentation."""
    rows = []
    for model, rs in _runs_by_model(runs):
        rows += [(MODEL_LABELS[model], r["aug"], 100 - 100 * r["test_acc"]) for r in rs] + [None]
    if ensemble:
        rows.append(("Ensemble", "ensemble", 100 - 100 * ensemble["test_acc"]))
    else:
        rows = rows[:-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 0.9))
    y, labels, last_model = 0, [], None
    ticks = []
    for row in rows:
        y -= 1
        if row is None:
            continue
        model, aug, err = row
        color = INK_2 if aug == "ensemble" else AUG_COLORS[aug]
        ax.barh(y, err, height=0.72, color=color)
        ax.text(err, y, f"  {err:.2f}%", va="center", ha="left", fontsize=9, color=INK)
        name = "MLP + CNN + ViT (full aug)" if aug == "ensemble" else AUG_LABELS[aug]
        labels.append(f"{model if model != last_model else ''}   {name}".strip())
        ticks.append(y)
        last_model = model
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r[2] for r in rows if r) * 1.18)
    ax.set_xlabel("Test error rate (%)  -  lower is better  (10,000 test images)")
    ax.set_title("MNIST test error by model and augmentation")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def learning_curves(runs, path):
    """Validation error per epoch, one panel per model."""
    groups = _runs_by_model(runs)
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (model, rs) in zip(axes, groups):
        for r in rs:
            h = r["history"]
            _line(ax, [e["epoch"] for e in h], [100 - 100 * e["val_acc"] for e in h], r["aug"], markersize=0)
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation error (%), log scale")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def robustness(runs, key, xlabel, path):
    """Test accuracy under a deterministic distortion, one panel per model."""
    groups = _runs_by_model(runs)
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (model, rs) in zip(axes, groups):
        for r in rs:
            pts = r["robustness"][key]
            xs = [float(k) for k in pts]
            _line(ax, xs, [100 * v for v in pts.values()], r["aug"])
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion_matrix(cm, path, title):
    """Off-diagonal counts on a sequential ramp; the diagonal (correct) is printed in grey."""
    cm = np.asarray(cm)
    off = cm.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(off, cmap=BLUES, vmin=0, vmax=max(1, np.nanmax(off)))
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if i == j:
                ax.text(j, i, v, ha="center", va="center", fontsize=7, color=MUTED)
            elif v:
                dark = v > 0.55 * np.nanmax(off)
                ax.text(j, i, v, ha="center", va="center", fontsize=8, color=SURFACE if dark else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(False)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.locator = matplotlib.ticker.MaxNLocator(integer=True)
    cb.update_ticks()
    cb.set_label("Misclassified images", color=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def image_grid(images, captions, path, title, ncols=10):
    n = len(images)
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.0, nrows * 1.35 + 0.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, cap in zip(np.atleast_1d(axes).ravel(), images, captions):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(cap, fontsize=7.5, color=INK_2, loc="center", fontweight="normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
