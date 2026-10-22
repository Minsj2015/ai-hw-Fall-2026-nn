"""Build presentation/MNIST_NN_presentation.pdf (16:9 slides) from the files in results/.

    python presentation/build_presentation.py

Needs reportlab (`pip install reportlab`) on top of requirements.txt.
All numbers are read from results/, so re-running after new experiments updates the slides.
"""
import csv
import json
import os
import sys
from pathlib import Path

import torch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import plots  # noqa: E402
from data import build_augment, load_mnist, to_input  # noqa: E402

RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
ASSETS = ROOT / "presentation" / "assets"
OUT = ROOT / "presentation" / "MNIST_NN_presentation.pdf"
REPO_URL = "https://github.com/{user}/ai-hw-Fall-2026-nn"
W, H = 960, 540  # 16:9 page in points
M = 48           # side margin

SURFACE, INK, INK_2, MUTED = HexColor(plots.SURFACE), HexColor(plots.INK), HexColor(plots.INK_2), HexColor(plots.MUTED)
HAIR, FILL, ACCENT = HexColor("#e1e0d9"), HexColor("#f0efec"), HexColor("#2a78d6")

try:  # embed Arial when available (macOS); otherwise fall back to the built-in Helvetica
    sup = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Sans", str(sup / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Bold", str(sup / "Arial Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", "/System/Library/Fonts/Supplemental/Courier New.ttf"))
    from reportlab.lib.fonts import addMapping
    addMapping("Sans", 0, 0, "Sans")
    addMapping("Sans", 1, 0, "Sans-Bold")
    SANS, BOLD, MONO = "Sans", "Sans-Bold", "Mono"
except Exception:
    SANS, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.c.setTitle("MNIST recognition with small neural networks")
        self.c.setAuthor("ai-hw-Fall-2026-nn")
        self.n = 0

    # ---- page furniture -------------------------------------------------
    def slide(self, title, kicker=None):
        if self.n:
            self.c.showPage()
        self.n += 1
        c = self.c
        c.setFillColor(SURFACE)
        c.rect(0, 0, W, H, stroke=0, fill=1)
        y = H - 44
        if kicker:
            c.setFillColor(ACCENT)
            c.setFont(BOLD, 11)
            c.drawString(M, y, kicker.upper())
            y -= 30
        c.setFillColor(INK)
        c.setFont(BOLD, 26)
        c.drawString(M, y, title)
        c.setFillColor(MUTED)
        c.setFont(SANS, 9)
        c.drawString(M, 20, "MNIST recognition with small neural networks  ·  ai-hw-Fall-2026-nn")
        c.drawRightString(W - M, 20, str(self.n))
        return y - 22  # top of the content area

    # ---- content helpers ------------------------------------------------
    def text(self, x, y_top, width, html, size=15, color=INK_2, leading=None, font=None, align=0):
        style = ParagraphStyle("t", fontName=font or SANS, fontSize=size, leading=leading or size * 1.32,
                               textColor=color, alignment=align)
        p = Paragraph(html, style)
        _, h = p.wrap(width, H)
        p.drawOn(self.c, x, y_top - h)
        return h

    def bullets(self, x, y_top, width, items, size=15, gap=8, color=INK_2):
        style = ParagraphStyle("b", fontName=SANS, fontSize=size, leading=size * 1.32, textColor=color,
                               leftIndent=16, bulletIndent=0, bulletFontName=SANS, bulletFontSize=size)
        y = y_top
        for item in items:
            p = Paragraph(item, style, bulletText="•")
            _, h = p.wrap(width, H)
            p.drawOn(self.c, x, y - h)
            y -= h + gap
        return y_top - y

    def image(self, path, x, y_top, max_w, max_h, align="left"):
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        s = min(max_w / iw, max_h / ih)
        w, h = iw * s, ih * s
        if align == "center":
            x += (max_w - w) / 2
        self.c.drawImage(img, x, y_top - h, w, h, mask="auto")
        return w, h

    def box(self, x, y, w, h, lines, fill=FILL, size=11, bold_first=True):
        c = self.c
        c.setFillColor(fill)
        c.setStrokeColor(fill)
        c.roundRect(x, y, w, h, 6, stroke=0, fill=1)
        total = len(lines) * size * 1.25
        ty = y + h / 2 + total / 2 - size
        for i, line in enumerate(lines):
            c.setFillColor(INK if i == 0 else INK_2)
            c.setFont(BOLD if (i == 0 and bold_first) else SANS, size if i == 0 else size - 1)
            c.drawCentredString(x + w / 2, ty, line)
            ty -= size * 1.25

    def arrow_down(self, x, y_from, y_to):
        c = self.c
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.2)
        c.line(x, y_from, x, y_to + 4)
        p = c.beginPath()
        p.moveTo(x - 4, y_to + 6)
        p.lineTo(x + 4, y_to + 6)
        p.lineTo(x, y_to)
        p.close()
        c.drawPath(p, stroke=0, fill=1)

    def table(self, x, y_top, col_w, rows, size=11.5, row_h=22, align=None, bold_rows=(), shade_rows=()):
        c = self.c
        align = align or ["l"] * len(col_w)
        y = y_top
        for r, row in enumerate(rows):
            if r in shade_rows:
                c.setFillColor(FILL)
                c.rect(x - 6, y - row_h + 5, sum(col_w) + 12, row_h, stroke=0, fill=1)
            cx = x
            for cell, w, a in zip(row, col_w, align):
                header = r == 0
                c.setFont(BOLD if header or r in bold_rows else SANS, size - (1 if header else 0))
                c.setFillColor(INK_2 if header else INK)
                ty = y - row_h + 11
                if a == "r":
                    c.drawRightString(cx + w - 6, ty, str(cell))
                else:
                    c.drawString(cx, ty, str(cell))
                cx += w
            if r == 0:
                c.setStrokeColor(HexColor("#c3c2b7"))
                c.setLineWidth(0.8)
                c.line(x - 6, y - row_h + 4, x + sum(col_w) + 6, y - row_h + 4)
            y -= row_h
        return y_top - y

    def stat(self, x, y_top, w, label, value, note=None):
        c = self.c
        c.setFillColor(INK_2)
        c.setFont(SANS, 12)
        c.drawString(x, y_top - 12, label)
        c.setFillColor(INK)
        c.setFont(BOLD, 40)
        c.drawString(x, y_top - 56, value)
        if note:
            self.text(x, y_top - 66, w, note, size=11.5, color=INK_2)

    def save(self):
        self.c.save()


# ---- data -----------------------------------------------------------------
def load_results():
    with open(RESULTS / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    runs = {r["run"]: r for r in rows}
    with open(RESULTS / "significance.csv") as f:
        sig = {(r["run_a"], r["run_b"]): r for r in csv.DictReader(f)}
    metrics = {p.parent.name: json.loads(p.read_text()) for p in RESULTS.glob("*/metrics.json")}
    missing = sorted(k for k, m in metrics.items() if "robustness" not in m)
    if missing:
        raise SystemExit(f"results for {', '.join(missing)} have no robustness numbers; "
                         "run `python src/test.py` without --no-robustness first")
    return runs, sig, metrics


def make_assets():
    """Augmentation examples and a strip of sample digits (from the *train* split)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    (x_tr, y_tr), _ = load_mnist(str(ROOT / "data"))
    idx = [int((y_tr == d).nonzero()[0]) for d in range(10)]
    x = to_input(x_tr[idx])
    g = torch.Generator().manual_seed(7)
    rows = [("Original", x)]
    for name in ("affine", "full"):
        aug = build_augment(name)
        rows += [(plots.AUG_LABELS[name], aug(x, g)) for _ in range(2)]
    fig, axes = plots.plt.subplots(len(rows), 10, figsize=(10, 1.02 * len(rows)))
    for r, (label, xs) in enumerate(rows):
        for j in range(10):
            axes[r, j].imshow(xs[j, 0], cmap="gray_r", vmin=0, vmax=1)
            axes[r, j].axis("off")
        axes[r, 0].text(-0.25, 0.5, label.replace(" + ", " +\n", 1) if r > 2 else label,
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=11, color=plots.INK_2)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.99, bottom=0.01, wspace=0.08, hspace=0.12)
    fig.savefig(ASSETS / "augmentations.png", dpi=160)
    plots.plt.close(fig)
    plots.image_grid([x[j, 0] for j in range(10)], [""] * 10, ASSETS / "digits.png", "", ncols=10)


def pct(v, digits=2):
    return f"{100 * float(v):.{digits}f}%"


# ---- slides ---------------------------------------------------------------
def build(user):
    runs, sig, metrics = load_results()
    make_assets()
    d = Deck(OUT)
    url = REPO_URL.format(user=user)
    name = {"mlp": "MLP", "cnn": "CNN", "vit": "Transformer"}

    # 1. Title
    d.slide("")
    c = d.c
    c.setFillColor(ACCENT)
    c.setFont(BOLD, 12)
    c.drawString(M, H - 120, "ASSIGNMENT #4.1  ·  AI HW FALL 2026")
    c.setFillColor(INK)
    c.setFont(BOLD, 40)
    c.drawString(M, H - 170, "MNIST recognition with")
    c.drawString(M, H - 216, "small neural networks")
    d.text(M, H - 240, 760, "Shallow MLP vs. CNN vs. Transformer encoder, and what data augmentation "
           "buys each of them. Trained on the train split, tested on the test split.", size=17)
    d.image(ASSETS / "digits.png", M - 8, H - 320, 620, 80)
    c.setFillColor(INK)
    c.setFont(BOLD, 14)
    c.drawString(M, 118, "Shengjie Min")
    d.text(M, 92, 800, f"Code, checkpoints and test results: <font color='#2a78d6'>{url}</font>", size=13)

    # 2. Protocol
    y = d.slide("Train on train, test on test", "Setup")
    c = d.c
    bx, by, bw = M, y - 70, 520
    c.setFillColor(FILL)
    c.roundRect(bx, by, bw * 55 / 60, 46, 6, stroke=0, fill=1)
    c.setFillColor(HexColor("#cde2fb"))
    c.roundRect(bx + bw * 55 / 60 + 3, by, bw * 5 / 60 - 3, 46, 6, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(bx + 12, by + 27, "Train split: 60,000 images")
    c.setFont(SANS, 11)
    c.setFillColor(INK_2)
    c.drawString(bx + 12, by + 11, "55,000 used to fit the weights (with or without augmentation)")
    c.drawCentredString(bx + bw * 57.5 / 60, by + 18, "5k val")
    d.text(bx, by - 8, bw, "The 5,000 validation images pick the best epoch. Nothing about training "
           "or model selection ever reads the test set.", size=12)
    ty = by - 90
    c.setFillColor(HexColor("#fbe3d7"))
    c.roundRect(bx, ty, bw * 10 / 60, 46, 6, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(bx + 12, ty + 27, "Test split")
    c.setFont(SANS, 11)
    c.setFillColor(INK_2)
    c.drawString(bx + 12, ty + 11, "10,000")
    d.text(bx + bw * 10 / 60 + 14, ty + 40, bw - 100, "Loaded only by <font name='Mono'>src/test.py</font>, "
           "once per final checkpoint. Results are saved per image in "
           "<font name='Mono'>results/&lt;run&gt;/predictions.csv</font>.", size=12)
    d.table(600, y - 4, [140, 190], [
        ["Shared recipe", ""],
        ["Optimizer", "AdamW"],
        ["Schedule", "One-cycle LR, 15% warm-up"],
        ["Batch size", "128"],
        ["Loss", "Cross-entropy, smoothing 0.1"],
        ["Grad clipping", "1.0"],
        ["Epochs", "MLP 30 · CNN 12 · ViT 20"],
        ["Hardware", "Laptop CPU (4 cores), no GPU"],
        ["Seed", "0 (fixed split + init)"],
    ], size=12, row_h=25)
    d.bullets(M, ty - 34, bw, [
        "Same data pipeline, optimizer and schedule for every model, so differences come from the "
        "architecture and the augmentation.",
        "Each architecture is trained without augmentation and with augmentation (MLP and CNN also "
        "with an affine-only ablation): 8 runs in total.",
    ], size=13)

    # 3. Models
    y = d.slide("Three architectures, three different assumptions", "Models")
    cols = [
        ("mlp", "Shallow MLP", "Every pixel is an independent input. No notion of locality or shift.",
         [["Image 28×28", "flattened to 784"], ["Linear 784 → 512", "ReLU, dropout 0.2"], ["Linear 512 → 10"]]),
        ("cnn", "CNN", "Local 3×3 filters shared across positions: built-in locality and shift tolerance.",
         [["2× Conv 3×3, 32 ch", "BatchNorm + ReLU"], ["MaxPool → 14×14", "2× Conv 3×3, 64 ch"],
          ["MaxPool → 7×7", "Conv 3×3, 128 ch"], ["Global avg pool", "dropout 0.3, Linear → 10"]]),
        ("vit", "Transformer encoder (ViT)", "Image as a sequence of patches; attention learns which patches matter.",
         [["4×4 patches → 49 tokens", "linear embed to 64-d"], ["+ [CLS] token", "+ learned positions"],
          ["4× encoder layer", "4-head self-attn + MLP 128"], ["[CLS] → LayerNorm", "Linear → 10"]]),
    ]
    cw, gap = 272, 24
    for i, (key, title, blurb, blocks) in enumerate(cols):
        x = M + i * (cw + gap)
        c.setFillColor(INK)
        c.setFont(BOLD, 16)
        c.drawString(x, y - 16, title)
        params = int(runs[f"{key}_none"]["params"]) if f"{key}_none" in runs else 0
        c.setFillColor(ACCENT)
        c.setFont(BOLD, 12)
        c.drawString(x, y - 34, f"{params:,} parameters")
        d.text(x, y - 44, cw, blurb, size=11.5)
        bh, bg = 40, 14
        top = y - 104
        for j, lines in enumerate(blocks):
            by = top - j * (bh + bg) - bh
            d.box(x + 16, by, cw - 32, bh, lines, size=11)
            if j:
                d.arrow_down(x + cw / 2, by + bh + bg, by + bh)

    # 4. Augmentation
    y = d.slide("Augmentation: new training images every epoch", "Data")
    d.image(ASSETS / "augmentations.png", M - 10, y - 6, 555, 330)
    d.bullets(M + 570, y - 4, 330, [
        "<b>Affine</b>: rotate ±10°, scale 0.9–1.1, shear ±10°, shift ±10% (2.8 px).",
        "<b>Full</b> = affine + <b>elastic</b> distortion (Simard et al. 2003: smoothed random "
        "displacement, like hand tremor) + <b>random erasing</b> of a small patch (30% of images).",
        "No flips: a mirrored 2 or 7 is not a digit.",
        "Done on the whole batch at once with <font name='Mono'>affine_grid</font> / "
        "<font name='Mono'>grid_sample</font>: about 12 ms per 128 images on the CPU.",
        "Applied to training batches only, never to validation or test images.",
    ], size=13)

    # 5+. result slides are added by add_result_slides()
    add_result_slides(d, runs, sig, metrics, name, url)
    d.save()
    print(f"wrote {OUT} ({d.n} slides)")


def add_result_slides(d, runs, sig, metrics, name, url):
    c = d.c
    acc = {k: float(r["test_acc"]) for k, r in runs.items()}
    err = {k: int(r["test_errors"]) for k, r in runs.items()}
    rob = {k: m["robustness"] for k, m in metrics.items()}

    # 5. Headline results: per model, the configuration with the best *validation* accuracy
    y = d.slide("Results on the 10,000 test images", "Results")
    aug_name = {"none": "no augmentation", "affine": "affine", "full": "full augmentation"}
    for i, key in enumerate(["mlp", "cnn", "vit"]):
        variants = [k for k in runs if k.startswith(key + "_")]
        pick = max(variants, key=lambda k: float(runs[k]["val_acc"]))
        others = ", ".join(f"{aug_name[runs[k]['aug']]} {pct(acc[k])}" for k in variants if k != pick)
        d.stat(M, y - i * 118, 260, f"{name[key]}  ·  {aug_name[runs[pick]['aug']]}", pct(acc[pick]),
               f"{err[pick]} errors. Other settings: {others}.")
    d.text(M, 70, 270, "Each model's setting is the one with the best validation accuracy, so the test set "
           "is not used to choose it.", size=10.5, color=MUTED)
    d.image(RESULTS / "figures" / "test_error.png", 330, y + 4, W - 330 - M + 10, 380)
    if "ensemble" in acc:
        d.text(330, 70, W - 330 - M, f"Averaging the probabilities of the three fully augmented models "
               f"(chosen in advance) gives {pct(acc['ensemble'])} ({err['ensemble']} errors).",
               size=10.5, color=MUTED)

    # 6. Learning curves
    y = d.slide("Training curves (validation error per epoch)", "Results")
    _, h = d.image(RESULTS / "figures" / "learning_curves.png", M - 6, y + 4, W - 2 * M + 12, 300)
    d.bullets(M, y - h - 16, W - 2 * M, LEARNING_NOTES(runs, metrics), size=14, gap=8)

    # 7. Significance
    y = d.slide("Are the differences real?", "Statistics")
    d.text(M, y, 860, "Each pair of models is compared on the same 10,000 test images with an exact McNemar "
           "test: only the images where exactly one of the two models is right carry information.", size=14)
    rows = [["Comparison", "Errors A → B", "Only A right", "Only B right", "p-value", "", "α = 0.05"]]
    for (a, b), r in sig.items():
        p = float(r["p_value"])
        rows.append([f"{a}  vs  {b}", f"{r['errors_a']} → {r['errors_b']}", r["only_a_correct"],
                     r["only_b_correct"], f"{p:.2g}", "", "significant" if p < 0.05 else "not significant"])
    d.table(M + 6, y - 58, [250, 130, 120, 120, 100, 36, 110], rows, size=12, row_h=25,
            align=["l", "r", "r", "r", "r", "l", "l"], shade_rows=[i for i in range(1, len(rows))
                                                               if float(list(sig.values())[i - 1]["p_value"]) < 0.05])
    d.text(M, 70, 860, "Shaded rows: difference is statistically significant at α = 0.05. "
           "With one training seed per configuration, differences of a few errors are within run-to-run noise.",
           size=11.5, color=MUTED)

    # 8-9. Robustness
    for key, fig, title, xs in [
        ("shift_px", "robustness_shift.png", "Shifted test digits: augmentation matters most here", "4"),
        ("rotation_deg", "robustness_rotation.png", "Rotated test digits", "30"),
    ]:
        y = d.slide(title, "Robustness")
        _, h = d.image(RESULTS / "figures" / fig, M - 6, y + 4, W - 2 * M + 12, 300)
        d.bullets(M, y - h - 16, W - 2 * M, ROBUSTNESS_NOTES(key, rob), size=14, gap=8)

    # 10. Error analysis for the best single model by validation accuracy (no test-set peeking)
    best = max((k for k in runs if k != "ensemble"), key=lambda k: float(runs[k]["val_acc"]))
    y = d.slide(f"What is still wrong: {name[runs[best]['model']]}, "
                f"{plots.AUG_LABELS[runs[best]['aug']].lower()}", "Errors")
    d.image(RESULTS / best / "confusion_matrix.png", M - 8, y + 4, 380, 345)
    d.image(RESULTS / best / "misclassified.png", M + 380, y + 2, W - 2 * M - 372, 300)
    d.text(M + 384, y - 306, W - 2 * M - 384, ERROR_NOTES(best, metrics), size=12.5)

    # 11. Takeaways
    y = d.slide("Takeaways", "Summary")
    d.bullets(M, y - 4, W - 2 * M, TAKEAWAYS(runs, metrics), size=16, gap=12)

    # 12. Reproduce
    y = d.slide("Reproduce it", "Code")
    code = [
        "git clone " + url + ".git",
        "cd ai-hw-Fall-2026-nn && pip install -r requirements.txt",
        "",
        "# evaluate the committed checkpoints on the test split (~12 min on a CPU)",
        "python src/test.py",
        "",
        "# train one model (train split only), then test it",
        "python src/train.py --model cnn --aug full",
        "python src/test.py --ckpt checkpoints/cnn_full.pt",
        "",
        "# everything: 8 training runs + testing (~2 h on a 4-core laptop CPU)",
        "./run_all.sh",
    ]
    c.setFillColor(FILL)
    c.roundRect(M, y - 26 - 22 * len(code), W - 2 * M, 22 * len(code) + 24, 6, stroke=0, fill=1)
    ty = y - 30
    for line in code:
        c.setFillColor(MUTED if line.startswith("#") else INK)
        c.setFont(MONO, 13.5)
        c.drawString(M + 18, ty, line)
        ty -= 22
    d.text(M, 92, 860, "<font name='Mono'>src/data.py</font> loading + augmentation  ·  "
           "<font name='Mono'>src/models.py</font> MLP / CNN / ViT  ·  "
           "<font name='Mono'>src/train.py</font>  ·  <font name='Mono'>src/test.py</font>  ·  "
           "<font name='Mono'>results/</font> metrics, per-image predictions, figures", size=12)


# Narrative text for the result slides; numbers are read from the result files.
def _last(metrics_dir, run):
    return json.loads((metrics_dir / run / "train_history.json").read_text())["history"][-1]


def LEARNING_NOTES(runs, metrics):
    mn, vf = _last(RESULTS, "mlp_none"), _last(RESULTS, "vit_full")
    return [
        f"<b>MLP</b> without augmentation memorises the training set ({pct(mn['train_acc'])} train vs "
        f"{pct(mn['val_acc'])} validation in the last epoch). Augmentation makes memorising impossible.",
        "<b>CNN</b> converges in about 8 epochs. With full augmentation it is still behind on clean images "
        "after 12 epochs: the distortions make the task harder and need a longer schedule.",
        f"<b>ViT</b> learns slowest and is still improving at epoch 20. With full augmentation it fits only "
        f"{pct(vf['train_acc'], 1)} of its distorted training batches, so it is under-trained, not over-fitted.",
    ]


def ROBUSTNESS_NOTES(key, rob):
    r = lambda run, k: pct(rob[run][key][k], 1)  # noqa: E731
    if key == "shift_px":
        return [
            f"Moving digits 4 px: <b>MLP {r('mlp_none', '4')} → {r('mlp_affine', '4')}</b> with affine "
            f"augmentation, <b>ViT {r('vit_none', '4')} → {r('vit_full', '4')}</b> with full augmentation.",
            f"The CNN keeps {r('cnn_none', '4')} <i>without</i> augmentation: convolution + pooling build shift "
            "tolerance into the architecture. The MLP and ViT have to learn it from the data.",
            "That is what augmentation is for: it teaches a model the invariance its architecture is missing.",
        ]
    return [
        "Training rotations were only ±10°, so every model degrades beyond about 20°.",
        f"At ±30°: CNN {r('cnn_none', '30')} → {r('cnn_affine', '30')}, MLP {r('mlp_none', '30')} → "
        f"{r('mlp_affine', '30')} (affine), ViT {r('vit_none', '30')} → {r('vit_full', '30')} (full).",
        f"Augmentation does not extrapolate: at ±60° every model is below "
        f"{max(rob[k][key]['60'] for k in rob) * 100:.0f}%, because a rotated 6 and a 9 are genuinely ambiguous.",
    ]


def ERROR_NOTES(best, metrics):
    m = metrics[best]
    cm = m["confusion_matrix"]
    pairs = sorted(((cm[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    top = ", ".join(f"{t} read as {p} ({n}×)" for n, t, p in pairs)
    return (f"Best single model by validation accuracy: <b>{m['test_errors']} errors in 10,000</b>. "
            f"Most frequent confusions: {top}. Many of the remaining mistakes are ambiguous even for a person.")


def TAKEAWAYS(runs, metrics):
    a = lambda k: pct(runs[k]["test_acc"])  # noqa: E731
    e = lambda k: runs[k]["test_errors"]  # noqa: E731
    rob = {k: m["robustness"] for k, m in metrics.items()}
    return [
        f"<b>Architecture matters most.</b> The CNN ({int(runs['cnn_none']['params']) // 1000}k parameters) "
        f"makes {e('cnn_none')}–{e('cnn_full')} errors; the MLP, with 3× more parameters, makes "
        f"{e('mlp_affine')}–{e('mlp_none')}. Locality and weight sharing are the right prior for images.",
        f"<b>Augmentation adds test % where the architecture lacks the invariance</b>: MLP {a('mlp_none')} → "
        f"{a('mlp_affine')} (p &lt; 0.001). For the CNN it did not help clean accuracy at 12 epochs "
        f"({a('cnn_none')} vs {a('cnn_affine')}, not significant; the strongest setting was worse).",
        f"<b>Its biggest payoff is robustness</b>: on 4-px-shifted digits the MLP goes "
        f"{pct(rob['mlp_none']['shift_px']['4'], 0)} → {pct(rob['mlp_affine']['shift_px']['4'], 0)} and the ViT "
        f"{pct(rob['vit_none']['shift_px']['4'], 0)} → {pct(rob['vit_full']['shift_px']['4'], 0)}.",
        f"<b>A small Transformer is data- and compute-hungry</b>: {a('vit_none')} after 20 CPU epochs, still "
        "improving. Without convolution's built-in priors it has to learn locality from scratch.",
        f"<b>Ensembling unequal models did not help</b>: {a('ensemble')} vs {a('cnn_full')} for the CNN "
        "alone (not significant). An average is only as good as its members are comparable.",
    ]


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "Minsj2015")
