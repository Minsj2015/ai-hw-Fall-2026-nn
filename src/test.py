"""Evaluate trained checkpoints on the official MNIST test split (10,000 images).

For every checkpoint this writes results/<run>/:
    metrics.json          accuracy, loss, per-class precision/recall/F1, confusion matrix,
                          accuracy on rotated / shifted test images
    predictions.csv       true label, predicted label and confidence for every test image
    confusion_matrix.png  where the errors are
    misclassified.png     the most confident mistakes
Then it rebuilds results/summary.md, results/summary.csv and results/figures/*.png
from every run evaluated so far.

    python src/test.py                                   # every checkpoint in checkpoints/
    python src/test.py --ckpt checkpoints/cnn_full.pt    # a single checkpoint
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import plots
from data import load_mnist, normalize, to_input
from models import build_model

ROTATIONS = [0, 10, 20, 30, 40, 50, 60]          # degrees; each tested clockwise and counter-clockwise
SHIFTS = [0, 1, 2, 3, 4, 5, 6]                   # pixels; each tested up, down, left and right
ENSEMBLE = ["mlp_full", "cnn_full", "vit_full"]  # fixed in advance, not picked by looking at test results


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


@torch.no_grad()
def predict_proba(model, x, batch_size=1000):
    return torch.cat([F.softmax(model(normalize(x[i:i + batch_size])), 1)
                      for i in range(0, len(x), batch_size)])


def warp(x, angle=0.0, dx=0, dy=0):
    """Rotate by `angle` degrees and shift by (dx, dy) pixels; uncovered area is black."""
    a = math.radians(angle)
    h, w = x.shape[-2:]
    theta = torch.tensor([[math.cos(a), -math.sin(a), -2.0 * dx / w],
                          [math.sin(a), math.cos(a), -2.0 * dy / h]], dtype=x.dtype)
    grid = F.affine_grid(theta.expand(len(x), 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def robustness(model, x, y):
    def acc(xx):
        return (predict_proba(model, xx).argmax(1) == y).float().mean().item()

    clean = acc(x)
    rotation = {a: (acc(warp(x, a)) + acc(warp(x, -a))) / 2 if a else clean for a in ROTATIONS}
    shift = {s: sum(acc(warp(x, dx=dx, dy=dy)) for dx, dy in [(s, 0), (-s, 0), (0, s), (0, -s)]) / 4
             if s else clean for s in SHIFTS}
    return dict(rotation_deg=rotation, shift_px=shift)


def classification_report(cm):
    cm = cm.double()
    tp = cm.diag()
    precision, recall = tp / cm.sum(0).clamp_min(1), tp / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {str(d): dict(precision=round(precision[d].item(), 4), recall=round(recall[d].item(), 4),
                         f1=round(f1[d].item(), 4), support=int(cm[d].sum().item())) for d in range(10)}


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on paired per-image correctness."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def evaluate_checkpoint(path, x, y, out_root, with_robustness=True):
    model, ckpt = load(path)
    name = ckpt["name"]
    t0 = time.time()
    proba = predict_proba(model, x)
    ms_per_image = (time.time() - t0) / len(x) * 1000
    conf, pred = proba.max(1)
    correct = pred == y
    cm = torch.zeros(10, 10, dtype=torch.long)
    cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)

    metrics = dict(
        name=name, model=ckpt["model"], aug=ckpt["aug"], params=ckpt["params"],
        epochs=ckpt["args"]["epochs"], best_epoch=ckpt["best_epoch"], best_val_acc=ckpt["best_val_acc"],
        train_time_s=round(ckpt["train_time_s"], 1),
        test_images=len(y), test_acc=correct.float().mean().item(), test_errors=int((~correct).sum()),
        test_nll=F.nll_loss(proba.clamp_min(1e-12).log(), y).item(),
        inference_ms_per_image=round(ms_per_image, 4),
        per_class=classification_report(cm), confusion_matrix=cm.tolist(),
    )
    if with_robustness:
        metrics["robustness"] = robustness(model, x, y)

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "prediction", "confidence"])
        w.writerows([i, int(t), int(p_), f"{c_:.4f}"] for i, (t, p_, c_) in enumerate(zip(y, pred, conf)))
    title = f"{plots.MODEL_LABELS[ckpt['model']]} / {plots.AUG_LABELS[ckpt['aug']].lower()}"
    plots.confusion_matrix(cm, out / "confusion_matrix.png",
                           f"{title}: {metrics['test_errors']} errors / {len(y):,}")
    wrong = (~correct).nonzero().squeeze(1)
    wrong = wrong[conf[wrong].argsort(descending=True)][:40]
    plots.image_grid([x[i, 0] for i in wrong],
                     [f"true {y[i]} / pred {pred[i]}\n{100 * conf[i]:.0f}% sure" for i in wrong],
                     out / "misclassified.png",
                     f"{title}: most confident mistakes")
    print(f"{name:12s} test acc {metrics['test_acc']:.4f} ({metrics['test_errors']} errors) | "
          f"val acc {metrics['best_val_acc']:.4f} | {ms_per_image:.3f} ms/image", flush=True)
    return proba


def summarize(out_root, ckpt_dir, x, y):
    out_root = Path(out_root)
    runs = []
    for m in sorted(out_root.glob("*/metrics.json")):
        r = json.loads(m.read_text())
        hist = m.parent / "train_history.json"
        r["history"] = json.loads(hist.read_text())["history"] if hist.exists() else []
        with open(m.parent / "predictions.csv") as f:
            r["correct"] = torch.tensor([row["label"] == row["prediction"] for row in csv.DictReader(f)])
        runs.append(r)
    order = {(m, a): i for i, (m, a) in enumerate((m, a) for m in plots.MODEL_LABELS for a in plots.AUG_COLORS)}
    runs.sort(key=lambda r: order.get((r["model"], r["aug"]), 99))
    by_name = {r["name"]: r for r in runs}

    ensemble = None
    if all((Path(ckpt_dir) / f"{n}.pt").exists() for n in ENSEMBLE):
        proba = sum(predict_proba(load(Path(ckpt_dir) / f"{n}.pt")[0], x) for n in ENSEMBLE) / len(ENSEMBLE)
        correct = proba.argmax(1) == y
        ensemble = dict(members=ENSEMBLE, test_acc=correct.float().mean().item(),
                        test_errors=int((~correct).sum()), correct=correct)
        print(f"{'ensemble':12s} test acc {ensemble['test_acc']:.4f} ({ensemble['test_errors']} errors) "
              f"= mean softmax of {', '.join(ENSEMBLE)}")

    def rob(r, key, k):
        v = r.get("robustness", {}).get(key, {}).get(str(k))
        return f"{100 * v:.2f}" if v is not None else "-"

    lines = [
        "# MNIST test results",
        "",
        "Every model is trained on 55,000 images from the official **train** split; the checkpoint is",
        "chosen on the remaining 5,000 train images (validation). The official **test** split",
        "(10,000 images) is used only by `src/test.py`, once per final checkpoint.",
        "",
        "| Run | Model | Augmentation | Params | Epochs (best) | Train time | Val acc % | **Test acc %** "
        "| Test errors | Rotated ±30° acc % | Shifted 4 px acc % |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in runs:
        lines.append(
            f"| `{r['name']}` | {plots.MODEL_LABELS[r['model']]} | {plots.AUG_LABELS[r['aug']]} | {r['params']:,} "
            f"| {r['epochs']} ({r['best_epoch']}) | {r['train_time_s'] / 60:.1f} min | {100 * r['best_val_acc']:.2f} "
            f"| **{100 * r['test_acc']:.2f}** | {r['test_errors']} | {rob(r, 'rotation_deg', 30)} "
            f"| {rob(r, 'shift_px', 4)} |")
    if ensemble:
        lines.append(f"| `ensemble` | MLP + CNN + ViT | each with full augmentation | - | - | - | - "
                     f"| **{100 * ensemble['test_acc']:.2f}** | {ensemble['test_errors']} | - | - |")

    pairs = [(f"{m}_none", f"{m}_{a}") for m in plots.MODEL_LABELS for a in ("affine", "full")]
    pairs += [("mlp_full", "cnn_full"), ("cnn_full", "vit_full"), ("mlp_full", "vit_full")]
    if ensemble:
        pairs.append(("cnn_full", "ensemble"))
    rows, sig = [], []
    for a, b in pairs:
        ra = by_name.get(a) or (ensemble if a == "ensemble" else None)
        rb = by_name.get(b) or (ensemble if b == "ensemble" else None)
        if ra and rb:
            only_a, only_b, p = mcnemar(ra["correct"], rb["correct"])
            verdict = "significant" if p < 0.05 else "not significant"
            rows.append(f"| `{a}` vs `{b}` | {ra['test_errors']} → {rb['test_errors']} | {only_a} | {only_b} "
                        f"| {p:.3g} | {verdict} |")
            sig.append([a, b, ra["test_errors"], rb["test_errors"], only_a, only_b, p])
    with open(out_root / "significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_a", "run_b", "errors_a", "errors_b", "only_a_correct", "only_b_correct", "p_value"])
        w.writerows(sig)
    if rows:
        lines += [
            "",
            "## Are the differences real? (exact McNemar test on the same 10,000 test images)",
            "",
            "| Comparison (A vs B) | Errors A → B | Only A correct | Only B correct | p-value | at α = 0.05 |",
            "|---|--:|--:|--:|--:|--:|",
        ] + rows

    lines += ["", "## Figures", ""]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    plots.test_error_bars(runs, fig_dir / "test_error.png", ensemble)
    lines.append("![Test error](figures/test_error.png)")
    if all(r["history"] for r in runs):
        plots.learning_curves(runs, fig_dir / "learning_curves.png")
        lines.append("![Learning curves](figures/learning_curves.png)")
    if all("robustness" in r for r in runs):
        plots.robustness(runs, "rotation_deg", "Rotation of test images (± degrees)",
                         fig_dir / "robustness_rotation.png")
        plots.robustness(runs, "shift_px", "Shift of test images (pixels)", fig_dir / "robustness_shift.png")
        lines += ["![Rotation robustness](figures/robustness_rotation.png)",
                  "![Shift robustness](figures/robustness_shift.png)"]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")

    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "aug", "params", "epochs", "best_epoch", "train_time_min", "val_acc",
                    "test_acc", "test_errors", "rot30_acc", "shift4_acc"])
        for r in runs:
            rb = r.get("robustness", {})
            w.writerow([r["name"], r["model"], r["aug"], r["params"], r["epochs"], r["best_epoch"],
                        round(r["train_time_s"] / 60, 2), r["best_val_acc"], r["test_acc"], r["test_errors"],
                        rb.get("rotation_deg", {}).get("30"), rb.get("shift_px", {}).get("4")])
        if ensemble:
            w.writerow(["ensemble", "+".join(ENSEMBLE), "full", "", "", "", "", "",
                        ensemble["test_acc"], ensemble["test_errors"], "", ""])
    print(f"wrote {out_root / 'summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", help="checkpoint files (default: every checkpoints/*.pt)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--no-robustness", action="store_true", help="skip the rotated/shifted test sets")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    _, (x_test, y_test) = load_mnist(args.data)  # official test split only
    x = to_input(x_test)
    paths = args.ckpt or sorted(Path(args.ckpt_dir).glob("*.pt"))
    if not paths:
        raise SystemExit(f"no checkpoints found in {args.ckpt_dir}/ - run src/train.py first")
    for path in paths:
        evaluate_checkpoint(path, x, y_test, args.out, with_robustness=not args.no_robustness)
    summarize(args.out, args.ckpt_dir, x, y_test)


if __name__ == "__main__":
    main()
