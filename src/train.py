"""Train one model on the MNIST train split.

The official 60k training set is split into 55k for training and 5k for
validation; the validation set picks the checkpoint. The 10k test set is never
loaded here, so it is only touched by test.py.

Examples:
    python src/train.py --model mlp --aug none
    python src/train.py --model cnn --aug full --epochs 12
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import AUGMENTATIONS, build_augment, iterate, load_mnist, normalize, to_input, train_val_split
from models import build_model, count_params

DEFAULTS = {
    "mlp": dict(epochs=30, lr=1e-3, weight_decay=1e-4),
    "cnn": dict(epochs=12, lr=3e-3, weight_decay=5e-4),
    "vit": dict(epochs=20, lr=1e-3, weight_decay=5e-2),
}


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    loss, correct = 0.0, 0
    for xb, yb in iterate(x, y, batch_size):
        out = model(normalize(to_input(xb)))
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss / len(x), correct / len(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=DEFAULTS)
    p.add_argument("--aug", default="none", choices=AUGMENTATIONS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--name", help="run name (default: <model>_<aug>)")
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.name = args.name or f"{args.model}_{args.aug}"

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    g = torch.Generator().manual_seed(args.seed)

    (x_all, y_all), _ = load_mnist(args.data)  # test split deliberately discarded
    (x_tr, y_tr), (x_val, y_val) = train_val_split(x_all, y_all, val_size=5000, seed=0)
    augment = build_augment(args.aug)

    model = build_model(args.model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(x_tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.15)
    print(f"[{args.name}] {count_params(model):,} params | train {len(x_tr)} / val {len(x_val)} | "
          f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} aug={args.aug}", flush=True)

    history, best = [], None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, tot_correct = time.time(), 0.0, 0
        for xb, yb in iterate(x_tr, y_tr, args.batch_size, shuffle=True, generator=g):
            xb = to_input(xb)
            if augment is not None:
                xb = augment(xb, g)
            out = model(normalize(xb))
            loss = F.cross_entropy(out, yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()

        val_loss, val_acc = evaluate(model, x_val, y_val)
        rec = dict(epoch=epoch, train_loss=tot_loss / len(x_tr), train_acc=tot_correct / len(x_tr),
                   val_loss=val_loss, val_acc=val_acc, epoch_time_s=time.time() - t0)
        history.append(rec)
        if best is None or val_acc >= best["val_acc"]:  # ties go to the later (lower-LR) epoch
            best = dict(epoch=epoch, val_acc=val_acc,
                        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"[{args.name}] epoch {epoch:2d}/{args.epochs} | train loss {rec['train_loss']:.4f} "
              f"acc {rec['train_acc']:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{rec['epoch_time_s']:.0f}s", flush=True)

    train_time = time.time() - t_start
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(name=args.name, model=args.model, aug=args.aug, args=vars(args),
                    params=count_params(model), best_epoch=best["epoch"], best_val_acc=best["val_acc"],
                    train_time_s=train_time, history=history, state_dict=best["state_dict"]),
               out_dir / f"{args.name}.pt")
    log_dir = Path("results") / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_history.json").write_text(json.dumps(
        dict(args=vars(args), best_epoch=best["epoch"], best_val_acc=best["val_acc"],
             train_time_s=train_time, history=history), indent=2))
    print(f"[{args.name}] done in {train_time / 60:.1f} min | best val acc {best['val_acc']:.4f} "
          f"(epoch {best['epoch']}) -> {out_dir / (args.name + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
