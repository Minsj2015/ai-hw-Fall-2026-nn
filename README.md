# ai-hw-Fall-2026-nn: MNIST recognition with small neural networks

Assignment #4.1. Three small networks (a **shallow MLP**, a **CNN** and a **Transformer encoder / ViT**)
are trained on the MNIST **train** split and evaluated on the MNIST **test** split, each with and
without **data augmentation**.

## Results (official MNIST test split, 10,000 images)

| Model | Augmentation | Params | Train time | **Test accuracy** | Test errors | Test digits shifted 4 px | Test digits rotated ±30° |
|---|---|--:|--:|--:|--:|--:|--:|
| MLP | No augmentation | 407,050 | 1.1 min | **98.59%** | 141 | 43.6% | 77.8% |
| MLP | Affine | 407,050 | 1.5 min | **99.04%** | 96 | 94.6% | 85.9% |
| MLP | Affine + elastic + erasing | 407,050 | 2.6 min | **98.88%** | 112 | 93.2% | 83.1% |
| CNN | No augmentation | 140,458 | 15.7 min | **99.64%** ⭐ | 36 | 98.9% | 88.5% |
| CNN | Affine | 140,458 | 14.2 min | **99.60%** | 40 | 99.6% | 94.3% |
| CNN | Affine + elastic + erasing | 140,458 | 14.1 min | **99.48%** | 52 | 99.4% | 92.8% |
| Transformer (ViT) | No augmentation | 139,018 | 26.5 min | **98.19%** | 181 | 51.8% | 80.7% |
| Transformer (ViT) | Affine + elastic + erasing | 139,018 | 33.1 min | **98.17%** | 183 | 95.0% | 86.0% |
| Ensemble (mean of the 3 `full` models) | full | | | **99.37%** | 63 | | |

Single seed (0) per configuration. Exact McNemar tests for every comparison are in
[results/summary.md](results/summary.md); per-image predictions for all 10,000 test images are in
`results/<run>/predictions.csv`.

![Test error by model and augmentation](results/figures/test_error.png)

### What we learned

1. **Architecture matters most.** The CNN (140k parameters) makes 36–52 errors; the shallow MLP, with
   3× more parameters, makes 96–141. Local filters and weight sharing are the right prior for images.
2. **Augmentation adds test % where the architecture lacks the invariance.** MLP: 98.59% → 99.04% with
   affine augmentation (McNemar p = 4×10⁻⁶). For the CNN it did not help clean accuracy at 12 epochs
   (99.64% vs 99.60%, not significant), and the strongest setting was significantly worse (99.48%,
   p = 0.02): elastic + erasing make training harder and would need a longer schedule.
3. **Its biggest payoff is robustness.** On test digits shifted by 4 px the MLP goes 43.6% → 94.6% and
   the ViT 51.8% → 95.0%. The CNN already holds 98.9% without augmentation, because convolution and
   pooling build shift tolerance into the architecture.
4. **A small Transformer is data- and compute-hungry.** 98.2% after 20 CPU epochs and still improving;
   without convolution's built-in priors it has to learn locality from scratch.
5. **Averaging unequal models did not help.** The 3-model ensemble (99.37%) is not better than the CNN
   alone (99.48%, p = 0.15).

![Accuracy on shifted test digits](results/figures/robustness_shift.png)

More figures: [learning curves](results/figures/learning_curves.png),
[rotated test digits](results/figures/robustness_rotation.png), and per-run confusion matrices and
misclassified examples, e.g. [`results/cnn_none/`](results/cnn_none/).

**Presentation:** [presentation/MNIST_NN_presentation.pdf](presentation/MNIST_NN_presentation.pdf) (12 slides).


## Protocol: train on train, test on test

| Split | Images | Used for |
|---|--:|---|
| official train, part 1 | 55,000 | fitting the weights (augmented on the fly when enabled) |
| official train, part 2 | 5,000 | validation: choosing the best epoch |
| official **test** | 10,000 | **only** `src/test.py`, once per final checkpoint |

`src/train.py` loads the test split and immediately discards it, so neither training nor checkpoint
selection can see a test image. Every model uses the same recipe: AdamW, one-cycle learning-rate
schedule (15% warm-up), batch size 128, cross-entropy with label smoothing 0.1, gradient clipping at 1.0,
seed 0. Everything ran on a 4-core laptop CPU (Intel i5, no GPU).

## Models (`src/models.py`)

| Model | Architecture | Parameters |
|---|---|--:|
| Shallow MLP | 784 → 512 (ReLU, dropout 0.2) → 10: one hidden layer | 407,050 |
| CNN | 2× conv3×3(32) → maxpool → 2× conv3×3(64) → maxpool → conv3×3(128) → global avg pool → 10, BatchNorm + ReLU after every conv | 140,458 |
| Transformer encoder (ViT) | 4×4 patches → 49 tokens embedded to 64-d, + [CLS] token + learned positions → 4 pre-norm encoder layers (4-head self-attention, MLP 128, GELU) → LayerNorm → 10 | 139,018 |

## Augmentation (`src/data.py`)

Done on whole batches with `affine_grid` / `grid_sample` (≈12 ms per 128 images on a CPU), training batches only.

| Setting | What it does |
|---|---|
| `none` | original images |
| `affine` | random rotation ±10°, scale 0.9–1.1, shear ±10°, translation ±10% (≈2.8 px) |
| `full` | `affine` + elastic distortion (Simard et al., 2003; α = 2 px, σ = 4) + random erasing (30% of images, 2–10% of the area) |

No horizontal flips: a mirrored digit is a different (or invalid) symbol.

## How to run

```bash
pip install -r requirements.txt          # torch, torchvision, numpy, matplotlib

# test the committed checkpoints on the official test split (MNIST downloads automatically)
python src/test.py                       # all checkpoints, ~12 min on a CPU (--no-robustness: ~1 min)
python src/test.py --ckpt checkpoints/cnn_full.pt

# train (train split only); writes checkpoints/<model>_<aug>.pt and results/<run>/train_history.json
python src/train.py --model mlp --aug none
python src/train.py --model cnn --aug full --epochs 12
python src/train.py --model vit --aug full

# everything: 8 training runs + testing
./run_all.sh                             # ~2 h on a 4-core CPU; PYTHON=/path/to/python ./run_all.sh

# rebuild the slides from results/ (needs reportlab and a full test.py run, i.e. with robustness)
python presentation/build_presentation.py
```

## Repository layout

```
src/data.py        MNIST loading, train/validation split, batched augmentation
src/models.py      MLP, CNN, ViT
src/train.py       training script (train split only)
src/test.py        evaluation on the test split, robustness tests, McNemar tests, ensemble, figures
src/plots.py       figure styling
run_all.sh         reproduces every experiment
checkpoints/       trained weights (best validation epoch) + training history
results/           summary.md / summary.csv / significance.csv, figures/, and per run:
                   metrics.json, predictions.csv (all 10,000 test images), confusion_matrix.png, misclassified.png
logs/              console output of the training and test runs
presentation/      MNIST_NN_presentation.pdf and the script that builds it
```
