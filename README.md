# DASE7506 MP1 — Small Language Model Challenge

A decoder-only transformer trained from scratch on the supplied WikiText-2
BPE-2048 data.

**Final test BPB (protocol `7506-mp1-wt2-v2`, FP32, CPU): 1.5139**
(classroom baseline ≈ 2.10).

Validation BPB of the final model: **1.4968**. Measured limits for the frozen
predictor: **3.7–3.9×** baseline CPU scoring time, **0.56 GiB** peak RAM,
**25.2 MB** uncompressed assets — all inside the 5× / 4 GiB / 64 MiB budget.

The full write-up is in [`REPORT.md`](REPORT.md).

---

## Repository layout

```
code/
  student.py          model (RMSNorm, RoPE, GELU MLP, tied embeddings, QK-norm)
  train_student.py    training recipe (AdamW, warmup+cosine, EMA, val selection)
  suite.py            runs a JSON list of experiments and writes a summary
  soup.py             weight averaging across checkpoints (negative result)
  measure_resources.py  measures the three evaluation limits
  configs/final.json  submitted configuration
  common.py, evaluate.py, model.py, data/, tests/   supplied, unchanged
final/
  checkpoint.pt       frozen predictor (implementation + config + weights)
  test_cpu_fp32.json  test-split result
results/              per-experiment summary tables
```

## Install

Use Python 3.12. From `code/`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Evaluate the frozen checkpoint (no retraining)

```powershell
python evaluate.py --checkpoint ..\final\checkpoint.pt --device cpu --precision fp32 --split test
```

`evaluate.py` rebuilds the predictor from the module and config stored in the
checkpoint, so this reproduces the reported **1.5139 test BPB**.

## Retrain the final model

```powershell
python train_student.py --implementation student --config configs/final.json `
    --device cuda --steps 20000 --batch-size 32 --lr 0.001 --warmup 500 `
    --eval-every 2000 --run-dir runs/final
```

Training writes `checkpoint.pt` (the validation-best state), `checkpoint_raw.pt`,
`checkpoint_ema.pt` and `metrics.json` into the run directory. Use validation for
all selection; score test only once the method is frozen.

## Method in one paragraph

The classroom baseline (1.09 M params, 1,200 steps ≈ 2.7 epochs) is
under-parameterised and under-trained. We use a compute-matched modern
architecture (RMSNorm, RoPE, bias-free linears, tied embeddings, and a **GELU**
feed-forward instead of SwiGLU) and train longer with decoupled weight decay,
cosine decay, dropout and **EMA**, keeping the **validation-best checkpoint**.
At equal tokens this improves validation BPB from 2.0795 to 1.6970; the final
model (width 256, depth 8, dropout 0.2, 20k steps) reaches 1.4968 validation /
1.5139 test. Ablations show the MLP choice is the main architectural lever
(GELU beats SwiGLU by 0.029 BPB with fewer parameters), depth is worth more than
width, RoPE/RMSNorm/QK-norm are individually neutral, and longer training only
helps with regularisation and validation-based checkpoint selection.

## Data attribution

WikiText-2 (Merity et al., 2016), text by Wikipedia contributors; CC BY-SA 3.0
and GNU FDL. The supplied `data/` splits, tokenizer and evaluator are unchanged.

## AI assistance disclosure

Substantive AI assistance (opencode) was used to write `student.py`,
`train_student.py`, `suite.py`, `soup.py`, `measure_resources.py`, the Kaggle
experiment notebook, and drafts of this README and `REPORT.md`. The human author
ran all training and evaluation, provided the compute, reviewed the code, and is
responsible for the submission. All reported numbers were produced by the
supplied evaluation pipeline. See `REPORT.md` §8.
