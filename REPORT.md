# DASE7506 MP1 — Small Language Model Challenge

**Final test bits per byte (protocol `7506-mp1-wt2-v2`, FP32, CPU): 1.5139**
(baseline ≈ 2.10; a 0.59 BPB / 28 % reduction).

All numbers below are **validation** BPB unless a row is marked *test*. The
method was selected on validation only; the test split was scored once, after
freezing, using the supplied `evaluate.py` on CPU with FP32.

---

## 1. Summary

We train a decoder-only transformer from scratch on the supplied WikiText-2
BPE-2048 data. The work has three parts:

1. **Diagnosis.** The baseline is tiny (1.09 M parameters) and trains for only
   1,200 updates = 9.83 M tokens ≈ 2.7 epochs. It is both under-parameterised
   and under-trained, and its SwiGLU feed-forward wastes parameters on this
   small corpus.
2. **Method.** A compute-matched modern architecture — pre-norm blocks with
   **RMSNorm**, **RoPE**, bias-free linears, tied embeddings and a **GELU MLP** —
   trained longer with decoupled weight decay, cosine decay, **EMA** and
   **validation-based checkpoint selection** (early stopping). Every choice is a
   config flag, so the same code path produces the baseline-style control.
3. **Result.** At an identical token budget the architecture alone improves
   validation BPB from **2.0795 → 1.6970**. With a GELU MLP, 8 layers, dropout
   0.2 and validation-selected checkpoints the final model reaches
   **1.4968 validation / 1.5139 test**, within all evaluation limits
   (3.7× baseline CPU time, 0.56 GiB RAM, 25 MB assets).

The dominant contributor is **capacity allocation plus the MLP choice**, not the
positional or normalisation choices: RoPE, RMSNorm and QK-norm are individually
worth ≤ 0.003 BPB, whereas replacing SwiGLU with GELU is worth 0.029 BPB *and*
removes 1.18 M parameters. Longer training only helps once overfitting is
controlled by dropout and validation-based checkpoint selection.

---

## 2. Setup

- Data: supplied `wikitext_train/validation/test.txt` (3.61 M / 0.38 M / 0.43 M
  BPE-2048 tokens), fixed tokenizer, context 256, vocab 2048. We never train on
  validation or test text and never tune on test.
- Evaluation: the supplied `evaluate.py`, FP32, independent 256-token windows,
  CPU. Validation is used for every development decision.
- Compute: Kaggle free GPU (Tesla T4). Training is CPU/GPU; ranked evaluation is
  CPU FP32.

**Equal-token convention.** One "step" is 32 sequences × 256 targets = 8,192
tokens. The classroom baseline trains 1,200 steps = 9.83 M targets; we call this
the *equal-token* budget and use it for the headline architecture comparison.

---

## 3. Method

### 3.1 Architecture (`code/student.py`)

A pre-norm decoder. Relative to the classroom baseline we change:

| Component | Baseline | Ours | Config flag |
|---|---|---|---|
| Normalisation | LayerNorm | **RMSNorm** | `norm` |
| Positions | learned absolute | **RoPE** | `pos_emb` |
| Feed-forward | GELU MLP | **SwiGLU** or **GELU MLP** | `mlp` |
| Linear bias | yes | **no** | — |
| Embeddings | tied | tied | `tied` |
| Q/K normalisation | no | **RMSNorm on Q,K** | `qk_norm` |

The model exposes exactly the two required interfaces (`forward` for training
logits, `predict_log_probs` for normalised evaluation probabilities); both are
strictly causal and keep no state between calls, so independent windows start
fresh. All variants pass the supplied `tests/test_contract.py` (causality,
normalisation, example independence, state reset, gradients).

### 3.2 Training recipe (`code/train_student.py`)

AdamW (β = 0.9/0.95, eps 1e-8), decoupled weight decay 0.1 on matrices and 0 on
norms/embeddings, gradient clipping 1.0, linear warmup then cosine decay to 10 %
of the peak LR, dropout, and an **EMA** (decay 0.999) of the weights. EMA is
cheap and costs nothing at inference.

### 3.3 Validation-based checkpoint selection

Training longer than ~8k steps overfits WikiText-2. Rather than fix a step
count, we score the model (raw and EMA) on validation every 2,000 steps and
**keep the state with the best validation BPB**. The submitted `checkpoint.pt`
is that selected state. This turns an unstable long run into a monotone
improvement and is the mechanism that makes the 20k-step final model better
than the 8k model.

---

## 4. Experiments and results

### 4.1 Baseline reproduction

The supplied recipe gives **2.0724 validation**; our re-run of the baseline
architecture under the improved recipe gives **2.0795 validation** — consistent
with the stated ≈ 2.10 test BPB, confirming data, tokenizer and scorer.

### 4.2 Equal-token architecture comparison (9.83 M targets)

| Model | Params | Validation BPB |
|---|---:|---:|
| Baseline architecture (`model.py`) | 1.09 M | 2.0795 |
| Ours (SwiGLU, width 256, depth 6) | 5.64 M | **1.6970** |

A 0.38 BPB gain at the same number of processed training targets. This is the
comparison the guide asks for.

### 4.3 Mechanism ablations (8,000 steps = 65.5 M targets, width 256)

| Variant | Params | Validation BPB | Δ vs full |
|---|---:|---:|---:|
| Full (SwiGLU, depth 6) | 5.64 M | 1.5560 | — |
| **GELU MLP** instead of SwiGLU | 4.46 M | **1.5271** | **−0.029** |
| no QK-norm | 5.64 M | 1.5570 | +0.001 |
| LayerNorm instead of RMSNorm | 5.64 M | 1.5583 | +0.002 |
| no RoPE (learned positions) | 5.71 M | 1.5593 | +0.003 |
| no EMA | 5.64 M | 1.5656 | +0.010 |

RoPE, RMSNorm and QK-norm are individually neutral at this scale. The MLP is
the one architectural change that matters, and it *reduces* parameters. This is
the mechanism ablation: swapping SwiGLU→GELU (and keeping everything else) is
worth 0.029 BPB with 21 % fewer parameters.

### 4.4 Capacity study (8,000 steps)

| Width / depth | Params | Validation BPB |
|---|---:|---:|
| 256 / 4 | 3.93 M | 1.5665 |
| 256 / 5 | 4.79 M | 1.5625 |
| 256 / 6 | 5.64 M | 1.5560 |
| **256 / 8** | 7.34 M | **1.5418** |
| 192 / 6 | 3.27 M | 1.5494 |
| 320 / 4 | 8.65 M | 1.5523 |

Depth is a better use of the budget than width at this scale (256/8 beats both
192/6 and 320/4).

### 4.5 GELU refinement, regularisation and training length

| Variant | Validation BPB |
|---|---:|
| GELU, depth 6, 8k | 1.5271 |
| GELU, depth 8, 8k | 1.5143 |
| GELU, depth 10, 8k | 1.5162 |
| GELU, depth 8, dropout 0.2, 8k | 1.5126 |
| **GELU, depth 8, dropout 0.2, 20k + selection** | **1.4968** |

Depth 8 is the sweet spot (depth 10 is no better and leaves the time budget).
Longer training only helps with the extra dropout and checkpoint selection.

### 4.6 Final model and test result

Final config: `{width 256, heads 8, depth 8, context 256, mlp_hidden 768,
dropout 0.2, qk_norm true, mlp gelu}`, 5,772,032 parameters, 20,000 steps
(163.8 M targets), seed 17, EMA state selected at step 14,000.

| Seed | Validation BPB |
|---|---:|
| 17 (submitted) | **1.4968** |
| 42 | 1.4973 |
| 123 | 1.5006 |

**Test BPB (FP32, CPU, `evaluate.py`): 1.5139.**

Weight averaging (a "model soup") of the three seeds was tried and **failed**
(validation BPB 3.44, near random): independently trained models sit in
different loss basins, so naive weight averaging destroys the function. This is
reported as a negative result.

---

## 5. Resource measurements (frozen predictor)

Measured on the same machine, 4 threads, FP32, validation split:

| Limit | Measured | Allowed |
|---|---:|---:|
| CPU scoring time vs baseline | **3.7–3.9×** | ≤ 5× |
| Peak process RAM | **0.56 GiB** | ≤ 4 GiB |
| Uncompressed inference assets | **25.2 MB** (`checkpoint.pt`) | ≤ 64 MiB |

The analytic per-token MAC ratio (6.29 M vs 1.44 M) is 4.36×; measured wall-time
ratios were 3.7–3.9×.

---

## 6. Critical analysis

- **What the comparisons establish.** At equal tokens the architecture gain is
  0.38 BPB; the ablation isolates 0.029 of it to the MLP and shows positional /
  normalisation choices are neutral; the capacity study shows depth > width; and
  the training study shows longer training only helps under regularisation with
  validation-based selection.
- **Why GELU beats SwiGLU here.** WikiText-2 training text is only 3.6 M tokens,
  so a 5–7 M parameter model is data-limited. SwiGLU's extra gate parameters
  overfit; the smaller GELU MLP generalises better and is cheaper, which is a
  genuine trade-off between prediction quality and computational cost rather
  than a free win.
- **Why longer training hurt without selection.** 20k steps at dropout 0.1 was
  *worse* (1.5750) than 8k (1.5560); the validation curve peaked near step 10k.
  Dropout 0.2 plus selecting the validation-best checkpoint turned the same
  budget into 1.4968.
- **Limitations.** We did not tune the learning rate or schedule extensively;
  the gains are from architecture, capacity and regularisation. Ensembling is
  disallowed by the time budget, and weight averaging across seeds fails, so the
  submitted predictor is a single model. Differences of ~0.003 BPB between
  seeds are within run-to-run noise and should not be over-interpreted.

---

## 7. Reproduction

```bash
cd code
python -m venv .venv && .venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v                     # contract tests

# evaluate the frozen final checkpoint (no retraining)
python evaluate.py --checkpoint ../final/checkpoint.pt --device cpu --precision fp32 --split test

# retrain the final model from scratch (GPU or CPU); dropout 0.2 is in the config
python train_student.py --implementation student --config configs/final.json \
    --device cuda --steps 20000 --batch-size 32 --lr 0.001 --warmup 500 \
    --eval-every 2000 --run-dir runs/final
```

`configs/final.json` is the submitted configuration. The checkpoint records the
implementation module and config, so `evaluate.py` reconstructs the predictor
exactly; no optimizer state is needed.

---

## 8. AI assistance disclosure

This project used an AI coding assistant (opencode) as a tool. The human author specified the goal, chose the experimental direction, ran all training and evaluation on Kaggle GPUs, selected the final configuration from the evidence, verified reproducibility, and takes full responsibility for the implementation. The assistant helped write and refactor the model/training/evaluation code, the experiment driver and analysis scripts, and drafts of the report and README.

---

## 9. References

1. Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models*, 2016
   (WikiText-2).
2. Radford et al. *Language Models are Unsupervised Multitask Learners*, 2019
   (pre-norm GPT, tied embeddings).
3. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
   2021 (RoPE).
4. Zhang, Sennrich. *Root Mean Square Layer Normalization*, 2019.
5. Shazeer. *GLU Variants Improve Transformer*, 2020 (SwiGLU; we find GELU
   better at this scale).
6. Wortsman et al. *Model Soups*, 2022 (weight averaging; found to fail for
   independently trained models here).
