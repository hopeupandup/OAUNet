# OAUNet — Selective Online Adaptation for Radar-Echo Nowcasting

Reference implementation for the manuscript

> *Selective Online Adaptation for Radar-Echo Nowcasting under Controlled Drift:
> Budget and Boundary Conditions*

This repository contains the **complete, self-contained code** needed to
reproduce the paper's experiments: the model, the online-adaptation framework,
all baselines, the drift-stream construction protocol and the dBZ evaluation
pipeline. No data or trained weights are shipped — see [Data preparation](#4-data-preparation).

---

## 1. What is here

OAUNet is a **selective** online-adaptation wrapper around a TrajGRU
radar-echo extrapolator. Four reported components can be switched on or off
independently, and one experimental module is retained for completeness (see
`components/`):

| Component | File | Role |
|---|---|---|
| Online variational inference | `components/online_vi.py` | Anchored-prior Bayesian update of the decoder; the anchor prior `N(θ₀, σ₀²)` acts as an anti-forgetting regulariser |
| Uncertainty-gated active learning | `components/active_learning.py` | Decides **whether** to adapt on a given sample (budget control) |
| ADWIN drift detection | `components/drift_detection.py` | Detects mean shifts in the bounded drift signal and classifies abrupt vs. gradual |
| Online variance recalibration (OVR) | `components/variance_recalibration.py` | Gradient-free per-sample scale correction of the predictive σ |
| Meta-learning (MAML) | `components/meta_learning.py` | **Experimental option, off by default.** Not part of the reported method; kept for completeness. See [Experimental option](#experimental-option-meta-learning) |

The main entry points are `train.py` (base model) and `run_experiments.py`
(offline / online / periodic-retrain baselines plus the ablation suite).

### Experimental option: meta-learning

`components/meta_learning.py` is retained but **disabled in the default
configuration**, and it is not part of the method reported in the paper. Two
reasons are worth recording for anyone reading the code:

- The reported method is the three used components: online VI, the gated
  update rule, and drift detection.
- With the module enabled, its drift-time branch fires on 4 of 3000 steps
  (0.13 %) and its removal changes MAE by 0.03 dBZ, below the three-seed
  spread of the main arms (0.05 dBZ).

Setting `enable_meta_learning: true` in `configs/oaunet_temporal.yaml` restores
two things: the module is instantiated and may act at detected abrupt drift,
and `results/checkpoints/meta_trained.pt` is loaded as the shared starting
point of the selective arms (it is generated once, on first use, by episodic
pre-training of the base model). The default configuration therefore starts the
selective arms from the plain base-model checkpoint; the 0.03 dBZ figure above
is the combined effect of those two roles, so it also bounds the difference
between the default and the runs reported in the paper.

Because `create_experiment_config` no longer overrides this flag per
experiment, the YAML value governs every arm except `oaunet_no_ml`,
`oaunet_vi_only`, `offline_ovr` and `offline_sigma`, which force it off by
design.

---

## 2. Repository structure

```
.
├── train.py                     # train the base TrajGRU (with heteroscedastic σ head)
├── run_experiments.py           # run all baselines and ablations
├── requirements.txt
├── configs/
│   ├── temporal.yaml            # base-model training config
│   └── oaunet_temporal.yaml     # online-experiment config (dBZ metrics, drift stream)
├── data/
│   ├── dataset.py               # datasets (mmap-based; nothing is copied)
│   ├── make_temporal.py         # build the natural time-split index (JSON)
│   ├── make_drift_stream.py     # build the constructed drift stream index (JSON)
│   └── make_compact.py          # legacy stratified split (NOT used in the paper)
├── models/trajgru.py            # TrajGRU + log σ² head
├── baselines/
│   ├── offline_baseline.py      # frozen model
│   ├── online_gd.py             # per-sample online gradient descent
│   └── periodic_retrain.py      # periodic retraining from a buffer
├── components/                  # the three adaptive components (+ an experimental MAML module)
├── oaunet/
│   ├── framework.py             # Algorithm 1: the per-sample processing loop
│   └── online_evaluator.py      # streaming evaluation, checkpointing, per-sample metrics
├── metrics/
│   ├── deterministic.py         # MSE / MAE / CSI / HSS in dBZ space
│   ├── online_metrics.py        # update ratio, recovery time, decay
│   └── probabilistic.py         # NLL / CRPS / ECE
└── utils/                       # paths, logger, losses, run modes
```

Generated at run time (all git-ignored):

```
data_raw/          # you place the CIKM2017 .npy files here (read-only source)
data_temporal/     # split-index JSONs written by data/make_*.py
results/           # checkpoints, metrics, figures, logs
results/verify/    # isolated output root for --run_mode verify
```

---

## 3. Requirements

Python ≥ 3.9 with:

```
pip install -r requirements.txt
# torch, numpy, matplotlib, pyyaml, tqdm, scipy
```

Install the PyTorch build matching your CUDA version from
<https://pytorch.org/get-started/locally/> before running the line above.

The paper's experiments were run on a **single consumer-grade NVIDIA RTX 4050
laptop GPU with 6 GB of memory**; the batch / buffer / support-query sizes in
`configs/` are set to fit that budget. A larger GPU will simply run faster —
note that the configs in `configs/oaunet_temporal.yaml` keep
`model.use_checkpoint: true`, which is required to fit 6 GB (it is
mathematically equivalent, just recomputed).

The code is single-process; do **not** launch several runs concurrently on a
6 GB card.

---

## 4. Data preparation

The experiments use the **CIKM 2017 AnalytiCup radar-echo dataset**
(radar reflectivity maps, 101 × 101 pixels, 5 input frames → 10 predicted
frames). Download the dataset from the official challenge page and place the
following three arrays directly inside `data_raw/`:

```
data_raw/
├── train_inputs.npy    # shape (N, 5, 101, 101), float32
├── train_outputs.npy   # shape (N, 10, 101, 101), float32
└── train_labels.npy    # shape (N,), rainfall intensity (mm)
```

Values are expected **normalised to [0, 1]** (`u = pixel / 255`). The pipeline
converts to dBZ internally with the standard CIKM2017 mapping

```
dBZ = u * 95 - 10        (equivalently, pixel * 95/255 - 10)
```

so the evaluation thresholds **5 / 15 / 30 dBZ** correspond to
`u = 0.1579 / 0.2632 / 0.4211`. A small number of negative pixels (missing-value
markers) are clipped to 0 on read.

The dataset order is chronological; the split scripts rely on that order.

---

## 5. Quick start (smoke test)

Before a full run, verify that the pipeline works end-to-end with 100 samples.
This takes well under a minute and writes into the isolated `results/verify/`
tree — it can never overwrite or interfere with a formal run.

```bash
# 1) build the split indices (JSON only, no data is copied)
python -m data.make_temporal --drift strong
python -m data.make_drift_stream

# 2) 100-sample smoke test of base training
python train.py --config configs/temporal.yaml --temporal --run_mode verify

# 3) 100-sample smoke test of one online experiment
python run_experiments.py --config configs/oaunet_temporal.yaml --temporal \
    --run_mode verify --experiment offline --force
```

---

## 6. Full reproduction

### Step 1 — build the split indices

Neither script copies data; they only record source indices in a small JSON
(a few tens of KB).

```bash
# natural time split: train = source [0, 2000), test = source [8000, 10000)
python -m data.make_temporal --drift strong

# constructed regime-switch drift stream (3000 test samples, abrupt point at
# stream step 2000); train segment is identical to the split above
python -m data.make_drift_stream
```

`make_temporal` also supports `--drift mid|weak`, which produce the drift-strength
gradient used for the sensitivity analysis.

### Step 2 — train the base model

```bash
python train.py --config configs/temporal.yaml --temporal
```

Trains on the whole training segment and writes
`results/checkpoints/model_final.pth`. The exported checkpoint is the one with
the **lowest training loss** over the run, not necessarily the last epoch.

> The training segment of the drift stream `temporal_drift3000` is exactly the
> same index range as `temporal_strong`, so a **single** base model is reused by
> every experiment and every drift setting.

### Step 3 — run the experiments

```bash
# the full main table + ablation suite (11 experiment arms)
python run_experiments.py --config configs/oaunet_temporal.yaml --temporal

# or a subset
python run_experiments.py --config configs/oaunet_temporal.yaml --temporal --experiment baselines
python run_experiments.py --config configs/oaunet_temporal.yaml --temporal --experiment ablation
python run_experiments.py --config configs/oaunet_temporal.yaml --temporal --experiment oaunet
```

Running without `--experiment` opens an interactive menu with progress display
and resume support. With `--experiment`, finished arms are skipped automatically
(a `summary.json` that already exists is never overwritten unless `--force` is
given). `--force` also deletes the episodic pre-training cache, so with the
optional MAML module enabled it implies re-running that pre-training — use it
deliberately.

### Available experiment arms

| `--experiment` value | Contents |
|---|---|
| `offline` | Frozen base model (no adaptation) |
| `offline_ovr` | Frozen μ + online **variance** recalibration (OVR) only |
| `offline_sigma` | Frozen μ + trained σ head, recalibration **off** — the matched control for `offline_ovr` |
| `online_gd` | Per-sample online gradient descent (updates every sample) |
| `periodic_retrain` | Periodic retraining from a recent buffer |
| `oaunet` | Full OAUNet (online VI + gated AL + drift detection) — the method reported in the paper |
| `oaunet_no_al` | OAUNet without the active-learning gate |
| `oaunet_no_dd` | OAUNet without drift detection |
| `oaunet_no_ml` | Same as `oaunet` with the optional MAML module forced off; identical to `oaunet` under the default config, retained so the comparison can be re-run with the module on |
| `oaunet_vi_only` | Online VI only — no gate, no drift detection |
| `oaunet_uniform` | Same components as `oaunet` but a uniform (non-difficulty) gate — separates the selection rule from the update count |
| `baselines` | shorthand for the baseline arms |
| `ablation` | shorthand for the ablation arms |
| `all` | every arm above |

---

## 7. Outputs

For a run on the drift stream, every arm writes to
`results/metrics/temporal_drift3000/<experiment>/`:

| File | Contents |
|---|---|
| `summary.json` | `deterministic` (CSI/HSS/MSE/MAE per threshold), `per_frame` (CSI vs. lead time), `per_sample_summary` (mean + decay), `online` (update ratio, recovery time), `timing`, `probabilistic` (NLL/CRPS/ECE) |
| `per_sample_csi.npz` | Per-sample CSI / HSS time series (decay curves) |
| `predictions.npz` | `preds` / `targets` in **dBZ** space, with a `metric_space` marker |
| `uncertainties.npy` | Predictive variance (dBZ²) |
| `online_history.json` | Per-step update / drift events |
| `comparison.json` | Cross-arm comparison (written when more than one arm runs in a single invocation) |

Figures land in `results/figures/temporal_drift3000/`, logs in `results/logs/`.

### Metrics and protocol notes

- All skill scores are computed in **dBZ** space at thresholds **5 / 15 / 30 dBZ**;
  CSI decreases monotonically with threshold, as it should.
- Because positive-class base rates are high (about 58.5 / 40.9 / 5.3 % at the
  three thresholds), CSI should always be read next to the random-forecast floor
  `p / (2 − p)`, and alongside the per-lead-time series rather than as a single
  scalar.
- The drift stream is a **constructed** regime-switch protocol: six blocks of
  real CIKM2017 observations concatenated in order of decreasing mean input
  intensity, with one abrupt change point at stream step 2000. The individual
  blocks are real observations; only their selection and ordering are designed.
  Please see the paper for the precise claims that this protocol does and does
  not support.

---

## 8. Command-line reference (most-used flags)

| Flag | Applies to | Effect |
|---|---|---|
| `--temporal` | both | use the time-split datasets |
| `--temporal_name NAME` | both | which split index (`temporal_strong`, `temporal_drift3000`, …) |
| `--run_mode formal\|verify` | both | full run vs. isolated 100-sample smoke test |
| `--limit_train N` / `--limit_test N` | both | override the verify-mode subset size |
| `--epochs N` | `train.py` | override `training.epochs` |
| `--device cpu\|cuda` | both | force the device |
| `--resume` | both | resume from the latest checkpoint |
| `--experiment NAME` | `run_experiments.py` | run a specific arm / shorthand / `all` |
| `--force` | `run_experiments.py` | ignore existing results (also clears the episodic pre-training cache) |
| `--gate_mode difficulty\|uniform` | `run_experiments.py` | override the active-learning gate |
| `--temporal_name` | `run_experiments.py` | choose the drift setting without editing YAML |

Because the split scripts only record indices, switching drift settings does not
require retraining: the same base model is reused.

---

## 9. Self-tests

```bash
python -m metrics.deterministic     # dBZ conversion round-trip, CSI sanity, base-rate monotonicity
```

---

## 10. Citation

If you use this code, please cite the paper:

```bibtex
@article{oaunet,
  title   = {Selective Online Adaptation for Radar-Echo Nowcasting under
             Controlled Drift: Budget and Boundary Conditions},
  author  = {Xiao, Chengguo and Guo, Jun},
  journal = {TBD},
  year    = {2026}
}
```

---

## 11. License

No license file is included yet. Add one (e.g. MIT or Apache-2.0) before making
the repository public.
