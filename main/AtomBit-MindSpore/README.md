# AtomBit-MindSpore (Ascend + MindSpore Usage Guide)

This part of the repository is focused on **training workflow on Ascend hardware with MindSpore**.  
This document only covers environment setup, data preparation, launch commands, and troubleshooting. It intentionally does **not** describe model internals.

> Training entrypoint: `Train_dist.py`

---

## 1. Project Layout and Responsibilities

```text
AtomBit-MindSpore/
├── Train_dist.py                 # Main training script (single-card / distributed)
├── train.sh                      # Example launch script for Ascend multi-card with msrun
├── scripts/
│   ├── Preprocess_h5.py          # Data preprocessing: xyz -> h5 chunks + metadata
│   ├── extxyz_to_pyg_custom.py   # Structure file parsing
│   ├── compute_average_e0.py
│   └── compute_save_e0.py
├── src/
│   ├── engine/                   # Training and validation pipeline
│   ├── data/                     # Dataset, sampler, and I/O
│   └── utils/                    # Config, scheduler, helper utilities
└── sharker/                      # Required path for dataloader imports
```

---

## 2. Ascend + MindSpore Environment Requirements

Make sure the following components are version-compatible:

- CANN (`8.3.RC1`)
- MindSpore (Ascend build, `2.7.1`)
- Python 3.9+

Common Python dependencies:

- `numpy`
- `h5py`
- `tqdm`

### 2.1 Required Checks

1. `msrun` is available (required for distributed training).
2. Ascend devices are visible (for example, check `npu-smi info` and confirm `ASCEND_RT_VISIBLE_DEVICES` is configured correctly).
3. Multi-node/multi-card communication environment is configured according to Ascend official guidance.

---

## 3. Data Preparation (HDF5)

Training uses **h5 chunks + metadata** by default.

### 3.1 Preprocessing Command

```bash
python scripts/Preprocess_h5.py
```

This script converts raw structure files into:

- Multiple `.h5` chunk files
- Metadata files (including fields such as `file_path`, `index_in_file`, `num_atoms`, `num_edges`)

### 3.2 Precompute E0 (Recommended)

Before training, generate and save reference energies (`e0_dict`) with:

```bash
python scripts/compute_save_e0.py
```

Then set `E0_PATH` in `Train_dist.py` to the generated file so training can load the precomputed E0 values.

### 3.3 Metadata Naming Consistency

A common issue is naming mismatch:

- Preprocessing may output: `*_metadata.pickle`
- Training config may load: `*.pkl`

Please make sure `TRAIN_META` and `TEST_META` in `Train_dist.py` match your actual metadata filenames.

---

## 4. Config Fields to Update Before Training

In `Config` inside `Train_dist.py`, verify at least:

- `DATA_DIR`: root directory for h5 data
- `TRAIN_META` / `TEST_META`: metadata filenames
- `LOG_DIR`: log and checkpoint output directory
- `E0_PATH`: optional external reference energy path

---

## 5. Launch Methods

### 5.1 Single-Card (Ascend)

```bash
python Train_dist.py
```

Use this first to validate environment and data pipeline.

### 5.2 Multi-Card (Ascend + msrun)

```bash
bash train.sh
```

`train.sh` typically includes:

- `PARALLEL_MODE=DATA_PARALLEL`
- `PYTHONPATH=$(pwd)/sharker:$PYTHONPATH`
- `msrun --worker_num=... Train_dist.py`

Adjust worker count and visible device settings based on your hardware.

---

## 6. Logs and Artifacts

Outputs are generated under `LOG_DIR`, including:

- Training logs
- Per-epoch checkpoints (for example, `model_epoch_{k}.ckpt`)

Track loss and validation metrics to verify convergence and stability.

---

## 7. Common Issues in Ascend Deployments

### 7.1 Metadata Not Found

- Check whether `DATA_DIR` is correct.
- Check whether `TRAIN_META` / `TEST_META` filename and extension match (`.pickle` vs `.pkl`).

### 7.2 Dataloader Errors After Distributed Launch

- Check whether `PYTHONPATH` includes `sharker`.
- Check whether all ranks can access the same data path.
- Check whether metadata `file_path` values point to existing files.

### 7.3 Training Starts but Is Unstable

- Validate data correctness on single-card first.
- Then scale to multi-card gradually and inspect batch settings, parallel parameters, and I/O bottlenecks.

---

## 8. Minimal Runbook

1. Prepare Ascend + MindSpore environment (including `msrun`).
2. Run `python scripts/Preprocess_h5.py` to generate h5 + metadata.
3. Run `python scripts/compute_save_e0.py` and set `E0_PATH` to the generated file.
4. Update data and output paths in `Train_dist.py`.
5. Validate with single-card: `python Train_dist.py`.
6. Scale with multi-card: `bash train.sh`.

---

For model architecture, loss formulation, and theoretical details, refer to external model documentation. This README is intentionally scoped to Ascend + MindSpore engineering usage.
