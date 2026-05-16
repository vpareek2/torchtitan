# Research Workflow

Use one environment: the repo-root `.venv`.

## Setup

From this directory:

```bash
./setup_env.sh
source ../.venv/bin/activate
```

This installs TorchTitan dev dependencies, Flash Linear Attention, and the
official source build of Mamba3 into `../.venv`.

## Tests

```bash
../.venv/bin/python -m pytest ../tests/unit_tests/test_train_spec.py ../tests/unit_tests/test_linear_attention_qwen3.py -q
```

## ClimbMix Subset

The 10M ClimbMix ablation configs read from a local subset at
`../data/climbmix_50m`.

From the repo root:

```bash
.venv/bin/python scripts/download_hf_assets.py --repo_id openai-community/gpt2 --assets tokenizer
.venv/bin/python scripts/download_climbmix_subset.py
```

The default subset writes:

- `data/climbmix_50m/train.jsonl`: 50,003,968 GPT-2 tokens
- `data/climbmix_50m/validation.jsonl`: 2,099,200 GPT-2 tokens

## Smoke Runs

Run from the repo root after activating the environment:

```bash
cd ..
source .venv/bin/activate
```

Fake-backend one-step smoke:

```bash
NGPU=1 COMM_MODE=fake_backend MODULE=linear_attention.qwen3 CONFIG=qwen3_10m_gdn_interleaved ./run_train.sh \
  --training.steps 1 \
  --training.local_batch_size 1 \
  --training.seq_len 128
```

Swap `CONFIG` for any of:

- `qwen3_10m` with `MODULE=qwen3` for the full dense baseline
- `qwen3_10m_gdn_interleaved`
- `qwen3_10m_kda_interleaved`
- `qwen3_10m_gdn_dense_3to1`
- `qwen3_10m_kda_dense_3to1`
- `qwen3_10m_kda_dense_1to1`
- `qwen3_10m_swa_dense_3to1`
- `qwen3_10m_swa_gdn_1to1`
- `qwen3_10m_swa_kda_1to1`

For a real single-GPU smoke, remove `COMM_MODE=fake_backend`.

The 3:1 configs use an 8-layer smoke model:

- `gdn_dense_3to1`: 6 GDN layers, dense anchors at layers 0 and 4
- `kda_dense_3to1`: 6 KDA layers, dense anchors at layers 0 and 4
- `kda_dense_1to1`: 1:1 alternating dense/KDA
- `swa_dense_3to1`: 6 SWA layers, dense anchors at layers 3 and 7
- `swa_gdn_1to1`: 1:1 alternating SWA/GDN
- `swa_kda_1to1`: 1:1 alternating SWA/KDA

## ClimbMix Ablation Matrix

From the repo root:

```bash
source .venv/bin/activate
wandb login
./run_ablation.sh
```

This runs the dense baseline and six hybrid ClimbMix configs sequentially,
sets a shared W&B group, writes per-run logs under `outputs/ablations/`, and
emits a local markdown report.

Useful overrides:

```bash
WANDB_PROJECT=linear-attention-climbmix \
WANDB_RUN_GROUP=qwen3-10m-climbmix-50m-v1 \
NGPU=1 \
./run_ablation.sh
```

For a one-step fake-backend validation of the runner:

```bash
COMM_MODE=fake_backend ENABLE_WANDB=0 ./run_ablation.sh --validator.steps 1
```
