# Linear Attention Experiments

Use one local environment: `.venv`.

## Setup

From a fresh clone:

```bash
git clone <repo-url> torchtitan
cd torchtitan
./scripts/setup_linear_attention_env.sh
source .venv/bin/activate
```

If you are working from `research/`, run the same setup through the local
wrapper:

```bash
cd research
./setup_env.sh
source ../.venv/bin/activate
```

The setup script installs the dev dependencies, Flash Linear Attention, and the
official source build of Mamba3 from `state-spaces/mamba`.

## Smoke Runs

```bash
NGPU=1 COMM_MODE=fake_backend MODULE=linear_attention.qwen3 CONFIG=qwen3_10m_gdn_interleaved ./run_train.sh \
  --training.steps 1 \
  --training.local_batch_size 1 \
  --training.seq_len 128
```

Swap `CONFIG` for:

- `qwen3_10m_gdn_interleaved`
- `qwen3_10m_kda_interleaved`
- `qwen3_10m_gdn_dense_3to1`
- `qwen3_10m_kda_dense_3to1`
- `qwen3_10m_kda_dense_1to1`
- `qwen3_10m_swa_dense_3to1`
- `qwen3_10m_swa_gdn_1to1`
- `qwen3_10m_swa_kda_1to1`
- `qwen3_10m_mamba3_interleaved`
- `qwen3_10m_raven_interleaved`

For a real single-GPU smoke, remove `COMM_MODE=fake_backend`.

The current baseline matrix focuses on dense Qwen3, GDN, KDA, and SWA:

- `MODULE=qwen3 CONFIG=qwen3_10m`: full dense Qwen3 baseline
- `qwen3_10m_gdn_dense_3to1`: 6 GDN layers, dense anchors at layers 0 and 4
- `qwen3_10m_kda_dense_3to1`: 6 KDA layers, dense anchors at layers 0 and 4
- `qwen3_10m_kda_dense_1to1`: 1:1 alternating dense/KDA
- `qwen3_10m_swa_dense_3to1`: 6 SWA layers, dense anchors at layers 3 and 7
- `qwen3_10m_swa_gdn_1to1`: 1:1 alternating SWA/GDN
- `qwen3_10m_swa_kda_1to1`: 1:1 alternating SWA/KDA

## Tests

```bash
.venv/bin/python -m pytest tests/unit_tests/test_train_spec.py tests/unit_tests/test_linear_attention_qwen3.py -q
```
