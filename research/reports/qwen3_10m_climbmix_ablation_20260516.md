# Qwen3 10M ClimbMix Ablation Results

Date: 2026-05-16

Run group: `qwen3-10m-climbmix-50m-20260516-100204`

W&B project: `linear-attention-climbmix`

Local artifacts:

- `outputs/ablations/qwen3-10m-climbmix-50m-20260516-100204/report.md`
- `outputs/ablations/qwen3-10m-climbmix-50m-20260516-100204/logs/`

## Setup

- Hardware: single NVIDIA GB10 / DGX Spark
- Training tokens: 50,003,968 GPT-2 tokens from local Nemotron-ClimbMix subset
- Validation split: local held-out Nemotron-ClimbMix subset
- Tokenizer: GPT-2 tokenizer
- Batch: local batch size 8
- Sequence length: 1024
- Steps: 6104
- Optimizer: plain AdamW via TorchTitan `OptimizersContainer.Config(lr=8e-4)`
- Scheduler: cosine decay, 122 warmup steps, `min_lr_factor=0.1`
- Checkpoints: disabled for this run

All seven runs completed successfully.

## Final Validation Ranking

Final validation was logged at step 6100.

| Rank | Config | Final validation loss | Delta vs dense | Final train loss | Runtime |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `qwen3_10m_climbmix_swa_kda_1to1` | 3.9886 | -0.1131 | 4.3269 | 1368s |
| 2 | `qwen3_10m_climbmix_kda_dense_3to1` | 4.0002 | -0.1015 | 4.3324 | 1339s |
| 3 | `qwen3_10m_climbmix_kda_dense_1to1` | 4.0593 | -0.0424 | 4.3744 | 1361s |
| 4 | `qwen3_10m_climbmix_dense` | 4.1017 | 0.0000 | 4.4312 | 1407s |
| 5 | `qwen3_10m_climbmix_swa_gdn_1to1` | 4.1232 | +0.0215 | 4.3625 | 1316s |
| 6 | `qwen3_10m_climbmix_swa_dense_3to1` | 4.1595 | +0.0578 | 4.4531 | 1412s |
| 7 | `qwen3_10m_climbmix_gdn_dense_3to1` | 4.1719 | +0.0702 | 4.4226 | 1266s |

## Takeaways

The winner in this first 10M-scale health ablation was `swa_kda_1to1`.

KDA was the strongest linear-attention candidate in this setup. Both KDA+dense
variants beat the dense baseline on validation, and SWA+KDA performed best
overall. This supports continuing the KDA branch before spending more time on
GDN.

SWA alone did not beat dense, so the top result should not be interpreted as
"SWA is generally better than dense" at this scale. The useful signal is more
specific: SWA+KDA beat dense, dense+KDA beat dense, and GDN variants did not.

This is still a single-seed 50M-token smoke-scale result. The next result worth
tracking is a repeat of the top candidates with another seed:

- `qwen3_10m_climbmix_dense`
- `qwen3_10m_climbmix_kda_dense_3to1`
- `qwen3_10m_climbmix_kda_dense_1to1`
- `qwen3_10m_climbmix_swa_kda_1to1`
