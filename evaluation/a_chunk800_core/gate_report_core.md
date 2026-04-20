# Gate Report (core)

- best_config: `similarity_k5`

| config | gate | pass_count | ratio |
|---|---:|---:|---:|
| similarity_k5 | FAIL | 4/5 | 0.80 |
| mmr_k5 | FAIL | 4/5 | 0.80 |
| hybrid_k5 | FAIL | 4/5 | 0.80 |
| similarity_k10 | FAIL | 4/5 | 0.80 |

## Thresholds

| metric | rule |
|---|---|
| p95_elapsed_time | <= 12.0 |
| avg_hit_at_5 | >= 0.8 |
| avg_ndcg_at_5 | >= 0.65 |
| avg_field_coverage | >= 0.55 |
| avg_grounded_token_ratio | >= 0.55 |
| decline_accuracy | >= 0.9 |
