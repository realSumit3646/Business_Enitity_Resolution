# Stage-1 feature-group ablations

Post-competition analysis of which stage-1 feature groups matter, and which ones transfer to a country the model has never seen.
Each script retrains the stage-1 XGBoost on the cached v6 features (GPU, 1–2 min per run) and reports macro F0.5 on the validation protocol.

| Script | Question | Log |
|---|---|---|
| `ablate.py` | In-country: F0.5 when one feature group is removed (half the training records) | `ablate.log` |
| `ablate_loco.py` | Unseen country: train on US and test on India, and the reverse, with groups removed | `ablate_loco.log` |
| `ablate_addr_loco.py` | Unseen country: address and house-number sub-groups | `ablate_addr_loco.log` |

Run from `src/` after the pipeline has built `cache/v6/{feat_trn,feat_val,model_meta.json}`:

```bash
cd src
ER_RUN=v6 ER_CAND_TAG=_v6 python ../experiments/feature_ablation/ablate.py > ../experiments/feature_ablation/ablate.log
```

Key findings (tables in [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md), Appendix B):
- In-country, **name** features carry the most (−0.0098 without them). Blocking-evidence features are almost redundant (−0.0002).
- Across countries, **context/rarity** features transfer best (about −0.008 without them in both directions).
- **House-number** features transfer worst. Removing them *improves* US → India by +0.0068, which motivated the no-house-number model C in the final France stack.
