# Business Entity Resolution: Team Game_Shift

Team: Sumit Barik (team leader), Shivanshu Vishwakarma, Sujal Verma.

Final submission: **v18, public leaderboard 0.985594** (`output/matching_results.tsv`, sha256 `052c8a2359a317d427b90cb4fec8f596db9a03427aa7f03b0a192065ae15ac76`).
v18 = v15 (below) with France re-decided by a **stack of three France models** (steps 10–11).

Pipeline, a cascade:

**normalise → block (TF-IDF views + exact keys + second pass) → stage 1 learned filter (XGBoost, p ≥ 0.01) →
stage 2 matcher (cluster context + pairwise evidence + multilingual cross-encoders) → best S1 per record**.

Countries without training labels (France) use a **self-trained** stage 1 and an **unseen-mode** stage 2 whose only
rerankers are **adapted to the country** on its own pseudo-labels.

`output/candidate_pairs.tsv` is the matcher's input: the pairs that pass the learned filter (7.88M pairs, 4.55 per Source-1 entity).
Final **test-shift corrections** remove link types that are over-represented on test compared with validation (US near-twins) or that carry
the decoy fingerprint (a few US/India cells).
No external data, APIs or lookups are used. Every learned resource (transliteration map, name vocabulary, admin-area map,
pseudo-labels, removed-link list) is derived from the provided files.

## Environment
- Python 3.10, 12-core CPU, 16 GB RAM, optional NVIDIA GPU. XGBoost uses CUDA if present. Stage 1 was trained on a 4 GB laptop GPU; the
  small adapted reranker `frs2` on a 6 GB laptop GPU.
- The cross-encoders were trained and run on Kaggle **GPU T4 ×2** (notebooks in `src/reranker/`). They also run on any CUDA GPU with `torchrun`.
- `pip install -r requirements.txt`
- Reranker base models: `intfloat/multilingual-e5-small` (MIT, 118M) and `intfloat/multilingual-e5-base` (MIT, 278M), from Hugging Face.

Data is expected at `../student_resource/dataset/{train,test}`. Override locations with these variables:

| Variable | Meaning | Value used |
|---|---|---|
| `ER_DATA` | dataset folder | `../student_resource/dataset` |
| `ER_CACHE` | intermediate files | `../cache` |
| `ER_OUTPUT` | where the submission files are written | `../output` |
| `ER_RUN` | cache sub-folder for models and predictions | `v6` |
| `ER_CAND_TAG` | suffix of the candidate folder | `_v6` |
| `ER_BLOCK_K` | top-k per TF-IDF view | `word=10,skel=5,numword=3` |
| `SP_TAG` | second-pass variant | `wgf` = second pass `w` (`SP_QP=0.9 SP_K=20`) + name-group (`g`) + fuzzy name-group (`f`) |
| `ADAPT_TAGS` | rerankers adapted to the unlabelled country (France) | `fr,frs2` |
| `CE_EXTRA` | folder with locally trained reranker scores | `../cache/ce_local/scores` |

Set `PYTHONIOENCODING=utf-8` on Windows. **Run heavy steps one at a time** on a 16 GB machine.

## Reproduce end to end (from `src/`)

```bash
export ER_BLOCK_K="word=10,skel=5,numword=3" ER_CAND_TAG=_v6 ER_RUN=v6 PYTHONIOENCODING=utf-8
SCORES=..              # folder that holds ce_{val,test}_full_scores_<tag>.parquet (small, base, v3, fr)
LOCAL=../cache/ce_local

# 1. Normalisation and resources learned from the provided files
python prepare.py train test            # normalise the six files          -> cache/{split}_s{1,2,3}.parquet
python learn_translit.py                # Indic -> English token map from TRAIN pairs -> resources/translit_map.json
python build_vocab.py                   # S1 name word frequencies (unlabelled)       -> resources/name_vocab.json
rm ../cache/*_s?.parquet && python prepare.py train test   # re-normalise with the maps + resources/admin_map.json (note 1)

# 2. Blocking and stage 1 (learned filter)
python run_blocking.py train            # per-country TF-IDF views + exact keys       -> cache/train_cand_v6/
python train_v5.py 0.12                 # 101 features, XGBoost, validation protocol  -> cache/v6/{xgb_model.json,val_pred.parquet}
python run_blocking.py test             #                                             -> cache/test_cand_v6/
python predict_v5.py test               # stage-1 p of every test candidate           -> cache/v6/test_pred_parts/
python predict_v5.py train              # stage-1 p of every TRAIN pair (complete stage-2 context) -> cache/v6/train_pred_parts/
SP_QP=0.9 SP_K=20 SP_TAG=w python second_pass.py train --val-only   # second pass (validation records)
SP_QP=0.9 SP_K=20 SP_TAG=w python second_pass.py test               # second pass (test)
SP_TAG=w  python name_group_pass.py train --val-only && SP_TAG=w  python name_group_pass.py test    # -> SP_TAG=wg
SP_TAG=wg python fuzzy_group_pass.py train --val-only && SP_TAG=wg python fuzzy_group_pass.py test  # -> SP_TAG=wgf

# 3. Countries without labels: leave-one-country-out models and self-training
python loco.py                          # stage 1 trained on one country, scored on the other -> val_pred_loco_*.parquet
python st_loco.py $SCORES India 2       # (validation only) self-training with India unseen
python st_loco.py $SCORES US 1          # (validation only) self-training with US unseen
python france_st.py feats               # stage-1 features of every pair of the unlabelled countries
python france_st.py teacher $SCORES     # unseen-mode stage 2 on current stage-1 scores (the teacher)
python france_st.py st 1                # self-trained stage 1, round 1 (cross-fitted)
ER_OUTPUT=/tmp/r1 python france_st.py final $SCORES 1 un 0.65 ../output   # round-1 stage 2 -> teacher for round 2
python france_st.py st 2 ../cache/v6/new_stage2_st1_un.parquet             # round 2

# 4. Cross-encoders (GPU; Kaggle notebooks in reranker/)
python export_ce_full.py ../kaggle/ce_handoff2     # rounds 1-2: small + e5-base training data and scoring pairs
python export_ce_v3.py   ../kaggle/ce_handoff3     # round 3: stage-1-mined hard negatives + every stage-2 pair
#   reranker/amazon_er_ce2_t4x2.ipynb  RUN "A" (small) and "B" (e5-base)  -> ce_{val,test}_full_scores_{small,base}.parquet
#   reranker/amazon_er_ce3_t4x2.ipynb  RUN "C" (e5-base v3)               -> ce_{val,test}_full_scores_v3.parquet
#   (the notebooks are resumable: checkpoints, out-of-memory batch halving, resume from a previous version's output)
#   put the score files in $SCORES

# 5. Stage 2 and the base submission (v10c)
python stage2_plus.py feats_test        # raw pairwise features of the test stage-2 pairs
SP_TAG=wgf ER_OUTPUT=../output_v10c python final_build_v10.py $SCORES 2
#   -> matching_results.tsv, candidate_pairs.tsv (learned filter), final_meta.json, quick_validate PASS

# 6. Test-shift correction of US near-twin links (v11a)
SP_TAG=wgf ER_OUTPUT=../output_v11 python final_build_v11.py fit $SCORES
SP_TAG=wgf ER_OUTPUT=../output_v11 python final_build_v11.py test_normal $SCORES
ER_OUTPUT=../output_v11 python patch_twins.py ../output_v10c 0.67 ../output_v11a    # removes 26,776 US links
#   (equivalent: python cell_shift.py ../cache/v6/stage2_val_oof_v10c.parquet 0.67 0.65 --unlabelled ../cache/v6/new_stage2_st2_un.parquet --apply ../output_v10c ../output_v11a --types twin --countries US)

# 7. France-adapted rerankers
#   fr   : training set from the France stage-2 output of self-training round 2 (new_stage2_st2_un.parquet: unseen-mode stage 2
#          with small+base+v3); then reranker/amazon_er_ce_fr_t4x2.ipynb (Kaggle, 2 x T4) continues reranker v3 on it
#          (LR 1e-5, 100-minute budget, batch 64/GPU) -> ce_{val,test}_full_scores_fr.parquet into $SCORES
python export_ce_fr.py ../kaggle/ce_fr 2          # -> ce_train_fr.parquet (French pseudo-pairs + 600k labelled US/India pairs)
#   frs2 : e5-small adapted locally, two rounds (round 2's teacher = the France decisions of the round-1 build)
python france_adapt.py pseudo ../output_v11 pseudo_fr                         # teacher: France stage 2 of v11
python ce_local.py train $LOCAL/frs  $LOCAL/sim_all $LOCAL/pseudo_fr.parquet  France 800000
python ce_local.py infer $LOCAL/frs  ../kaggle/ce_handoff3/ce_val_v3.parquet  $LOCAL/scores/ce_val_full_scores_frs.parquet
python ce_local.py infer $LOCAL/frs  ../kaggle/ce_handoff3/ce_test_v3.parquet $LOCAL/scores/ce_test_full_scores_frs.parquet France
ADAPT_TAGS=frs CE_EXTRA=$LOCAL/scores ER_OUTPUT=../output_v12 python final_build_v12.py fit $SCORES
ADAPT_TAGS=frs CE_EXTRA=$LOCAL/scores ER_OUTPUT=../output_v12 python final_build_v12.py test $SCORES frs
python france_adapt.py pseudo ../output_v12 pseudo_fr2                        # round 2 teacher
python ce_local.py train $LOCAL/frs2 $LOCAL/sim_all $LOCAL/pseudo_fr2.parquet France 800000
python ce_local.py infer $LOCAL/frs2 ../kaggle/ce_handoff3/ce_val_v3.parquet  $LOCAL/scores/ce_val_full_scores_frs2.parquet
python ce_local.py infer $LOCAL/frs2 ../kaggle/ce_handoff3/ce_test_v3.parquet $LOCAL/scores/ce_test_full_scores_frs2.parquet France
#   ($LOCAL/sim_all = e5-small trained on labelled US + India pairs: python ce_local.py train $LOCAL/sim_all intfloat/multilingual-e5-small ../kaggle/ce_handoff3/ce_train_v3.parquet US,India <n>)

# 8. France re-decided with the adapted rerankers ONLY (v14b)
export ADAPT_TAGS=fr,frs2 CE_EXTRA=$LOCAL/scores ER_OUTPUT=../output_v14b
python final_build_v12.py fit $SCORES                  # unseen-mode stage 2 variants (incl. fr+frs2) on leave-one-country-out pairs
python final_build_v12.py test $SCORES fr+frs2         # France test pairs -> output_v14b/stage2_test_unlabelled.parquet
python france_threshold.py ../output_v14b        # label-free threshold table: pick empty-S1 ~5.65-5.7% -> t = 0.70
python final_build_v12.py write ../output_v11a 0.70   # France rows replaced; US/India byte-identical to v11a; quick_validate
unset ADAPT_TAGS CE_EXTRA ER_OUTPUT

# 9. Decoy-fingerprint cells on US/India (v15, LB 0.985345)
#   the list of removed links is shipped in resources/ (produced by the cell analysis:
#   python apply_cells.py ../cache/v6/stage2_val_oof_v10c.parquet <v11a_dir> <out_dir> "US:other:0.00-0.80,US:other:0.80-0.90,US:other:0.90-0.95,India:other:0.00-0.80,India:twin:0.00-0.80,India:twin:0.90-0.95")
python patch_france_decoys.py ../output_v14b ../output_v15 resources/usindia_cells_dropped_links.tsv   # removes 5,538 links

# 10. France word-swap corrections (v17b; France rows only, US/India byte-identical)
python calibrate_wordswap.py ../output_v15     # v16b's calibrated word-swap list -> resources/france_wordswap_calibrated_dropped_links.tsv
#   the two lists used below are shipped in resources/ (note 4): filler-swap links to add, content-word swaps to drop
python patch_france_filler.py ../output_v15 ../output_v17b \
       resources/france_filler_added_links.tsv resources/france_wordswap_content_dropped_links.tsv
#   -> +4,470 filler-swap links, -2,473 content-word swaps; added IDs are appended to candidate_pairs.tsv (+52 pairs)

# 11. France stack (final v18, LB 0.985594)
#   A = v15's France (step 8, output_v14b/stage2_test_unlabelled.parquet: adapted rerankers fr+frs2)
#   B = unseen-mode stage 2 with rerankers small+base+v3+fr on the round-2 self-trained stage 1
python france_variant.py $SCORES small,base,v3,fr ../output_v15 ../output_B --round 2   # -> cache/v6/new_stage2_st2_small_base_v3_fr.parquet
#   C = B's recipe with stage 1 WITHOUT the 5 house-number features (note 5), in its own run folder cache/v6nh:
#       junctions (or copies) of v6's feat_trn/, feat_val/, feat_new/, train_pred_parts/; copies of model_meta.json and
#       val_pred.parquet; teacher = model B's France output
mkdir -p ../cache/v6nh && cp ../cache/v6/{model_meta.json,val_pred.parquet} ../cache/v6nh/
for d in feat_trn feat_val feat_new train_pred_parts; do ln -s ../v6/$d ../cache/v6nh/$d; done
cp ../cache/v6/new_stage2_st2_small_base_v3_fr.parquet ../cache/v6nh/new_teacher.parquet
ER_RUN=v6nh ER_FEAT_DROP=house python loco.py                  # leave-one-country-out stage 1 without house features
ER_RUN=v6nh ER_FEAT_DROP=house python france_st.py st 1        # self-trained stage 1 (1 round, teacher = B)
ER_RUN=v6nh ER_FEAT_DROP=house python france_variant.py $SCORES small,base,v3,fr ../output_v15 ../output_C --round 1
#       -> cache/v6nh/new_stage2_st1_small_base_v3_fr.parquet (held-out-country F0.5 0.98783, t = 0.70)
python export_stack_inputs.py       # A, B, C -> stack_inputs/france_probs/{A_v15_fr_frs2,B_small_base_v3_fr,C_no_housenum}.parquet (ID-keyed, p >= 0.01)
#   A, B and C are shipped in stack_inputs/france_probs/ (repository root), so this step runs without re-training anything:
ER_EMPTY=0.05649 ER_AGREE=C_no_housenum python stack_france.py write AB ../output
#   France = mean p of A and B, best S1 per record, threshold set so France's empty-S1 share equals v15's (5.649%, t = 0.646);
#   a change vs v15 is kept only where C makes the same call; then step 10's corrections. US/India rows = v15.
python quick_validate.py ../output ../student_resource/dataset/test     # PASS; sha256 052c8a23... (the uploaded file)
```

Step 11 reproduces the uploaded `matching_results.tsv` byte for byte from `output_v15/`, `output_v17b/` and the shipped
probability files (checked). `stack_france.py report` prints the label-free statistics of every stack variant that was considered.

**Note 1: `resources/admin_map.json`.** This map rewrites French departments to regions ("Gironde" → "Nouvelle-Aquitaine"). It is learned without labels by `learn_admin_map.py` from high-confidence pseudo-pairs of a first-pass model on the test sources. The shipped file is the one used. To regenerate it: run steps 1–2 once without it, run `ER_RUN=<run> python learn_admin_map.py`, then re-run steps 1–2.

**Note 2: validation out-of-fold file.** `cell_shift.py` and `apply_cells.py` need the stage-2 out-of-fold probabilities of the validation pairs (`cache/v6/stage2_val_oof_v10c.parquet`). They are produced by refitting the chosen stage-2 variant out-of-fold, exactly as `final_build_v10.py` does when it selects the variant: `python oof_stage2.py wgf 0.67 v10c`.

**Note 3: France threshold.** France has no labels. `france_threshold.py` prints links per S1 and the share of S1 left empty for each threshold. The generator's singleton share is 5.58% in every training country, and a validation-quality output sits 0.1–0.2 points above it. So the threshold whose empty share is about 5.65–5.7% is used (0.70 for `fr+frs2`).

**Note 4: France word-swap lists (step 10).** Both lists come from a label-free comparison of v15 with the France links of
our earlier, independently built v4 pipeline (own fine-tuned cross-encoder, LB 0.982439). Links only v4 makes are mostly
decoys, except *filler swaps*: the record replaces S1 name words only with filler words (Groupe, Développement, & Fils, SARL…)
at the same house number and street, or has no address (40 of 40 sampled were right). `france_filler_added_links.tsv` holds
the 4,470 such links whose record is unlinked in v15. v16b's calibrated drop list turned out to be half filler swaps, so
`france_wordswap_content_dropped_links.tsv` keeps only its 2,473 content-word swaps ("Club" → "Amicale").

**Note 5: model C.** Same rerankers and stage-2 recipe as B. `ER_FEAT_DROP=house` makes `loco.py` and `france_st.py`
train stage 1 without the five `house_*` features (`features_v5.stage1_features`; unset, all 101 features are used, so
every other step is unaffected). France is self-trained for one round with model B's France output as teacher. Held-out-country
F0.5 0.98783 at t = 0.70. Its France probabilities are shipped (`stack_inputs/france_probs/C_no_housenum.parquet`). Leave-one-country-out ablations
(Documentation, Appendix B) show that the house-number features transfer worst to an unseen country, which is why C serves as the
independent check on the stack's changes.

**Checks.** `quick_validate.py <output dir> <test dir>` applies the official validator's rules block by block (the official one needs a lot of RAM on a large candidate file). The final output also passes the official `utils/validate_submission.py`.

## Modules

| File | Role |
|---|---|
| `normalize.py`, `prepare.py`, `io_utils.py` | Unicode folding, transliteration, leetspeak repair, junk and URL stripping, dotted legal forms, vocabulary-based segmentation, abbreviation / state / admin-area canonicalisation, skeletons; parallel normalisation; paths and I/O |
| `learn_translit.py`, `build_vocab.py`, `learn_admin_map.py` | resources learned from the provided data |
| `blocking.py`, `run_blocking.py`, `second_pass.py`, `name_group_pass.py`, `fuzzy_group_pass.py` | per-country TF-IDF top-k views, exact keys and full cosines; second-pass retrieval by name character 3-grams and aliases; name-group retrieval (exact and fuzzy name groups re-ranked by address) |
| `features_v5.py`, `train_v5.py`, `predict_v5.py` | 101 stage-1 features; training and validation protocol; memory-bounded scoring |
| `collective.py` | stage-2 features (cluster context, cross-source similarity, cross-encoder features), out-of-fold stacker, threshold curve |
| `stage2_plus.py`, `oof_stage2.py` | pairwise stage-1 features for stage 2; out-of-fold stage-2 probabilities of the validation pairs |
| `loco.py`, `unseen_stage2.py`, `unseen_check.py`, `st_loco.py` | leave-one-country-out stage 1, unseen-mode stage 2, self-training validation |
| `france_st.py` | self-training applied to countries without labels (features, teacher, cross-fitted rounds, output) |
| `export_ce.py`, `export_ce_full.py`, `export_ce_v3.py`, `export_ce_fr.py`, `reranker/` | cross-encoder data exports; Kaggle notebooks and `ce.py` (DDP fine-tuning and sharded scoring) |
| `ce_unseen_sim.py`, `ce_adapt_sim.py` | India-as-unseen simulation of France: unseen vs adapted rerankers (why France uses adapted rerankers only) |
| `france_adapt.py`, `ce_local.py` | France reranker adaptation: pseudo-labelled training pairs; single-GPU fine-tuning and scoring |
| `final_build_v10.py` | base build (v10c): stage 2 per country type, submission files, validation |
| `final_build_v11.py`, `patch_twins.py`, `cell_shift.py` | **test-shift correction**: link rates per S1 by country × link type × p band, test vs validation; US near-twin cells discounted |
| `final_build_v12.py`, `france_threshold.py` | **France with adapted rerankers only**: unseen-mode stage 2 variants, France decision spliced onto the base; label-free threshold table |
| `apply_cells.py`, `patch_france_decoys.py` | **decoy-fingerprint cells**: select (country, type, p-band) cells; remove an exact list of links from a submission (every other row byte-identical) |
| `calibrate_wordswap.py`, `patch_france_filler.py` | **France word-swap corrections** (v16b calibrated drop list; v17b filler-swap adds and content-word drops) |
| `france_variant.py`, `export_stack_inputs.py`, `stack_france.py` | **France stack (final v18)**: France stage 2 with a chosen reranker set; ID-keyed France probability files; mean-p stack with the empty-share threshold and the model-C agreement filter |
| `data_shift.py`, `distractor_profile.py`, `link_type_shift.py`, `band_shift.py` | label-free train-vs-test analyses (distractor kind, link types, probability bands) |
| `qwen_probe.py`, `qwen_lora_probe.py` | decoder-LLM reranker tests on the India-as-unseen simulation (not used: weaker than e5) |
| `write_candidates.py`, `quick_validate.py`, `metrics.py` | candidate file (learned filter), low-memory submission checks, the official macro F0.5 |
| `failure_study.py`, `probe_country.py`, `shift_exp.py`, `loss_breakdown.py`, `tune_stage2.py`, `decode.py`, `s2_learning_curve.py` | diagnostics and experiments (not needed for the output) |

Earlier iterations kept for reference only: `features.py`, `context.py`, `train_model.py`, `predict.py` (LightGBM v1–v3), `blend_ce.py`, `rethreshold.py`, `final_build.py` (v9).
