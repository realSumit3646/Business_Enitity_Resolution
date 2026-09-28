# Results log (overnight 27–28 Sep 2026)

Each version is a full submission in `output_<version>/` (`matching_results.tsv` + `candidate_pairs.tsv`, both passing
`quick_validate.py`).

## How the leaderboard is estimated
LB ≈ 0.8502 · F(US+India on test) + 0.1498 · F(France). The shares are exact S1 counts, and France is spread evenly
through the test file.

| Known point | LB | Implied |
|---|---|---|
| v10c | 0.983825 | with F(France) ≈ 0.957 → F(US+India) ≈ 0.9886 |
| v11a | 0.985128 | F(US+India) ≈ 0.9901 = validation 0.9901, so after the near-twin fix US/India test ≈ validation |
| v10a_g, France emptied | 0.848 | F(France) ≈ 0.957 ± 0.007 |

- **US/India:** validation F0.5 (110,550 held-out S1), since v11a shows test ≈ validation.
- **France** (no labels): a labelled **France proxy**, India treated as a fully unseen country. Stage 2 is fitted on
  US only and every reranker is trained without India labels (`ce_unseen_sim.py`, `ce_adapt_sim.py`). The proxy
  reproduces France's measured score: an unseen reranker gives **0.9575 on India, vs France ≈ 0.957 on the leaderboard**.
  The estimated France gain of a change is its gain on the proxy.

## Key findings
1. **France's gap comes from its rerankers being unseen.** The old leave-one-country-out validation (0.988) kept the rerankers seen.
   | India as unseen country (proxy), stage 2 fitted on US | F0.5 @0.66 | P | R |
   |---|---|---|---|
   | reranker seen (trained on US + India) | 0.9781 | 0.994 | 0.954 |
   | reranker unseen (US only): France's situation | 0.9575 | 0.990 | 0.918 |
   | + stage 2 fitted on unseen-reranker features | 0.9615 | 0.987 | 0.930 |
   | **reranker adapted on pseudo-labels (self-training)** | **0.9741** (best 0.9744 @0.80) | 0.988 | 0.954 |
   | adapted + an unadapted reranker together | 0.9627 | 0.991 | 0.925 |
   | no reranker | 0.9447 | 0.968 | 0.929 |
   Reranker AUC on India pairs: unseen 0.900 → adapted 0.980 (seen 0.986).
   **So France should use an adapted reranker ALONE; the unadapted base/small next to it hurt.**
2. Density is not the cause. Re-weighting validation to France's street density (59% of French S1 on streets shared by
   ≥ 10 S1) predicts 0.9885, not 0.957.
3. Round-2 collective features: +0.00005 on validation (dropped).

## Versions
| Version | Change | Val F0.5 (US/India) | P | R | France estimate | **Estimated LB** | Actual LB |
|---|---|---|---|---|---|---|---|
| v10c | baseline | 0.99011 | 0.99858 | 0.97291 | 0.957 | 0.9838 | 0.983825 |
| v11a | v10c − US near-twin links | 0.99011 (test ≈ val) | 0.99858 | 0.97291 | 0.957 | 0.9851 | **0.985128** |
| v12 | v11a + France: local French-adapted reranker `frs` (e5-small, 600k French pseudo-labels + 200k labelled) as France's only reranker, France t = 0.85 | unchanged | unchanged | unchanged | 0.970 (proxy +0.017, discounted) | **0.9871** (0.9856–0.9886) | |
| **v13** | v12 with a round-2 adapted reranker `frs2` (teacher = v12's France decisions), France t = 0.85; 10,192 French rows differ from v12 | unchanged | unchanged | unchanged | 0.971 (proxy +0.018) | **0.9872** (0.9857–0.9887) | |

| v14 | v11a + France: Sumit's `fr` (e5-base continued from v3 on French pseudo-labels; stopped at 54% of its schedule) as France's only reranker, France t = 0.75 (empty S1 5.66%, 3.383 links/S1); 14,279 French rows differ from v11a, US/India byte-identical; official validator PASS | unchanged | unchanged | unchanged | ~0.97 (no proxy for e5-base adapted) | ~0.987 | |

| v14b | v11a + France stage 2 on the adapted rerankers only (`fr`+`frs2`), France t = 0.70; 14,500 French rows differ from v11a | unchanged | | | ~0.97 | ~0.987 | |
| **v15 (FINAL)** | v14b − 5,538 US/India decoy-fingerprint cell links (Sumit's `apply_cells.py`; estimated test precision 0.24–0.56 for 4,935 of them, ~0.70 for 604); France byte-identical to v14b; official validator PASS; package `final_submission/Game_Shift_submission.zip` | unchanged | | | ~0.97 | ~0.987 (0.986–0.989) | **0.985345** |

**27 Sep midday:** v13 and v14 agree on only about a third of their France changes (both add 3,263 / both remove 4,414; v13 alone +10.1k/−5.1k, v14 alone +3.6k/−3.7k). No label-free check separates them: link profiles are all at validation-output level (total-variation distance 0.005–0.007, v11a included, because France errors are swaps). A "same address, one word swapped" decoy test is invalid: that pattern is 93% true matches on labelled US/India pairs. So **v14b** = France stage 2 on both adapted rerankers (`fr`+`frs2`) as the hedge.
Also ruled out today: frozen Qwen2.5-0.5B + trained head (India-unseen AUC 0.76; adapted 0.82 vs e5 0.90 / 0.98), and LoRA-fine-tuned Qwen2.5-0.5B on US only (India AUC 0.807 vs e5 unseen 0.902); Sumit's `m3` (bge-reranker-v2-m3) has AUC 0.987, below stage 1's 0.992 (12% of its training done).

Proxy steps behind v12/v13 (India as unseen, F0.5 at the best threshold): unseen 0.9613 → adapted 0.9744 → round 2 0.9753;
both adapted rerankers together 0.9745 (not used).

### France threshold (label-free)
The adapted reranker makes France's top probabilities very confident. At the leave-one-country-out threshold (0.61),
France has 3.445 links per S1 and 5.41% empty S1. That is fewer empty S1 than the generator's singleton rate (5.58%),
so singletons are being linked. On the proxy the adapted pipeline is flat over t = 0.5–0.9 (best 0.80), so France uses
t = 0.85: 3.40 links per S1, 5.68% empty, like validation's profile (3.38, singletons + 0.18 points).

## Tried and not used (28 Sep morning)
| Idea | Result |
|---|---|
| Two adapted rerankers together (round 1 + round 2) on the proxy | 0.9745 vs 0.9753 for round 2 alone |
| Stage 2 trained on *adapted-unseen* reranker scores (India-only reranker adapted to US pseudo-labels, AUC 0.973) | proxy 0.97503 vs 0.97525 (tie); France stays on the v13 design |
| Legal form (LLC/Inc/Pvt Ltd) to break shared-name ties | only 295 of 6,619 lost ambiguous pairs are fixable this way (~+0.0002), and stage 2 already has legal features |
| Local e5-base reranker | e5-base download failed (Xet error, then DNS / no internet). This lever stays with the friend's `fr` |

## Where 0.991 would have to come from
LB = 0.8502 · F(US+India) + 0.1498 · F(France). With US/India at validation level (0.990), 0.991 needs France at 0.996,
higher than our labelled countries. With US/India at 0.992 it needs France 0.985. The failure study of v10c
(`cache/failure_study_v10c.md`) puts **all** non-ambiguous US/India losses at +0.0046 of validation F0.5 (blocking misses
~2,100 pairs, p below t ~890, wrong S1 ~585, false links 531). The ambiguous ones (address-less, S1 name shared by a
median of 59 S1 in blocking misses) are coin flips. Realistic best: France ≈ 0.98 (e5-base adapted) and US/India ≈ 0.990–0.991
→ **LB ≈ 0.989–0.990**.

## Why 0.995 is out of reach for this pipeline (honest ceiling)
- US + India are 85% of the score and are already at validation level on test (0.990). The failure study puts the
  ceiling at about 0.9946 even if every non-ambiguous miss and every false link were fixed (63% of misses are
  address-less records whose name is shared by several S1).
- Even a perfect France (1.0) gives 0.8502 · 0.990 + 0.1498 · 1.0 = 0.9915.
- An estimate above 0.995 would need US/India on test above 0.9953, beyond that ceiling.
  **Realistic targets:** France ≈ 0.975–0.98 with the friend's e5-base `fr` → LB ≈ 0.988–0.989.

## When the friend's `fr` scores arrive (e5-base continued from v3 on French pseudo-labels)
Put `ce_val_full_scores_fr.parquet` and `ce_test_full_scores_fr.parquet` in the project root, then from `src/`:
```
export PYTHONIOENCODING=utf-8 ER_CAND_TAG=_v6 ER_RUN=v6 ADAPT_TAGS=fr ER_OUTPUT=../../../output_v14
python final_build_v12.py fit ../../..          # ~5 min; variants v11 / fr / fr+v3
python final_build_v12.py test ../../.. fr      # adapted-only, as the proxy says
python france_threshold.py ../../../output_v14  # pick t: empty-S1 about 5.65-5.7%, links per S1 about 3.40
python final_build_v12.py write ../../../output_v11a <t>
```
Prefer v14 over v13: e5-base continued from v3 is much stronger (proxy: seen e5-base 0.9856 vs seen e5-small 0.9781).

## Final result (27 Sep)
v15 scored **0.985345** (v11a 0.985128, +0.0002). That's about the US/India cells' estimate alone, so France's adapted rerankers gained about nothing on the leaderboard, even though the India-as-unseen proxy predicted about +0.017 France F0.5. Likely cause: the proxy's distractors are training-style, while France has the densest engineered decoys (0.59 orphan same-address groups per S1), and self-training on our own French decisions turns already-linked decoys into pseudo-positives. Lesson: validate unseen-country methods on a proxy with synthetic decoys.

## v16b: calibrated France word-swap removal (27 Sep, built for the last upload)
v16 dropped all 19,401 France "same address, one name word swapped" links (p 0.70–0.99), which puts France at 5.92% empty S1. That is past the label-free target of about 5.75% (generator singleton rate 5.58% + validation offset). France's total word-swap rate (281 per 1k S1) is also below validation's (328), so many of those links are probably true. v16b drops only the lowest-p part: links are taken in ascending p while France's empty share stays ≤ 5.75% and p < 0.95 (`calibrate_wordswap.py` → `resources/france_wordswap_calibrated_dropped_links.tsv`, applied with `patch_france_decoys.py`).
The 5.75% target was hit first: **5,124 links removed, p 0.7001–0.8763**. France empty 5.649% → 5.750%, links/S1 3.384 → 3.364. 5,061 French rows changed vs v15 (pure removals); every other row is byte-identical. Official validator PASS. matching_results sha256 prefix fa7f5cb0e7d78a96.

## v17 / v17b: France filler-swap links from v4 as a second opinion (27 Sep evening)
Label-free comparison with the older workspace's v4 (LB 0.982439; France from its own fine-tuned CE):
- Blocking is not the bottleneck: each pipeline's candidates contain 99.85% of the other's France links.
- France links: 837,754 in both, 40,204 only in v15, 14,791 only in v4. v15-only at p <= 0.95 looked ~80% right (keep).
  v4-only looked ~45% right overall (content-word swaps, one-letter edits = decoys), but the **filler-swap** subset
  (name words replaced only by Groupe / Developpement / & Fils / SARL..., same house number and street, or no address)
  was 40/40 right. 4,643 such links, 4,470 whose record is free in v15.
- **v16b's drop list is half filler swaps**: of 5,124 dropped links, 2,651 are filler swaps (20/20 looked true);
  2,473 are content-word swaps (Club -> Amicale).
- House-number-conflict links could not be judged (half have number "1"); the cross-country rate test does not work
  for them (first-number rule differs by address format: France 35/1k S1 vs train 203-285). No change.

| Version | Change (France only; US/India byte-identical to v15) | France links/S1 | France empty | Est. LB vs v15 |
|---|---|---|---|---|
| v17 | v15 + 4,470 filler-swap links (`resources/france_filler_added_links.tsv`) | 3.401 | 5.649% | +0.0001 |
| **v17b** | v17 − v16b's 2,473 content-word swaps (`resources/france_wordswap_content_dropped_links.tsv`) | 3.392 | 5.715% | +0.0001 to +0.0003 |

Built with `patch_france_filler.py`; official validator PASS for both. sha256 prefixes: v17 26ac2b0e47dfa837,
v17b 255290bfcc2a67e4. v17b ≥ v17 unless content-word swaps are more than ~80% true.

## v18_final: France stack (27 Sep 22:42)
`stack_inputs/` (8+1 ID-keyed France probability sets from ours, Sumit's cache/v6 and v6nh, old v4). `stack_france.py`: France =
mean p of A (v15 fr+frs2) and B (Sumit's small+base+v3+fr), best S1 per record, t 0.646 (empty share held at v15's 5.649%).
AB's changes vs v15 are kept only where C (Sumit's no-house-number model) makes the same call (C confirms 85% of AB's adds,
91% of its removals), then v17b's France corrections. ABC mean was 3.4% changed S1 (over the 3% limit), so not used.
US/India identical to v15. vs v17b: +3,011 / −1,903 French links. quick_validate + official validator PASS.
sha256 prefix 052c8a2359a317d4. Package `final_submission/Game_Shift_submission_v18.zip` (output byte-identical).
