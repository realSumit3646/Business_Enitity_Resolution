# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Game_Shift  
**Team Members:** Sumit Barik (team leader), Shivanshu Vishwakarma, Sujal Verma  
**Submission Date:** 28 September 2026 (final leaderboard submission: 27 September 2026)

---

## 1. Executive Summary
The pipeline is a **cascade**: cheap blocking, then a learned filter, then an expensive matcher. Each stage sees far fewer pairs than the one before it.

1. **Blocking.** Text is normalised with maps learned from the provided data (transliteration, name segmentation, admin areas). Candidates are then retrieved per country from three TF-IDF views plus exact keys, with a character-trigram **second pass** for weak records. That's about 116 candidates per Source-1 entity.
2. **Learned filter (stage 1).** A gradient-boosted classifier scores every blocked pair on 101 string, number and context features. Only pairs with p ≥ 0.01 go on: **4.5 candidates per S1**, which keep **99.93%** of the true pairs blocking found. This is the set in `candidate_pairs.tsv`.
3. **Matcher (stage 2).** A stacker combines:
   - the stage-1 evidence;
   - **fine-tuned multilingual cross-encoders** (MIT-licensed `multilingual-e5-small` and `-base`), including one trained on hard negatives mined by stage 1;
   - **collective, cluster-level features**: what the other records say about the same S1, including cross-source S2↔S3 agreement.
4. **A test-shift correction.** Comparing link rates per S1 on test with validation, by link type and confidence, shows that US *near-twin* links (identical name, different house number) are 1.6–8× over-represented on test: decoys that validation under-samples. Links in those cells are discounted by the validation/test rate ratio (+0.0013 on the leaderboard). The same analysis, plus a *decoy fingerprint* (decoy links attach to Source-1 entities whose cluster is already complete), flags a few more US/India cells, whose estimated test precision (0.24–0.56) is below the removal break-even. Those 5,538 links are removed as well.
5. **Countries without training labels (France)** go through the same cascade with **self-training** (pseudo-labels from our own confident predictions, cross-fitted) and **rerankers adapted to the new country**. France's rerankers are re-trained on French pseudo-labelled pairs, and France's stage 2 uses *only* these adapted rerankers. Both choices are validated by treating a training country as unseen: self-training gives +0.0018 (India) and +0.0004 (US). With India as the unseen country, an adapted reranker lifts India from 0.9575 (unadapted, which reproduces France's leaderboard-derived 0.957) to 0.974–0.975.

6. **A France stack (final step).** France is re-decided by averaging two independently built France models (v15's adapted-reranker model and a second stage 2 on all four rerankers). The threshold keeps France's empty-S1 share unchanged, and a change is accepted only where a third model, whose stage 1 has no house-number features, makes the same call. Label-free word-swap corrections come from comparison with an earlier independent pipeline.

Held-out macro F0.5 is **0.9901** on US/India (precision 0.9986, recall 0.9729). The final submission (v18) scores **0.985594** on the public leaderboard: 0.985128 after the test-shift correction, 0.985345 after the decoy-fingerprint cells, and +0.00025 from the France stack. No external data, APIs or lookups are used.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory analysis of the 2.2M / 5.0M / 5.3M training records (and, without labels, of the test files) showed the following.

- **Structure of the ground truth.**
  - Every S2/S3 record matches **at most one** S1 entity, so the task is a per-record decision: "which S1, if any?"
  - Only 5.6% of S1 entities are singletons. A cluster has 1–5 S2 and 1–6 S3 records (never more), 3.46 on average, **identically in both countries**.
  - About 26% of training S2/S3 records match nothing (distractors). No true pair crosses countries.
- **Names are not identifiers.** **49.5% of S1 entities share their exact normalised name with another S1** in the same country (mean group size 22, largest 556). The address has to carry the match, and an address-less record with a shared name is inherently ambiguous.
- **Source 1 is clean.** Title Case, full street words and ASCII. Sources 2 and 3 carry the noise.
- **Name noise:** native-script Indic names (about 10% of S2 and 5% of S3, nine scripts); website-only names; digit-for-letter substitutions; injected accents; duplicated words; junk tags; legal suffixes moved, dropped or dotted ("L.L.C."); acronyms; invented names attached to the true address.
- **Address noise:** reordered components; abbreviations; damaged house numbers (11% of US and 23% of Indian true pairs have a different house number); N/A and PO Box insertions; truncation; about 3% empty.
- **The test set is harder than training** (label-free analysis, `data_shift.py`, `distractor_profile.py`):
  - Source 1 is statistically identical in train and test (every noise trait, lengths, state mix; 92–94% of test cities appear in train), and so are the matched records' noise patterns.
  - Test has about 2× the distractors per S1. They are a **different kind of record**: in training, distractors are lone, loosely related records. On test, many are **engineered hard negatives built from real Source-1 entities**: the same name with a different house number (near-twins), the same address with one name word swapped, or a one-character name edit at the same address.
  - Unlinked records that sit in a same-address, similar-name group: US 0.85 per S1 on test vs 0.03 among training distractors; India 0.41 vs 0.014; France 0.59.
  - 66% of the US near-twin links removed by the test-shift correction have such an unlinked sibling record, vs 1.7% of ordinary links. The near-twin excess is one symptom of these decoys.
  - One-character decoys do not leak into US/India links: single-substitution name edits occur at 8.5 per 1,000 test links vs 8.3 per 1,000 true training pairs.
  - The US test population is half of training's (663k vs 1.32M S1; India 810k vs 883k), so count features such as name frequency halve on US test. They carry only 1.3% of stage-1 gain.
- **France (test only, 15% of S1):**
  - S1 names the **region** ("Nouvelle-Aquitaine"); about 40% of S2/S3 name the **department** ("Gironde"). This is handled by the learned admin map.
  - Names come from a small vocabulary of generic French words (Comité, Amicale, Parents…).
  - France is **dense**: 87% of French S1 entities share a street with another S1, and 12% share an exact address (US 31% / 4%, India 9% / 5%).

### 2.2 Solution Strategy
**Approach Type:** Hybrid cascade. Blocking (TF-IDF + keys + second pass), then a learned filter (gradient boosting), then the matcher (collective stacker over cross-encoder rerankers), with a best-S1-per-record assignment.  
**Core Innovations:**
1. **Normalisation learned from the provided data:**
   - a transliteration map learned from training positives;
   - Viterbi segmentation of concatenated names over S1 name frequencies;
   - consonant skeletons;
   - an **admin-area map learned without labels** (department → region) from high-confidence pseudo-pairs.
2. **A cascade built for scale.** Each stage is about 20× more selective than the one before, and the transformer only reads the 4.5 pairs per S1 that survive the learned filter.
3. **Cross-encoder rerankers** read the raw "name | address" pair jointly and handle accents, scripts and abbreviations that hand-built features miss. The round-3 reranker is trained on **negatives mined by stage 1**: near-twins, the same name elsewhere, the same address under another name.
4. **Collective, cross-source reasoning with complete context.** Stage 2 sees:
   - how many confident matches the S1 already has, and from which source;
   - how similar the record is to that S1's most confident other match.

   The context is computed from *all* records of the split: stage 1 is run on every training pair, so validation clusters look like test clusters.
5. **Unseen-country handling, validated.** Leave-one-country-out experiments measure what an unseen country costs:
   - stage 1 loses 0.047 (India) and 0.013 (US);
   - an **unseen-mode** stage 2 (fitted on cross-country scores) plus **self-training** recovers almost all of it.

`country` is treated as an open set throughout. Every stage loops over the labels present; the "countries without labels" branch applies to any label absent from training.

---

## 3. Candidate Generation (Blocking)
Blocking runs **per country label**, with S2/S3 records as queries against an S1 index. IDF is fit on S1 only.

- **Blocking keys used:**
  - **`word`:** TF-IDF of word 1–2-grams of `name_core | address`, top-10 (sparse top-n matrix product).
  - **`skel`:** consonant-skeleton 1–2-grams + numbers, top-5. Robust to typos and transliteration.
  - **`numword`:** (number × address-word) tokens, top-3. Handles truncated or reordered addresses.
  - **Exact keys:**
    - `key_compact`: space-free name;
    - `key_sorted`: sorted name tokens;
    - **`key_name_num`**: space-free name + address number, which resolves a shared name by its house number.

    Keys shared by more than 25 S1 entities and TF-IDF features with S1 document frequency above 3000 are skipped.
  - **Second pass:** records with no confident candidate (best stage-1 p < 0.9) are re-queried by character 3-grams of the space-free name (immune to segmentation errors, robust to typos) and by alias names ("X f/k/a Y"), top-20. It only adds candidates.
- **Candidate pairs generated (test):**

  | Stage | Pairs | Per S1 |
  |---|---|---|
  | Blocking + second pass | 200,442,250 | 115.7 |
  | After the learned filter (stage-1 p ≥ 0.01) | 7,876,590 | 4.55 |
  | **`candidate_pairs.tsv` (+ 52 French filler-swap pairs appended by the v17b correction)** | **7,876,642** | **4.55** |

  The matcher (stage 2, cross-encoders) scores exactly the pairs that pass the learned filter. Every other blocked pair keeps p < 0.01, below the decision threshold, so no model match can come from outside the file. The France correction of step 10 (Appendix A) adds 4,470 filler-swap links, 52 of them outside the learned-filter set; their IDs are appended to the candidate file, so every match is a candidate.
- **How we ensured true matches were not lost:**
  - Recall was measured per view and for the union, on training queries against the full S1 index. The union reaches **0.984** (0.9945 for records with an address). The name + number key adds 0.17 points.
  - The second pass recovers about 17% of the remaining blocking misses.
  - The learned filter keeps **99.93%** of the true pairs that blocking found (validation). Raising its cut to 0.05 would keep 99.72%; we kept 0.01.
  - Most remaining misses are address-less records whose name is shared by many S1 entities, which can't be resolved.

---

## 4. Matching Model

### Stage 1: learned filter, pairwise XGBoost (101 features)
- **Name:** rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the core name; space-free and skeleton similarities; token Jaccard and overlap; legal-form equality and conflict; acronym match.
- **Address:** ratio, token-set, token-sort and partial on the address and its words; skeleton token-set; Jaccard of words, numbers and alphanumeric tokens; house number (exact, truncated, log difference, edit distance, presence in the S1 address); digit-string similarity; address equality.
- **TF-IDF cosines:** 5 views, each with its rank among the record's candidates and its gap to the best.
- **Context:** blocking ranks; number of candidates; name- and address-tied candidates; how many S1 entities share the name or address; how contested the S1 is; how "invented" the record's name is.
- **Flags:** website name, non-Latin text, empty address, PO Box, source.
- **Model:** XGBoost `hist` on GPU (depth 10, learning rate 0.05, subsample 0.8, column sample 0.7, early stopping), trained on 14.5M pairs from a random 12% of training records, disjoint from validation.

### Cross-encoder rerankers
- **Models:** `intfloat/multilingual-e5-small` (MIT, 118M) and `intfloat/multilingual-e5-base` (MIT, 278M), each with a single-logit head. Input is the raw pair `"name | address" [SEP] "name | address"`, 96 tokens.
- **Training data:** training records that never touch the validation entities: positives plus the hardest wrong candidates by blocking similarity.
  - **Round 3** (e5-base, continued) adds the 3 highest-stage-1-p wrong candidates per record over 800k new records, plus 750k confidently pseudo-labelled French pairs (best stage-2 p ≥ 0.97 as positive, its other candidates as negatives), 4.3M pairs in total, with a 200-minute training budget on 2 × T4.
  - Validation AUC: stage 1 0.99214, e5-small 0.99187, e5-base 0.99287, **round 3 0.9936**.
- **France-adapted rerankers** (used for France only):
  - **`fr`**: round 3 (e5-base) continued on 2M French pseudo-labelled pairs + 600k labelled US/India pairs (no forgetting), 2 × T4, stopped by its time budget at 54% of the schedule.
  - **`frs2`**: `multilingual-e5-small` adapted on a laptop GPU (`france_adapt.py`, `ce_local.py`): 300k confident French positives with their 3 hardest candidates as negatives, 300k distractor negatives and 200k labelled pairs. Two rounds; round 2 uses the round-1 France decisions as teacher.
  - Why adapted and *only* adapted (India-as-unseen simulation, `ce_unseen_sim.py`, `ce_adapt_sim.py`; stage 2 fitted on US, applied to India; macro F0.5 on India):

    | India's reranker | F0.5 | Reranker AUC on India |
    |---|---|---|
    | none | 0.9447 | – |
    | unseen (trained on US only): France's situation | 0.9575 | 0.900 |
    | adapted on India pseudo-labels (round 1 / round 2) | **0.9741 / 0.9753** | 0.980 |
    | adapted + unadapted together | 0.963 | – |
    | seen (trained on US + India labels) | 0.9781 | 0.986 |

    The unseen row reproduces France's leaderboard-derived score (about 0.957).
- **Training:** binary cross-entropy, AdamW with warm-up and linear decay, fp16, PyTorch DDP on **2 × T4 (Kaggle)**.
- **Scoring:** every pair entering stage 2 (stage-1 p ≥ 0.01, top-5 per record), including second-pass pairs; length-sorted batches, fp16.

### Stage 2: the matcher (collective + cross-source + reranker stacker)
Stage 2 scores every candidate pair (stage-1 p ≥ 0.01). Its features:
- **Record-level:** stage-1 logit, rank and gap among the record's candidates; candidate count.
- **Pairwise evidence:** 25 raw stage-1 features, so stage 2 sees *why* stage 1 scored the pair: name and address similarities, house-number equality and difference, digits, legal conflict, empty address, name and address frequencies.
- **Cluster-level** (for the pair's S1, excluding the record itself): number of other records that chose this S1, how many with p ≥ 0.9, their share and mean (size-free), the maximum, per source; the S1's number of plausible records and this pair's rank among them.
- **Cross-source similarity:** name and address token-set similarity to the S1's most confident other match, and whether it is from the other source.
- **Cross-encoders:** logit, presence, rank and gap within the record, per reranker.
- **Second pass:** flag and retrieval cosine.

It is an XGBoost stacker (depth 6, 500 rounds) fitted **out-of-fold** on the validation records, with 5 folds grouped by S1 cluster, then refitted on all of them.

**Countries without labels (France):**
1. **Unseen-mode stage 2.** Fitted on leave-one-country-out stage-1 scores (India scored by a US-only model and vice versa), so it learns how to correct a stage 1 that has never seen the country.
2. **Self-training of stage 1.** Pseudo-labels from the unseen-mode output: best p ≥ 0.9 gives a positive (its other candidates negative), best p < 0.1 gives all-negative; 95% of French records qualify. Stage 1 is retrained on all labelled training pairs plus these pseudo-pairs, **cross-fitted** in 2 folds grouped by S1, so no record is scored by a model that saw its own pseudo-label. Two rounds, then stage 2 as in step 1.
3. **Adapted rerankers only.** France's unseen-mode stage 2 uses the France-adapted rerankers `fr` and `frs2` as its only reranker features (`final_build_v12.py`, variant `fr+frs2`), following the simulation above.

**Threshold selection method:** official macro F0.5 on the validation protocol, scanned 0.30–0.95 in steps of 0.01. US/India: t = 0.67. France has no labels, so its threshold is chosen label-free (`france_threshold.py`). The generator gives every training country the same singleton share (5.58%), and a validation-quality output leaves about 0.1–0.2 points more S1 empty. France uses the threshold whose empty-S1 share is 5.65% (t = 0.70, 3.38 links per S1). A lower threshold would link singletons, and on the India simulation the adapted pipeline is flat over t = 0.5–0.9. **Decision rule:** each record goes to its highest-probability candidate if p ≥ t, otherwise to nothing.

**Test-shift correction (prior shift by link type).** The generator yields identical per-S1 link rates in every training country, so links per S1 are a label-free yardstick. Every test link is typed (near-twin = identical name, the record has an address, the house numbers differ; no address; invented name; other) and binned by stage-2 p. For US near-twin links with p < 0.99 the test rate is 1.6–8.2× the validation rate: decoys that the validation set under-samples. In those cells a link is kept only if p × (validation rate / test rate) ≥ t, which removes 27,027 US links (1.2% of US links). India's and France's near-twin cells show no comparable excess and are left unchanged. This single change moved the public leaderboard from 0.9838 to 0.9851 (`cell_shift.py`).

**Decoy fingerprint (final step, `apply_cells.py`).** Decoy records attach to Source-1 entities whose cluster is already complete. So on test, a cell full of decoys shows S1 entities with more *other* confident links (p ≥ 0.99) than validation's true links of the same cell. The US near-twin cells show it clearly (+0.27 to +0.44 links), and the fingerprint's predicted gain for them (+0.0007) was below the measured +0.0013, so the method is conservative. Cells with both a rate excess (1.4–3.1×) and a fingerprint of +0.35 to +0.75 get the same correction: US 'other' links with p < 0.95, India 'other' with p < 0.80, and India near-twins in two bands. Each cell's test precision is estimated as its validation precision divided by the test/validation rate ratio (the excess links are the decoys): 0.24–0.56 for 4,935 of the links and about 0.70 (break-even) for 604. Validation precision of the same probability bands is 0.74–0.96, so removing them only pays because of the shift. 5,538 links are removed (`resources/usindia_cells_dropped_links.tsv`, applied by `patch_france_decoys.py`). France is left to its adapted rerankers. It has no validation reference, and its crowded structure (24% of French S1 share their name with another S1 in the same city, vs 1% in the US; 59% sit on streets with ≥ 10 S1) makes the fingerprint ambiguous there.

**France word-swap corrections (v17b, `patch_france_filler.py`).** A label-free comparison with the France links of our earlier, independently built v4 pipeline (own fine-tuned cross-encoder, LB 0.982439) shows the following. Each pipeline's candidates contain 99.85% of the other's France links, so blocking is not the bottleneck. The 14,791 links only v4 makes are mostly decoys (content-word swaps, one-letter edits; about 45% right by inspection), but one subset is clean: *filler swaps*, where S1 name words are replaced only by filler words (Groupe, Développement, & Fils, SARL…) at the same house number and street, or where the record has no address (40 of 40 sampled were right). 4,470 such links whose record is unlinked are added. An earlier calibrated drop list of "same address, one name word swapped" links turned out to be half filler swaps, so only its 2,473 content-word swaps ("Club" → "Amicale") are removed. France ends at 3.39 links per S1 and 5.72% empty S1 (target 5.65–5.75%).

**France stack (final v18, `stack_france.py`).** Several France stage-2 models exist, built by two team members on identical caches. None can be scored on French labels, so they are combined conservatively:
- **A:** v15's France: unseen-mode stage 2 on the France-adapted rerankers `fr` + `frs2` (self-trained stage 1, round 2).
- **B:** unseen-mode stage 2 on all four rerankers, `small` + `base` + `v3` + `fr`, on the same self-trained stage 1 (`france_variant.py`).
- **C:** B's recipe with stage 1 retrained **without the five house-number features** (leave-one-country-out models and one self-training round with B's France output as teacher; held-out-country F0.5 0.98783). Leave-one-country-out ablations (Appendix B) show house-number features transfer worst to an unseen country (removing them *raises* US→India F0.5 by 0.0068), so C is an independent opinion on house-number decisions, where France's near-twin decoys live.

France pairs get the mean p of A and B (a missing p counts as 0). Each record goes to its best S1, and the threshold (t = 0.646) is set so France's empty-S1 share equals v15's 5.649%, so the stack re-ranks rather than shifts the link count. A change relative to v15 is kept only where C makes the same call at its own empty-share threshold. C confirms 85% of the stack's added links and 91% of its removals. Then the v17b corrections are applied. Averaging all three models (ABC) changed 3.4% of French S1 rows, over our 3% change limit, so it was not used. The final output changes 4.4% of French S1 rows vs v15 (+7,480 / −4,375 links), leaves US/India byte-identical, and moved the leaderboard from 0.985345 to **0.985594**.

**Validation protocol:**
- **V:** the 5% of S1 entities with id mod 20 = 0 (110,550).
- **Validation records:** every record whose true S1 is in V or whose top-ranked candidate is in V, so wrong links *into* V are counted.
- **Metric:** the official macro F0.5 over all of V, including singletons, against the full ground truth. Blocking misses count as lost recall.
- **No reuse:** stage-1 training records and cross-encoder training records are disjoint from these records.

---

## 5. Results & Error Analysis

| Version | Change | Validation F0.5 | Precision | Recall | Public LB |
|---|---|---|---|---|---|
| v1 | base normalisation, 3 TF-IDF views + 2 keys, LightGBM | 0.9726 | 0.989 | 0.948 | |
| v3 | + ambiguity and house-number features, word top-10 | 0.9788 | 0.993 | 0.957 | 0.962 |
| v5 | + segmentation, name+number key, full cosines, XGBoost | 0.9822 | 0.995 | 0.961 | 0.968 |
| v6 | + dotted legal forms, admin map; stage 2 (collective + e5-small reranker) | 0.9883 | 0.998 | 0.969 | 0.9815 |
| v9 | complete-context stage 2, full reranker coverage, e5-base, second pass; unseen-mode stage 2 for France | 0.98938 | 0.9985 | 0.9713 | 0.982177 |
| v10a-g | + France self-training, pairwise features in stage 2, wider second pass, name-group retrieval | 0.98978 | 0.9986 | 0.9720 | |
| v10c | + reranker v3, fuzzy name-group retrieval | 0.99011 | 0.9986 | 0.9729 | 0.983825 |
| v11a | + test-shift correction of US near-twin links (26,776 links) | 0.99011 (validation unchanged by design) | 0.9986 | 0.9729 | **0.985128** |
| v14b | + France's stage 2 on the France-adapted rerankers only (`fr` + `frs2`), label-free France threshold 0.70 (14,500 French rows change) | US/India unchanged; India-as-unseen simulation of adaptation +0.017 | | | |
| v15 | + decoy-fingerprint cells: 5,538 US/India links removed (France unchanged) | (validation unchanged by design) | | | 0.985345 |
| v17b | + France word-swap corrections: 4,470 filler-swap links added, 2,473 content-word swaps removed | (France only, no labels) | | | (not uploaded) |
| **v18 (final)** | + France stack: mean of A + B, empty share held at v15's, changes kept only where C (no house-number features) agrees; then v17b's corrections | (France only, no labels) | | | **0.985594** |

The leaderboard decomposes by country share: LB ≈ 0.850 · F(US+India) + 0.150 · F(France). A diagnostic submission with France's rows emptied put France at about 0.957. After the near-twin correction, US/India on test (≈ 0.990) equals validation. v15 scored 0.985345, +0.0002 over v11a. The France stack (v18, US/India unchanged) then added +0.00025, about +0.0017 on France's F0.5. That is about the size estimated for the US/India cells alone, so the France-adapted rerankers gained little or nothing on the leaderboard, although the India-as-unseen simulation predicted about +0.017 on France. The likely reason is that the simulation's unseen country has training-style distractors, while France has the densest engineered decoys of the test set (unlinked same-address, similar-name groups: 0.59 per S1). Adaptation learns from our own French decisions, so decoys that were already linked become pseudo-positives. A simulation with synthetic decoys would be needed to measure that.

**Unseen-country validation** (each training country treated as unseen; macro F0.5 on its held-out entities):

| Unseen country | Stage 1, other country only | + unseen-mode stage 2 | + self-training | Precision | Recall |
|---|---|---|---|---|---|
| India | 0.93250 | 0.98540 | **0.98723** (2 rounds) | 0.9974 → 0.9973 | 0.9625 → 0.9669 |
| US | 0.97191 | 0.98862 | **0.98904** (1 round) | 0.9978 → 0.9980 | 0.9708 → 0.9716 |

Self-training alone lifts stage 1 from 0.932 to 0.977 on India. With real India labels it reaches 0.979.

- **Where the remaining validation loss comes from:** blocking misses (~1.4% of true pairs, mostly address-less records with typo'd or shared names); the correct S1 below threshold (~1.0%); another S1 ranked first (~0.7%, mostly address-less records with a shared name). Precision is 0.998.
- **Common false positives (wrong merges):**
  1. Address-less records whose name matches one of several S1 entities sharing it.
  2. Near-twin decoys: the same street, a house number off by a little, one name word changed. In France, where 12% of S1 entities share an exact address, a damaged house number often lands on a neighbour.
- **Common false negatives (missed matches):**
  1. Address-less records with shared or heavily typo'd names.
  2. True matches whose house number was damaged as much as a decoy's.
  3. Invented names with truncated addresses.
- **Validation vs leaderboard.** A leaderboard submission with France's rows emptied (a diagnostic) put US/India at about 0.9875 on test against 0.990 on validation, and France at about 0.957. The US part of the gap was traced to near-twin decoys over-represented on test (corrected above; afterwards US/India on test ≈ validation). France's gap is explained by its rerankers being unseen: the India-as-unseen simulation with an unseen reranker gives 0.9575, the same as France. That is why France uses rerankers adapted on French pseudo-labels.
- **Checked and ruled out:**
  - The distractor-rate and near-twin shift in test: reweighting validation to the test mix costs 0.0004 with no threshold change.
  - A France house-number recall loss: a label-free link-count test shows the unlinked near-twins are decoys.
  - Row-order or ID leakage: none; correlations are about 0.001.
- **Late reranker experiments:**
  - `BAAI/bge-reranker-v2-m3` (568M, Apache-2.0), fine-tuned for 240 min on 2 × T4, reached validation AUC 0.987 (under-trained; 12% of its schedule) and added nothing to stage 2 (0.99010 vs 0.99011).
  - Larger decoder LLMs for the unseen country (Qwen2.5-0.5B, Apache-2.0), tested on the India-as-unseen simulation: a frozen model with a trained head reaches AUC 0.76 (0.82 with India pseudo-labels), and LoRA fine-tuning on US labels reaches 0.807, vs 0.902 for the unseen e5-small. Training is also slow (about 30 pairs/s on a laptop GPU). Not used.
  - The held-out-country score cannot choose France's rerankers: the adapted rerankers continue from models trained on US and India labels, so on either held-out country they are *seen*. The unseen-mode stage 2 scores 0.9876–0.9879 there with any reranker set. The India-as-unseen simulation, where the reranker has never seen India, is the relevant test.
  - The two adapted rerankers agree on about a third of their France changes relative to the unadapted build. The final stage 2 uses both, so it keeps most of the changes they agree on (2,855 of 3,263 added links and 3,577 of 4,414 removed ones).
- **Late France leaderboard lessons:** a submission that dropped French links with p < 0.99 lost 0.00036 (those links were about 80% true), and one that dropped 7,444 French near-twin links lost 0.00038 (about 100% true). Removing French links does not pay, which is why the final stack keeps France's link count fixed and only re-ranks.
- **Tried and dropped:** metric-aware per-S1 decoding (+0.0002, noise); stage-2 capacity tuning (saturated); per-country thresholds and per-source caps (+0.00001); distractor-weighted and importance-weighted stage 2; a stage 2 fitted on the unseen country's own pseudo-labels (worse on both held-out countries); a third self-training round (changed 0.6% of French links); more stage-2 training data (learning curve flat).

---

## 6. Conclusion
The largest gains came from understanding the data:
- names are shared by many businesses, so the address has to carry the match;
- most of the generator's corruptions can be reversed with normalisation learned from the data itself;
- a validation set is only trustworthy when its context looks like the test set's.

A cheap blocking stage and a learned filter narrow 17 trillion possible pairs to 4.5 per S1 entity. Multilingual cross-encoders and cluster-level evidence then decide at 99.8% precision. For a country with no labels, self-training of stage 1 and rerankers adapted on the country's own pseudo-labels recover most of the unseen-country loss, each validated by treating a training country as unseen. A label-free comparison of test with validation then exposed decoys that validation under-samples (near-twins and other engineered hard negatives), and correcting for them was the largest single leaderboard gain after the rerankers. For France, where no labels exist, the final gain came from combining independently built models conservatively: link count held fixed, and changes accepted only when a model with a different feature view agrees.

---

## Appendix

### A. Code Artefacts
`src/` holds all source. [`REPRODUCE.md`](REPRODUCE.md) gives the exact commands and `requirements.txt` the pinned versions. The pipeline, in order:
1. **Normalisation:** `prepare.py`, `learn_translit.py`, `build_vocab.py`, `learn_admin_map.py`.
2. **Blocking and stage 1:** `run_blocking.py`, `train_v5.py`, `predict_v5.py test` and `predict_v5.py train` (stage-1 scores of every training pair, for complete stage-2 context), `second_pass.py`.
3. **Unseen countries:** `loco.py` (cross-country stage 1), `st_loco.py` (validation of self-training), `france_st.py feats | teacher | st | final`.
4. **Cross-encoders:** `export_ce_full.py`, `export_ce_v3.py`, then the Kaggle notebooks (`reranker/`, GPU T4 ×2).
5. **Stage 2 and output:** `stage2_plus.py feats_test`, then **`final_build_v10.py <score_dir>`** (writes `matching_results.tsv`, the learned-filter `candidate_pairs.tsv`, runs `quick_validate.py`).
6. **Test-shift correction:** `final_build_v11.py` + `patch_twins.py` (US near-twin cells; equivalently `cell_shift.py … --apply … --types twin --countries US`).
7. **France-adapted rerankers:** `fr` from the Kaggle notebook `reranker/amazon_er_ce_fr_t4x2.ipynb`; `frs2` from `france_adapt.py pseudo` + `ce_local.py train / infer` (two rounds).
8. **France with adapted rerankers only:** `final_build_v12.py fit | test fr+frs2 | write <v11a> 0.70` (`ADAPT_TAGS=fr,frs2`), threshold table from `france_threshold.py`.
9. **Decoy-fingerprint cells:** `apply_cells.py` / `cell_shift.py` produce `resources/usindia_cells_dropped_links.tsv`; `patch_france_decoys.py <v14b> <v15> resources/usindia_cells_dropped_links.tsv` removes exactly those links (see README).
10. **France word-swap corrections (v17b):** `calibrate_wordswap.py`, then `patch_france_filler.py <v15> <v17b> resources/france_filler_added_links.tsv resources/france_wordswap_content_dropped_links.tsv`.
11. **France stack (final v18):** `france_variant.py` (model B; model C with the house-number features removed from stage 1), `export_stack_inputs.py` (ID-keyed France probabilities), then `ER_EMPTY=0.05649 ER_AGREE=C_no_housenum stack_france.py write AB <out>`. The A, B and C probability files are shipped in `stack_inputs/france_probs/`, and this step reproduces the uploaded `matching_results.tsv` byte for byte.

**Models and licences:**
- XGBoost: Apache-2.0.
- `intfloat/multilingual-e5-small` (118M) and `intfloat/multilingual-e5-base` (278M): MIT, well under the 8B limit.
- Libraries: polars, scikit-learn, rapidfuzz, sparse_dot_topn, Unidecode, PyTorch, transformers.

**Data:** no external data, APIs or geocoding. The transliteration map is learned from training pairs. The name vocabulary comes from unlabelled S1 files. The admin-area map and the French pseudo-labels come from unlabelled test data, with no test labels.

### B. Additional Results
- Blocking recall (100k training queries): union 0.9844 (0.9945 with an address); exact keys alone 0.47–0.50; address-less records India 0.72, US 0.80.
- Learned-filter retention on validation: p ≥ 0.01 keeps 99.93% of blocked true pairs (3.8 pairs per S1); p ≥ 0.05 keeps 99.72%.
- Stage-2 ablation on validation (US/India, out-of-fold):

  | Stage 2 | F0.5 |
  |---|---|
  | collective + reranker (small) | 0.98851 |
  | + second pass | 0.98902 |
  | + e5-base reranker | 0.98938 |
  | + pairwise features | 0.98952 |
  | + wider second pass | 0.98966 |
  | + reranker v3 | 0.99001 |
  | + fuzzy name-group retrieval (v10c) | 0.99011 |

- **Stage-1 feature-group ablation** (validation, half the training records, one group removed at a time; macro F0.5 at each run's best threshold). Every group helps; name features carry the most, and blocking-evidence features are almost redundant with the similarity features:

  | Stage 1 | Features | F0.5 | Change |
  |---|---|---|---|
  | all features | 101 | 0.98091 | – |
  | − name | 76 | 0.97108 | −0.0098 |
  | − context / rarity | 85 | 0.97641 | −0.0045 |
  | − address | 76 | 0.97661 | −0.0043 |
  | − house number | 96 | 0.97913 | −0.0018 |
  | − flags | 95 | 0.98041 | −0.0005 |
  | − blocking evidence | 77 | 0.98074 | −0.0002 |

- **Which features transfer to an unseen country** (stage 1 trained on one country, macro F0.5 on the other's validation entities; this drove model C of the France stack):

  | Stage 1 | US → India | India → US |
  |---|---|---|
  | all features | 0.93173 | 0.97142 |
  | − context / rarity | 0.92397 | 0.96393 |
  | − blocking evidence | 0.93408 | 0.97120 |
  | − rank / gap features | 0.92845 | 0.97139 |
  | − digit-string features | 0.92658 | 0.97054 |
  | **− house-number features** | **0.93851** | 0.97047 |

  Context features transfer best (removing them costs about 0.008 both ways). House-number features transfer worst: the countries write house numbers differently, and dropping them *improves* US → India by 0.0068. Unseen-country thresholds are also higher (best t 0.72–0.86 vs 0.64–0.70 in-country).
