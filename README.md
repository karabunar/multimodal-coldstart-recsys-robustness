# MAMEX robustness analysis

Reproduction of MAMEX (Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation, CIKM 2025) on Amazon Baby, a stress test with corrupted text on cold items, and a proposed fix, Visual Grounding Regularization (VGR).

## Source paper and code
This project studies and extends the following work. It is not the authors' repository.

- **Paper:** V.-K. Nguyen, D.-H. Pham, H.-S. Nguyen, C.-V. T. Nguyen, H.-Q. Le, D.-T. Le, "MAMEX: Multi-modal adaptive mixture of experts for cold-start recommendation", Proc. 34th ACM Int. Conf. on Information and Knowledge Management (CIKM), 2025, pp. 5053-5057, doi:[10.1145/3746252.3760837](https://doi.org/10.1145/3746252.3760837).
- **Authors' code:** <https://github.com/L2R-UET/MAMEX>, commit `88a932eca821a5657b2b7fc0b6445e4685f4dfab`. This repository restructures that code into a package; model, losses and training loop are unchanged except for two switches that default to off (`docs/reused_components.md`). The upstream repository states no licence, so its code is used here for study and reproduction only, with attribution.
- The paper's published scores (Baby: Rec@10 0.0512, Rec@20 0.0771, NDCG@10 0.0292, NDCG@20 0.0363) are kept apart from the scores measured here throughout the report.

The upstream README is kept in `docs/upstream_README.md`. The full write-up is the submitted report (PDF, IEEE conference format).

## Research question
Does MAMEX's adaptive gate keep trusting a text embedding that has been corrupted, and does a parameter-free image-text agreement signal (VGR) reduce the damage without hurting clean accuracy? Hypotheses, written before full-training results were read: `docs/hypotheses.md`.

## What was added to MAMEX
- `--mode baseline | stress_test | improved` (`main.py`, `src/mamex_robustness/`).
- Test-time corruption of cold items' text features (hashtag / shuffle / gaussian). Training and validation stay clean.
- VGR: `w_t ← w_t · sigmoid(γ · cos(img_emb, text_emb))`, renormalised to `w_v + w_t = 1`. No learned parameters, applied at evaluation on the clean-trained checkpoint (`--train_with_vgr 1` also applies it in training, as an ablation). γ is chosen on noisy validation data (`--gamma_grid`).
- Controls: `--vgr_control shuffled` and `constant`.
- Ablation: `--train_noise_rate` corrupts that share of warm items' text in every training epoch (default 0, off); results carry the suffix `noise_aug` and are reported apart from the main tables.
- Evaluation report per run: clean / noisy, with and without VGR, gate weights, latency, GPU memory.

## Repository layout
```
src/mamex_robustness/   code: cli.py (training and evaluation), repair_cli.py, gate_cause_cli.py;
                        data/ (loader, noise), domain/ (gating, noise, repair), models/ (MAMEX), application/ (session, evaluation, diagnostics), reporting/
tests/                  166 unit and end-to-end tests on a synthetic dataset (no data needed)
scripts/                run_*.sh (experiment drivers), repro_*, rq1_* to rq4_* (analysis scripts grouped by research question, each writes results/<same name>.md), build_cold_dataset.py, demo.sh
results/                result JSONs and tables; top level = main runs and hNN.md/json; subfolders: ckpt291 and ckpt_aug (clean and augmented checkpoints, *.pt ignored),
                        repair, diagnostics, gate_cause, sports, curves291; rq3_* and rq4_* folders hold the runs of those analyses
docs/                   hypotheses.md, reproducibility.md, literature.md, references.bib,
                        reused_components.md, upstream_README.md
main.py                 entry point (adds src/ to the path)
```
Not tracked: the data (`processed_data_baby/`), checkpoints (`*.pt`) and the demo app.

## Setup and commands
See `docs/reproducibility.md` (versions, data checksums, seeds, exact commands). Short form:
```
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q
DATA_PATH=<folder with train.txt, val.txt, test.txt, cold_item_index.npy, image_feat.npy, text_feat.npy> ./scripts/run_experiments.sh
python -m mamex_robustness.reporting.plot_results --results_dir results --k 20
```

## Data
Not included in the repository. Download `processed_data_baby.zip` from the Google Drive link in the upstream README: <https://drive.google.com/file/d/1C-18Y84lMS5xsRGwKQYa9P0Tot-gxgYY/view>. Unzip it and pass the `cold/` folder as `DATA_PATH` (`processed_data_baby/cold`). It must contain `train.txt`, `val.txt`, `test.txt`, `cold_item_index.npy`, `image_feat.npy` and `text_feat.npy` (7,050 items, 1,410 cold items, 19,442 training users).

SHA-256 checksums of the six files, the feature-provenance notes and the licence status are in `docs/reproducibility.md`. Amazon review data is distributed under its own research terms; check the original dataset page before redistributing. No data is committed here.

## Results
Test split, cold items only, hashtag noise on 40% of the cold items, mean Recall@20 over fifteen seeds (2025 to 2039, pinned PyTorch 2.9.1). The gains are over the unrepaired noisy model.

| Condition | Recall@20 |
|---|---|
| Clean | 0.0770 |
| Noisy text | 0.0677 |
| Noisy text + VGR (gain +0.0026) | 0.0703 |
| Noisy text + M1, blind neighbour-text blend (+0.0059) | 0.0736 |
| Noisy text + M3, flagged-item neighbour-text blend (+0.0078) | 0.0755 |

The paper's published Baby scores (Rec@20 0.0771, NDCG@20 0.0363) are not mixed with these. Recall@10 stays 0.0025 short of the published 0.0512 in all fifteen seeds; the cause is not identified.

What was found (details and per-hypothesis outcomes: `docs/hypotheses.md`, tables in `results/`):
- Corrupted text lowers Recall@20 by about 14%; the clean-trained gate is nearly constant (text weight 0.55), so it neither adapts nor is misled item by item.
- VGR, the one proposed fix, recovers 27 to 38% of the loss, but about two thirds of that is also reached by controls that ignore the image (bootstrap interval of the difference to the shuffled control: +0.0012 [0.0007, 0.0018]).
- A training-free repair that blends in the text of the visually nearest warm items does better and beats random-neighbour and permuted-score controls. On a second dataset (Amazon Sports, built from LATTICE's files) the ranking holds with a smaller effect.
- The gains depend on the cold items' clean features having been in the alignment loss during training. With the cold items left out (`--warm_only_aux 1`), the AUROC of VGR's cosine falls from 0.999 to 0.72 and M3's gain from +0.0078 to +0.0004; two inductive remedies failed. These are not inductive cold-start results.
- Confirmation: five unused seeds (2040 to 2044), with six predictions fixed beforehand (`results/rq3_confirmation.md`), all hold: M3 0.0762 against noisy 0.0671 and clean 0.0770.
- The features are LATTICE's, not CLIP, and the noise is simulated on embeddings, never real text.

## Tests
`python -m pytest -q` runs 166 tests in about 3 seconds on CPU, with a tiny synthetic dataset (no real data needed). They cover the noise simulator and its mask rules (18), the VGR gating and controls (8 + 4), the session evaluation and report (14 + 4), the CLI (9 + 5 end to end), the legacy behaviour of the upstream code before the restructure (5), result tables, plots, failure analysis and bootstrap metrics (13 + 8 + 8 + 6). The newest ones (18) check the ablation and its probe: training noise touches only warm items, at the requested share, is redrawn each epoch, never reaches validation or the stored model state, and does not overlap the test mask; the ablation table pairs every seed with its clean-trained runs and is left out when no augmented run exists. Status on 2026-11-05: 166 passed. The tests check the code, not the scientific results.

## Quick demo
`scripts/demo.sh` runs three short steps (about 2.5 minutes on one GPU, measured; steps 1 and 2 take seconds and need neither data nor GPU):
```
pip install -r requirements.txt
PY=python DATA_PATH=<folder with the Baby data, see Data> ./scripts/demo.sh
```
1. The tables of `results/` are rebuilt from the saved result files (`results/tables.md`, `results/repro_epoch_rule.md`); `git diff` shows no change when they are reproduced exactly.
2. The 166 unit tests run on a tiny synthetic dataset.
3. A clean baseline is trained for 21 epochs with a checkpoint, evaluated under corrupted cold-item text (40% of the cold items), with VGR, and with the training-free text repair (M1, M3), and the Recall@20 of each is printed.

Step 3 is a demonstration of the pipeline, not a result: a 21-epoch model is far from converged (best epochs in the reported runs are 230 to 730), so the effects of VGR and of the repair that the report measures on converged models are not reliably visible in it. The reported numbers come from `scripts/run_experiments.sh`, `scripts/run_repair.sh` and the commands in `docs/reproducibility.md`.

## Reused components
`docs/reused_components.md`: what comes from the upstream MAMEX code (compared line by line with commit `88a932e`), the data and the literature, and what was written for this project.

## Literature
`docs/literature.md`, `docs/references.bib`.

## AI use
Disclosed in the submitted report (Appendix A).

## Credit
MAMEX paper and code: see "Source paper and code" at the top. Data: Amazon Baby item cold-start split as distributed by the MAMEX authors (features from LATTICE, not CLIP).
