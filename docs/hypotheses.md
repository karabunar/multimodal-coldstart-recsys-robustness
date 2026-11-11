# Research questions and hypotheses

Four research questions, each with one main hypothesis, its prediction and what was measured. Supporting analyses are named by the script that runs them (`scripts/<name>.py`, output `results/<name>.md`). Failed predictions are kept.

**Status of the evidence.** The repository has two commits made on the day of publication, so its history cannot show when the hypotheses were written relative to the experiments. The supporting analyses are exploratory. Only the confirmation run (`rq3_confirmation_run.py`) is fully pre-specified: new seeds, frozen rules, six predictions. No correction for multiple comparisons was made.

## RQ1. Does the gate keep trusting corrupted text?

**Hypothesis 1.** Corrupting cold-item text lowers accuracy and the gate does not move away from text. *Supported.* Corrupting 40% of the cold items' text lowers Recall@20 by 14%, while the text weight moves only from 0.550 to 0.563. The clean-trained gate is nearly constant (std of the text weight across cold items 0.050), so it neither reacts nor is misled item by item (`rq1_gate_spread.py`).

## RQ2. How much do VGR and the neighbour-text repairs recover, and is it grounding or muting?

**Hypothesis 2.** VGR recovers part of the loss at small cost, the gain needs the image, and a repair that blends in the text of image-nearest warm items does better and beats random-neighbour and permuted-score controls. *Partly.*
- **VGR:** recovers 27 to 38% of the loss in 3 of 3 seeds at a clean-accuracy cost inside one seed std, but latency rose 5.6% against a 5% limit. Shuffled and constant controls acting among the cold items recover about two thirds of the gain; the true image cosine adds +0.0012 Recall@20 (bootstrap interval [0.0007, 0.0018]) (`rq2_vgr_control_scope.py`, `rq2_vgr_bootstrap.py`). A gamma rule that favours cosine-specific gains did not hold (`rq2_gamma_rule.py`).
- **Neighbour-text repair:** the blind blend (M1) and the flagged-item blend (M3) beat both controls. M3 exceeds M1 by +0.0019 over fifteen seeds, not in every setting; about 72% of M3's gain comes from the corrupted items alone (`rq2_repair_seeds.py`, `rq2_repair_decomposition.py`, `rq2_repair_additivity.py`).
- **What the repair is:** a light blend raises clean Recall@20 in 15 of 15 seeds, so it is partly a smoothing of cold text; under subtler noise detection stays easy (AUROC 0.99), contrary to the prediction, and the blind blend is best (`rq2_repair_regulariser.py`, `rq2_subtle_noise.py`).
- **Other data:** on Amazon Sports the ranking holds with a smaller effect (M3 recovers about 38%); on the 49% of items with a 2023 catalogue entry the image-nearest warm items' titles are 8.2 times more similar than random ones, so the repair's premise holds for real products (`rq2_second_dataset_table.py`, `rq2_neighbour_text.py`, `rq2_real_text_source.py`).

## RQ3. Do the gains depend on the transductive setting, and do they replicate?

**Hypothesis 3.** The cosine detector and the repairs survive when the cold items are left out of the training losses, and an inductive repair exists. *Not supported.* With the cold items left out (15 seeds): AUROC of VGR's cosine 0.721 against 0.999, M3 gain +0.0004 against +0.0078, VGR beats the noisy model in 4 of 15 seeds against 15 of 15. The neighbour-text oracle still gains +0.0029, so this is mostly a detection failure. Two inductive remedies fail: ridge detector AUROC 0.565, neutral-text oracle +0.0002, flagged muting loses (`rq3_inductive_remedies.py`; runs in `results/rq3_warm_only/`).

**Confirmation on unused seeds** (`rq3_confirmation_run.py`): all six frozen predictions hold on seeds 2040 to 2044: noise loss 0.0099 (5/5), VGR above noisy (5/5), M3 above noisy (5/5), M3 at least M1 (4/5, mean +0.0025), VGR above shuffled (5/5), AUROC 0.9991 (5/5). This replicates the transductive results; it does not test the inductive setting.

## RQ4. Why does the gate trust text? (exploratory; no main claim rests on it)

**Hypothesis 4.** The gate's preference for text on cold items comes from the training interactions, not from the item's content. *Supported on a noise-augmented model, through routes other than those first guessed* (two to three seeds):
- Training on corrupted text at rate 0.4 lowers noisy Recall@20 (0.0568 against 0.0668) and raises the text weight to 0.917; at rate 0.1 it helps in two seeds. Noise on the image side keeps the text weight low but is worse than clean training (`rq4_lower_rates_table.py`, `rq4_augmentation_designs_table.py`).
- The preference follows from an item having no training interactions, not from gate bias, expert-output norm or distance to the warm centroid: warm items with removed interactions get text weight 0.97 like cold items, trained items 0.35. Among warm items it falls with log interaction count (Spearman $-0.7$; clean-trained control $-0.4$) (`rq4_held_out_table.py`, `rq4_dose_response.py`).
- The gate reads a trace of the training interactions in the image expert's output: swapping the component along the mean user direction moves the text weight by 0.18 to 0.42 (random directions 0.001); a 20-dimensional user subspace carries about half of what the gate reads. The rest of the trace is not identified (`rq4_trace.py`, `rq4_user_direction.py`, `rq4_user_subspace.py`, `rq4_gate_subspace.py`, `rq4_gate_perp.py`).

## Real clickbait check (follow-up to RQ1 and RQ2)

The stress test so far mixes each corrupted item's text with prototypes drawn from warm items' own text. This check replaces the prototypes by embeddings of real clickbait strings (`data_maps/clickbait_strings.txt`, 41 strings, fixed before the runs) made with the encoder that reproduces the distributed text features (`scripts/build_clickbait_prototypes.py`). Setting: the saved clean-trained checkpoints of seeds 2025--2039, noise rate 0.4, mixing weight 0.8, repairs and parameter grids unchanged, parameters chosen on validation data with the same kind of corruption. Predictions, written before the first run:

- **a.** Corruption lowers Recall@20 by 5 to 25% relative to clean (the synthetic hashtag noise gives 12%).
- **b.** The AUROC of VGR's cosine for finding the corrupted items is at least 0.95 (mean over seeds).
- **c.** VGR beats the unrepaired model in at least 12 of 15 seeds.
- **d.** M3 beats the unrepaired model in at least 12 of 15 seeds, with a mean gain of at least +0.003.
- **e.** The mean M3 gain is within a factor of two of the synthetic one (between +0.0039 and +0.0156).

What would make me doubt the earlier result: b failing says the cosine finds prototypes drawn from warm items but not real clickbait; d failing says the repair does not transfer to realistic corruption. A replacement weight of 1.0 is run as an exploratory sensitivity check only.

## Real item text check (follow-up to RQ1 and RQ2, written 2026-11-11 before any run)

Until now the corruption was made on embeddings. Here the corruption is made on the text of real products and the corrupted text is embedded again with the encoder identified for the distributed features (`roberta-large-nli-stsb-mean-tokens`, mean pooling, raw scale). The items are the cold items with a 2023 catalogue entry (the set S). Two corruptions of the title: *wrap*, a clickbait string from `data_maps/clickbait_strings.txt` put in front of the title, and *mismatch*, the title of another item of S. Every item of S is first embedded from its clean title alone, so that all items of S come from the same pipeline; 80% of S, chosen by the usual test and validation seeds, then get a corrupted title. Reference for the loss is the model on the title-only embeddings, not the original features. Clean-trained checkpoints of seeds 2025 to 2039, no training; parameters chosen on the validation mask, as in the main grid.

Predictions, written before the runs:
(a) Both corruptions lower Recall@20 against the title-only reference in 15 of 15 seeds; the loss from mismatch is larger than from wrap, since wrap keeps the title.
(b) VGR's cosine finds mismatched items with an AUROC of at least 0.85 and wrapped items with a lower AUROC, between 0.60 and 0.85.
(c) M3 beats the unrepaired model in at least 12 of 15 seeds for both corruptions, and beats the random-neighbour and permuted controls in at least 12 of 15 for mismatch.
(d) Using the title alone costs clean accuracy: the title-only reference is at least 0.0010 below the 0.0770 of the original features (the original text holds more than the title).
Doubt: if the cost in (d) is larger than the loss from the corruptions, this design cannot separate the two and the result is reported as uninformative.

## Image-only baseline (added after the other results)

For the same flagged cold items the text is dropped and the item is ranked by its image embedding alone (`scripts/run_image_only.sh`, `scripts/rq2_image_only_table.py`, `results/rq2_image_only.md`). It recovers 79% of the loss against 85% for M3 and 64% for M1; M3 is above it by +0.0005 Recall@20 (11 of 15 seeds) and below it on NDCG@20. On clean input the same rule costs 0.0028 Recall@20, while M3 raises it. The repairs' advantage is therefore mainly kept clean accuracy.

## Reproduction check

Epoch rule, candidate set or seed variation explains the Rec@10 gap to the paper: *not supported.* Mean Rec@10 0.0486, 0 of 15 seeds reach the published 0.0512; the gap is systematic and unexplained (`repro_epoch_rule_tables.py`, `repro_rec10_candidates.py`).

## RQ5. Is a better detector enough to make the repair work when the cold items were not seen in training? (written before the runs, answered after them)

**Hypothesis 5.** The detection failure of RQ3 belongs to the two detectors (VGR's cosine, AUROC 0.72; the ridge map, 0.565), not to the features. A small two-tower network trained with an InfoNCE loss on warm items only, on clean text and without knowledge of the noise, finds the corrupted items better, and with its score driving M3 the inductive gain rises above +0.0004. *Supported* only if on fifteen seeds (2025 to 2039) the mean AUROC on the cold items is at least 0.72 and the mean M3 gain at least +0.0015. Controls: VGR's cosine, the ridge score, a random score, the true mask (+0.0029). The blend weight, quantile and steepness are chosen on noisy validation data and the network's early stopping on held-out warm items; the test set is not used (`rq5_contrastive_detector.py`, `rq5_contrastive_noise_types.py`; the prediction was committed before the runs).

*Partly supported.* The AUROC is 0.868 (range 0.845 to 0.887), but the M3 gain is +0.0014 (9 of 15 seeds positive, bootstrap interval over seeds -0.0001 to +0.0028), below the threshold and half of the oracle's +0.0029. Random scores lose (-0.0015) and the clean Recall@20 falls by 0.0045 (0.0757 to 0.0712). The detector reads image-text agreement and not the look of the hashtags: AUROC 0.910 for shuffled text (the natural text of another item) and 0.508 for Gaussian noise on the embedding (VGR's cosine 0.847 and 0.717). I expected an AUROC of 0.60 to 0.70, so the prediction was wrong on detection and right on the gain. Detection is therefore not the only obstacle.
