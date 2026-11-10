## Image-only baseline, Recall@20, hashtag noise 0.4, 15 seeds

| condition | noisy Recall@20 | share of the loss recovered | noisy NDCG@20 | clean-input Recall@20 |
|---|---|---|---|---|
| Unrepaired | 0.0677 |  | 0.0313 | 0.0770 |
| Image only, flagged items | 0.0751 | 79% | 0.0349 | 0.0742 |
| Image only, all cold items | 0.0739 | 66% | 0.0341 |  |
| Image only, true mask (oracle) | 0.0755 | 84% | 0.0350 |  |
| M1 (blind blend) | 0.0737 | 64% | 0.0335 | 0.0765 |
| M3 (flagged blend) | 0.0756 | 85% | 0.0345 | 0.0786 |

Clean (no corruption) Recall@20 0.0770; unrepaired noisy 0.0677.
M3 minus image-only on flagged items: +0.0005 (std over seeds 0.0019, s.e. 0.0005), M3 higher in 11 of 15 seeds.
The flagged share q is chosen on noisy validation data (NDCG@10) from 0.3, 0.4, 0.5, as for M3. The baseline was added after the other results.
