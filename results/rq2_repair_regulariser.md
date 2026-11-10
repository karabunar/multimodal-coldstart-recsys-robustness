| condition | mean Recall@20 | seeds above reference |
|---|---|---|
| clean text (reference a) | 0.0770 | |
| clean text blended with t_knn, lambda 0.1 | 0.0776 (+0.0006) | 13/15 |
| clean text blended with t_knn, lambda 0.25 | 0.0788 (+0.0018) | 15/15 |
| clean text blended with t_knn, lambda 0.5 | 0.0788 (+0.0018) | 12/15 |
| clean text blended with t_knn, lambda 1.0 | 0.0720 (-0.0050) | 0/15 |
| noisy text, unrepaired | 0.0677 | |
| oracle: corrupted items lambda 1 (reference b) | 0.0739 | |
| oracle plus clean items lambda 0.25 | 0.0753 (+0.0014) | 13/15 |
