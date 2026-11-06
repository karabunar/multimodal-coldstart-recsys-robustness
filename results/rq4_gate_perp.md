## the part of the gate-sensitive subspace outside the user directions (augmented models, clean text)

| seed | w_t cold | drop for G-perp (10 directions) | G-perp directions with AUROC >= 0.75 or <= 0.25 (of 10) | AUROC range | Spearman(change of w_t on warm items, log n) | mean change on warm items |
|---|---|---|---|---|---|---|
| 2025 | 0.991 | +0.372 | 1 | 0.45 to 0.94 | -0.449 | +0.190 |
| 2026 | 0.981 | +0.290 | 1 | 0.13 to 0.67 | -0.233 | +0.241 |
| 2027 | 0.983 | +0.388 | 1 | 0.08 to 0.57 | -0.480 | +0.170 |

Exploratory (not a written hypothesis): the G-perp direction whose coordinate separates cold from warm items most, on warm items.

| seed | direction | AUROC cold against warm | Spearman with log n | Spearman with output norm |
|---|---|---|---|---|
| 2025 | 0 | 0.94 | +0.010 | -0.010 |
| 2026 | 1 | 0.13 | +0.019 | +0.079 |
| 2027 | 0 | 0.08 | -0.038 | -0.006 |
