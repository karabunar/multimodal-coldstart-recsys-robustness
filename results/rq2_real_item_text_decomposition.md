Seeds: 15. Gains are Recall@20 over the unrepaired model; M1 weight chosen on validation data, applied to one group at a time.

| quantity | wrap | mismatch |
|---|---|---|
| items in group "all" | 1410 | 1410 |
| items in group "corrupted" | 558 | 558 |
| items in group "s_clean" | 140 | 140 |
| items in group "outside_s" | 712 | 712 |
| M1 gain, blended into "all" | +0.0047 | +0.0077 |
| M1 gain, blended into "corrupted" | +0.0018 | +0.0046 |
| M1 gain, blended into "s_clean" | +0.0002 | -0.0000 |
| M1 gain, blended into "outside_s" | +0.0022 | +0.0014 |
| m3_gain | 0.0017 | 0.0063 |
| m3_flagged_share | 0.4902 | 0.4991 |
| m3_precision | 0.7954 | 0.7928 |
| m3_recall | 0.9845 | 1.0000 |
| m3_flagged_outside_s | 0.0096 | 0.0099 |
| wrap: seeds with (a) corrupted-only below all, (b) S-clean above outside-S | 14 of 15, 1 of 15 |  |
| mismatch: seeds with (a) corrupted-only below all, (b) S-clean above outside-S | 14 of 15, 3 of 15 |  |
