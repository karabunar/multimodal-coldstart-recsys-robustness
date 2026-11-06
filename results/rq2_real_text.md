## real text for the items of the Baby split

Items with a 2023 catalogue entry: 3443 of 7050; cold items among them 698 of 1410, warm items 2745. Items with a second-level category: 3353.

### overlap@10 of the 10 nearest items in `text_feat` with the 10 nearest in TF-IDF space

| source | overlap@10 |
|---|---|
| T title | 0.1962 |
| P title+features+description | 0.2467 |
| R reviews (2014) | 0.1865 |
| permuted control (P) | 0.0033 |
| chance | 0.0029 |

H32a: P/R = 1.32 (predicted at least 1.25). H32b: P - T = +0.0505 (predicted at least 0). H32c: best source 0.247 (predicted 0.05 to 0.30), control 0.0033 (predicted below 0.01).

### neighbours of the cold items in the subset (10 nearest warm items by each space; those with a catalogue entry are kept)

| neighbours | title similarity (TF-IDF cosine) | same second-level category | kept per item |
|---|---|---|---|
| image | 0.1181 | 0.510 | 5.6 |
| text | 0.1777 | 0.684 | 5.8 |
| random reference (20 draws) | 0.0143 | 0.146 | 10.0 |

H33a: title similarity image / random = 8.23 (predicted at least 1.5). H33b: category share image / random = 3.49 (predicted at least 2). H33c: text minus image title similarity = +0.0595 (predicted above 0).
