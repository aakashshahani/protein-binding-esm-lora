# Data card — UniProtSMB (via CLAPE-SMB)

- Snapshot date: **2024-04-17**
- Source: https://github.com/JueWangTHU/CLAPE-SMB (Raw_data/UniProtSMB)
- Task: per-residue small-molecule binding-site classification
- Label: `1` = binding residue, `0` = non-binding

| split | proteins | residues | binding residues | pos. fraction | mean len |
| --- | --- | --- | --- | --- | --- |
| train | 3972 | 1682806 | 46294 | 0.0275 | 424 |
| valid | 496 | 205218 | 5898 | 0.0287 | 414 |
| test | 496 | 205848 | 5568 | 0.0270 | 415 |

Binding residues are rare (see pos. fraction) — training uses focal / class-weighted loss to handle the imbalance.
