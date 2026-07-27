# Ablation Module Summary

Reference full method: FullClusterAdd

## Drop-one @ K=5 (Rescue / Harm / Net)

- **P_layout**: rescue=19, harm=7, net=12
- **S_link**: rescue=18, harm=10, net=8
- **LocalWindow**: rescue=18, harm=10, net=8
- **P_type**: rescue=4, harm=1, net=3
- **ClusterPrior**: rescue=3, harm=3, net=0

## Incremental first-hit @ K=5

- DirectOnly: 43
- never: 21
- Direct+Link+Layout: 15
- Direct+Link: 12
- LG-JSSF+ClusterAdd: 2
- Direct+Link+Layout+Type: 2
