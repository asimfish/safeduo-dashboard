# Scientific evaluation matrix

- Inputs: 1 summary files
- Episodes: 192; violations: 8
- Episode violation rate: 0.0417
- Wilson 95% upper bound: 0.0801

## Stratified results

| factor | cell | episodes | violations | rate | Wilson 95% upper |
|---|---:|---:|---:|---:|---:|
| flow | directed_all | 96 | 0 | 0.0000 | 0.0385 |
| flow | l1_full | 96 | 8 | 0.0833 | 0.1559 |
| seed | 0 | 64 | 2 | 0.0312 | 0.1070 |
| seed | 1 | 64 | 3 | 0.0469 | 0.1290 |
| seed | 2 | 64 | 3 | 0.0469 | 0.1290 |
| amp | 0.005 | 192 | 8 | 0.0417 | 0.0801 |

## Six arm-pair exposure

| pair | exposed episodes |
|---|---:|
| F_L-F_R | 62 |
| F_L-U_L | 9 |
| F_L-U_R | 35 |
| F_R-U_L | 40 |
| F_R-U_R | 9 |
| U_L-U_R | 39 |

## Design completeness

- Complete: **True**
