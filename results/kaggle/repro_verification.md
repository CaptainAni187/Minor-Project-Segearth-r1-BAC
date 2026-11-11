# Verification against reference results

New results: `../kaggle/outputs/repro_v2/unz/results`  |  Reference: `../kaggle/outputs/repro_v2/ref`

## Claims (checked on the new run alone)

| claim                                                                                       | status   | detail                                          |
|:--------------------------------------------------------------------------------------------|:---------|:------------------------------------------------|
| Released model reproduces the paper (|gIoU - 68.60| <= 1.0)                                 | PASS     | 68.40                                           |
| Zeroing the 64 visual tokens does not lower accuracy (CI upper bound > 0, |delta| < 1)      | PASS     | +0.38 [-0.03, +1.08], r=0.985                   |
| A different-category image's tokens leave the masks unchanged (per-image r >= 0.95)         | PASS     | +0.00, r=1.000                                  |
| No visual tokens at all: accuracy within 1 gIoU                                             | PASS     | -0.18 [-0.52, +0.08]                            |
| dproj: removing the D-Projector's image input does not lower accuracy (|delta| < 1.5)       | PASS     | +0.24 [-0.65, +1.23], r=0.967                   |
| dproj_zero: removing the D-Projector's image input does not lower accuracy (|delta| < 1.5)  | PASS     | +0.60 [-0.01, +1.47], r=0.978                   |
| Mismatched question: the question matters (drop > 1.5) but there is no collapse (drop < 15) | PASS     | -4.10 [-7.42, -0.62], r=0.580                   |
| Every trained connector at every budget stays within 1.5 gIoU of the released model         | PASS     | 19 connectors; largest |delta|: uniform64 -0.80 |

## Configuration by configuration

| config            | kind      |   n_images |   gIoU_new |   gIoU_ref |   delta |   tolerance | status   |
|:------------------|:----------|-----------:|-----------:|-----------:|--------:|------------:|:---------|
| bac128            | trained   |        300 |      68.37 |      68.75 |   -0.37 |         1.5 | PASS     |
| bac16             | trained   |        300 |      68.12 |      68.11 |    0.01 |         1.5 | PASS     |
| bac32             | trained   |        300 |      68.59 |      67.83 |    0.75 |         1.5 | PASS     |
| bac64             | trained   |        300 |      68.11 |      68.26 |   -0.14 |         1.5 | PASS     |
| bac64_lam0        | trained   |        300 |      67.65 |      68.32 |   -0.66 |         1.5 | PASS     |
| bac64_learned     | trained   |        300 |      68.39 |      68.29 |    0.1  |         1.5 | PASS     |
| bac64_nonms       | trained   |        300 |      67.67 |      67.62 |    0.05 |         1.5 | PASS     |
| bac64_nope        | trained   |        300 |      68.34 |      68.5  |   -0.16 |         1.5 | PASS     |
| bac64_rho25       | trained   |        300 |      67.99 |      68.3  |   -0.3  |         1.5 | PASS     |
| bac64_rho75       | trained   |        300 |      68.19 |      69.1  |   -0.9  |         1.5 | PASS     |
| bac64_sobel       | trained   |        300 |      67.97 |      67.78 |    0.19 |         1.5 | PASS     |
| dproj             | inference |        300 |      68.65 |      68.65 |    0    |         0.5 | PASS     |
| dproj_zero        | inference |        300 |      69    |      69    |    0    |         0.5 | PASS     |
| dproj_zero_wrongq | inference |        300 |      63.04 |      63.04 |    0    |         0.5 | PASS     |
| none              | inference |        300 |      68.22 |      68.22 |   -0    |         0.5 | PASS     |
| none_tta2         | inference |        300 |      68.73 |      68.73 |   -0    |         0.5 | PASS     |
| pool128           | trained   |        300 |      68.24 |      68.85 |   -0.61 |         1.5 | PASS     |
| pool16            | trained   |        300 |      68.74 |      68.95 |   -0.21 |         1.5 | PASS     |
| pool32            | trained   |        300 |      68.36 |      68.78 |   -0.43 |         1.5 | PASS     |
| pool64            | trained   |        300 |      68.59 |      69.26 |   -0.67 |         1.5 | PASS     |
| random64          | trained   |        300 |      68.61 |      68.62 |   -0    |         1.5 | PASS     |
| released          | inference |        300 |      68.4  |      68.4  |    0    |         0.5 | PASS     |
| released_tta2     | inference |        300 |      68.89 |      68.89 |   -0    |         0.5 | PASS     |
| tome64            | trained   |        300 |      68.3  |      68.8  |   -0.49 |         1.5 | PASS     |
| uniform16         | trained   |        300 |      68.38 |      68.8  |   -0.42 |         1.5 | PASS     |
| uniform64         | trained   |        300 |      67.6  |      68.28 |   -0.68 |         1.5 | PASS     |
| wrong             | inference |        300 |      68.4  |      68.4  |   -0    |         0.5 | PASS     |
| wrong_far         | inference |        300 |      68.4  |      68.4  |   -0    |         0.5 | PASS     |
| wrongq            | inference |        300 |      64.3  |      64.3  |   -0    |         0.5 | PASS     |
| zero              | inference |        300 |      68.78 |      68.78 |   -0    |         0.5 | PASS     |
