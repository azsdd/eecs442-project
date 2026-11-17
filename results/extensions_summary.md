## Backbone study (last-block FT or linear probe, brightness-rotation augmentation, 100% data, 3 seeds)

| Backbone | Adaptation | Trainable params | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | CPU ms/img |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeiT-Tiny (5.7M) | linear probe | 579 | 0.777 ± 0.006 | 0.777 ± 0.007 | 0.685 ± 0.029 | 0.695 ± 0.029 | 15 |
| DeiT-Tiny (5.7M) | last-block FT | 445,827 | 0.803 ± 0.012 | 0.801 ± 0.011 | 0.768 ± 0.028 | 0.775 ± 0.028 | 16 |
| DeiT-Small (22M) | linear probe | 1,155 | 0.860 ± 0.017 | 0.857 ± 0.017 | 0.786 ± 0.029 | 0.790 ± 0.027 | 29 |
| DeiT-Small (22M) | last-block FT | 1,776,387 | 0.867 ± 0.021 | 0.865 ± 0.020 | 0.808 ± 0.027 | 0.811 ± 0.028 | 31 |
| DINOv2-S (22M) | linear probe | 1,155 | 0.847 ± 0.025 | 0.842 ± 0.027 | 0.765 ± 0.065 | 0.772 ± 0.065 | 36 |
| DINOv2-S (22M) | last-block FT | 1,777,155 | 0.883 ± 0.012 | 0.880 ± 0.013 | 0.866 ± 0.019 | 0.870 ± 0.017 | 37 |
| CLIP ViT-B/32 (88M) | linear probe | 2,307 | 0.850 ± 0.010 | 0.845 ± 0.011 | 0.854 ± 0.008 | 0.860 ± 0.007 | 30 |
| CLIP ViT-B/32 (88M) | last-block FT | 7,091,715 | 0.847 ± 0.012 | 0.844 ± 0.012 | 0.885 ± 0.018 | 0.889 ± 0.017 | 32 |

## Item classifier + rule table (two-stage) and CLIP zero-shot

| Model | Training | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | Item accuracy (RW / proxy) |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT-Tiny, direct 3-way head (main model) | 461 reviewed images | 0.803 ± 0.012 | 0.801 ± 0.011 | 0.768 ± 0.028 | 0.775 ± 0.028 | - |
| DeiT-Tiny, 10-way item head + rule table | 461 reviewed images | 0.803 ± 0.015 | 0.802 ± 0.015 | 0.786 ± 0.022 | 0.794 ± 0.022 | 0.723 / 0.638 |
| CLIP ViT-B-16/openai zero-shot, item prompts + rule table | none | 0.790 | 0.794 | 0.859 | 0.862 | 0.700 / 0.796 |
| CLIP ViT-B-16/openai zero-shot, direct bin prompts | none | 0.620 | 0.518 | 0.704 | 0.709 | - |
| CLIP ViT-B-32/openai zero-shot, item prompts + rule table | none | 0.790 | 0.794 | 0.880 | 0.881 | 0.680 / 0.789 |
| CLIP ViT-B-32/openai zero-shot, direct bin prompts | none | 0.540 | 0.453 | 0.585 | 0.542 | - |
| CLIP ViT-L-14/openai zero-shot, item prompts + rule table | none | 0.870 | 0.867 | 0.951 | 0.952 | 0.810 / 0.894 |
| CLIP ViT-L-14/openai zero-shot, direct bin prompts | none | 0.630 | 0.515 | 0.683 | 0.674 | - |

Accuracy on campus-rule items (plastic film + glass, bin = landfill by rule), rigid plastic and all other items:

| System | RW film | RW glass | RW rigid plastic | RW other | Proxy film | Proxy glass | Proxy rigid plastic | Proxy other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct 3-way (main model) | 0.39 (n=6) | 0.85 (n=9) | 0.72 (n=12) | 0.83 | 0.00 (n=3) | 0.70 (n=20) | 0.44 (n=9) | 0.80 |
| Two-stage item + rule | 0.50 (n=6) | 0.67 (n=9) | 0.81 (n=12) | 0.84 | 0.00 (n=3) | 0.75 (n=20) | 0.48 (n=9) | 0.81 |
| CLIP ViT-B/32 zero-shot items | 0.83 (n=6) | 0.67 (n=9) | 0.83 (n=12) | 0.80 | 0.33 (n=3) | 0.85 (n=20) | 0.67 (n=9) | 0.90 |
| CLIP ViT-L/14 zero-shot items | 1.00 (n=6) | 0.89 (n=9) | 0.75 (n=12) | 0.86 | 1.00 (n=3) | 0.90 (n=20) | 0.78 (n=9) | 0.96 |

## Augmentation variants (DeiT-Tiny last-block FT, 100% data, 3 seeds)

| Augmentation | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | Train time (s) |
| --- | --- | --- | --- | --- | --- |
| none | 0.770 ± 0.010 | 0.767 ± 0.014 | 0.744 ± 0.016 | 0.754 ± 0.015 | 16 |
| brightness + rotation (proposal) | 0.803 ± 0.012 | 0.801 ± 0.011 | 0.768 ± 0.028 | 0.775 ± 0.028 | 73 |
| strong: crop, flip, colour, rotation | 0.777 ± 0.006 | 0.771 ± 0.010 | 0.770 ± 0.051 | 0.777 ± 0.051 | 43 |
| strong + background swap (GrabCut mask, synthetic backgrounds, p=0.5) | 0.760 ± 0.017 | 0.753 ± 0.021 | 0.768 ± 0.025 | 0.771 ± 0.025 | 46 |

## Targeted vs random extra annotation (DeiT-Tiny last-block FT + aug, 3 seeds)

| Training set | Images | RealWaste acc | RealWaste F1 | Proxy acc | Proxy F1 | RW film acc | Proxy film acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (461) | 461 | 0.803 ± 0.012 | 0.801 ± 0.011 | 0.768 ± 0.028 | 0.775 ± 0.028 | 0.39 | 0.00 |
| + random extra images | 480 | 0.793 ± 0.031 | 0.788 ± 0.033 | 0.779 ± 0.018 | 0.786 ± 0.020 | 0.39 | 0.00 |
| + targeted plastic-film images | 480 | 0.780 ± 0.020 | 0.778 ± 0.021 | 0.770 ± 0.011 | 0.777 ± 0.010 | 0.44 | 0.00 |

## Calibration and abstention (clipb32; T fitted on validation, mean T = 2.20)

| Test set | Accuracy | ECE raw → calibrated | Acc at 90% coverage | Acc at 80% coverage | Acc at 70% coverage | Coverage / acc at conf ≥ 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| RealWaste test | 0.847 | 0.128 → 0.079 | 0.889 | 0.942 | 0.967 | 0.92 / 0.884 |
| Proxy new-photo test | 0.885 | 0.064 → 0.074 | 0.904 | 0.924 | 0.936 | 0.79 / 0.925 |

## Calibration and abstention (dinov2; T fitted on validation, mean T = 1.97)

| Test set | Accuracy | ECE raw → calibrated | Acc at 90% coverage | Acc at 80% coverage | Acc at 70% coverage | Coverage / acc at conf ≥ 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| RealWaste test | 0.883 | 0.078 → 0.038 | 0.930 | 0.946 | 0.957 | 0.90 / 0.929 |
| Proxy new-photo test | 0.866 | 0.076 → 0.047 | 0.891 | 0.912 | 0.926 | 0.83 / 0.903 |

## Calibration and abstention (main; T fitted on validation, mean T = 1.41)

| Test set | Accuracy | ECE raw → calibrated | Acc at 90% coverage | Acc at 80% coverage | Acc at 70% coverage | Coverage / acc at conf ≥ 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| RealWaste test | 0.803 | 0.087 → 0.061 | 0.833 | 0.875 | 0.924 | 0.81 / 0.868 |
| Proxy new-photo test | 0.768 | 0.076 → 0.064 | 0.802 | 0.833 | 0.843 | 0.64 / 0.850 |

## Calibration and abstention (twostage; T fitted on validation, mean T = 0.77)

| Test set | Accuracy | ECE raw → calibrated | Acc at 90% coverage | Acc at 80% coverage | Acc at 70% coverage | Coverage / acc at conf ≥ 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| RealWaste test | 0.803 | 0.091 → 0.088 | 0.863 | 0.908 | 0.938 | 0.84 / 0.893 |
| Proxy new-photo test | 0.786 | 0.088 → 0.052 | 0.807 | 0.836 | 0.860 | 0.67 / 0.857 |

## Bootstrap 95% confidence intervals (2,000 resamples of the test images, seed-averaged)

| Test set | Comparison | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | P(ΔF1 ≤ 0) |
| --- | --- | --- | --- | --- |
| RealWaste | main model vs plain linear probe | +0.043 [+0.007, +0.083] | +0.043 [+0.004, +0.086] | 0.013 |
| RealWaste | augmentation (last-block FT, 100%) | +0.033 [+0.007, +0.067] | +0.033 [+0.005, +0.066] | 0.010 |
| RealWaste | last-block FT vs linear (aug, 100%) | +0.027 [-0.003, +0.060] | +0.024 [-0.006, +0.057] | 0.057 |
| RealWaste | 100% vs 25% data (last-block FT, aug) | +0.027 [-0.013, +0.070] | +0.030 [-0.012, +0.076] | 0.076 |
| RealWaste | augmentation (last-block FT, 25%) | +0.027 [-0.017, +0.070] | +0.026 [-0.020, +0.072] | 0.137 |
| Proxy | main model vs plain linear probe | +0.052 [+0.009, +0.099] | +0.050 [+0.009, +0.095] | 0.004 |
| Proxy | augmentation (last-block FT, 100%) | +0.023 [+0.000, +0.047] | +0.022 [-0.001, +0.046] | 0.029 |
| Proxy | last-block FT vs linear (aug, 100%) | +0.082 [+0.035, +0.134] | +0.080 [+0.034, +0.132] | 0.000 |
| Proxy | 100% vs 25% data (last-block FT, aug) | +0.040 [-0.002, +0.080] | +0.041 [+0.001, +0.079] | 0.021 |
| Proxy | augmentation (last-block FT, 25%) | +0.085 [+0.052, +0.117] | +0.091 [+0.058, +0.126] | 0.000 |

| Test set | Configuration | Accuracy [95% CI] | Macro-F1 [95% CI] |
| --- | --- | --- | --- |
| RealWaste | lastblock_brightrot_f1.0 | 0.803 [0.727, 0.870] | 0.801 [0.724, 0.864] |
| RealWaste | lastblock_none_f1.0 | 0.770 [0.687, 0.843] | 0.767 [0.686, 0.837] |
| RealWaste | linear_brightrot_f1.0 | 0.777 [0.697, 0.850] | 0.777 [0.697, 0.848] |
| RealWaste | linear_none_f1.0 | 0.760 [0.677, 0.837] | 0.758 [0.674, 0.832] |
| RealWaste | lastblock_brightrot_f0.25 | 0.777 [0.707, 0.837] | 0.771 [0.701, 0.828] |
| RealWaste | lastblock_none_f0.25 | 0.750 [0.677, 0.820] | 0.745 [0.671, 0.809] |
| Proxy | lastblock_brightrot_f1.0 | 0.768 [0.707, 0.829] | 0.775 [0.715, 0.833] |
| Proxy | lastblock_none_f1.0 | 0.744 [0.678, 0.808] | 0.754 [0.688, 0.813] |
| Proxy | linear_brightrot_f1.0 | 0.685 [0.620, 0.751] | 0.695 [0.628, 0.758] |
| Proxy | linear_none_f1.0 | 0.716 [0.646, 0.784] | 0.725 [0.655, 0.789] |
| Proxy | lastblock_brightrot_f0.25 | 0.728 [0.669, 0.786] | 0.735 [0.676, 0.788] |
| Proxy | lastblock_none_f0.25 | 0.643 [0.587, 0.702] | 0.644 [0.586, 0.699] |