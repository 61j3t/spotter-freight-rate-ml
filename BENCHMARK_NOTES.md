# Public benchmark notes

Public assessment repositories were used only to identify testable modeling
hypotheses. Their code was not copied or executed, and their report text was not
reused. Every candidate below was implemented independently and evaluated on
this repository's existing time-based validation protocol.

## Reference point

The strongest directly comparable public claim I found was
[S-V-J/spotter-freight-rate-ml](https://github.com/S-V-J/spotter-freight-rate-ml/tree/eae839b7359a402521c07d9990cd9b91e0379db0),
which reports RMSE 607.05 on a single chronological holdout beginning
2025-09-15. Its report motivated tests of native LightGBM categories, calendar
features, and market/quote interactions.

Using that exact cutoff, this implementation scores RMSE **598.45**, MAE
**101.90**, and MAPE **4.71%**. This is a secondary apples-to-apples check, not
the model-selection score.

## Selection protocol

The primary decision still uses the stricter pooled August, September, and
October expanding-window folds, with every held-out row included in metrics.
Under that protocol:

| Approach | RMSE | Decision |
|---|---:|---|
| Native-categorical LightGBM | **628.73** | Selected |
| Native LightGBM + LightGBM + CatBoost | 629.04 | Rejected |
| Previous LightGBM + CatBoost + Ridge | 629.31 | Replaced |
| Log-rate-per-mile target | 630.21 | Rejected |
| Cross-fold stacking | 630.36 | Rejected |
| Rate-per-mile target | 630.71 | Rejected |
| Log quote-ratio target | 638.14 | Rejected |

The selected change is limited and explainable: pickup, delivery, equipment,
and lane remain native categorical variables for LightGBM, while pickup,
delivery, and lane frequencies are fitted on training rows only within each
fold.
