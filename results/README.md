# Reference results

Final six-candidate Original and augmented ensemble controls across 15 completed dataset–backbone settings.

- `metrics.csv`: aggregate metrics for candidate banks and committee controls.
- `contrasts.csv`: paired differences and student-bootstrap intervals.
- `per_run_metrics.csv`: 180 single-predictor evaluations.

Source: the frozen six-versus-six analysis in the source snapshot cited in the root README. Columns ending in `_percent` report percentages, including RMSE multiplied by 100. Columns ending in `_pp` report percentage-point differences or interval bounds.

The 95% intervals are pointwise and conditional on fitted predictors and selection decisions; they do not quantify independent retraining uncertainty or apply multiplicity adjustment. Cross-setting means and win counts are descriptive. Candidate counts and epoch caps match; seed identities and wall-clock training costs do not.

These tables do not provide raw student-level predictions or checkpoint files.

