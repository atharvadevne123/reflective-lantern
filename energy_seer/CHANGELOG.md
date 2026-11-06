# Energy Seer — Changelog

All notable changes to the `energy_seer` sub-project are documented here.
This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [Unreleased]

### Added
- `demand_spike`: expanded `__all__` to include `spike_windows`, `demand_trend`,
  `demand_percentile`, and `smooth_demand` (functions defined after the original
  `__all__` block were silently excluded from the public API).
- `monitoring`: new `get_drift_summary` aggregation helper; added `__all__`.
- `forecaster`: new `forecast_confidence_band` helper; added `__all__`.
- `features`: new `describe_dataframe` summary helper.
- `grid_report`: new `report_health_score` numeric health metric.
- `psi`: new `psi_batch_alert` convenience wrapper; updated `__all__`.
- `aws_stub`: new `delete_model` and `model_exists` stub helpers; added `__all__`.

### Tests
- `test_demand_spike_extended.py` — expanded with `TestDemandVariabilityExtended`,
  `TestDemandTrendEdgeCases`.
- `test_monitoring_extended.py` — added `TestGetDriftSummaryExtended`,
  `TestSummariseAnomaliesExtended`.
- `test_forecaster_extended.py` — new file with 6 test classes covering all
  public functions including `forecast_confidence_band`.
- `test_features_extended.py` — new file covering `validate_dataframe_columns`,
  `describe_dataframe`, `TemporalFeatureTransformer`, `BuildingEncoderTransformer`.
- `test_psi_extended.py` — added `TestPsiReportExtended`, `TestPsiSeriesTrendExtended`,
  `TestPsiBatchAlert`.
- `test_efficiency_score_extended.py` — added `TestBuildingBenchmarksExtended`,
  `TestComputeEfficiencyScoreEdgeCasesExtended`.
- `test_cache_extended.py` — added `TestCacheDefaultTTLExtended`.
- `test_telemetry_extended.py` — added `TestTelemetryIncrementByExtended`.
- `test_validators_extended.py` — added `TestValidateForecastLengthExtended`.
- `test_peak_demand_extended.py` — added `TestPeakDemandWindowEdgeCases`.
- `test_grid_report_extended.py` — added `TestReportHealthScore`.
- `test_aws_stub_extended.py` — new file covering upload, download, list, delete,
  model_exists stubs.
- `test_mlflow_stub_extended.py` — added `TestLogParamsTypesExtended`.
