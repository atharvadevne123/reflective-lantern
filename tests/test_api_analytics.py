"""API tests for the tariff, load-profile, weather-normalization and
demand-response analytics endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

FLAT_DAY = [1.0] * 24
PEAKY_DAY = [1.0] * 20 + [20.0, 22.0, 21.0, 1.0]


class TestTariffCompareEndpoint:
    def test_returns_all_three_schemes(self, client: TestClient) -> None:
        r = client.post("/api/v1/tariff/compare", json=FLAT_DAY)
        assert r.status_code == 200
        data = r.json()
        assert data["flat_cost"] > 0
        assert data["time_of_use_cost"] > 0
        assert data["tiered_cost"] > 0
        assert data["hours_priced"] == 24

    def test_cheapest_scheme_is_a_known_name(self, client: TestClient) -> None:
        r = client.post("/api/v1/tariff/compare", json=FLAT_DAY)
        assert r.json()["cheapest_scheme"] in {"flat", "time_of_use", "tiered"}

    def test_start_hour_shifts_time_of_use_cost(self, client: TestClient) -> None:
        off_peak = client.post("/api/v1/tariff/compare?start_hour=0", json=[1.0] * 5)
        on_peak = client.post("/api/v1/tariff/compare?start_hour=16", json=[1.0] * 5)
        assert on_peak.json()["time_of_use_cost"] > off_peak.json()["time_of_use_cost"]

    def test_empty_series_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/tariff/compare", json=[])
        assert r.status_code == 422

    def test_negative_consumption_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/tariff/compare", json=[1.0, -5.0])
        assert r.status_code == 422

    def test_invalid_start_hour_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/tariff/compare?start_hour=99", json=FLAT_DAY)
        assert r.status_code == 422


class TestLoadProfileEndpoint:
    def test_flat_day_classified_flat(self, client: TestClient) -> None:
        r = client.post("/api/v1/load-profile", json=FLAT_DAY)
        assert r.status_code == 200
        data = r.json()
        assert data["profile_class"] == "flat"
        assert data["load_factor"] == pytest.approx(1.0)

    def test_peaky_day_classified_peaky(self, client: TestClient) -> None:
        r = client.post("/api/v1/load-profile", json=PEAKY_DAY)
        assert r.status_code == 200
        data = r.json()
        assert data["profile_class"] == "peaky"
        assert data["peak_kwh"] == pytest.approx(22.0)

    def test_all_shape_metrics_present(self, client: TestClient) -> None:
        data = client.post("/api/v1/load-profile", json=PEAKY_DAY).json()
        for key in (
            "base_load_kwh",
            "peak_kwh",
            "mean_kwh",
            "load_factor",
            "peak_to_average",
            "max_ramp_kwh",
            "profile_class",
        ):
            assert key in data

    def test_empty_series_rejected(self, client: TestClient) -> None:
        assert client.post("/api/v1/load-profile", json=[]).status_code == 422


class TestWeatherNormalizeEndpoint:
    def test_identical_weather_matches_raw_change(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/weather-normalize"
            "?baseline_kwh=1000&current_kwh=900&baseline_degree_days=500&current_degree_days=500"
        )
        assert r.status_code == 200
        data = r.json()
        assert data["raw_change_pct"] == pytest.approx(-10.0)
        assert data["normalized_change_pct"] == pytest.approx(-10.0)
        assert data["weather_effect_pct"] == pytest.approx(0.0)

    def test_mild_period_reveals_hidden_regression(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/weather-normalize"
            "?baseline_kwh=1000&current_kwh=900&baseline_degree_days=500&current_degree_days=400"
        )
        data = r.json()
        assert data["raw_change_pct"] < 0
        assert data["normalized_change_pct"] > 0

    def test_zero_baseline_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/weather-normalize?baseline_kwh=0&current_kwh=900&baseline_degree_days=500&current_degree_days=500"
        )
        assert r.status_code == 422

    def test_negative_current_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/weather-normalize"
            "?baseline_kwh=1000&current_kwh=-5&baseline_degree_days=500&current_degree_days=500"
        )
        assert r.status_code == 422


class TestDemandResponseEndpoint:
    def test_full_delivery_pays_without_penalty(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demand-response/evaluate?committed_kwh=16",
            json={"baseline_hourly_kwh": [10.0] * 4, "actual_hourly_kwh": [6.0] * 4},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["curtailed_kwh"] == pytest.approx(16.0)
        assert data["penalty"] == 0.0
        assert data["performance_score"] == pytest.approx(1.0)

    def test_shortfall_incurs_penalty(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demand-response/evaluate?committed_kwh=16",
            json={"baseline_hourly_kwh": [10.0] * 4, "actual_hourly_kwh": [8.0] * 4},
        )
        data = r.json()
        assert data["shortfall_kwh"] == pytest.approx(8.0)
        assert data["penalty"] > 0
        assert data["performance_score"] < 1.0

    def test_net_payment_is_incentive_minus_penalty(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demand-response/evaluate?committed_kwh=20",
            json={"baseline_hourly_kwh": [10.0] * 4, "actual_hourly_kwh": [9.0] * 4},
        )
        data = r.json()
        assert data["net_payment"] == pytest.approx(round(data["incentive"] - data["penalty"], 2))

    def test_mismatched_series_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demand-response/evaluate?committed_kwh=5",
            json={"baseline_hourly_kwh": [10.0] * 4, "actual_hourly_kwh": [6.0]},
        )
        assert r.status_code == 422


class TestPowerQualityEndpoint:
    def test_healthy_site_reports_good_factor(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/power-quality?real_power_kw=100&reactive_power_kvar=10",
            json=[230.0, 230.0, 230.0],
        )
        assert r.status_code == 200
        data = r.json()
        assert data["power_factor_rating"] == "good"
        assert data["imbalance_within_limit"] is True

    def test_poor_factor_is_reported(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/power-quality?real_power_kw=100&reactive_power_kvar=90",
            json=[230.0, 230.0, 230.0],
        )
        assert r.json()["power_factor_rating"] == "poor"

    def test_imbalanced_phases_flagged(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/power-quality?real_power_kw=100&reactive_power_kvar=10",
            json=[230.0, 280.0, 190.0],
        )
        assert r.json()["imbalance_within_limit"] is False

    def test_single_phase_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/power-quality?real_power_kw=100&reactive_power_kvar=10",
            json=[230.0],
        )
        assert r.status_code == 422

    def test_correction_zero_when_already_at_target(self, client: TestClient) -> None:
        r = client.get("/api/v1/power-quality/correction?real_power_kw=100&current_power_factor=0.98")
        assert r.status_code == 200
        assert r.json()["required_kvar"] == 0.0

    def test_correction_positive_for_poor_factor(self, client: TestClient) -> None:
        r = client.get("/api/v1/power-quality/correction?real_power_kw=100&current_power_factor=0.75")
        assert r.json()["required_kvar"] > 0

    def test_correction_invalid_target_rejected(self, client: TestClient) -> None:
        r = client.get(
            "/api/v1/power-quality/correction?real_power_kw=100&current_power_factor=0.8&target_power_factor=1.5"
        )
        assert r.status_code == 422


class TestSolarEndpoints:
    def test_economics_reports_full_split(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/solar/economics",
            json={
                "generation_hourly_kwh": [0.0, 2.0, 8.0, 2.0],
                "consumption_hourly_kwh": [3.0, 3.0, 3.0, 3.0],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["self_consumed_kwh"] + data["exported_kwh"] == pytest.approx(data["generated_kwh"])
        assert data["self_consumed_kwh"] + data["imported_kwh"] == pytest.approx(data["consumed_kwh"])

    def test_economics_total_benefit_sums_components(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/solar/economics",
            json={"generation_hourly_kwh": [5.0] * 4, "consumption_hourly_kwh": [3.0] * 4},
        )
        data = r.json()
        assert data["total_benefit"] == pytest.approx(round(data["bill_saving"] + data["export_revenue"], 2))

    def test_economics_mismatched_series_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/solar/economics",
            json={"generation_hourly_kwh": [5.0] * 4, "consumption_hourly_kwh": [3.0]},
        )
        assert r.status_code == 422

    def test_payback_returns_years(self, client: TestClient) -> None:
        r = client.get("/api/v1/solar/payback?system_cost=10000&annual_benefit=1000")
        assert r.status_code == 200
        data = r.json()
        assert data["repays_within_lifetime"] is True
        assert data["payback_years"] > 0

    def test_payback_null_when_never_repaid(self, client: TestClient) -> None:
        r = client.get("/api/v1/solar/payback?system_cost=10000&annual_benefit=0")
        data = r.json()
        assert data["repays_within_lifetime"] is False
        assert data["payback_years"] is None

    def test_payback_negative_cost_rejected(self, client: TestClient) -> None:
        r = client.get("/api/v1/solar/payback?system_cost=-1&annual_benefit=1000")
        assert r.status_code == 422


class TestBatteryEndpoints:
    SPIKY = [10.0, 10.0, 50.0, 60.0, 55.0, 10.0, 10.0]

    def _shave(self, client: TestClient, **params: float):
        query = {
            "capacity_kwh": 200.0,
            "max_charge_kw": 100.0,
            "max_discharge_kw": 100.0,
            "target_peak_kw": 30.0,
        }
        query.update(params)
        qs = "&".join(f"{k}={v}" for k, v in query.items())
        return client.post(f"/api/v1/battery/peak-shave?{qs}", json=self.SPIKY)

    def test_ample_battery_meets_target(self, client: TestClient) -> None:
        r = self._shave(client)
        assert r.status_code == 200
        data = r.json()
        assert data["target_met"] is True
        assert data["peak_after_kw"] <= 30.0

    def test_power_limit_prevents_meeting_target(self, client: TestClient) -> None:
        data = self._shave(client, max_discharge_kw=25.0).json()
        assert data["target_met"] is False
        assert data["peak_after_kw"] == pytest.approx(35.0)

    def test_saving_scales_with_reduction(self, client: TestClient) -> None:
        data = self._shave(client, demand_charge_per_kw=15.0).json()
        assert data["demand_charge_saving"] == pytest.approx(round(data["peak_reduction_kw"] * 15.0, 2))

    def test_invalid_capacity_rejected(self, client: TestClient) -> None:
        assert self._shave(client, capacity_kwh=0.0).status_code == 422

    def test_negative_target_rejected(self, client: TestClient) -> None:
        assert self._shave(client, target_peak_kw=-5.0).status_code == 422

    def test_sizing_reports_required_capacity(self, client: TestClient) -> None:
        r = client.post("/api/v1/battery/sizing?target_peak_kw=30", json=self.SPIKY)
        assert r.status_code == 200
        data = r.json()
        assert data["required_usable_kwh"] == pytest.approx(75.0)
        assert data["peak_load_kw"] == pytest.approx(60.0)

    def test_sizing_zero_when_load_under_target(self, client: TestClient) -> None:
        r = client.post("/api/v1/battery/sizing?target_peak_kw=100", json=self.SPIKY)
        assert r.json()["required_usable_kwh"] == 0.0

    def test_sizing_negative_target_rejected(self, client: TestClient) -> None:
        r = client.post("/api/v1/battery/sizing?target_peak_kw=-1", json=self.SPIKY)
        assert r.status_code == 422


class TestCohortBenchmarkEndpoint:
    COHORT = [80.0, 90.0, 110.0, 120.0, 150.0]

    def test_efficient_building_grades_well(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/benchmark/cohort?annual_kwh=50000&floor_area_m2=1000",
            json=self.COHORT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["grade"] == "A"
        assert data["percentile_rank"] == pytest.approx(100.0)

    def test_wasteful_building_grades_poorly(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/benchmark/cohort?annual_kwh=200000&floor_area_m2=1000",
            json=self.COHORT,
        )
        assert r.json()["grade"] == "F"

    def test_reports_cohort_context(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/benchmark/cohort?annual_kwh=100000&floor_area_m2=1000",
            json=self.COHORT,
        )
        data = r.json()
        assert data["cohort_size"] == 5
        assert data["cohort_median_eui"] == pytest.approx(110.0)

    def test_undersized_cohort_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/benchmark/cohort?annual_kwh=100000&floor_area_m2=1000",
            json=[100.0, 110.0],
        )
        assert r.status_code == 422

    def test_zero_floor_area_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/benchmark/cohort?annual_kwh=100000&floor_area_m2=0",
            json=self.COHORT,
        )
        assert r.status_code == 422
