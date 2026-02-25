"""Tests for learned forecaster integration into the experiment runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import VALID_POLICIES, run_experiment
from hmarl_mvp.learned_forecaster import (
    LearnedForecaster,
    build_forecast_dataset,
    collect_queue_traces,
    train_forecaster,
)


@pytest.fixture()
def small_config() -> dict[str, object]:
    return get_default_config(num_ports=3, num_vessels=4, rollout_steps=15)


@pytest.fixture()
def trained_forecaster(small_config: dict[str, object]) -> LearnedForecaster:
    """Train a small forecaster for integration tests."""
    cfg = dict(small_config)
    num_ports = cfg["num_ports"]
    assert isinstance(num_ports, int)
    traces = collect_queue_traces(
        num_episodes=2,
        steps_per_episode=15,
        config=cfg,
        seed=42,
    )
    dataset = build_forecast_dataset(traces, horizon=5, features_per_port=3)
    forecaster = LearnedForecaster(
        num_ports=num_ports,
        horizon=5,
        features_per_port=3,
        hidden_dims=[32, 32],
    )
    train_forecaster(forecaster, dataset, epochs=10, batch_size=16, verbose=False)
    return forecaster


class TestLearnedForecastPolicyType:
    """Test learned_forecast as a valid policy_type in run_experiment."""

    def test_learned_forecast_in_valid_policies(self) -> None:
        assert "learned_forecast" in VALID_POLICIES

    def test_learned_forecast_requires_forecaster(
        self, small_config: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="learned_forecaster must be provided"):
            run_experiment(
                policy_type="learned_forecast",
                steps=5,
                config=small_config,
            )

    def test_learned_forecast_runs(
        self,
        small_config: dict[str, object],
        trained_forecaster: LearnedForecaster,
    ) -> None:
        df = run_experiment(
            policy_type="learned_forecast",
            steps=10,
            config=small_config,
            learned_forecaster=trained_forecaster,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "avg_queue" in df.columns
        assert "avg_vessel_reward" in df.columns

    def test_learned_forecast_policy_label(
        self,
        small_config: dict[str, object],
        trained_forecaster: LearnedForecaster,
    ) -> None:
        df = run_experiment(
            policy_type="learned_forecast",
            steps=5,
            config=small_config,
            learned_forecaster=trained_forecaster,
        )
        assert (df["policy"] == "learned_forecast").all()

    def test_learned_forecast_produces_finite_metrics(
        self,
        small_config: dict[str, object],
        trained_forecaster: LearnedForecaster,
    ) -> None:
        df = run_experiment(
            policy_type="learned_forecast",
            steps=10,
            config=small_config,
            learned_forecaster=trained_forecaster,
        )
        assert df["avg_queue"].notna().all()
        assert df["total_fuel_used"].notna().all()
        assert np.all(np.isfinite(df["avg_vessel_reward"].values))


class TestLearnedForecasterPredict:
    """Test that a trained LearnedForecaster produces valid forecast shapes."""

    def test_predict_shape(self, trained_forecaster: LearnedForecaster) -> None:
        from hmarl_mvp.state import PortState

        ports = [
            PortState(port_id=i, queue=2, docks=3, occupied=1)
            for i in range(trained_forecaster.num_ports)
        ]
        pred = trained_forecaster.predict(ports)
        assert pred.shape == (trained_forecaster.num_ports, trained_forecaster.horizon)

    def test_predict_non_negative(self, trained_forecaster: LearnedForecaster) -> None:
        from hmarl_mvp.state import PortState

        ports = [
            PortState(port_id=i, queue=0, docks=3, occupied=0)
            for i in range(trained_forecaster.num_ports)
        ]
        pred = trained_forecaster.predict(ports)
        # ReLU output should be non-negative
        assert (pred >= 0).all()
