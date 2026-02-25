"""Tests for learned forecaster: dataset, model, training, and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hmarl_mvp.config import get_default_config
from hmarl_mvp.learned_forecaster import (
    ForecastDataset,
    LearnedForecaster,
    LearnedForecasterNet,
    build_forecast_dataset,
    collect_queue_traces,
    train_forecaster,
)
from hmarl_mvp.state import PortState


class ForecastDatasetTests(unittest.TestCase):
    def test_build_from_traces(self) -> None:
        # 2 episodes, 10 steps each, 3 ports, 3 features per port
        num_ports = 3
        features_per_port = 3
        horizon = 4
        traces = []
        for _ in range(2):
            episode: list[list[float]] = []
            for t in range(10):
                snapshot: list[float] = []
                for p in range(num_ports):
                    snapshot.extend([float(t + p), float(p), 3.0])  # queue, occupied, docks
                episode.append(snapshot)
            traces.append(episode)

        dataset = build_forecast_dataset(traces, horizon=horizon, features_per_port=features_per_port)
        # Each episode has 10 - 4 = 6 samples → 2 * 6 = 12
        self.assertEqual(len(dataset), 12)
        self.assertEqual(dataset.inputs.shape[1], num_ports * features_per_port)
        self.assertEqual(dataset.targets.shape[1], num_ports * horizon)

    def test_empty_traces_produce_empty_dataset(self) -> None:
        dataset = build_forecast_dataset([], horizon=5)
        self.assertEqual(len(dataset), 0)

    def test_short_episode_skipped(self) -> None:
        # Episode shorter than horizon
        traces = [[[1.0, 0.0, 2.0] for _ in range(3)]]
        dataset = build_forecast_dataset(traces, horizon=5, features_per_port=3)
        self.assertEqual(len(dataset), 0)

    def test_to_tensors(self) -> None:
        inputs = np.random.randn(5, 6).astype(np.float32)
        targets = np.random.randn(5, 12).astype(np.float32)
        ds = ForecastDataset(inputs=inputs, targets=targets)
        t_in, t_tgt = ds.to_tensors()
        self.assertEqual(t_in.shape, (5, 6))
        self.assertEqual(t_tgt.shape, (5, 12))


class LearnedForecasterNetTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        net = LearnedForecasterNet(input_dim=9, output_dim=15, hidden_dims=[32, 16])
        x = torch.randn(4, 9)
        y = net(x)
        self.assertEqual(y.shape, (4, 15))

    def test_output_non_negative(self) -> None:
        net = LearnedForecasterNet(input_dim=6, output_dim=10)
        x = torch.randn(10, 6)
        y = net(x)
        self.assertTrue((y >= 0).all())


class LearnedForecasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.forecaster = LearnedForecaster(num_ports=3, horizon=5, hidden_dims=[32, 16])

    def test_predict_shape(self) -> None:
        ports = [
            PortState(port_id=0, queue=2, docks=3, occupied=1),
            PortState(port_id=1, queue=4, docks=3, occupied=2),
            PortState(port_id=2, queue=1, docks=3, occupied=0),
        ]
        pred = self.forecaster.predict(ports)
        self.assertEqual(pred.shape, (3, 5))

    def test_predict_non_negative(self) -> None:
        ports = [
            PortState(port_id=i, queue=i, docks=3, occupied=i % 3) for i in range(3)
        ]
        pred = self.forecaster.predict(ports)
        self.assertTrue(np.all(pred >= 0))

    def test_save_and_load(self) -> None:
        ports = [
            PortState(port_id=0, queue=3, docks=4, occupied=2),
            PortState(port_id=1, queue=1, docks=4, occupied=0),
            PortState(port_id=2, queue=5, docks=4, occupied=3),
        ]
        pred_before = self.forecaster.predict(ports)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            self.forecaster.save(path)
            new_forecaster = LearnedForecaster(num_ports=3, horizon=5, hidden_dims=[32, 16])
            new_forecaster.load(path)
            pred_after = new_forecaster.predict(ports)
        np.testing.assert_array_almost_equal(pred_before, pred_after)


class TrainForecasterTests(unittest.TestCase):
    def test_training_reduces_loss(self) -> None:
        # Create a simple synthetic dataset
        np.random.seed(42)
        num_ports = 2
        horizon = 3
        n_samples = 100
        inputs = np.random.randn(n_samples, num_ports * 3).astype(np.float32)
        # Targets are a simple function of inputs (queue values grow linearly)
        targets = np.abs(np.random.randn(n_samples, num_ports * horizon).astype(np.float32))
        dataset = ForecastDataset(inputs=inputs, targets=targets)

        forecaster = LearnedForecaster(num_ports=num_ports, horizon=horizon, hidden_dims=[32, 16])
        result = train_forecaster(
            forecaster=forecaster,
            dataset=dataset,
            epochs=50,
            batch_size=32,
            lr=1e-3,
        )
        self.assertGreater(len(result.epoch_losses), 0)
        # Loss should decrease from first to last epoch
        self.assertLess(result.epoch_losses[-1], result.epoch_losses[0])
        self.assertEqual(result.num_samples, n_samples)

    def test_empty_dataset_returns_inf(self) -> None:
        forecaster = LearnedForecaster(num_ports=2, horizon=3, hidden_dims=[16])
        empty_ds = ForecastDataset(
            inputs=np.zeros((0, 6), dtype=np.float32),
            targets=np.zeros((0, 6), dtype=np.float32),
        )
        result = train_forecaster(forecaster=forecaster, dataset=empty_ds, epochs=10)
        self.assertEqual(result.final_loss, float("inf"))
        self.assertEqual(result.num_samples, 0)


class CollectQueueTracesTests(unittest.TestCase):
    def test_traces_shape(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=10)
        traces = collect_queue_traces(
            num_episodes=2,
            steps_per_episode=10,
            config=cfg,
            seed=42,
        )
        self.assertEqual(len(traces), 2)
        # Each episode should have 10 snapshots
        self.assertEqual(len(traces[0]), 10)
        # Each snapshot: 3 ports * 3 features = 9 values
        self.assertEqual(len(traces[0][0]), 9)


if __name__ == "__main__":
    unittest.main()
