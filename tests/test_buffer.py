"""Tests for RolloutBuffer and MultiAgentRolloutBuffer."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.buffer import MultiAgentRolloutBuffer, RolloutBuffer


class RolloutBufferTests(unittest.TestCase):
    def test_empty_buffer_properties(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=4)
        self.assertEqual(buf.size, 0)
        self.assertFalse(buf.full)

    def test_add_and_size(self) -> None:
        buf = RolloutBuffer(capacity=5, obs_dim=3)
        for i in range(5):
            buf.add(
                obs=np.zeros(3),
                action=0.5,
                reward=1.0,
                done=False,
                log_prob=-0.1,
                value=2.0,
            )
        self.assertEqual(buf.size, 5)
        self.assertTrue(buf.full)

    def test_add_raises_when_full(self) -> None:
        buf = RolloutBuffer(capacity=2, obs_dim=2)
        buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)
        buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=True)
        with self.assertRaises(RuntimeError):
            buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)

    def test_reset_clears_buffer(self) -> None:
        buf = RolloutBuffer(capacity=3, obs_dim=2)
        buf.add(obs=np.ones(2), action=1.0, reward=1.0, done=False)
        buf.reset()
        self.assertEqual(buf.size, 0)
        self.assertFalse(buf.full)

    def test_compute_returns_simple(self) -> None:
        buf = RolloutBuffer(capacity=3, obs_dim=1, gamma=0.99, lam=0.95)
        buf.add(obs=np.array([1.0]), action=0.0, reward=1.0, done=False, value=0.5)
        buf.add(obs=np.array([2.0]), action=0.0, reward=2.0, done=False, value=1.0)
        buf.add(obs=np.array([3.0]), action=0.0, reward=3.0, done=True, value=1.5)
        buf.compute_returns(last_value=0.0)
        data = buf.get_tensors()
        self.assertEqual(data["advantages"].shape[0], 3)
        self.assertEqual(data["returns"].shape[0], 3)
        # Returns should not be all zeros
        self.assertGreater(data["returns"].abs().sum().item(), 0.0)

    def test_get_tensors_shape(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, act_dim=2)
        for _ in range(4):
            buf.add(
                obs=np.random.randn(3),
                action=np.random.randn(2),
                reward=1.0,
                done=False,
            )
        data = buf.get_tensors()
        self.assertEqual(data["obs"].shape, (4, 3))
        self.assertEqual(data["actions"].shape, (4, 2))
        self.assertEqual(data["rewards"].shape, (4,))

    def test_minibatch_iter_covers_all_data(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=2)
        for _ in range(10):
            buf.add(obs=np.random.randn(2), action=0.0, reward=1.0, done=False)
        buf.compute_returns()
        total = 0
        for batch in buf.minibatch_iter(batch_size=3, shuffle=False):
            total += batch["obs"].shape[0]
        self.assertEqual(total, 10)


class MultiAgentRolloutBufferTests(unittest.TestCase):
    def test_creation_and_indexing(self) -> None:
        mab = MultiAgentRolloutBuffer(num_agents=3, capacity=5, obs_dim=4)
        self.assertIsInstance(mab[0], RolloutBuffer)
        self.assertIsInstance(mab[2], RolloutBuffer)

    def test_full_requires_all_agents(self) -> None:
        mab = MultiAgentRolloutBuffer(num_agents=2, capacity=2, obs_dim=1)
        mab[0].add(obs=np.zeros(1), action=0.0, reward=0.0, done=False)
        mab[0].add(obs=np.zeros(1), action=0.0, reward=0.0, done=True)
        self.assertFalse(mab.full)  # agent 1 still empty
        mab[1].add(obs=np.zeros(1), action=0.0, reward=0.0, done=False)
        mab[1].add(obs=np.zeros(1), action=0.0, reward=0.0, done=True)
        self.assertTrue(mab.full)

    def test_reset_clears_all(self) -> None:
        mab = MultiAgentRolloutBuffer(num_agents=2, capacity=3, obs_dim=2)
        mab[0].add(obs=np.ones(2), action=0.0, reward=0.0, done=False)
        mab[1].add(obs=np.ones(2), action=0.0, reward=0.0, done=False)
        mab.reset()
        self.assertEqual(mab[0].size, 0)
        self.assertEqual(mab[1].size, 0)


if __name__ == "__main__":
    unittest.main()
