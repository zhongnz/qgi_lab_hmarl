# Learned Forecaster

## Overview

The learned forecaster replaces heuristic queue prediction with a trainable
MLP that maps current port state features to future queue trajectories.
It integrates with the existing `MediumTermForecaster` / `ShortTermForecaster`
interface so it can be used as a drop-in replacement.

## Module Map

| Module | Purpose |
|--------|---------|
| `learned_forecaster.py` | `LearnedForecaster`, `LearnedForecasterNet`, dataset helpers, `train_forecaster()` |
| `experiment.py` | `run_experiment(policy_type="learned_forecast", learned_forecaster=...)` |
| `scripts/train_forecaster.py` | CLI: collect traces → build dataset → train → evaluate |

## Training Pipeline

```
1. collect_queue_traces()
   └── Run N heuristic rollouts → per-step [queue, occupied, docks] snapshots

2. build_forecast_dataset(traces, horizon)
   └── (current_state) → (future_queue_trajectory) supervised pairs

3. train_forecaster(forecaster, dataset, epochs, ...)
   └── MLP training with MSE loss, train/val split, early stopping

4. forecaster.predict(ports)
   └── Returns (num_ports, horizon) array — same as heuristic API
```

## Network Architecture

`LearnedForecasterNet` is a simple feed-forward MLP:

- **Input**: `num_ports × features_per_port` (default 3: queue, occupied, docks)
- **Hidden**: configurable (default `[128, 64]`) with ReLU activations
- **Output**: `num_ports × horizon` with ReLU (queues are non-negative)

## Dataset Format

`ForecastDataset` holds numpy arrays:
- `inputs`: shape `(N, num_ports × features_per_port)`
- `targets`: shape `(N, num_ports × horizon)` — future queue values

Built from `queue_traces`: list of episodes, each a list of per-step
flattened port snapshots.

## Integration with Experiment Runner

```python
from hmarl_mvp import LearnedForecaster, run_experiment

forecaster = LearnedForecaster(num_ports=5, horizon=7)
forecaster.load("runs/forecaster/forecaster_weights.pt")

df = run_experiment(
    policy_type="learned_forecast",
    learned_forecaster=forecaster,
)
```

When `policy_type="learned_forecast"`, the experiment runner:
- Uses `forecaster.predict()` for both medium and short forecasts
- Applies the same heuristic policy logic (mode `"forecast"`) but with
  learned predictions instead of synthetic noise

## CLI Usage

```bash
python scripts/train_forecaster.py \
    --episodes 20 --steps 40 --epochs 200 --verbose
```

Outputs:
- `runs/forecaster/forecaster_weights.pt` — model weights
- `runs/forecaster/eval_metrics.json` — MAE/RMSE on validation set

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_ports` | 5 | Number of ports in the environment |
| `horizon` | 7 | Future steps to predict |
| `features_per_port` | 3 | Features per port (queue, occupied, docks) |
| `hidden_dims` | [128, 64] | MLP hidden layer sizes |
| `epochs` | 100 | Training epochs |
| `batch_size` | 64 | Minibatch size |
| `lr` | 1e-3 | Learning rate |
| `val_fraction` | 0.1 | Validation split ratio |

---

## RNN Forecaster (GRU) — April Ablation (E9)

`RNNForecaster` is a GRU-based alternative that consumes a rolling window of
historical port state to predict future congestion. It is used in the E9
forecaster ablation experiment to compare seq2seq forecasting against the
stateless MLP baseline.

### Module map

| Class / Function | Purpose |
|------------------|---------|
| `RNNForecasterNet` | GRU encoder → linear decoder producing `(num_ports, horizon)` |
| `RNNForecaster` | Wrapper maintaining a history buffer; `predict(ports)` API identical to `LearnedForecaster` |
| `RNNForecastDataset` | PyTorch Dataset — `(seq_len, num_ports × 5)` inputs, `(num_ports × horizon)` targets |
| `build_rnn_dataset` | Converts expanded queue traces to `RNNForecastDataset` |
| `train_rnn_forecaster` | Training loop: MSE loss, Adam, train/val split, early stopping |
| `collect_expanded_queue_traces` | Rollout collector producing 5 features per port per step |

### Input features (5 per port)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `queue` | Number of vessels waiting |
| 1 | `occupied` | Number of docks currently in service |
| 2 | `docks` | Total dock capacity |
| 3 | `wait_delta` | Change in queue length from previous step (∆ queue) |
| 4 | `served_delta` | Change in occupied docks from previous step (∆ occupied) |

### Network architecture

```
Input: (batch, seq_len, num_ports × 5)
  └── GRU encoder (hidden_size=64, num_layers=1)
       └── Last hidden state → Linear → (batch, num_ports × horizon)
            └── ReLU → reshape to (batch, num_ports, horizon)
```

Default: `seq_len=8`, `hidden_size=64`, `horizon=config.short_horizon_hours`.

### History buffer

`RNNForecaster` maintains a `deque(maxlen=seq_len)` of port state features.
On each `predict(ports)` call it appends the current state and pads with zeros
if fewer than `seq_len` steps have been observed. This makes it usable from
the first simulation step without special warm-up logic.

### CLI usage

```bash
# Collect expanded traces and train GRU forecaster
python scripts/train_forecaster.py --model rnn --seq-len 8 --epochs 200 --verbose
```

Outputs `runs/forecaster/rnn_forecaster_weights.pt` and `rnn_eval_metrics.json`.
