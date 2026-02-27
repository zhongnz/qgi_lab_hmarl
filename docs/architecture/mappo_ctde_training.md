# MAPPO with Centralised Training Decentralised Execution (CTDE)

## Overview

The project uses Multi-Agent PPO (MAPPO) with a Centralised Training
Decentralised Execution paradigm.  During training, a shared critic
observes the full global state; at execution time, each agent's actor
uses only its local observation.

## Module Map

| Module | Purpose |
|--------|---------|
| `mappo.py` | `MAPPOTrainer`, `MAPPOConfig`, reward normalisation (`RunningMeanStd`), action translators, PPO update loop |
| `networks.py` | `ActorCritic` (per agent type), `ContinuousActor`, `DiscreteActor`, `Critic`, `build_actor_critics`, `obs_dim_from_env` |
| `buffer.py` | `RolloutBuffer`, `MultiAgentRolloutBuffer` — per-step storage with GAE-Lambda returns |
| `checkpointing.py` | `TrainingCheckpoint` (periodic model saves), `EarlyStopping` (patience-based) |
| `experiment.py` | `run_mappo_comparison()` — train + evaluate against heuristic baselines |
| `plotting.py` | `plot_training_curves()`, `plot_mappo_comparison()` |
| `scripts/run_mappo.py` | CLI entry point: `train`, `compare`, `sweep`, `ablate` subcommands |

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     MAPPOTrainer                           │
│                                                            │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │    Vessel     │  │    Port    │  │   Coordinator    │   │
│  │  ActorCritic  │  │ ActorCritic│  │   ActorCritic    │   │
│  │ ┌──────────┐  │  │ ┌────────┐│  │  ┌────────────┐  │   │
│  │ │  Actor   │  │  │ │ Actor  ││  │  │   Actor    │  │   │
│  │ │(ContinuousA) │  │ │(Discrete)│  │  │ (Discrete) │  │   │
│  │ ├──────────┤  │  │ ├────────┤│  │  ├────────────┤  │   │
│  │ │  Critic  │  │  │ │ Critic ││  │  │   Critic   │  │   │
│  │ │(global   │  │  │ │(global ││  │  │  (global   │  │   │
│  │ │ state)   │  │  │  state) ││  │  │   state)   │  │   │
│  │ └──────────┘  │  │ └────────┘│  │  └────────────┘  │   │
│  │ (param-shared │  │ (param-   │  │  (param-shared   │   │
│  │  across all   │  │  shared   │  │   across all     │   │
│  │  vessels)     │  │  across   │  │   coordinators)  │   │
│  └──────────────┘  │  all ports)│  └──────────────────┘   │
│                    └────────────┘                          │
│  NOTE: Each agent type has its OWN Critic network.         │
│  "CTDE" means critics observe the global state during      │
│  training — not that all types share a single critic.      │
│  All critics receive: env.get_global_state()               │
│    = concat(coordinator_obs * N_c,                         │
│             vessel_obs * N_v, port_obs * N_p,              │
│             global_congestion, total_emissions)            │
│                                                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │         MultiAgentRolloutBuffer (per type)        │      │
│  │  Stores: obs, actions, rewards, log_probs,        │      │
│  │          values, global_states, dones             │      │
│  │  Computes: GAE-Lambda advantages, returns         │      │
│  └──────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
```

> **CTDE clarification**: The current implementation has **three separate
> `ActorCritic` modules** (one per agent type), each containing its own critic.
> "Parameter sharing" means all agents of the same type share one network
> (e.g., all vessels use the same `ActorCritic`), not that there is a single
> monolithic shared critic across types. Each critic independently receives the
> full global state during training, satisfying CTDE. A true single shared
> critic across all agent types is a possible future extension.

## Actor / Action Spaces

| Agent Type | Actor | Action Space | Translation |
|------------|-------|-------------|-------------|
| Vessel | `ContinuousActor` | `[target_speed]` (1-D) | Clamp to `[speed_min, speed_max]` |
| Port | `DiscreteActor` | `service_rate` (docks + 1 choices) | Index → `service_rate` dict |
| Coordinator | `DiscreteActor` | `dest_port` (num_ports choices) | Index → per-vessel destination dict |

## Training Loop

```python
trainer = MAPPOTrainer(env_config={...}, mappo_config=MAPPOConfig(...))
for iteration in range(num_iterations):
    rollout_info = trainer.collect_rollout()   # Fill buffers
    update_info  = trainer.update()            # PPO clipped updates
```

Each `collect_rollout()`:
1. Resets buffers and environment
2. For each step: query actors, translate actions, step env, write rewards
3. Normalises rewards if `normalize_rewards=True` (Welford online stats)
4. Computes GAE-Lambda advantages using per-step global states

Each `update()`:
1. Concatenates all agent data per type into flat tensors
2. Normalises advantages (zero-mean, unit-std)
3. Runs `num_epochs` of minibatch PPO with clipped surrogate + clipped value loss
4. Steps the linear LR scheduler

## Key Hyperparameters (`MAPPOConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_length` | 64 | Steps per rollout |
| `num_epochs` | 4 | PPO epochs per update |
| `minibatch_size` | 32 | Minibatch size |
| `clip_eps` | 0.2 | PPO clip epsilon |
| `value_clip_eps` | 0.2 | Value function clip epsilon |
| `value_coeff` | 0.5 | Value loss coefficient |
| `entropy_coeff` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `lr` | 3e-4 | Initial learning rate |
| `lr_end` | 0.0 | Final LR (linear annealing) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `normalize_rewards` | True | Welford reward normalisation |

## Reward Normalisation

Each agent type has an independent `RunningMeanStd` tracker.  Raw rewards
are passed through Welford online updates, then normalised before being
stored in the rollout buffer.  This stabilises training when reward scales
differ across agent types (vessel fuel costs vs coordinator emissions).

## Checkpointing & Early Stopping

- `TrainingCheckpoint`: saves model weights periodically and tracks best
  reward.  Supports `cleanup_old_checkpoints()` and JSON history.
- `EarlyStopping`: monitors a metric (typically `mean_reward`) with
  configurable patience and `min_delta`.  Returns `should_stop=True` when
  improvement stalls.

## Global State Dimension

The critic input dimension is deterministic:

```
N_c × coordinator_obs_dim
+ N_v × vessel_obs_dim
+ N_p × port_obs_dim
+ N_p  (global congestion)
+ 1    (total emissions)
```

Coordinator observations are zero-padded to the maximum coordinator
dimension so the global state size is fixed regardless of how many
vessels each coordinator manages.

## Training Diagnostics

`PPOUpdateResult` carries per-update metrics for each agent type:

| Field | Description |
|-------|-------------|
| `policy_loss` | PPO clipped surrogate loss |
| `value_loss` | Critic MSE loss |
| `entropy` | Policy entropy bonus |
| `total_loss` | Combined loss |
| `clip_fraction` | Fraction of samples clipped |
| `grad_norm` | Gradient L2 norm after clipping |
| `weight_norm` | Total parameter L2 norm after update |

`MAPPOTrainer.get_diagnostics()` returns a summary dict containing:
- Per-agent-type weight norms and parameter counts
- Reward normaliser running statistics (mean, var, count)
- Current learning rate and iteration number

## Multi-Seed MAPPO Evaluation

`run_multi_seed_mappo_comparison()` trains MAPPO independently per seed,
evaluates each against heuristic baselines, and returns a single
`DataFrame` with `seed` + `policy` columns compatible with
`summarize_multi_seed()` for mean ± std aggregation.
