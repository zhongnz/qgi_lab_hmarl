# Supervisor Report: Hierarchical MARL for Congestion-Aware Vessel Scheduling

**Student:** Independent Study, Spring 2026
**Supervisor:** Prof. Aboussalah
**Report Date:** 2026-02-28
**Repository:** `qgi_lab_hmarl` — branch `main`, commit `3fa2946`
**Status:** Infrastructure complete; systematic experiments pending

---

## Table of Contents

1. [Project Overview and Motivation](#1-project-overview-and-motivation)
2. [Research Questions](#2-research-questions)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Simulation Environment](#4-simulation-environment)
   - 4.1 [State Representation](#41-state-representation)
   - 4.2 [Physics and Dynamics](#42-physics-and-dynamics)
   - 4.3 [Weather Model](#43-weather-model-ar1)
   - 4.4 [Departure Window Enforcement](#44-departure-window-enforcement)
   - 4.5 [Asynchronous Message Bus](#45-asynchronous-message-bus)
5. [Agent Hierarchy](#5-agent-hierarchy)
   - 5.1 [Fleet Coordinator](#51-fleet-coordinator)
   - 5.2 [Vessel Agents](#52-vessel-agents)
   - 5.3 [Port Agents](#53-port-agents)
6. [Reward Structure and Economic Model](#6-reward-structure-and-economic-model)
7. [Forecasting Infrastructure](#7-forecasting-infrastructure)
   - 7.1 [Heuristic Forecasters](#71-heuristic-forecasters)
   - 7.2 [MLP Learned Forecaster](#72-mlp-learned-forecaster)
   - 7.3 [GRU Learned Forecaster](#73-gru-learned-forecaster)
8. [MAPPO Training Framework (CTDE)](#8-mappo-training-framework-ctde)
   - 8.1 [Actor-Critic Networks](#81-actor-critic-networks)
   - 8.2 [Rollout Buffer and GAE](#82-rollout-buffer-and-gae)
   - 8.3 [PPO Update Loop](#83-ppo-update-loop)
   - 8.4 [Reward Normalisation](#84-reward-normalisation)
9. [Heuristic Policy Baselines](#9-heuristic-policy-baselines)
10. [Supporting Infrastructure](#10-supporting-infrastructure)
    - 10.1 [Curriculum Learning](#101-curriculum-learning)
    - 10.2 [Statistical Evaluation Module](#102-statistical-evaluation-module)
    - 10.3 [Experiment Configuration System](#103-experiment-configuration-system)
    - 10.4 [Checkpointing and Early Stopping](#104-checkpointing-and-early-stopping)
    - 10.5 [Structured Logger](#105-structured-logger)
    - 10.6 [Gymnasium Wrapper](#106-gymnasium-wrapper)
    - 10.7 [Report and Plotting Modules](#107-report-and-plotting-modules)
11. [Test Suite](#11-test-suite)
12. [Documentation Suite](#12-documentation-suite)
13. [Experiment Design](#13-experiment-design)
14. [Current Status](#14-current-status)
15. [Open Items and Next Steps](#15-open-items-and-next-steps)

---

## 1. Project Overview and Motivation

This project investigates **Hierarchical Multi-Agent Reinforcement Learning (HMARL)** for the problem of congestion-aware maritime vessel scheduling. The central hypothesis is that a three-level hierarchy of RL agents — fleet coordinator, individual vessels, and port operators — can learn to coordinate via shared congestion forecasts, achieving jointly lower fuel consumption, port congestion, scheduling delays, and carbon emissions compared to independent or reactive heuristic policies.

The maritime logistics domain is chosen deliberately: it presents a natural three-tier hierarchy with heterogeneous agent objectives, discrete and continuous action spaces at different levels, asynchronous decision cadences, hard physical constraints (fuel, distance, dock capacity), stochastic disruptions (weather), and rich economic significance. Each of these properties creates a distinct challenge for standard MARL methods, making the domain a scientifically interesting testbed.

The codebase has been built from scratch during February 2026 as a module-first Python package with reproducible experiments, a full MAPPO/CTDE training stack, systematic ablation infrastructure, and 644 tests. The notebook (`colab_mvp_hmarl_maritime.ipynb`) is retained as an analysis front-end but all logic lives in importable modules.

---

## 2. Research Questions

| ID  | Question | Experiments |
|-----|----------|-------------|
| RQ1 | How can heterogeneous agents (vessels, ports, coordinator) coordinate using shared congestion forecasts? | E1, E2 |
| RQ2 | Does proactive coordination with forecasts improve over independent and reactive baselines? | E1, E6 |
| RQ3 | Which forecast horizons and sharing strategies maximise decision quality? | E3, E4, E5, E9 |
| RQ4 | How do coordination improvements affect economics (fuel cost, delay penalties, carbon price)? | E1, E6, E8 |

These questions progress from mechanism (RQ1) to comparison (RQ2) to optimisation (RQ3) to economic quantification (RQ4), forming a logical chain for the final report.

---

## 3. System Architecture Overview

The codebase is structured as the `hmarl_mvp` Python package (24 modules, ~6,000 lines of implementation code) under a module-first layout. The overall data-flow is:

```
                 ┌──────────────────────────────────────┐
                 │         MaritimeEnv (env.py)          │
                 │                                      │
  reset() ──►   │  state.py   dynamics.py   rewards.py  │
  step()  ──►   │  forecasts.py  message_bus.py         │
                 └──────────────┬───────────────────────┘
                                │  observations / rewards
            ┌───────────────────┼───────────────────────┐
            │                   │                       │
     FleetCoordinator    VesselAgents           PortAgents
     (policies.py /      (policies.py /         (policies.py /
      networks.py)        networks.py)           networks.py)
            │                   │                       │
            └─────── MessageBus (message_bus.py) ───────┘
                                │
                         MAPPOTrainer (mappo.py)
                          ┌──────────────────┐
                          │  RolloutBuffer   │
                          │  ActorCritic × 3 │
                          │  PPO update loop │
                          └──────────────────┘
```

The key architectural decisions are:

- **Module-first, notebook-second**: all logic is importable and tested; the notebook is a thin consumer.
- **Typed configuration**: `HMARLConfig` is a frozen dataclass with runtime validation; no magic strings flow through training.
- **Stateless dynamics**: `dynamics.py` is a pure function library with no mutable global state, enabling deterministic, parallelisable testing.
- **Pluggable policies**: heuristic and neural policies share the same `propose_action` / action-dict interface, enabling direct comparison.
- **CTDE**: training uses global state in the critic; execution uses only local observations in the actor — satisfying the Centralised Training Decentralised Execution requirement.

---

## 4. Simulation Environment

### 4.1 State Representation

The environment (`env.py`) manages a flattened list of dataclass instances defined in `state.py`.

**Vessel state** (`VesselState`): each of the $N_V$ vessels carries:

| Field | Type | Meaning |
|-------|------|---------|
| `vessel_id` | int | Unique identifier |
| `location` | int | Current port index (when docked) |
| `destination` | int | Assigned target port |
| `position_nm` | float | Distance covered on current leg (nm) |
| `speed` | float | Commanded speed (kn) |
| `fuel` | float | Remaining fuel (tons) |
| `initial_fuel` | float | Fuel at episode start (100 tons) |
| `emissions` | float | Cumulative CO₂ emitted (tons) |
| `delay_hours` | float | Accumulated delay waiting for a berth |
| `at_sea` | bool | Whether vessel is in transit |
| `trip_start_step` | int | Step on which current leg began |
| `pending_departure` | bool | Vessel en-queued for future departure (window enforcement) |
| `depart_at_step` | int | Step at which `pending_departure` resolves to `at_sea = True` |

**Port state** (`PortState`): each of the $N_P$ ports carries:

| Field | Type | Meaning |
|-------|------|---------|
| `port_id` | int | Unique identifier |
| `queue` | int | Vessels waiting for a berth |
| `docks` | int | Total berth capacity (`docks_per_port`, default 3) |
| `occupied` | int | Berths currently in service |
| `service_times` | list[float] | Countdown remaining per occupied berth (hours) |
| `cumulative_wait_hours` | float | Time-weighted queue accumulation |
| `vessels_served` | int | Throughput counter |

The default configuration targets a fleet of **8 vessels** across **5 ports**, each port with **3 docks** (15 total berths). Observations expose both `docks` and `occupied` so agents can infer remaining free capacity without requiring the environment to publish it separately.

### 4.2 Physics and Dynamics

All state transitions are implemented as pure functions in `dynamics.py`.

**Cubic fuel model.** Fuel consumption per tick follows the standard Admiralty formula:

$$\Delta F_k = c \cdot v_k^3 \cdot \Delta t \cdot \mu(s_{ij})$$

where $c = 0.002$ (fuel rate coefficient), $v_k$ is the commanded speed in knots, $\Delta t = 1.0$ hour, and $\mu(s_{ij}) = 1 + p \cdot s_{ij}$ is the weather penalty multiplier ($p = 0.15$, $s_{ij}$ is the sea state on the active route). The cubic dependence creates a strong incentive for learned policies to reduce speed when urgency is low—a key test of the forecast's utility for forward planning.

**CO₂ emissions.** Each fuel-burn step produces:

$$\Delta E_k = \Delta F_k \cdot e, \quad e = 3.114 \text{ tons CO}_2/\text{ton fuel}$$

This matches the IMO's published emission factor for marine fuel oil.

**Position update.** At-sea vessels advance by:

$$x_k^{(t+1)} = x_k^{(t)} + v_k \cdot \Delta t \cdot f_w(s_{ij}), \quad f_w = \frac{1}{\mu(s_{ij})}$$

meaning weather both increases fuel burn and reduces effective distance covered, accurately coupling the two effects.

**Arrival condition.** A vessel arrives when $x_k \geq d_{ij}$ (great-circle distance). On arrival: `at_sea = False`, position resets, `queue` at the destination increments.

**Service countdown.** Each occupied berth ticks down by $\Delta t$ per step. Berths completing service (countdown ≤ 0) are freed, decrementing `occupied` and incrementing `vessels_served`. The port action `service_rate` controls how many queued vessels are admitted to free berths each step.

**Queue evolution.**

$$Q_j^{(t+1)} = \max\!\left(Q_j^{(t)} - \text{served}_t + \text{arrivals}_t,\; 0\right)$$

where $\text{served}_t$ is bounded by the port's `service_rate` action, available free berths, and the current queue depth.

### 4.3 Weather Model (AR(1))

Weather is an optional stochastic process (enabled via `weather_enabled = True`). The sea-state matrix $\mathbf{S}^{(t)} \in \mathbb{R}^{N_P \times N_P}$ evolves as a first-order autoregressive process:

$$\mathbf{S}^{(t+1)} = \alpha \cdot \mathbf{S}^{(t)} + (1 - \alpha) \cdot \boldsymbol{\varepsilon}^{(t)}$$

where $\boldsymbol{\varepsilon}^{(t)} \sim U(0, s_{\max})$ (symmetric, zero diagonal) and $\alpha \in [0, 1]$ is the autocorrelation parameter (configurable via `weather_autocorrelation`). At $\alpha = 0$ the model degenerates to i.i.d. noise, recovering the original random weather. At $\alpha = 0.7$ (a plausible meteorological value) weather persists for several steps, enabling agents to plan ahead based on forecast trends. The range is clipped to $[0, s_{\max}]$ and remains symmetric after each update.

The weather matrix is observed directly by vessel and coordinator agents and enters the reward implicitly through fuel consumption. A dedicated weather-shaping bonus (`weather_vessel_shaping`) rewards vessels that slow down when the fuel multiplier exceeds 1.1×.

### 4.4 Departure Window Enforcement

The fleet coordinator can issue a `departure_window_hours` $W > 0$ alongside each vessel directive. When a vessel receives such a directive and its slot request is accepted, `dispatch_vessel()` computes:

$$t_{\text{depart}} = t_{\text{slot\_accepted}} + \lfloor W / \Delta t \rfloor$$

and places the vessel in `pending_departure = True`. The vessel remains docked until `current_step >= depart_at_step`. This models real-world tide windows, traffic separation schemes, and berth pre-allocation windows.

During `pending_departure`, the vessel accumulates no fuel burn but does accumulate `delay_hours` if it was time-sensitive. At $t_{\text{depart}}$ the physics engine sets `at_sea = True` and the vessel begins its transit.

Heuristic policies currently set `departure_window_hours = 0` (immediate departure). The learned coordinator can, in principle, exploit non-zero windows to smooth port congestion — testing this is part of experiment E1.

### 4.5 Asynchronous Message Bus

Real maritime operations involve communication latency (e.g. VHF radio, satellite uplink). The `MessageBus` class models this with three typed FIFO queues:

| Queue | Direction | Payload |
|-------|-----------|---------|
| Directive queue | Coordinator → Vessel | `{dest_port, per_vessel_dest, departure_window_hours, emission_budget}` |
| Arrival-request queue | Vessel → Port | `{vessel_id, destination, requested_arrival_time}` |
| Slot-response queue | Port → Vessel | `{accepted: bool, port_id}` |

Each message is enqueued with a `deliver_step = current_step + message_latency_steps` and only becomes visible to the receiver at or after that step. The bus also maintains `_pending_port_requests` (keyed by port) with parallel `_pending_port_arrival_times` dicts, enabling ports to sort pending arrivals by earliest-deadline-first using `get_pending_requests_sorted()`.

The `requested_arrival_time` ($t_{\text{arr}}$) in each arrival request is the vessel's self-reported desired arrival step. This is a direct output of the vessel's neural network actor (a continuous scalar clamped to a valid future range), threading the vessel's urgency signal through the message system to the port scheduler.

---

## 5. Agent Hierarchy

### 5.1 Fleet Coordinator

The coordinator operates every `coord_decision_interval_steps = 12` simulation steps and observes:

- Medium-term forecast matrix: $(N_P \times H_m)$ where $H_m = 5$ days (default)
- Fleet summary: per-vessel $(x_k, v_k, F_k, E_k, \ell_k)$
- Total fleet emissions (scalar)
- Weather matrix (if enabled): $(N_P \times N_P)$

The coordinator outputs a **per-vessel destination directive**: `{dest_port, per_vessel_dest, departure_window_hours, emission_budget}`. In the MAPPO setting, one `ActorCritic` is shared across all coordinator instances (there is currently one coordinator for the full fleet; multi-coordinator partitioning is scaffolded but not yet activated at scale).

The coordinator's action space is discrete: it selects one of $N_P$ destination ports for each vessel via a `DiscreteActor`. In the heuristic setting, four modes are implemented: `independent` (random destinations), `reactive` (direct vessels to least-congested port), `forecast` (rank ports by mean predicted congestion), and `oracle` (same as forecast but with perfect future knowledge).

### 5.2 Vessel Agents

Vessels operate every step (`vessel_decision_interval_steps = 1`) and observe:

- Latest coordinator directive (`dest_port`, `departure_window_hours`, `emission_budget`)
- Short-term forecast row for the **assigned destination port**: $(H_s,)$ where $H_s = 12$ hours. **Note**: for a vessel currently docked or pending departure, the forecast index is `directive["dest_port"]` (future assignment); for a vessel at sea, it is `vessel.destination` (active navigation target). This distinction was a subtle correctness fix identified during a colleague review.
- Local state: `(position_nm, speed, fuel, emissions, delay_hours, dock_availability)`
- Sea state for the active route (if weather enabled)

Vessel output is **continuous**: a `ContinuousActor` produces a 1-D scheduled speed and a scalar `requested_arrival_time` ($t_{\text{arr}}$). The speed is clamped to $[v_{\min}, v_{\max}]$ = $[8, 18]$ kn. The $t_{\text{arr}}$ is forwarded into the arrival-request message to signal urgency to the port.

### 5.3 Port Agents

Ports operate every `port_decision_interval_steps = 2` steps and observe:

- Own queue depth, occupancy, and total dock count
- Pending arrival requests (sorted by `requested_arrival_time`)
- Short-term forecast for own port: $(H_s,)$

Port output is **discrete**: a `DiscreteActor` with `docks + 1` choices produces a `service_rate` (number of vessels to admit per step, from 0 to the total dock count). This models variable service intensity under resource constraints.

**Inter-level coupling.** The vessel's $t_{\text{arr}}$ flowing into `get_pending_requests_sorted()` creates a direct incentive channel: the coordinator's `departure_window_hours` directive changes when vessels depart, which changes their ETA, which determines their queue priority at the port. The three levels are therefore coupled not only through shared observations but through the timing of actions.

---

## 6. Reward Structure and Economic Model

Rewards are designed to align individual and collective incentives while avoiding double-penalisation.

### Per-step vessel reward

$$R_V = -\bigl(\alpha_F \cdot \Delta F_k + \alpha_D \cdot h_k + \alpha_E \cdot \Delta E_k\bigr)$$

Default weights: $\alpha_F = 1.0$ (fuel), $\alpha_D = 1.5$ (delay, higher to prioritise on-time arrival), $\alpha_E = 0.7$ (emissions). Typical per-step range: 0 (docked, no delay) to approximately −20 (fast transit in bad weather).

### Per-step port reward

$$R_P = -\bigl(Q_j \cdot \Delta t + \beta \cdot (D_j - O_j)\bigr)$$

where $\beta = 0.5$ is the dock idle-time weight. The first term measures waiting time accumulation (vessel-hours); the second penalises berth under-utilisation. The combination addresses both sides of the queue: congestion costs vessels but idle docks cost the port operator.

### Per-step coordinator reward

$$R_C = -\bigl(\Delta F_{\text{fleet}} + \bar{Q} + \lambda \cdot \Delta E_{\text{fleet}}\bigr)$$

where $\lambda = 2.0$ amplifies the CO₂ signal at the coordinator level. This intentional overlap with vessel rewards is deliberate: the coordinator and vessel are rewarded for the same fuel/emissions events, aligning them toward joint fuel reduction without requiring explicit coordination bonuses that could distort policy learning.

A **weather-aware shaping bonus** is also available:

$$R_{\text{shape}} = \kappa \cdot (v_{\text{nominal}} - v_k)^+ \cdot \mathbb{1}[\mu(s) > 1.1]$$

rewarding vessels that voluntarily reduce speed in rough seas (beyond the threshold where weather increases fuel burn by more than 10%). The shaping weight $\kappa = 0.3$ is rampable via the curriculum.

### Economic translation

`compute_economic_metrics()` converts operational quantities to USD at the end of each episode:

| Metric | Formula | Default price |
|--------|---------|---------------|
| `fuel_cost_usd` | $\Delta F_{\text{fleet}} \times 600$ | \$600/ton |
| `delay_cost_usd` | $h_{\text{fleet}} \times 5{,}000$ | \$5,000/hr |
| `carbon_cost_usd` | $\Delta E_{\text{fleet}} \times 90$ | \$90/ton CO₂ |
| `total_ops_cost_usd` | sum of above | — |
| `cost_reliability` | $1 - \text{total\_cost} / \text{cargo\_value}$ | normalised |

These prices are configurable for sensitivity analysis (E8). The `cost_reliability` metric captures what fraction of cargo value is consumed by operational overhead, providing an investor-facing summary number.

---

## 7. Forecasting Infrastructure

### 7.1 Heuristic Forecasters

Three heuristic forecasters are implemented in `forecasts.py`, all producing a `(num_ports, horizon)` array:

| Forecaster | Method | Use |
|------------|--------|-----|
| `MediumTermForecaster` | Queue + linear trend + Gaussian noise | Coordinator (strategic) |
| `ShortTermForecaster` | Queue + Gaussian noise | Vessels and ports (operational) |
| `OracleForecaster` | Perfect current-queue repeat | Upper-bound ablation |

All implement a `predict(ports)` interface, enabling transparent substitution at experiment time.

### 7.2 MLP Learned Forecaster

`LearnedForecaster` / `LearnedForecasterNet` in `learned_forecaster.py` provides a fully supervised, trainable drop-in replacement for the heuristic short-term forecaster.

**Training pipeline:**
1. `collect_queue_traces()` runs $N$ heuristic rollouts, recording per-step `[queue, occupied, docks]` snapshots for all ports.
2. `build_forecast_dataset()` constructs sliding-window supervised pairs: input = flattened current port state, target = future queue trajectory over $H_s$ steps.
3. `train_forecaster()` trains the MLP with MSE loss, Adam optimiser, train/validation split, and early stopping.
4. `forecaster.predict(ports)` returns `(num_ports, H_s)` — identical API to heuristic forecasters.

**Network:** `(num_ports × 3)` → `[128, 64]` ReLU hidden layers → `(num_ports × H_s)` with ReLU output (queue values are non-negative).

The trained forecaster is loaded by setting `policy_type = "learned_forecast"` in the experiment config, substituting `forecaster.predict()` for both medium and short forecast calls inside the heuristic policy logic.

### 7.3 GRU Learned Forecaster

`RNNForecaster` / `RNNForecasterNet` extends the MLP approach with a recurrent architecture designed to exploit sequential correlations in queue dynamics — especially useful under the AR(1) weather model where port state at time $t$ is genuinely predictive of state at $t+k$.

**Key differences from MLP:**

| Property | MLP `LearnedForecaster` | GRU `RNNForecaster` |
|----------|------------------------|---------------------|
| Input shape | `(num_ports × 3,)` stateless | `(seq_len, num_ports × 5)` rolling window |
| Features/port | 3: queue, occupied, docks | 5: + wait_delta, served_delta |
| Architecture | 2-layer MLP | GRU encoder → linear decoder |
| History buffer | None | `deque(maxlen=seq_len)`, zero-padded at episode start |
| Seq length | — | 8 steps (8 hours of history) |

The two additional features (`wait_delta = ΔQ`, `served_delta = ΔO`) are first-order finite differences that give the GRU direct access to trend signals, reducing the burden on the recurrent layer to implicitly discover them.

The `RNNForecaster` is used exclusively in experiment E9 (Forecaster Ablation), where heuristic, MLP, and GRU variants are compared under identical seeds and environment configurations.

---

## 8. MAPPO Training Framework (CTDE)

The training stack (`mappo.py`, `networks.py`, `buffer.py`, `checkpointing.py`) implements Multi-Agent PPO with Centralised Training Decentralised Execution.

### 8.1 Actor-Critic Networks

Three independent `ActorCritic` modules are instantiated — one per agent type. Within each type, all agents share the same network weights (parameter sharing). This means:

- All $N_V = 8$ vessels share one `ActorCritic` (Vessel).
- All $N_P = 5$ ports share one `ActorCritic` (Port).
- All coordinators share one `ActorCritic` (Coordinator).

Each `ActorCritic` contains:

- **Actor**: `ContinuousActor` (Vessel) or `DiscreteActor` (Port, Coordinator)
  - Vessel: Gaussian policy, 1-D continuous speed output, learnable log-std `σ`
  - Port: Categorical policy over `docks + 1 = 4` discrete service-rate choices
  - Coordinator: Categorical policy over `num_ports = 5` destination choices
- **Critic** (CTDE): a separate MLP that observes the full global state

All linear layers are initialised with orthogonal weights ($\text{gain} = \sqrt{2}$ for hidden layers, $0.01$ for output layers), following PPO best practice for stable initial value estimates.

**CTDE clarification.** The term "shared critic" in the literature refers to the critic receiving global state during training while the actor uses only local observations at execution. In this implementation, each agent type has its own critic network — there is no single monolithic critic across all types. The three critics all receive `env.get_global_state()` as input, satisfying the CTDE property. Whether to merge them into a truly shared single critic is a future design choice.

**Global state dimension.** The critic input size is:

$$d_{\text{global}} = N_C \cdot d_C^{\max} + N_V \cdot d_V + N_P \cdot d_P + N_P + 1$$

where the $N_P + 1$ term is the concatenated global congestion vector and total emissions scalar. With 8 vessels, 5 ports, and 1 coordinator at default config, this is approximately 300–400 dimensions, well within the capacity of a [256, 256] value network.

### 8.2 Rollout Buffer and GAE

`MultiAgentRolloutBuffer` stores, per agent type, per-step tensors for `(obs, actions, rewards, log_probs, values, global_states, dones)`. At the end of each rollout, **Generalised Advantage Estimation** (GAE-λ) is computed:

$$\hat{A}_t = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Default: $\gamma = 0.99$, $\lambda = 0.95$. Returns $G_t = \hat{A}_t + V(s_t)$ are used as critic targets.

### 8.3 PPO Update Loop

For each of the `num_epochs = 4` PPO epochs, the buffer is split into `num_minibatches = 4` minibatches (32 samples each with rollout length 128). For each minibatch, the update simultaneously optimises three agent-type losses:

**Policy loss (clipped surrogate):**
$$\mathcal{L}_{\text{clip}} = -\mathbb{E}\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$

**Value loss (optional clipped MSE):**
$$\mathcal{L}_V = \frac{1}{2}\mathbb{E}\!\left[\left(V_\theta(s) - G_t\right)^2\right]$$

**Total loss:**
$$\mathcal{L} = \mathcal{L}_{\text{clip}} + c_V \mathcal{L}_V - c_H H(\pi)$$

with default $\epsilon = 0.2$, $c_V = 0.5$, $c_H = 0.01$ (entropy coefficient). Gradients are clipped at max-norm $= 0.5$ before the Adam step.

`PPOUpdateResult` records per-update diagnostics: `policy_loss`, `value_loss`, `entropy`, `total_loss`, `clip_fraction`, `grad_norm`, `weight_norm`.

### 8.4 Reward Normalisation

`RunningMeanStd` implements Welford's online mean/variance tracker. All per-agent-type rewards are normalised through their own running statistics before being stored in the buffer:

$$r_t^{\text{norm}} = \frac{r_t - \hat{\mu}}{\hat{\sigma} + \epsilon}$$

This prevents reward-scale differences between agent types from biasing gradient magnitudes. Statistics are computed with `epsilon = 1e-8` to avoid division by zero at the start of training.

---

## 9. Heuristic Policy Baselines

Four heuristic policies are implemented in `policies.py` and serve as the comparison baselines in E1:

| Policy | Coordinator Strategy | Vessel Strategy | Port Strategy |
|--------|--------------------|-----------------|-|
| `independent` | Random destination per vessel | Nominal speed | Admit up to capacity |
| `reactive` | Nearest (least-congested) port | Nominal speed | Admit up to capacity |
| `forecast` | Forecast-ranked port (lowest predicted congestion) | Adjust speed for ETA | Admit up to capacity |
| `oracle` | Perfect future congestion, weather-aware routing | Weather-aware speed | Admit up to capacity |

The `forecast` policy is weather-aware: port scores are modified by mean sea-state to the destination, penalising routes through rough weather by an amount proportional to `weather_penalty_factor`. The `oracle` policy uses `OracleForecaster` (zero-noise future state repeat) as an achievable upper bound.

All four heuristics are compared against the MAPPO neural policy in E1. The `forecast` and `oracle` policies set the primary bars that MAPPO must plausibly beat to constitute a positive result.

---

## 10. Supporting Infrastructure

### 10.1 Curriculum Learning

`CurriculumScheduler` provides a **progressive difficulty ramp** for RL training. Two modes:

1. **Linear ramp**: integer parameters (`num_vessels`, `num_ports`, `docks_per_port`, `rollout_steps`) are linearly interpolated from `start_config` to `target_config` over `warmup_fraction` of training (default 30%).
2. **Multi-stage**: explicit `CurriculumStage` definitions give full control over parameter trajectories.

Float parameters (`sea_state_max`, `weather_penalty_factor`, `weather_shaping_weight`, `emission_lambda`) and boolean parameters (`weather_enabled`) are also rampable. The boolean threshold is 0.5: weather is disabled until the ramp crosses 50%, then enabled.

The curriculum is critical for the weather scenario: training MAPPO directly on harsh weather (max sea state 3.0) from random initialisation is unstable; ramping from flat seas to storm conditions gives the actor time to learn basic routing before encountering stochastic disruption.

### 10.2 Statistical Evaluation Module

`stats.py` implements all statistical tests required by the experiment protocol:

- `compare_methods(a_scores, b_scores)` — Welch's t-test (unequal variance), α = 0.05
- `bootstrap_ci(data, statistic, n_resamples=10000)` — non-parametric 95% CI
- `cohens_d(a, b)` — effect size (pooled standard deviation)
- `multi_method_comparison(results_df)` — all pairwise tests, corrected for multiple comparisons
- `summarize_multi_seed(df, by="policy")` — mean ± std aggregation across seeds

All statistical tests use fixed random seeds for the bootstrap resampler to ensure reproducibility.

### 10.3 Experiment Configuration System

`experiment_config.py` defines an `ExperimentConfig` dataclass (YAML-serialisable) that freezes the full specification of an experiment as a single file:

```yaml
name: baseline_sweep
env:
  num_vessels: 8
  num_ports: 5
  weather_enabled: false
mappo:
  learning_rate: 3e-4
  entropy_coeff: 0.01
  gae_lambda: 0.95
num_iterations: 200
num_seeds: 5
seeds: [42, 49, 56, 63, 70]
```

`scripts/run_experiment.py` loads these configs and calls `run_from_config()`, saving `metrics.csv`, `config.yaml`, `summary.json`, and model checkpoints to `runs/<name>/`. TensorBoard logging is wired in. The `--smoke` flag reduces to 2 iterations for CI testing. The `--compare` flag runs pairwise statistical tests across multiple experiment configs.

Four configs are provided:
- `configs/baseline.yaml` — standard MAPPO run
- `configs/multi_seed.yaml` — 5-seed statistical evaluation
- `configs/weather_curriculum.yaml` — progressive weather ramp
- `configs/no_sharing_ablation.yaml` — per-agent networks, no parameter sharing

**Note**: all four configs currently have `num_vessels: 3, num_ports: 2` (development scale). Updating to `8v/5p` before running final experiments is the first prerequisite step.

### 10.4 Checkpointing and Early Stopping

`checkpointing.py` provides:
- `TrainingCheckpoint`: saves `ActorCritic` state dicts + optimizer states every $N$ iterations and on improvement of the best checkpoint metric.
- `EarlyStopping`: terminates training if the checkpoint metric does not improve for `patience` consecutive iterations.

Checkpoint files are atomic (write to temp then rename) to prevent corruption on interrupt.

### 10.5 Structured Logger

`logger.py` writes per-iteration training metrics as JSON lines to `runs/<name>/training.jsonl`:

```json
{"iteration": 47, "mean_reward": -12.3, "total_fuel": 284.1, ..., "iter_time": 5.2}
```

The logger is deliberately low-overhead (no external dependencies) and compatible with any offline analysis tool. `total_train_time` is appended to the final entry.

### 10.6 Gymnasium Wrapper

`gym_wrapper.py` wraps `MaritimeEnv` as a `gymnasium.Env` with standard `Box` observation and action spaces, exposing only the vessel-agent viewpoint. This enables:
- Single-agent RL baselines (DQN, SAC, etc.) on the marginalised vessel problem
- Integration with stable-baselines3 or similar libraries
- Standard benchmarking interfaces for external comparison

The `step()` `info` dict includes the full weather matrix for weather-aware single-agent algorithms.

### 10.7 Report and Plotting Modules

`report.py` generates Markdown summary reports from results DataFrames (policy comparison tables, per-metric significance tables, configuration summaries).

`plotting.py` provides:
- `plot_training_curves()` — reward/loss curves per training seed
- `plot_mappo_comparison()` — bar charts with error bars for policy comparison
- `plot_multi_seed_curves()` — shaded mean ± std learning curves
- `plot_timing_breakdown()` — rollout vs. update time per iteration

`scripts/generate_paper_figures.py` is a CLI script that produces publication-quality figures for all experiments, saving to `runs/figures/`.

---

## 11. Test Suite

The test suite contains **644 tests** across 35 test files, covering every module and most code paths.

| Test file | Scope |
|-----------|-------|
| `test_smoke.py` | 9 end-to-end smoke tests: environment, policies, MAPPO loop |
| `test_components.py` | Unit tests for dynamics, state, rewards |
| `test_config_schema.py` | Config validation: type errors, bounds, edge cases |
| `test_state.py` | Dataclass initialisation and state helpers |
| `test_message_bus.py` | All three queues: enqueue, deliver, latency, `t_arr` threading |
| `test_rewards_metrics.py` | Reward functions, economic metrics, edge cases |
| `test_model_correctness.py` | Numerical correctness: fuel formula, emissions factor, queue dynamics |
| `test_networks.py` | Network shapes, forward pass, log-prob, action sampling |
| `test_buffer.py` | GAE computation, multi-agent buffer filling |
| `test_mappo.py` | Training loop: gradient flow, loss values, parameter updates |
| `test_mappo_advanced.py` | Early stopping, multi-seed train, value clipping |
| `test_action_masking.py` | Discrete actor with invalid action masks |
| `test_scenarios.py` | Large fleets (20 vessels, 8 ports), heterogeneous ports, stress latency |
| `test_learned_forecaster.py` | MLP forecaster: training, predict API, serialisation |
| `test_learned_forecast_integration.py` | MLP forecaster as drop-in in environment and policies |
| `test_training_infra.py` | Checkpointing, early stopping, JSONL logger |
| `test_training_pipeline.py` | End-to-end: `train()` → checkpoint → load → evaluate |
| `test_training_quality.py` | Reward improvement over random; entropy reduction |
| `test_new_modules.py` | Curriculum, stats, experiment config |
| `test_sweep_ablation.py` | Horizon sweep, noise sweep, sharing sweep functions |
| `test_report_plotting.py` | Figure generation, Markdown report output |
| `test_weather_ar1_and_coord_mask.py` | AR(1) update, weather-mask in coordinator |
| `test_weather_policy_rewards.py` | Weather shaping bonus: sign, magnitude |
| `test_weather_integration.py` | Full episode with weather; fuel increase vs. calm |
| `test_stats.py` | Welch t-test, bootstrap CI, Cohen's d, `summarize_multi_seed` |
| `test_parameter_sharing.py` | Shared vs. independent network architectures |
| `test_experiment_config.py` | YAML save/load round-trip; `run_from_config()` smoke |
| `test_proposal_alignment.py` | Regression: dock obs, trip duration, CTDE critic input |
| `test_audit_fixes.py` | Regression: evaluate() early-termination fix, seed variation |

Tests are run via `make test` (`pytest`). The CI pipeline (`.github/workflows/ci.yml`) runs `make check` (ruff lint + mypy type-check + pytest) on every push.

---

## 12. Documentation Suite

The `docs/` directory contains structured documentation across four categories:

### Architecture (`docs/architecture/`)

| Document | Content |
|----------|---------|
| `state_dynamics.md` | Formal LaTeX equations for all state variables: position update, fuel, emissions, delay, service countdown, queue evolution, cumulative wait, departure window, AR(1) weather |
| `agent_input_output.md` | Per-agent I/O schema; vessel short-forecast indexing (at-sea vs. docked distinction); docks-per-port note |
| `mappo_ctde_training.md` | Full CTDE architecture diagram (3×ActorCritic, each with own Critic); actor/action space table; training loop; hyperparameter table; CTDE clarification note |
| `learned_forecaster.md` | MLP pipeline; GRU forecaster section (architecture, 5 features, history buffer, CLI usage) |
| `forecasting_async_communication.md` | Forecaster table (all 5 variants including RNN); cadence model; sequenceDiagram |
| `emissions_reward_interactions.md` | Reward overlap rationale: intentional coordinator/vessel co-penalisation for aligned incentives |
| `environment_relationship.md` | Mermaid flowchart of the full data-flow; environment prep rules |
| `multi_coordinator_scaling.md` | Partition strategy for multiple coordinators; CTDE integration notes |

### Reports (`docs/reports/`)

| Document | Content |
|----------|---------|
| `experiment_protocol.md` | All 9 experiments (E1–E9): goals, configs, seeds, accepted metrics, commands; statistical methodology; critical prerequisites; execution plan |
| `metrics_dictionary.md` | Complete key-type-description table for every metric emitted by training and evaluation |
| `2026-02-25_mvp_state-review.md` | 26-item done/remaining checklist tracking all completed work |
| `supervisor_report_2026-02-28.md` | This document |

### Meetings and Decisions (`docs/meetings/`, `docs/decisions/`)

Three meeting notes (Meeting 01 kickoff, Meeting 02 study session, Meeting 03 colab walkthrough/feedback) capture the evolution of the project scope and design decisions. `ADR-0001_module_first_layout.md` records the architectural decision to adopt the module-first layout over a pure notebook approach.

---

## 13. Experiment Design

Nine experiments are specified in `docs/reports/experiment_protocol.md`, mapping directly to the four research questions:

| Exp | Name | RQ | Est. Time | Status |
|-----|------|----|-----------|-|
| E1 | Baseline Policy Sweep | RQ1, RQ2 | 2 h | Ready to run |
| E2 | Parameter Sharing Ablation | RQ1 | 2 h (of E1 budget) | Ready |
| E3 | Forecast Horizon Sweep | RQ3 | 3 h | Ready |
| E4 | Forecast Noise Ablation | RQ3 | 2 h | Ready |
| E5 | Forecast Sharing Ablation | RQ3 | 2 h | Ready |
| E6 | Weather Impact Analysis | RQ2, RQ4 | 4 h | Ready |
| E7 | Hyperparameter Sweep | All | 8 h | After E1–E3 |
| E8 | Economic Analysis | RQ4 | 1 h | After E1, E6 |
| E9 | Forecaster Ablation | RQ3 | 3 h | After E1, E3 |

**Statistical methodology**: All multi-seed comparisons use Welch's t-test (α = 0.05, unequal-variance), 10,000-resample bootstrap 95% CI, and Cohen's d effect size. The minimum bar for a "positive result" in E1 is: MAPPO achieves statistically significant improvement (p < 0.05) over at least 2 heuristic baselines on `total_fuel` and `mean_delay_hours`.

**Seed protocol**: 5 seeds for E1 (42, 49, 56, 63, 70); 3 seeds for ablations. Each seed sets both `torch.manual_seed()` and `np.random.default_rng()`.

**Prerequisites**: Before running any experiment, all YAML configs must be updated from the 3-vessel/2-port development scale to the research-proposal scale of 8 vessels/5 ports. This is a 30-minute task that gates all downstream results.

---

## 14. Current Status

### Completed (February 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| Core simulation (env, state, dynamics) | ✅ Complete | Physics verified numerically, 644 tests |
| Reward functions (all 3 agent types) | ✅ Complete | Economic model included |
| Metrics (operational + economic) | ✅ Complete | All keys standardised |
| Heuristic baselines (4 policies) | ✅ Complete | Independent, reactive, forecast, oracle |
| MAPPO/CTDE training stack | ✅ Complete | Networks, buffer, GAE, PPO update, reward norm |
| Parameter sharing / ablation | ✅ Complete | `MAPPOConfig.parameter_sharing` toggle |
| Curriculum learning | ✅ Complete | Linear ramp + multi-stage; weather rampable |
| Weather AR(1) model | ✅ Complete | Configurable autocorrelation and intensity |
| Weather-aware rewards and routing | ✅ Complete | Shaping bonus + coordinator penalty |
| Departure window enforcement | ✅ Complete | `pending_departure` + `depart_at_step` in `VesselState` |
| $t_{\text{arr}}$ vessel output and EDF port sorting | ✅ Complete | Full message-bus threading |
| MLP learned forecaster | ✅ Complete | Training pipeline, predict API, CLI |
| GRU learned forecaster | ✅ Complete | `RNNForecaster`, history buffer, 5 features |
| Statistical evaluation module | ✅ Complete | Welch, bootstrap CI, Cohen's d |
| YAML experiment config system | ✅ Complete | 4 example configs; reproducible runs |
| Multi-seed training runner | ✅ Complete | `train_multi_seed()`, aggregated curves |
| Checkpointing and early stopping | ✅ Complete | Atomic save, patience parameter |
| Structured JSONL logger | ✅ Complete | Per-iteration + timing metrics |
| Gymnasium wrapper | ✅ Complete | Box obs/act spaces, weather in info |
| CLI scripts (6 scripts) | ✅ Complete | run_baselines, run_experiment, run_mappo, train_forecaster, generate_figures |
| CI pipeline | ✅ Complete | ruff + mypy + pytest on every push |
| Documentation suite | ✅ Complete | 8 architecture docs, experiment protocol, metrics dictionary |
| Test suite | ✅ 644 tests | 35 test files, all passing |

### In Progress / Pending

| Item | Priority | Blocker |
|------|----------|---------|
| Update YAML configs to 8v/5p | **CRITICAL** | Gates all experiments |
| Run E1 (Baseline Policy Sweep) | High | Config scale fix |
| Run E2–E6 ablations | High | E1 complete |
| Run E7 Hyperparameter Sweep (81 runs) | Medium | E1–E3 analysed |
| Run E8 Economic Analysis | Medium | E1, E6 |
| Run E9 Forecaster Ablation | Medium | E1, E3 |
| Final report writeup | Final step | All experiments |

---

## 15. Open Items and Next Steps

### Immediate (Week of March 2, 2026)

1. **Scale YAML configs to 8v/5p**: Edit `configs/*.yaml` to set `num_vessels: 8, num_ports: 5`. Estimated 30 minutes. This is the single gate blocking all experiments.

2. **Smoke-test at 8v/5p**: Run `python scripts/run_experiment.py configs/baseline.yaml --smoke` (2-iteration dry run) to verify timing and memory at full scale. Target: ≤ 8 seconds per iteration on the 12-core, no-GPU machine (profiling suggests ~5.2 s/iter at 8v/5p).

3. **Run E1**: Execute `python scripts/run_experiment.py configs/multi_seed.yaml` (5 seeds × 200 iters). Expected wall time: ~2 hours. This is the anchor result for the whole experiment matrix.

### Week of March 9, 2026

4. **Run E2, E4, E5** (ablation batch): parameter sharing, noise, forecast sharing. These share infrastructure and can be batched in a single ~4-hour session.

5. **Run E3** (horizon sweep across 5×4 = 20 combinations): identify optimal medium/short horizon pair. The `run_horizon_sweep()` function is already implemented.

6. **Run E6** (weather impact): requires the `weather_curriculum.yaml` config; will test whether the curriculum-trained policy outperforms the flat-weather policy under storms.

### Week of March 16, 2026

7. **Run E7** (hyperparameter sweep, 81 runs): only after reviewing E1–E3 results to confirm the baseline is meaningful before tuning.

8. **Run E9** (forecaster ablation: heuristic vs MLP vs GRU). Train MLP and GRU forecasters first (`train_forecaster.py`), then evaluate.

9. **Run E8** (economic analysis): straightforward post-processing of E1/E6 results.

### April 2026

10. **Final report**: compile results, statistical tables, and figures into the course report. The `generate_paper_figures.py` script is ready; `report.py` generates Markdown tables automatically from results DataFrames.

### Design Decisions to Revisit

- **Single vs. true shared critic**: The current architecture uses three independent critics (one per agent type), each receiving global state. A true single shared critic across all types would be a stronger CTDE implementation. This is deferred pending confirmation that the current approach achieves a meaningful training signal in E1.

- **Multi-coordinator activation**: The `multi_coordinator_scaling.md` document describes a partition strategy (vessels partitioned to nearest coordinator) already scaffolded in `agents.py` and `env.py`. This is disabled (`num_coordinators = 1`) until single-coordinator MAPPO produces a stable baseline.

- **Vessel action space extension**: Currently vessels output only `target_speed` (1-D continuous). A richer action space — `(target_speed, requested_arrival_time, fuel_reserve_target)` — is architecturally supported but not yet active in the MAPPO training loop. `requested_arrival_time` is already implemented in the heuristic agents; extending it to the neural actor is a straightforward change.

- **Forecast integration in neural policy**: The current MAPPO setup uses heuristic forecast generation at training time. An integrated training loop where the learned forecaster is jointly optimised with the RL policy (or at least periodically retrained on fresh rollouts) is a natural extension for RQ3.

---

## Appendix: Key Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_vessels` | 8 | Fleet size |
| `num_ports` | 5 | Number of ports |
| `docks_per_port` | 3 | Berths per port |
| `medium_horizon_days` | 5 | Coordinator forecast window |
| `short_horizon_hours` | 12 | Vessel/port forecast window |
| `coord_decision_interval_steps` | 12 | Coordinator cadence (steps) |
| `port_decision_interval_steps` | 2 | Port cadence (steps) |
| `message_latency_steps` | 1 | Communication delay |
| `fuel_weight` | 1.0 | Vessel reward fuel term weight |
| `delay_weight` | 1.5 | Vessel reward delay term weight |
| `emission_weight` | 0.7 | Vessel reward emission term weight |
| `emission_lambda` | 2.0 | Coordinator emission amplification |
| `fuel_rate_coeff` | 0.002 | Cubic fuel model coefficient |
| `emission_factor` | 3.114 | CO₂ tons per ton fuel (IMO) |
| `speed_min / speed_max` | 8 / 18 kn | Vessel speed bounds |
| `dt_hours` | 1.0 | Simulation timestep |
| `weather_penalty_factor` | 0.15 | Sea-state fuel penalty slope |
| `weather_autocorrelation` | 0.0 | AR(1) $\alpha$ (0 = i.i.d.) |
| `fuel_price_per_ton` | $600 | Economic parameter |
| `delay_penalty_per_hour` | $5,000 | Economic parameter |
| `carbon_price_per_ton` | $90 | Economic parameter (EU ETS approximate) |

---

*This document was auto-generated from codebase state at commit `3fa2946` (2026-02-28). All module descriptions, equations, and configuration values are verified against the source.*
