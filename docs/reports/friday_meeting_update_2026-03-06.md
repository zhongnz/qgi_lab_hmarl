# Friday Meeting Update — HMARL Maritime Scheduling

**Date:** 2026-03-06  
**Scope:** dynamics summary, reward audit, forecasting note, and extended-training artifacts for the Friday discussion.

## 1. Time-evolving variables by agent

### Vessel agent

For vessel $k$, the time-evolving state is

$$s_{V,k}^{(t)} = \left(x_k^{(t)}, v_k^{(t)}, F_k^{(t)}, E_k^{(t)}, \ell_k^{(t)}, h_k^{(t)}, p_k^{(t)}, t_{\mathrm{dep},k}^{(t)}\right)$$

with updates:

$$x_k^{(t+1)} = x_k^{(t)} + v_k^{(t)} \cdot dt \cdot f_w\!\left(s_{\ell_k,d_k}^{(t)}\right)$$

$$\Delta F_k^{(t)} = c \cdot \left(v_k^{(t)}\right)^3 \cdot dt \cdot \mu\!\left(s_{\ell_k,d_k}^{(t)}\right), \qquad F_k^{(t+1)} = \max\!\left(F_k^{(t)} - \Delta F_k^{(t)}, 0\right)$$

$$\Delta E_k^{(t)} = e \cdot \Delta F_k^{(t)}, \qquad E_k^{(t+1)} = E_k^{(t)} + \Delta E_k^{(t)}$$

$$h_k^{(t+1)} = h_k^{(t)} + dt \quad \text{if the vessel is docked and still waiting for / denied a slot}$$

Pending-departure state is controlled by the coordinator directive:

$$t_{\mathrm{dep},k} = t_{\mathrm{accept},k} + \lfloor W_k / dt \rfloor$$

The vessel remains with `pending_departure=True` until $t \ge t_{\mathrm{dep},k}$, after which `at_sea=True` and the transit update resumes.

### Port agent

For port $j$, the time-evolving state is

$$s_{P,j}^{(t)} = \left(Q_j^{(t)}, O_j^{(t)}, \tau_j^{(t)}, W_j^{(t)}, N_j^{(t)}\right)$$

with updates:

$$\tau_{j,b}^{(t+1)} = \max\!\left(\tau_{j,b}^{(t)} - dt, 0\right)$$

$$Q_j^{(t+1)} = \max\!\left(Q_j^{(t)} - \mathrm{served}_j^{(t)} + \mathrm{arrivals}_j^{(t)}, 0\right)$$

$$W_j^{(t+1)} = W_j^{(t)} + Q_j^{(t)} \cdot dt$$

$$N_j^{(t+1)} = N_j^{(t)} + \mathrm{completed\_services}_j^{(t)}$$

The occupancy variable is induced by active service timers:

$$O_j^{(t)} = \left|\left\{b : \tau_{j,b}^{(t)} > 0\right\}\right|$$

### Fleet coordinator agent

For coordinator $c$, the tracked internal state is

$$s_C^{(t)} = \left(d_C^{(t)}, W_C^{(t)}, B_e^{(t)}, E_{\mathrm{total}}^{(t)}\right)$$

where $d_C$ is the last primary destination, $W_C$ is the last departure-window directive, $B_e$ is the last emission-budget directive, and

$$E_{\mathrm{total}}^{(t)} = \sum_k E_k^{(t)}$$

If the coordinator is due to act at time $t$ and selects

$$a_C^{(t)} = \left(d_C^{\star (t)}, W_C^{\star (t)}, B_e^{\star (t)}\right),$$

then the directive state updates as

$$d_C^{(t+1)} = d_C^{\star (t)}, \qquad W_C^{(t+1)} = W_C^{\star (t)}, \qquad B_e^{(t+1)} = B_e^{\star (t)}$$

and otherwise those three variables remain unchanged. In the current heuristic coordinator used for Friday baselines,

$$B_e^{\star (t)} = \max\!\left(50.0 - 0.1E_{\mathrm{total}}^{(t)}, 10.0\right), \qquad W_C^{\star (t)} = 0$$

More detail and code references are collected in [state_dynamics.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/state_dynamics.md).

## 2. Reward audit and emissions consistency

The implemented rewards are stepwise and use **incremental** physical quantities:

$$R_{V,k}^{(t)} = -\left(\alpha \Delta F_k^{(t)} + \beta \Delta h_k^{(t)} + \gamma \Delta E_k^{(t)}\right)$$

$$R_{P,j}^{(t)} = -\left(Q_j^{(t)} dt + w_{\mathrm{idle}} \cdot \mathrm{idle\_docks}_j^{(t)}\right)$$

$$R_C^{(t)} = -\left(\Delta F_{\mathrm{fleet}}^{(t)} + \overline{Q}^{(t)} + \lambda \Delta E_{\mathrm{fleet}}^{(t)}\right)$$

Key clarification: the emissions penalty uses $\Delta E^{(t)}$ (CO2 emitted during the current step), not the cumulative state $E^{(t)}$. The cumulative emissions are still tracked in state and fed into coordinator observations, but the reward itself is incremental so that past emissions are not penalized repeatedly at every later step.

This is consistent with the proposal's per-step decomposition and preserves the intended hierarchy:

- Vessel reward penalizes each vessel's own current-step fuel, delay, and emissions.
- Coordinator reward penalizes the fleet-level current-step fuel, congestion, and emissions.
- The overlap between vessel and coordinator emissions terms is intentional: it aligns local and global incentives.

Reference: [emissions_reward_interactions.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/emissions_reward_interactions.md).

## 3. Forecasting note for Friday experiments

The forecasters used in the current baseline and MAPPO comparison runs are **heuristic forecasters**, not learned predictors:

- `ShortTermForecaster` provides heuristic short-horizon congestion estimates.
- `MediumTermForecaster` provides heuristic medium-horizon congestion estimates.
- `OracleForecaster` is only an upper-bound diagnostic baseline.
- Learned MLP / GRU forecasters exist in the codebase, but they are **not** used in the Friday training curves reported here.

This point should be stated explicitly in the meeting discussion so the current results are framed as HMARL with heuristic forecast signals rather than learned forecasting.

## 4. Extended training run

An extended full-scale training run was completed at:

- output directory: `runs/friday_meeting_2026-03-06/`
- topology: 8 vessels, 5 ports, 3 docks per port
- weather: enabled, AR(1) with $\alpha = 0.7$
- coordinator departure-window options: $\{0, 6, 12, 24\}$ hours
- training budget: 80 MAPPO iterations with 64-step rollouts

Key outcomes from `train_history.csv` / `eval_result.json`:

- best training mean reward: `-19.46` at iteration `38`
- final training mean reward: `-46.74`
- final per-agent rewards:
  - vessel: `-2.75`
  - port: `-1.42`
  - coordinator: `-43.99`
- early vs late training average reward:
  - first 8 iterations: `-56.05`
  - last 8 iterations: `-45.26`
  - improvement: `+10.79`
- 5-episode evaluation mean total reward: `-3439.16`
- 5-episode evaluation mean emissions: `1358.77` tons CO2
- 5-episode evaluation mean total operating cost: `$1.031M`
- 5-episode evaluation mean delay: `16.18` hours per vessel

Artifacts in that directory:

- `train_history.csv`
- `training_curves.png` (includes vessel / port / coordinator reward curves)
- `report.md`
- `eval_result.json`
- `final_model_*`

The training plot used for the meeting discussion is:

- [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)

The generated run report is:

- [report.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/report.md)
