# MAPPO Training Report

*Generated: 2026-03-06 19:43 UTC*

## Configuration

| Parameter | Value |
|-----------|-------|
| `env.coordinator_departure_window_options` | `(0, 6, 12, 24)` |
| `env.num_ports` | `5` |
| `env.num_vessels` | `8` |
| `env.rollout_steps` | `69` |
| `env.sea_state_max` | `3.0` |
| `env.weather_autocorrelation` | `0.7` |
| `env.weather_enabled` | `True` |
| `mappo.clip_eps` | `0.2` |
| `mappo.device` | `cpu` |
| `mappo.entropy_coeff` | `0.01` |
| `mappo.entropy_coeff_end` | `None` |
| `mappo.gae_lambda` | `0.95` |
| `mappo.gamma` | `0.99` |
| `mappo.grad_accumulation_steps` | `1` |
| `mappo.hidden_dims` | `[64, 64]` |
| `mappo.lr` | `0.0003` |
| `mappo.lr_end` | `0.0` |
| `mappo.lr_warmup_fraction` | `0.0` |
| `mappo.max_grad_norm` | `0.5` |
| `mappo.minibatch_size` | `32` |
| `mappo.normalize_observations` | `True` |
| `mappo.normalize_rewards` | `True` |
| `mappo.num_epochs` | `4` |
| `mappo.parameter_sharing` | `True` |
| `mappo.rollout_length` | `64` |
| `mappo.target_kl` | `0.02` |
| `mappo.total_iterations` | `80` |
| `mappo.value_clip_eps` | `0.2` |
| `mappo.value_coeff` | `0.5` |
| `mappo.weight_decay` | `0.0` |

## Training Summary

- **Iterations**: 80
- **Wall-clock time**: 5.8m
- **Time per iteration**: 4.37s
- **Final mean reward**: -48.586040
- **Best mean reward**: -22.807349
- **Worst mean reward**: -65.154776
- **Reward std**: 8.938388
- **Final vessel mean reward**: -2.905885
- **Final port mean reward**: -1.406250
- **Final coordinator mean reward**: -45.680155
- **Early avg (first 10%)**: -43.448915
- **Late avg (last 10%)**: -41.828500
- **Improvement (late - early)**: +1.620416
- **Final LR**: 0.00e+00
- **Final entropy coeff**: 0.0100

## Per-Agent Training Metrics (Final Iteration)

| Agent | Policy Loss | Value Loss | Entropy | Clip Frac | KL |
|-------|------------|------------|---------|-----------|-----|
| coordinator | -9.45e-06 | 39.6746 | 2.9948 | 0.0000 | 0.0000 |
| port | -3.34e-05 | 2.4979 | 1.3968 | 0.0000 | 1.90e-08 |
| vessel | -9.84e-06 | 30.2337 | 1.8507 | 0.0000 | 4.37e-09 |

## Evaluation (Multi-Episode)

- **Episodes**: 5

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| mean_vessel_reward | -2.3696 | 0.3147 | -2.7432 | -1.9528 |
| mean_port_reward | -1.4200 | 0.0068 | -1.4304 | -1.4130 |
| mean_coordinator_reward | -34.4182 | 5.7170 | -43.2837 | -26.3397 |
| total_reward | -2636.3393 | 414.9570 | -3273.3561 | -2050.0805 |
| avg_speed | 8.0000 | 0.0000 | 8.0000 | 8.0000 |
| avg_fuel_remaining | 58.7523 | 6.8433 | 48.1633 | 68.4579 |
| total_fuel_used | 329.9820 | 54.7461 | 252.3364 | 414.6934 |
| total_emissions_co2 | 1027.5639 | 170.4795 | 785.7756 | 1291.3554 |
| avg_delay_hours | 23.0500 | 4.8564 | 16.8750 | 29.5000 |
| on_time_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| avg_queue | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dock_utilization | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| total_wait_hours | 25.8000 | 10.4575 | 13.0000 | 44.0000 |
| total_vessels_served | 10.6000 | 2.2450 | 7.0000 | 13.0000 |
| avg_wait_per_vessel | 2.3595 | 0.5516 | 1.8571 | 3.3846 |
| fuel_cost_usd | 197989.1879 | 32847.6891 | 151401.8417 | 248816.0641 |
| delay_cost_usd | 922000.0000 | 194257.5610 | 675000.0000 | 1180000.0000 |
| carbon_cost_usd | 92480.7497 | 15343.1556 | 70719.8002 | 116221.9836 |
| total_ops_cost_usd | 1212469.9375 | 194152.5800 | 940994.3164 | 1497122.1346 |
| price_per_vessel_usd | 151558.7422 | 24269.0725 | 117624.2895 | 187140.2668 |
| cost_reliability | 0.8484 | 0.0243 | 0.8129 | 0.8824 |
