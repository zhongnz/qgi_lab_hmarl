"""Quick diagnostic run to verify training fixes.

Runs 30 MAPPO iterations and prints key metrics:
- Per-agent EV (explained variance) — target: vessel >> 0.036
- Coordinator gradient norm and KL — target: KL >> 10^-6
- Reward trajectory
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

NUM_ITERATIONS = 50

cfg = get_default_config(
    num_ports=5,
    num_vessels=8,
    rollout_steps=69,
    weather_enabled=True,
    forecast_source="ground_truth",
    docks_per_port=1,
    emission_weight=0.0,
    coordinator_compliance_weight=1.5,
)

mappo_cfg = MAPPOConfig(
    rollout_length=128,
    num_epochs=4,
    minibatch_size=128,
    lr=3e-4,
    lr_end=1e-4,
    hidden_dims=[64, 64],
    vessel_hidden_dims=[128, 128],
    entropy_coeff=0.08,
    entropy_coeff_end=0.01,
    coordinator_use_attention=True,
    use_encoded_critic=True,
    vessel_use_recurrence=True,
    total_iterations=NUM_ITERATIONS,
    normalize_observations=True,
    normalize_rewards=True,
)

print("=" * 70)
print(f"Quick diagnostic: {NUM_ITERATIONS} iterations")
print(f"  docks_per_port={cfg['docks_per_port']}, num_vessels={cfg['num_vessels']}, num_ports={cfg['num_ports']}")
print(f"  vessel_lr={mappo_cfg.vessel_lr}, coordinator_lr={mappo_cfg.coordinator_lr}")
print(f"  coordinator_max_grad_norm={mappo_cfg.coordinator_max_grad_norm}")
print(f"  max_grad_norm={mappo_cfg.max_grad_norm} (warmup from {mappo_cfg.max_grad_norm_start})")
print(f"  value_clip_eps={mappo_cfg.value_clip_eps}")
print("=" * 70)

trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=42)

for i in range(NUM_ITERATIONS):
    rollout_info = trainer.collect_rollout()
    update_results = trainer.update()

    if i % 5 == 0 or i == NUM_ITERATIONS - 1:
        v = update_results["vessel"]
        p = update_results["port"]
        c = update_results["coordinator"]
        print(
            f"[{i+1:3d}/{NUM_ITERATIONS}]  "
            f"reward={rollout_info['mean_reward']:+7.2f}  "
            f"v_EV={v.explained_variance:+.3f}  "
            f"p_EV={p.explained_variance:+.3f}  "
            f"c_EV={c.explained_variance:+.3f}  "
            f"v_grad={v.grad_norm:.1f}  "
            f"c_grad={c.grad_norm:.1f}  "
            f"c_KL={c.approx_kl:.2e}  "
            f"v_KL={v.approx_kl:.2e}  "
            f"c_clip={c.clip_fraction:.3f}"
        )

print("\n" + "=" * 70)
print("Evaluation (5-step deterministic):")
eval_metrics = trainer.evaluate(num_steps=30)
for k, v in sorted(eval_metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
print("=" * 70)
