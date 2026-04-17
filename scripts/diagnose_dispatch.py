"""Diagnose why MAPPO produces 0 vessel dispatches.

Traces the first 15 steps of an episode with:
1. Heuristic (stub) policy — expected to dispatch vessels
2. Random MAPPO policy (untrained) — test whether the mechanism works
3. Trained MAPPO policy (50 iterations) — test learned behavior

Prints per-step: port mask, port action, accept count, pending requests,
delivered responses, dispatch events, vessel awaiting state.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    _nn_to_port_action,
    _nn_to_vessel_action,
    _nn_to_coordinator_action,
    _decode_port_action,
    _port_action_levels,
)

SEED = 42
NUM_STEPS = 15

ENV_KWARGS = dict(
    num_ports=3,
    num_vessels=15,
    num_coordinators=1,
    docks_per_port=2,
    rollout_steps=69,
    forecast_source="ground_truth",
    emission_weight=0.0,
    coordinator_compliance_weight=4.5,
)


def trace_episode(trainer: MAPPOTrainer, label: str, train_iters: int = 0):
    """Trace dispatch chain for NUM_STEPS."""
    # Train if requested
    for _ in range(train_iters):
        trainer.collect_rollout()
        trainer.update()

    env = trainer.env
    obs = env.reset(seed=SEED)
    cfg = env.cfg

    print(f"\n{'=' * 80}")
    print(f"TRACE: {label} ({train_iters} training iterations)")
    print(f"{'=' * 80}")

    levels = _port_action_levels(cfg)
    print(f"Port action levels: {levels} (docks={cfg['docks_per_port']})")
    print(f"Port actions: ", end="")
    for idx in range(levels * levels):
        sr, ar = _decode_port_action(idx, cfg)
        print(f"  {idx}→(svc={sr},acc={ar})", end="")
    print()
    print(f"Cadence: coord={cfg['coord_decision_interval_steps']}, "
          f"vessel={cfg['vessel_decision_interval_steps']}, "
          f"port={cfg['port_decision_interval_steps']}, "
          f"latency={cfg.get('message_latency_steps', 1)}")

    # Initial state
    print(f"\nInitial state:")
    for v in env.vessels[:5]:
        print(f"  V{v.vessel_id}: loc={v.location} dest={v.destination} "
              f"at_sea={v.at_sea} fuel={v.fuel:.0f} "
              f"port_busy={int(getattr(v, 'port_service_state', 0))}")
    print(f"  ... ({len(env.vessels)} vessels total)")
    for p in env.ports:
        print(f"  P{p.port_id}: queue={p.queue} occupied={p.occupied}/{p.docks} "
              f"queued_ids={getattr(p, 'queued_vessel_ids', [])} "
              f"servicing_ids={getattr(p, 'servicing_vessel_ids', [])}")

    dispatched_total = 0
    for step in range(NUM_STEPS):
        # Build actions using the learned policy
        global_state = env.get_global_state()
        gs_np = np.array(global_state, dtype=np.float32)
        gs_tensor = torch.as_tensor(gs_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Vessel actions
            vessel_actions = []
            for i, v_obs in enumerate(obs["vessels"]):
                v_t = torch.as_tensor(
                    np.array(v_obs, dtype=np.float32), dtype=torch.float32
                ).unsqueeze(0)
                ac = trainer._get_ac("vessel", i)
                a, _, _, _ = ac.get_action_and_value(v_t, gs_tensor, deterministic=True)
                vessel_actions.append(
                    _nn_to_vessel_action(a.squeeze(0), cfg, current_step=env.t)
                )

            # Port actions
            port_actions = []
            port_details = []
            for i, p_obs in enumerate(obs["ports"]):
                p_t = torch.as_tensor(
                    np.array(p_obs, dtype=np.float32), dtype=torch.float32
                ).unsqueeze(0)
                p_mask = trainer._port_mask_tensor(i)
                ac = trainer._get_ac("port", i)
                a, _, _, _ = ac.get_action_and_value(
                    p_t, gs_tensor, deterministic=True, action_mask=p_mask
                )
                action_dict = _nn_to_port_action(a.squeeze(0), i, env)
                port_actions.append(action_dict)

                # Record details
                mask_np = p_mask.squeeze(0).cpu().numpy()
                backlog = len(env.bus.get_pending_requests(i))
                port_details.append({
                    "action_idx": int(a.item()),
                    "accept": action_dict["accept_requests"],
                    "service": action_dict["service_rate"],
                    "backlog": backlog,
                    "mask": mask_np.astype(int).tolist(),
                })

            # Coordinator actions
            assignments = env._build_assignments()
            coord_actions = []
            c_mask = trainer._coordinator_mask_tensor()
            for i, c_obs in enumerate(obs["coordinators"]):
                c_t = torch.as_tensor(
                    np.array(c_obs, dtype=np.float32), dtype=torch.float32
                ).unsqueeze(0)
                ac = trainer._get_ac("coordinator", i)
                a, _, _, _ = ac.get_action_and_value(
                    c_t, gs_tensor, deterministic=True, action_mask=c_mask
                )
                coord_actions.append(
                    _nn_to_coordinator_action(
                        a.squeeze(0), i, env, assignments,
                        per_vessel=trainer._use_per_vessel_coordinator,
                    )
                )

        actions_dict = {
            "coordinator": coord_actions[0] if coord_actions else {},
            "coordinators": coord_actions,
            "vessels": vessel_actions,
            "ports": port_actions,
        }

        # Count states before step
        awaiting = sum(1 for v in env.vessels if env.bus.is_awaiting(v.vessel_id))
        at_sea = sum(1 for v in env.vessels if v.at_sea)
        port_busy = sum(1 for v in env.vessels
                       if int(getattr(v, 'port_service_state', 0)) > 0)

        # Step
        obs, rewards, done, info = env.step(actions_dict)

        # Count dispatches from events
        events = info.get("events", [])
        dispatches = [e for e in events if e.get("event_type") == "vessel_dispatched"]
        requests_enqueued = [e for e in events if e.get("event_type") == "arrival_request_enqueued"]
        slot_responses = [e for e in events if e.get("event_type") == "slot_response_enqueued"]
        dispatched_total += len(dispatches)

        due = info.get("cadence_due", {})

        print(f"\n--- Step {step} (t={env.t - 1}) due={due} ---")
        print(f"  Vessels: awaiting={awaiting} at_sea={at_sea} port_busy={port_busy}")
        for i, pd in enumerate(port_details):
            due_port = due.get("port", False)
            print(f"  Port {i}: backlog={pd['backlog']} mask={pd['mask']} "
                  f"action={pd['action_idx']}→(svc={pd['service']},acc={pd['accept']}) "
                  f"{'[DUE]' if due_port else '[skip]'}")
        print(f"  Events: {len(requests_enqueued)} slot_reqs, "
              f"{len(slot_responses)} slot_responses, "
              f"{len(dispatches)} dispatches")
        if dispatches:
            for d in dispatches:
                print(f"    DISPATCH: V{d['vessel_id']}→P{d['port_id']} "
                      f"speed={d.get('applied_speed', '?')}")
        print(f"  Rewards: vessel={np.mean(rewards['vessels']):+.2f} "
              f"port={np.mean(rewards['ports']):+.2f} "
              f"coord={rewards['coordinator']:+.2f}")

    print(f"\n  TOTAL DISPATCHES after {NUM_STEPS} steps: {dispatched_total}")
    at_sea_final = sum(1 for v in env.vessels if v.at_sea)
    print(f"  Vessels at sea: {at_sea_final}")
    print(f"  Completed arrivals: {sum(v.completed_arrivals for v in env.vessels)}")


if __name__ == "__main__":
    cfg = get_default_config(**ENV_KWARGS, weather_enabled=False)
    mappo_cfg = MAPPOConfig(
        rollout_length=128, num_epochs=4, minibatch_size=64, total_iterations=100,
    )

    # Test 1: Untrained (random) policy
    trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=SEED)
    trace_episode(trainer, "RANDOM (untrained)", train_iters=0)

    # Test 2: After 50 iterations of training
    trainer2 = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=SEED)
    trace_episode(trainer2, "TRAINED (50 iters)", train_iters=50)
