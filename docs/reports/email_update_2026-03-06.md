# Reply Draft — Mehdi

**Date:** 2026-03-06  
**To:** Mehdi  
**From:** [Student Name]  
**Subject:** HMARL Friday meeting updates

Hi Mehdi,

Thank you for the clear feedback. I have incorporated the requested updates for Friday.

1. For each agent, I wrote out the time-evolving variables and their update rules in mathematical form.
   - Vessel: position, speed, fuel, emissions, delay, and pending-departure state.
   - Port: queue, occupancy, service timers, cumulative waiting time, and throughput.
   - Fleet coordinator: destination directive, departure window, emission budget, and fleet-level cumulative emissions summary.

   The main writeups are:
   - [state_dynamics.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/state_dynamics.md)
   - [friday_meeting_update_2026-03-06.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/reports/friday_meeting_update_2026-03-06.md)

2. I double-checked the reward definitions, especially the emissions term.
   - The reward implementation uses **step-level deltas**, not cumulative emissions.
   - Vessel reward is:
     `R_V^(t) = -(alpha * Δfuel + beta * Δdelay + gamma * ΔCO2)`
   - Port reward is:
     `R_P^(t) = -(queue * dt + idle_dock_penalty)`
   - Coordinator reward is:
     `R_C^(t) = -(Δfleet_fuel + avg_queue + lambda * Δfleet_CO2)`
   - This means past emissions are not penalized repeatedly at every later step. The cumulative emissions are still tracked in state and observations, but the reward itself is incremental. That is the correct interpretation and is now stated explicitly in the documentation.

   The reward note is here:
   - [emissions_reward_interactions.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/emissions_reward_interactions.md)

3. I made the forecasting assumption explicit.
   - The current Friday experiments use **heuristic forecasters** (`ShortTermForecaster` and `MediumTermForecaster`).
   - Learned MLP / GRU forecasters exist in the codebase but are not the forecasters used in the current training curves and discussion figures.

4. I trained for longer and generated updated learning curves.
   - Full-scale run: 8 vessels, 5 ports, weather enabled, AR(1) weather persistence, departure-window options `{0, 6, 12, 24}`.
   - Training budget: 80 MAPPO iterations with 64-step rollouts.
   - The updated training plot now includes **reward vs training iterations for each agent type**: vessel, port, and coordinator.

   Main results from the completed run:
   - Best training mean reward: `-19.46` at iteration `38`
   - Final training mean reward: `-46.74`
   - Final per-agent rewards:
     - vessel: `-2.75`
     - port: `-1.42`
     - coordinator: `-43.99`
   - Early vs late average training reward:
     - first 8 iterations: `-56.05`
     - last 8 iterations: `-45.26`
     - improvement: `+10.79`
   - 5-episode evaluation mean total reward: `-3439.16`
   - 5-episode evaluation mean emissions: `1358.77` tons CO2
   - 5-episode evaluation mean operating cost: `$1.031M`
   - 5-episode evaluation mean delay: `16.18` hours per vessel

5. I also prepared a presentation-style update for Friday, including interpretation of the current diagrams and the main discussion points.

   Main files:
   - [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)
   - [report.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/report.md)
   - [friday_presentation_update_2026-03-07.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/reports/friday_presentation_update_2026-03-07.md)

If you want, I can also turn the presentation update into a shorter speaking version for the actual Friday delivery.

Best,  
[Student Name]
