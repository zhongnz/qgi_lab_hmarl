# Meeting 04: Notebook-to-Scripts Refactor

## Meeting

- Title: Meeting 4 - Notebook-to-Scripts Refactor
- Date: 2026-02-28
- Location/Call: QGI Lab (in person)
- Note taker: Peter

## Attendees

- Professor Amine Aboussalah
- Idriss Malek
- El Mehdi Nezahi

## Agenda

1. Review the refactoring from notebook workflow to Python scripts
2. Identify remaining documentation work for the transition kernel and rewards
3. Define the next validation run and plotting tasks

## Discussion Notes

- The team discussed the refactoring of the codebase from a notebook-driven workflow into Python scripts.
- The refactor was framed as the main path forward for making the project easier to run, inspect, and maintain.
- The next technical priority identified was to fully examine and document the transition kernel and reward structure.
- The team also agreed on a short validation step: run a test model from the refactored codebase and generate plots for loss and rewards.

## Decisions Made

- Continue with the Python-script refactor as the primary development path.
- Prioritize clear documentation of the transition kernel and rewards before broader experimentation.
- Validate the refactored pipeline with a test run and basic loss/reward plots.

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Fully examine and document the transition kernel and reward definitions | Peter | Next meeting | Open |
| Run a test model from the refactored scripts and generate loss/reward plots | Peter | Next meeting | Open |

## Open Questions

- Does the refactored script pipeline fully preserve the notebook behavior?
- Which transition-kernel and reward assumptions still need explicit written documentation?
