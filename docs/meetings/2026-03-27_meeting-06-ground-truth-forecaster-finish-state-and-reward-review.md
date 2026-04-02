# Meeting 06: Ground-Truth Forecaster, Finish State, and Reward Review

## Meeting

- Title: Meeting 6 - Ground-Truth Forecaster, Finish State, and Reward Review
- Date: 2026-03-27
- Time: 2:00 PM
- Location/Call: QGI Lab (in person)
- Note taker: Peter

## Attendees

- Professor Amine Aboussalah
- Idriss Malek
- Abdessalam Ed-dib
- El Mehdi Nezahi
- Peter Zhong

## Agenda

1. Decide how to simplify the forecaster setup for the next validation phase
2. Discuss using a finish-state formulation instead of continuous running
3. Review and explain the reward-function changes that improved behavior
4. Convert the presentation slides from Figma to Google Slides

## Discussion Notes

- The team agreed that, for the immediate validation stage, the forecaster should use ground-truth information rather than an estimated forecast.
- The reason for this simplification is to fine-tune and verify that the core functions of the model are working before reintroducing a harder forecasting component.
- A finish-state formulation was proposed as an alternative to the current continuously running episode structure.
- One concrete example discussed was to limit each vessel to one trip per episode so that reaching the destination port can be treated as a clear success outcome.
- The team also requested a clearer explanation of the reward-function changes that led to improved model behavior, so that the effect of those changes is technically understandable and easier to present.
- The team also noted that the presentation slides should be converted from Figma into Google Slides for easier presentation and sharing.

## Decisions Made

- Use ground-truth forecaster inputs for now as a validation-focused setup.
- Investigate a finish-state episode design rather than relying only on continuous running.
- Explore a one-trip-per-episode version where each vessel has at most one mission.
- Prepare a clear explanation of the reward-function changes and why they improved training behavior.
- Convert the current presentation deck from Figma into Google Slides.

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Configure and test a ground-truth forecaster setup for model validation | Peter | Next meeting | Open |
| Investigate a finish-state formulation with one trip per episode per vessel | Peter | Next meeting | Open |
| Summarize the reward-function changes that improved learning and explain their impact | Peter | Next meeting | Open |
| Convert the current presentation slides from Figma to Google Slides | Peter | Next meeting | Open |

## Open Questions

- What is the cleanest way to define "ground truth" for the forecaster in the current codebase?
- Does a finish-state / one-trip-per-episode setup improve convergence compared with the current ongoing scheduling formulation?
- Which reward-function changes had the largest impact on making the learning signal more informative?
