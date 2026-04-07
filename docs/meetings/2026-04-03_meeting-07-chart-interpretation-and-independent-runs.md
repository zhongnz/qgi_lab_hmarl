# Meeting 07: Chart Interpretation and Independent Model Runs

## Meeting

- Title: Meeting 7 - Chart Interpretation and Independent Model Runs
- Date: 2026-04-03
- Time: 4:30 PM
- Location/Call: QGI Lab (in person)
- Note taker: Peter

## Attendees

- QGI Lab members

## Agenda

1. Share progress updates from each team member
2. Review the HMARL comparison chart and related figures
3. Clarify what the current benchmark numbers mean
4. Agree on the next experimental runs

## Discussion Notes

- Each member presented their recent work and current progress.
- For the HMARL project, the team reviewed the comparison figures, especially the `continuous + ground truth` versus `single mission + ground truth` benchmark chart.
- The team noted that these figures need a clearer explanation so that the audience understands what each metric means and how the two settings differ conceptually.
- In particular, the benchmark should be explained as a comparison between:
  - a continuous scheduling environment that measures rolling operational performance, and
  - a simplified single-mission environment that is mainly intended as a validation benchmark.
- The team agreed that the chart and associated figures should be interpreted carefully, since some metrics improve in the simplified benchmark because the task is shorter and easier, not necessarily because the policy is better in the operational sense.
- It was also agreed that separate model runs are needed so that Mehdi and Peter can independently run and compare the model from their own sides.

## Decisions Made

- Add a clear explanation of the current HMARL benchmark chart and related figures.
- Treat the `continuous + ground truth` and `single mission + ground truth` comparison as an interpretation task as well as a numerical comparison.
- Have Mehdi and Peter run separate instances of the model independently.

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Prepare a clear explanation of the benchmark chart and related HMARL figures | Peter | Next meeting | Open |
| Run an independent model experiment and record results | El Mehdi Nezahi | Next meeting | Open |
| Run an independent model experiment and record results | Peter | Next meeting | Open |

## Open Questions

- What is the clearest way to explain the benchmark figures so that the audience understands the difference between validation success and operational performance?
- How similar or different will Mehdi's and Peter's independent runs be under the current setup?
