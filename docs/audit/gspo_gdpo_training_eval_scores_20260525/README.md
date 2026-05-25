# GSPO/GDPO Training-Time Eval Score Record 2026-05-25

Scope: existing local training logs plus training-time validation/generation outputs only. This record intentionally excludes LVB, OVO, and StreamingBench benchmark eval results.

No new training run, restart, or model eval was launched for this export.

## Files
- `training_raw_metrics_wide.csv`: observed per-step train metrics parsed from every available run log.
- `training_preferred_chain_wide.csv`: preferred restart-aware train curve with GSPO step70 estimated.
- `training_preferred_chain_long.csv`: long-format plotting table for reward/outcome/recall/memory metrics.
- `training_time_validation_eval_chain_wide.csv`: training-time validation/eval checkpoint curve.
- `training_time_validation_eval_chain_long.csv`: long-format validation/eval plotting table.
- `estimated_metric_values.csv`: every explicitly estimated value and the method used.
- `plot_metric_plan.csv`: recommended 4-5 plot groupings and metric choices.
- `validation_by_category_observed.csv`, `validation_by_task_observed.csv`: observed validation breakdowns.

## Restart-Aware Chains
- `gspo_preferred`: estimated train steps 1-10, `gspo_main` steps 11-60, then `gspo_kl_from60` steps 61-69, plus estimated step70.
- `gdpo_preferred`: estimated train steps 1-10, `gdpo_main` steps 11-50, then `gdpo_kl_from50` steps 51-80.

## Estimation Policy
- Missing per-metric values inside an observed validation row are filled from the nearest observed checkpoint in the same run.
- Missing training steps 1-10 are backfilled from the median of the first observed training steps 11-15.
- Missing GSPO training step70 is estimated as the median of GSPO preferred training steps 65-69.
- Missing GSPO validation/eval step70 is estimated by half-step linear extrapolation from validation steps 50 and 60.
- All estimated rows are marked `estimated`; mixed observed rows are marked `mixed_observed_estimated`.

## Key Values
- Observed train rows exported: 148.
- Preferred chain train rows exported: 150, including 21 estimated rows.
- GSPO estimated validation/eval step70 accuracy: 80.78%; score: 1.447.
- GDPO preferred validation/eval step80 accuracy: 77.11%; score: 1.313.
- GSPO train score first/last observed in preferred chain: 1.360 -> 1.250.
- GDPO train score first/last observed in preferred chain: 1.354 -> 1.301.
- GSPO compact memory quality first/last observed: 0.269 -> 0.705.
- GDPO compact memory quality first/last observed: 0.302 -> 0.380.

## Plot Recommendations
1. Training Reward / Outcome: plot `train/thinkstream/reward/score/mean` and `train/thinkstream/reward/outcome/mean` with `roll5`; conclusion should say no stable monotonic rise.
2. Recall Usage and Request Hit: plot `call_traj_frac`, `recall_call_count/mean`, `recall_support_request_hit_rate/mean`.
3. Recall Effectiveness: plot `answer_used_per_labeled/rate`, `answer_success_per_labeled/rate`, and optionally `post_recall_outcome_mean/mean`.
4. Compact Memory Quality: plot `compress_quality/mean`, `cover_ok/mean`, `boundary_score/mean`, `source_precision/mean`.
5. Training-Time Eval Accuracy: plot `trajectory_mean_correct_question_weighted` from `training_time_validation_eval_chain_long.csv`.
