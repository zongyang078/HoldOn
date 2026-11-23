# Evaluation — rtmw-x-384_2026-11-18_wrist_ankle/events_lopo.csv

Macro average over 22 videos; 949 predicted and 939 ground-truth events (31 wall pushes `w`, never predicted, count as FN). Protocol: docs/evaluation_protocol.md.

## All limbs, hands, feet

| Setting | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| all | 0.757 / 0.755 / **0.752** | 0.737 / 0.734 / **0.731** | 0.654 / 0.649 / **0.648** | 0.820 |
| hands | 0.856 / 0.848 / **0.847** | 0.835 / 0.826 / **0.825** | 0.762 / 0.745 / **0.749** | 0.835 |
| feet | 0.656 / 0.662 / **0.652** | 0.638 / 0.644 / **0.634** | 0.550 / 0.554 / **0.546** | 0.800 |

## Route (macro over the route's videos)

| Route | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| green (11 videos) | 0.812 / 0.801 / **0.803** | 0.799 / 0.789 / **0.791** | 0.713 / 0.703 / **0.706** | 0.834 |
| orange (11 videos) | 0.701 / 0.709 / **0.700** | 0.674 / 0.679 / **0.672** | 0.595 / 0.595 / **0.590** | 0.803 |

## Occluded vs visible contact frames (frame recall, pooled over all events)

| Limbs | occluded frames | recall occluded | visible frames | recall visible |
| --- | --- | --- | --- | --- |
| all | 35303 | 0.833 | 89491 | 0.753 |
| hands | 34313 | 0.833 | 33015 | 0.864 |
| feet | 990 | 0.859 | 56476 | 0.688 |

## Climber experience (group averages only; macro over the group's videos)

| Climbing for | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| less than 1 year + 1-2 years (6 participants, 12 videos) | 0.730 / 0.745 / **0.734** | 0.706 / 0.719 / **0.709** | 0.637 / 0.646 / **0.638** | 0.824 |
| 5-10 years (2 participants, 4 videos) | 0.826 / 0.767 / **0.793** | 0.820 / 0.762 / **0.787** | 0.727 / 0.674 / **0.697** | 0.834 |
| more than 10 years (2 participants, 6 videos) | 0.764 / 0.766 / **0.760** | 0.743 / 0.746 / **0.739** | 0.640 / 0.638 / **0.634** | 0.801 |
