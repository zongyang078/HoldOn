# Evaluation — rtmw-x-384_2026-11-18_fingertip_toe/events_lopo.csv

Macro average over 22 videos; 904 predicted and 939 ground-truth events (31 wall pushes `w`, never predicted, count as FN). Protocol: docs/evaluation_protocol.md.

## All limbs, hands, feet

| Setting | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| all | 0.890 / 0.848 / **0.864** | 0.869 / 0.827 / **0.844** | 0.814 / 0.773 / **0.790** | 0.836 |
| hands | 0.834 / 0.832 / **0.828** | 0.799 / 0.796 / **0.792** | 0.726 / 0.721 / **0.719** | 0.835 |
| feet | 0.959 / 0.868 / **0.908** | 0.956 / 0.865 / **0.905** | 0.923 / 0.835 / **0.874** | 0.837 |

## Route (macro over the route's videos)

| Route | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| green (11 videos) | 0.928 / 0.851 / **0.884** | 0.910 / 0.835 / **0.867** | 0.855 / 0.783 / **0.814** | 0.849 |
| orange (11 videos) | 0.852 / 0.845 / **0.845** | 0.829 / 0.820 / **0.821** | 0.774 / 0.763 / **0.765** | 0.821 |

## Occluded vs visible contact frames (frame recall, pooled over all events)

| Limbs | occluded frames | recall occluded | visible frames | recall visible |
| --- | --- | --- | --- | --- |
| all | 35303 | 0.781 | 89491 | 0.904 |
| hands | 34313 | 0.779 | 33015 | 0.869 |
| feet | 990 | 0.845 | 56476 | 0.925 |

## Climber experience (group averages only; macro over the group's videos)

| Climbing for | P / R / F1 @>0 | P / R / F1 @≥0.3 | P / R / F1 @≥0.5 | mean tIoU of matches @≥0.5 |
| --- | --- | --- | --- | --- |
| less than 1 year + 1-2 years (6 participants, 12 videos) | 0.874 / 0.861 / **0.864** | 0.846 / 0.834 / **0.837** | 0.783 / 0.769 / **0.773** | 0.838 |
| 5-10 years (2 participants, 4 videos) | 0.944 / 0.839 / **0.885** | 0.937 / 0.834 / **0.879** | 0.902 / 0.801 / **0.845** | 0.843 |
| more than 10 years (2 participants, 6 videos) | 0.887 / 0.826 / **0.851** | 0.871 / 0.810 / **0.835** | 0.819 / 0.762 / **0.785** | 0.827 |
