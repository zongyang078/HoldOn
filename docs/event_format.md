# Shared contact-event format

All tracks emit contact events in this format and are scored by `holdon.evaluation`.
Changing it requires agreement from the whole team.

| Field | Type | Meaning |
| --- | --- | --- |
| `video_id` | str | Video identifier, e.g. participant + route |
| `limb` | str | One of `left_hand`, `right_hand`, `left_foot`, `right_foot` |
| `hold_id` | str | Hold identifier as given by the route annotation |
| `start_frame` | int | First contact frame (inclusive) |
| `end_frame` | int | Last contact frame (inclusive) |
| `score` | float | Confidence used to order greedy matching; ground truth uses 1.0 |

Events are stored as CSV with exactly these columns, one row per event.
Defined in code by `holdon.events.ContactEvent`.
