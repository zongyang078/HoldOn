# Shared contact-event format

All tracks emit contact events in this format and are scored by the same evaluation
([evaluation_protocol.md](evaluation_protocol.md)). Code: `holdon.events.ContactEvent`.

| Field | Type | Meaning |
| --- | --- | --- |
| `video_id` | str | `<recording>_<route>`, e.g. `p2b_orange` |
| `limb` | str | `left_hand`, `right_hand`, `left_foot` or `right_foot` |
| `hold_id` | str | index of the hold in `{route}_routeAnnotation.json` as a string (`"0"`, `"12"`); `"w"` only in ground truth |
| `start_frame` | int | first contact frame, 0-based, inclusive |
| `end_frame` | int | last contact frame, inclusive (`end_frame >= start_frame`) |
| `score` | float | confidence; orders predictions in greedy matching (higher first); ground truth uses 1.0 |

Stored as CSV with exactly these columns in this order, one row per event, header included:

```
video_id,limb,hold_id,start_frame,end_frame,score
p1_green,left_hand,2,68,113,0.94
```

Write and read only with `holdon.events.write_events` / `read_events`. Where runs are stored:
[interfaces.md](interfaces.md) section 4.
