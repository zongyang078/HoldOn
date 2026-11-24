# Track 2 candidate coverage

Feature table: 394759 candidate rows, 113269 labelled as contact.

Ground-truth contact frames (limb on a hold; wall pushes `w` excluded). A frame whose hold is not a candidate cannot be predicted by Track 2.

| Limbs | GT frames | in candidates | lost: no pose | lost: hold not a candidate |
| --- | --- | --- | --- | --- |
| all | 123446 | 0.918 | 0.063 | 0.020 |
| hands | 67108 | 0.907 | 0.064 | 0.029 |
| feet | 56338 | 0.930 | 0.061 | 0.010 |
