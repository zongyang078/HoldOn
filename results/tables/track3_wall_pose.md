# Track 3 wall pose (bolt holes)

Made by `scripts/wall_pose.py`. Pinhole camera, principal point at the image centre (720×1280),
one focal length for all 22 videos fitted to the bolt holes (`holdon.preprocessing.homography`).

- **Shared focal length: 590 px** (WHAM default sqrt(w² + h²) = 1468.6 px).
  Summed squared bolt-hole error: 29.7 px² at 590 px vs
  2323.7 px² at the WHAM default.
- Bolt-hole RMS error with the shared focal: median 0.86 px, max 2.45 px.
- 20 of 22 files list the two top bolt holes in swapped order
  (fixed by `fix_bolt_hole_order`; the stored `homography_mat` of these files is wrong).
- Hold boxes mapped to the wall: median 10.1 × 9.4 cm,
  90th percentile 18.5 × 18.9 cm
  (plausible hold sizes, so wall units are centimetres).

| Video | Bolt holes | Swapped | RMS px (shared f) | RMS px (WHAM f) | Camera in wall coords x, y, distance (m) |
| --- | --- | --- | --- | --- | --- |
| p1_green | 4 | no | 1.17 | 9.67 | 0.49, 0.89, 2.63 |
| p1_orange | 4 | yes | 0.79 | 9.07 | 0.46, 0.82, 2.49 |
| p2a_green | 350 | no | 2.45 | 7.09 | 0.29, 0.82, 2.79 |
| p2a_orange | 4 | yes | 1.69 | 8.27 | 0.35, 0.83, 2.82 |
| p2b_green | 4 | yes | 0.68 | 8.69 | 0.37, 0.78, 2.81 |
| p2b_orange | 4 | yes | 0.59 | 10.42 | 0.42, 0.87, 2.57 |
| p3_green | 4 | yes | 0.64 | 10.16 | 0.59, 0.83, 2.57 |
| p3_orange | 4 | yes | 2.02 | 10.53 | 0.54, 0.84, 2.57 |
| p4_green | 4 | yes | 0.28 | 8.75 | 0.42, 0.83, 2.46 |
| p4_orange | 4 | yes | 1.11 | 10.33 | 0.45, 0.85, 2.55 |
| p5_green | 4 | yes | 0.89 | 9.52 | 0.36, 0.85, 2.70 |
| p5_orange | 4 | yes | 0.39 | 11.52 | 0.41, 0.88, 2.60 |
| p6_green | 4 | yes | 0.83 | 12.60 | 0.32, 0.85, 2.63 |
| p6_orange | 4 | yes | 0.52 | 9.18 | 0.34, 0.85, 2.62 |
| p7_green | 4 | yes | 0.42 | 11.37 | 0.24, 0.85, 2.51 |
| p7_orange | 4 | yes | 1.63 | 13.10 | 0.33, 0.87, 2.51 |
| p8_green | 4 | yes | 1.81 | 12.95 | 0.28, 0.85, 2.56 |
| p8_orange | 4 | yes | 0.74 | 10.87 | 0.18, 0.86, 2.55 |
| p9_green | 4 | yes | 0.89 | 8.82 | 0.23, 0.87, 2.63 |
| p9_orange | 4 | yes | 0.55 | 9.93 | 0.37, 0.90, 2.72 |
| p10_green | 4 | yes | 0.90 | 10.09 | 0.34, 0.86, 2.63 |
| p10_orange | 4 | yes | 1.33 | 10.52 | 0.27, 0.86, 2.60 |
