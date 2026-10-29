EPCC-5 candidate-library groups. Selected from the launcher with `+candidates=<task>`
(Hydra appends the group; the base cell does not need to declare it).

  mode      select | refine | full_cand    — the paper's Full EPCC does NOT use this policy
  library   P (frozen DP chunk) | R (InvDyn on the DTG contour) | PR (both)
  rates     accumulate-k grid. k=1 = demo pace (ν≈c). k=3 exactly exhausts the
            16-waypoint contour over the 5-step horizon (5·k ≤ 15) and is the cap.
  n_dp      DDPM draws per replan — densifies family P at ~no extra cost.

Family P's backbone is the SAME frozen DP the DP and SuP baseline rows use, so the
{P, k=1} library reproduces the DP row (round-trip gate G2).
