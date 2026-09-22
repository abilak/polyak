# Completed validation runs

These results verify the implementation and provide preliminary evidence only.  The NCATS
and O'Neil package objects are small illustrative subsets, and the ALMANAC pilots use too
few panels/seeds for scientific claims.  The preregistered full configurations are separate.

## Main synthetic benchmark

The full main benchmark completed all 24,000 preregistered runs: 30 sparse problems formed by
five objectives, three ambient dimensions, and two sparsity levels; eight algorithms; and 100
paired seeds. There were no duplicate, missing, nonfinite, or truncated run summaries.

| Algorithm | Mean rank over 30 tasks | Task wins | Median final regret | Mean proposals/query |
|---|---:|---:|---:|---:|
| Sparse ECP | **1.67** | **15** | **0.0818** | 2,054 |
| GP-UCB | 2.90 | 13 | 0.1889 | 1,933 |
| Hard-threshold ECP | 2.97 | 2 | 0.1737 | 2,145 |
| Support-balanced random | 4.00 | 0 | 0.2849 | **1** |
| Maximin space filling | 4.10 | 0 | 0.2390 | 1,021 |
| Hard-threshold random | 5.37 | 0 | 0.4833 | **1** |
| Dense ECP | 7.27 | 0 | 2.1745 | 1,714 |
| Dense random | 7.73 | 0 | 2.3183 | **1** |

Paired same-seed comparisons support three conclusions. Sparse ECP beat both dense methods on
all 30 tasks with 95% bootstrap intervals excluding zero in every task, showing that exploiting
sparsity is essential. It beat support-balanced random search on 28 of 30 tasks, significantly
on 26, showing a benefit from ECP filtering beyond the support-first proposal alone. It beat the
hard-threshold ECP heuristic on 27 tasks, significantly on 20; the heuristic significantly won
one task. The comparison with GP-UCB was mixed: Sparse ECP won 17 tasks and GP-UCB won 13, with
ten significant wins for each and ten inconclusive comparisons.

The GP-UCB comparison depends strongly on intrinsic sparsity. Sparse ECP won 12 of the 15
`s=1` tasks but only three of the 15 `s=2` tasks. This is an important qualification: the
benchmark supports Sparse ECP as the best overall method in this collection, not uniform
dominance over GP-UCB.

The fitted Sparse ECP budget slope was negative in all 30 settings. For `s=1`, the median slope
was -2.31 against the asymptotic minimax exponent -1; for `s=2`, it was -0.375 against -0.5.
These finite-budget objective-dependent slopes are broadly consistent with decreasing regret,
but the main benchmark alone should not be presented as a precise validation of the minimax
exponent. The dedicated theorem battery is the appropriate source for that claim.

## ECP hyperparameter ablation

The full ablation completed all 19,200 runs: 16 problems, 12 parameter/method conditions, and
100 paired seeds. The most conservative tolerance growth, `tau=1.001`, ranked first on every
problem and was significantly better than the `epsilon_1=0.01`, `tau=1.01`, `C=1000` baseline
on all 16. Its computational cost was also extreme: a mean 22,621 proposals per accepted
evaluation, versus 2,100 for the baseline.

| Setting | Mean rank | Task wins | Median final regret | Mean proposals/query |
|---|---:|---:|---:|---:|
| `tau=1.001` | **1.00** | **16** | **0.2339** | 22,621 |
| `C=5000` | 3.69 | 0 | 0.4928 | 10,412 |
| `epsilon_1=0.01` (baseline) | 4.25 | 0 | 0.5394 | 2,100 |
| `epsilon_1=0.1` | 4.50 | 0 | 0.5605 | 1,327 |
| `epsilon_1=0.001` | 4.50 | 0 | 0.5470 | 2,871 |
| `epsilon_1=1.0` | 6.09 | 0 | 0.5663 | 578 |
| `C=100` | 6.31 | 0 | 0.6534 | 215 |
| `C=10` | 7.56 | 0 | 0.9791 | 23 |
| `tau=1.05` | 9.12 | 0 | 1.0946 | 392 |
| `C=1` | 9.62 | 0 | 1.0883 | 3 |
| `tau=1.1` | 9.81 | 0 | 1.1252 | 191 |
| Sparse random | 11.53 | 0 | 1.1805 | **1** |

The initial tolerance had a comparatively modest effect on regret but a large effect on cheap
proposal cost. Increasing `epsilon_1` from 0.01 to 1.0 reduced mean proposals per query from
2,100 to 578. Faster tolerance growth (`tau=1.05` or `1.1`) and short rejection patience
(`C=1` or `10`) reduced proposals further but materially worsened regret. The ablation therefore
exposes a genuine statistical-computational frontier rather than a universally best parameter.

## Theorem-suite smoke check

The small configuration in `configs/theory_validation_smoke.yaml` completed end to end. It
covered 32 noiseless finite-bound endpoints, four approximate-sparsity endpoints, eight noisy
ECP endpoints, and eight proposal-complexity endpoints; every observed value was within its
corresponding conservative high-probability bound. The filtering diagnostic retained 22
eligible histories and its mean empirical next-query target-hit rate exceeded the mean
Theorem 9 lower bound in the reported acceptance-mass bin.

This is an engineering validation with two seeds and two budgets. Its fitted slopes are noisy
and are not scientific evidence for the asymptotic exponents. The 100-seed preregistered
configuration is `configs/theory_validation.yaml` and remains distinct from this smoke run.

## NCATS triple-combination example

The complete configured run used the 2,400-point, two-block package example, 300 hidden
response queries, ten paired seeds, and six algorithms.

| Algorithm | Final regret | Censored queries to 95% | 95% reach rate | Trajectory AUC | Proposals/query |
|---|---:|---:|---:|---:|---:|
| Sparse ECP | 0.588 | 20.8 | 100% | **0.9735** | 175.0 |
| Candidate-uniform ECP | 0.661 | **20.7** | 100% | 0.9727 | 174.7 |
| GP-UCB | **0.000** | 27.2 | 100% | 0.9622 | 2186.6 |
| Maximin space filling | 2.814 | 101.9 | 90% | 0.9455 | 1993.3 |
| Support-balanced random | 2.708 | 88.0 | 90% | 0.9356 | 1.0 |
| Candidate-uniform random | 3.640 | 130.0 | 100% | 0.9354 | 1.0 |

Sparse ECP and candidate-uniform ECP are expected to be close here: the example has only
two equal-sized supports.  It is an end-to-end higher-order dose-optimization test, not a
strong support-discovery benchmark.  Proposal counts measure candidates considered, not
wall-clock-equivalent operations across methods.

## NCI-ALMANAC pilots

The full dose archive was converted into 20 panels at each of d = 8, 12, 16, and 20 for
MCF7, A549/ATCC, and K-562, producing 240 panel/cell-line tasks.  The initial d = 8 pilot
used three tasks, two paired seeds, and 50 queries (six paired runs per method).

| Algorithm | Final regret | Censored queries to 95% | 95% reach rate | Trajectory AUC |
|---|---:|---:|---:|---:|
| Sparse ECP | **2.580** | **16.3** | **83%** | **0.9414** |
| Candidate-uniform ECP | 2.700 | 22.5 | 67% | 0.9253 |
| Maximin space filling | 9.232 | 27.3 | 50% | 0.9183 |
| Candidate-uniform random | 9.856 | 41.8 | 33% | 0.8426 |
| Support-balanced random | 17.253 | 37.0 | 33% | 0.8239 |
| GP-UCB | 23.070 | 36.3 | 50% | 0.7880 |

This pilot motivated no post-hoc algorithm change.  It only confirms that the full study is
worth running.  Confidence intervals remain wide with six runs.

The robust pilot took the minimum response across the three cell lines for the same d = 8
panel.  Exact grid intersection left 252 candidates on all three cell lines.  Sparse ECP
had the best AUC (0.9575), narrowly ahead of maximin space filling (0.9525).  With only two
seeds this remains a pipeline check, not evidence of a robust-objective win.

## O'Neil package example

The released illustration contains two tasks with only 25 aggregated candidates each, so a
50-query budget exhausts each task and every algorithm reaches zero final regret.  Sparse
ECP and ordinary ECP tie at AUC 0.9406.  This validates ingestion and replicate aggregation;
it is not the full 583-pair replication study.

## NCATS pair-score support-discovery control

The recovered paper-linked table has 1,162 pair scores and no dose surfaces.  With one
candidate per support, support-first and candidate-uniform sampling become identical.  The
300-query, ten-seed run confirms that: Sparse ECP exactly matches ordinary ECP (AUC 0.4718),
and support-balanced random exactly matches candidate-uniform random (AUC 0.4771).  Sparse
ECP reached the 95% response threshold in only 10% of runs versus 30% for random.  This is
the expected support-information barrier, not evidence against within-support dose search.

## Reproducibility artifacts

Every result directory contains per-query trajectories, one-row-per-run summaries,
bootstrap aggregates, paired same-seed comparisons against Sparse ECP, and a PNG trajectory
plot.  Full ALMANAC configurations are provided for all four panel sizes; they are not
confused with the small pilot reported above.

Threshold-query aggregates include both reach rate and a budget-censored mean, where a
failure is assigned budget + 1.  Conditional-on-success means are retained in the CSV but
are not used alone in the tables above.

## Materials and catalyst pipeline checks

Two deliberately tiny, two-seed runs verify the added pipeline; neither is a scientific
result. On the 30-query steel smoke run, Sparse ECP had trajectory AUC 0.9204 and reached a
top-1% material after a censored mean of 3.0 queries. Candidate-uniform ECP was essentially
tied at AUC 0.9220. This table has only 12 distinct elemental supports, so a large separation
between those variants was not expected.

On the 50-query OCM pilot, Sparse ECP had the best trajectory AUC (0.6911) and reached the
measured top 1% after a censored mean of 22.5 queries; GP-UCB was second by AUC (0.6483).
No method reached 95% of the measured response range in either seed. The result is promising
enough to justify the paired ten-seed run, but two seeds are far too few for a claim of
superiority. Full configurations are provided and remain unrun while the larger ALMANAC job
is active.
