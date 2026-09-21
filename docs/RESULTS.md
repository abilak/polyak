# Completed validation runs

These results verify the implementation and provide preliminary evidence only.  The NCATS
and O'Neil package objects are small illustrative subsets, and the ALMANAC pilots use too
few panels/seeds for scientific claims.  The preregistered full configurations are separate.

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
