# Completed validation runs

These results verify the implementation and report the completed synthetic, ablation,
theorem-validation, and real-data suites. The NCATS and O'Neil package objects remain small
illustrative subsets, but the four NCI-ALMANAC panel studies and four materials/catalyst
studies below are the full configured runs rather than pilots.

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

## Full theorem-validation battery

The preregistered 100-seed configuration completed without missing or duplicate endpoint
rows. It produced 48,000 noiseless endpoints (24,000 Sparse ECP), 500 approximate-sparsity
endpoints, 14,400 noisy endpoints (4,800 Sparse ECP bound checks), 6,000 proposal-complexity
checks, and 59,697 eligible filtering histories. The only absent numeric entries are the
intentional bound fields for the multiscale noisy method, for which this experiment does not
compute a finite-sample bound.

| Study | Successful checks | Empirical coverage |
|---|---:|---:|
| Noiseless Sparse ECP | 23,999 / 24,000 | 99.9958% |
| Noiseless, nontrivial local-bound rows only | 10,499 / 10,500 | 99.9905% |
| Approximate-sparsity oracle inequality | 500 / 500 | 100% |
| Noisy Sparse ECP | 4,800 / 4,800 | 100% |
| Proposal high-probability bound | 6,000 / 6,000 | 100% |

The lone miss was the complexity-adaptive proposal at `d=16`, `s=1`, seed 47, and budget
800: regret 0.2296 versus bound 0.2057. This is compatible with a 95% high-probability claim.
Coverage should not be oversold: 13,500 of the 24,000 noiseless ECP checks were still in the
global-diameter fallback regime. Among the 10,500 rows where the local rate bound was active,
only that one row missed.

The dimension control behaved as predicted qualitatively. At budget 800, known-support
median regret was nearly invariant to ambient dimension, whereas exact-unknown-support regret
increased with `d`. For example, with `s=2` it rose from 0.134 at `d=8` to 0.274 at `d=32`,
while the known-support medians were 0.00571 and 0.00522. Complexity-adaptive size weights
beat uniform size weights at the final budget for every tested sparsity.

The finite-budget slopes do not uniformly establish the asymptotic exponents. All 48
noiseless Sparse ECP slopes were negative, but exact-unknown-support median slopes for
`s=1,2,3,4` were -1.183, -0.290, -0.186, and -0.147 against references -1, -0.5, -0.333,
and -0.25. The noisy base method matched closely for unknown-support `s=1` (-0.321 versus
-0.333) and known-support `s=2` (-0.261 versus -0.25), but several higher-sparsity slopes
were shallower. Moreover, base noisy ECP and sparse random search were exactly identical at
all recorded endpoints for unknown-support `s=2,3,4`, and at more than 99.6% of endpoints for
known-support `s=3,4`. The confidence correction makes filtering effectively inactive in
those finite-budget regimes. These outcomes support conservative coverage and monotone
improvement, not a blanket empirical claim that every asymptotic exponent has been reached.

The filtering diagnostic is consistent with Theorem 9 but imbalanced. Across all histories,
the empirical next-query target-hit rate was 0.1122% versus a mean lower bound of 0.1006%.
A cluster bootstrap over `(d,s,seed)` put their difference at -0.0131 to 0.0386 percentage
points. Two acceptance-mass bins had empirical means below the lower-bound mean, but neither
deficit was resolved statistically: one contained only five histories, and the other had 317
histories with a cluster-bootstrap difference interval of -2.05 to 1.10 percentage points.
The plot now displays bin counts so these sparse bins cannot be mistaken for equally precise
evidence.

The approximate-sparsity bound became substantially tighter with budget while retaining full
coverage: at budget 2,000, median regret was 0.358, the median oracle bound was 0.489, and the
median bound/regret ratio was 1.37. Proposal complexity also had full coverage but was highly
conservative and heavy-tailed. At budget 800, the median observed count was 809 proposals
against a median high-probability bound of 419,108; the 99th percentile was 106,087 and the
maximum observed/bound ratio over all budgets was 0.162.

Because the primary coverage questions are answered, adding more omnibus seeds would be
wasteful. `configs/theory_rate_followup.yaml` instead extends the tractable regimes whose
rate estimates had not stabilized. It is a post-review diagnostic and must remain separate
from the preregistered full battery.

## Full real-data suite: completion and integrity

The server archive contains all ten configured real-data studies and 29,220 one-row-per-run
summaries. Every study has all six algorithms for every task/seed pair. An independent audit
found no duplicate keys, incomplete task/seed blocks, nonfinite primary metrics, out-of-range
regret/AUC/coverage values, or inconsistencies between proposal totals and reported proposal
overhead. Every aggregate table has six algorithm rows, and every paired table has all five
comparisons against Sparse ECP with the expected number of paired runs.

| Study | Tasks | Paired seeds | Effective budget | Run summaries |
|---|---:|---:|---:|---:|
| NCATS triple example | 1 | 10 | 300 | 60 |
| O'Neil two-block example | 2 | 10 | 25 | 120 |
| NCI-ALMANAC, d = 8 | 60 | 20 | 180 | 7,200 |
| NCI-ALMANAC, d = 12 | 60 | 20 | 250 | 7,200 |
| NCI-ALMANAC, d = 16 | 60 | 20 | 300 | 7,200 |
| NCI-ALMANAC, d = 20 | 60 | 20 | 300 | 7,200 |
| Matbench steels | 1 | 10 | 200 | 60 |
| Matbench experimental band gap | 1 | 10 | 300 | 60 |
| Matbench perovskites | 1 | 10 | 300 | 60 |
| CADS oxidative methane coupling | 1 | 10 | 300 | 60 |
| **Total** | **247** | — | — | **29,220** |

Across the ten study-level AUC summaries, Sparse ECP had the best mean rank (1.8), followed
by candidate-uniform ECP (2.6), maximin space filling (3.1), GP-UCB (3.2), candidate-uniform
random (4.9), and support-balanced random (5.4). This cross-study rank is descriptive: the
studies differ greatly in the number of tasks, candidate count, response scale, and budget.

## NCATS triple-combination example

The complete configured run used the 2,400-point, two-block package example, 300 hidden
response queries, ten paired seeds, and six algorithms.

| Algorithm | Final regret | Censored queries to 95% | 95% reach rate | Trajectory AUC | Proposals/query |
|---|---:|---:|---:|---:|---:|
| Sparse ECP | 0.588 | 20.8 | 100% | **0.9735** | 175.0 |
| Candidate-uniform ECP | 0.661 | **20.7** | 100% | 0.9727 | 174.7 |
| GP-UCB | **0.000** | 27.2 | 100% | 0.9622 | 2186.6 |
| Maximin space filling | 3.142 | 101.7 | 90% | 0.9446 | 1993.3 |
| Support-balanced random | 2.708 | 88.0 | 90% | 0.9356 | 1.0 |
| Candidate-uniform random | 3.640 | 130.0 | 100% | 0.9354 | 1.0 |

Sparse ECP and candidate-uniform ECP are expected to be close here: the example has only
two equal-sized supports.  It is an end-to-end higher-order dose-optimization test, not a
strong support-discovery benchmark.  Proposal counts measure candidates considered, not
wall-clock-equivalent operations across methods.

## Full NCI-ALMANAC panel study

The dose archive was converted into 20 independently sampled drug panels at each of
`d = 8, 12, 16, 20` for MCF7, A549/ATCC, and K-562. This produced 240 panel/cell-line tasks,
20 paired seeds per task, six algorithms, and 28,800 runs. The table reports mean normalized
trajectory AUC; larger is better.

| Panel dimension | Sparse ECP | Candidate-uniform ECP | Space filling | GP-UCB | Uniform random | Support random |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | **0.9804** | 0.9800 | 0.9778 | 0.9555 | 0.9321 | 0.9304 |
| 12 | **0.9676** | 0.9674 | 0.9623 | 0.9338 | 0.9053 | 0.9038 |
| 16 | 0.9631 | 0.9628 | **0.9686** | 0.9192 | 0.8951 | 0.8935 |
| 20 | 0.9391 | 0.9416 | **0.9465** | 0.8941 | 0.8782 | 0.8762 |

For inference across the panel study, each metric difference was first averaged over the 20
paired seeds within a task, then bootstrapped over the 240 panel/cell-line tasks. Positive
differences favor Sparse ECP: AUC is `Sparse ECP - competitor`; query and regret differences
are `competitor - Sparse ECP`.

| Competitor | AUC difference (95% CI) | Queries-to-95 difference (95% CI) | Final-regret difference (95% CI) |
|---|---:|---:|---:|
| Candidate-uniform ECP | -0.0004 [-0.0020, 0.0011] | -1.5 [-4.5, 1.3] | -0.317 [-0.626, -0.041] |
| Support-balanced random | **0.0616 [0.0567, 0.0667]** | **75.7 [69.5, 82.2]** | **4.188 [3.598, 4.839]** |
| Candidate-uniform random | **0.0599 [0.0549, 0.0651]** | **72.8 [66.6, 79.2]** | **3.579 [2.987, 4.229]** |
| Maximin space filling | -0.0012 [-0.0047, 0.0021] | 0.5 [-5.7, 6.6] | -0.289 [-0.916, 0.285] |
| GP-UCB | **0.0369 [0.0324, 0.0416]** | **23.4 [17.6, 29.0]** | **-1.876 [-2.411, -1.412]** |

Sparse ECP therefore has a clear anytime advantage over both random baselines and GP-UCB:
relative to GP-UCB it gains 0.0369 normalized AUC and reaches 95% about 23 queries earlier.
GP-UCB nevertheless has lower final regret, so the correct conclusion is faster early
optimization rather than endpoint dominance. Sparse ECP and candidate-uniform ECP are
indistinguishable in AUC and threshold time, and candidate-uniform ECP has slightly lower
final regret. The real panel study therefore does **not** demonstrate an additional benefit
from support-balanced proposals. Space filling crosses over with dimension: Sparse ECP has
the larger mean AUC at `d = 8, 12`, while space filling leads at `d = 16, 20`; their pooled
task-cluster AUC difference is unresolved.

## O'Neil package example

The released illustration contains two tasks with only 25 aggregated candidates each, so a
50-query configured budget exhausts each task after 25 effective queries and every algorithm
reaches zero final regret. Sparse ECP and ordinary ECP tie at AUC 0.9406. This validates
ingestion and replicate aggregation;
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
plot. Full ALMANAC configurations are provided for all four panel sizes.

Threshold-query aggregates include both reach rate and a budget-censored mean, where a
failure is assigned budget + 1.  Conditional-on-success means are retained in the CSV but
are not used alone in the tables above.

## Full materials and catalyst studies

Each full study used ten paired seeds and six algorithms. These are single-dataset
retrospective studies, so their paired intervals quantify seed variation but do not establish
generalization to new materials datasets.

| Dataset | Best AUC (method) | Sparse ECP AUC | Sparse final regret | Sparse queries to top 1% |
|---|---:|---:|---:|---:|
| Matbench steels | **0.9703 (Sparse ECP)** | **0.9703** | 4.600 | 12.5 |
| Matbench experimental band gap | **0.9946 (Sparse ECP)** | **0.9946** | 0.012 | 44.5 |
| Matbench perovskites | **0.9104 (GP-UCB)** | 0.8889 | 0.466 | 55.4 |
| CADS oxidative methane coupling | **0.8778 (GP-UCB)** | 0.7827 | 2.734 | 80.9 |

On steels, Sparse ECP significantly beats GP-UCB, uniform random, and support-balanced random
in paired AUC, but its AUC differences from candidate-uniform ECP and space filling are not
resolved. Space filling and GP-UCB attain zero mean final regret, illustrating again that AUC
and endpoint regret measure different behavior. On experimental band gap, Sparse ECP has the
largest mean AUC, but only its small AUC advantage over space filling is resolved; space
filling reaches the 95% threshold faster while Sparse ECP reaches a top-1% candidate much
earlier. Sparse ECP and candidate-uniform ECP are exactly identical on perovskites because
every candidate has its own support in this encoding. On OCM, GP-UCB significantly beats
Sparse ECP in AUC, final regret, and time to 95%; Sparse ECP significantly beats space filling
in AUC and top-1% discovery, while comparisons with the remaining methods are unresolved.
