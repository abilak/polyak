# Experiments

## Experimental questions

We evaluate Sparse ECP along four axes. First, we test whether its empirical simple-regret
scaling reflects the intrinsic sparsity predicted by the theory and whether unknown support
produces the separate combinatorial cost. Second, we isolate the contribution of ECP's
acceptance filter from that of the sparse proposal distribution. Third, we test adaptation to
unknown and approximate sparsity, noisy observations, and the candidate-generation bound.
Finally, we evaluate whether support-first search improves query efficiency on sparse
scientific candidate spaces.

The synthetic studies are controlled diagnostics with known optima. The real-data studies are
retrospective reveal-only-selected-row simulations. We do not interpret performance on a
finite historical table as a proof of a continuous theorem or as prospective experimental
validation.

## Algorithms

Our primary method is Sparse ECP with direct support-first proposals. We compare it with Sparse
Random Search (the identical proposal distribution without filtering), dense ECP and dense
random search, maximin space filling, Gaussian-process UCB, and dense-draw/hard-threshold ECP
with its random-search control. The hard-threshold method is included only as an ablation: its
induced slice distribution is not the proposal measure analyzed in the theory.

When the exact sparsity is unknown, we compare uniform size weights with the scale-free prior

    w_r = (s+1) / (s r(r+1)),  r=1,...,s.

Known-support runs place all proposal mass on the true slice. They are an oracle control, not a
deployable competitor. Finite scientific tables also include candidate-uniform ECP and
candidate-uniform random search because unequal numbers of rows per support can otherwise
confound support-first and candidate-first sampling.

All methods use the same expensive-query budget, task definition, hidden response table, and
paired random seed. We hold each method's hyperparameters fixed across tasks within a study.
Rejected ECP proposals do not consume the expensive-query budget, but we report them as a
separate computational-cost endpoint.

## Synthetic landscapes and protocol

We embed a hidden s-sparse maximizer in ambient dimensions d in {8,16,32}. The benchmark suite
contains translated cone, quadratic, Rastrigin, Ackley, Griewank, Levy, and Salomon landscapes.
For each landscape we test s in {1,2,3,4}, known and unknown support, budgets 25, 50, 100, and
300, and 100 paired repetitions. This gives 168 sparse problem settings before budgets and
seeds. The cone is the primary Lipschitz-theory diagnostic; the other landscapes test
multimodality and model misspecification rather than the minimax theorem.

The primary paper table uses a preregistered 30-problem subset: five landscapes, three ambient
dimensions, and two sparsities, each with 100 repetitions. This matches the problem/repetition
scale of the original ECP comparison. The 168-setting factorial is an extended robustness
study and is reported separately so its size does not dominate the main presentation.

We report median simple-regret trajectories with interquartile bands, endpoint means with
bootstrap confidence intervals, algorithm ranks across tasks, and per-(d,s,objective) paired
differences. We estimate log-regret/log-budget slopes only when at least two positive median
regrets are available. The line -1/s is shown as a theoretical reference, not fitted to the
data.

## Noiseless theorem diagnostics

### Intrinsic rate and support discovery

On centered sparse cubes, a hidden Lipschitz cone supplies an exact optimum and Lipschitz
constant. We compare known support, unknown exact support, uniform adaptation up to an upper
sparsity, and the scale-free complexity prior. For every endpoint we compute the explicit
finite-budget upper bound using the guaranteed post-burn-in design count. We report empirical
coverage but do not treat coverage as proof; the bound is intentionally conservative.

The rate prediction is n^{-1/s}. The support-discovery comparison fixes s and n while varying
d, and relates regret to binomial(d,s)^{1/s}. Known-support runs remove this factor and isolate
the within-slice geometry.

### Filtering gain

To distinguish Sparse ECP from Sparse Random Search, we fix a target value below the optimum.
At selected post-burn-in histories before the target is reached, held-out cheap proposals
estimate the acceptance mass q_t and target mass p_y. We compare the observed probability that
the next expensive ECP evaluation reaches the target with the conditional lower bound

    p_y Psi_C(q_t),  Psi_C(q) = [1-(1-q)^C]/q.

Diagnostic proposals never enter the optimization history. We preregister their number, the
target, the history stride, and acceptance-mass bins. We additionally compare target-hitting
times and proposals per accepted evaluation against Sparse Random Search.

We repeat the analysis under one-factor-at-a-time ablations of the initial tolerance
epsilon_1, geometric growth factor tau, and rejection patience C. The C=1 condition is the
near-unfiltered control; progressively larger C values expose the tradeoff between expensive
evaluations and cheap proposals predicted by Theorems 9 and 11.

### Approximate sparsity

We use a dense maximizer whose coordinates decay polynomially. For every r, the best r-sparse
comparator is known by truncation, giving the exact approximation error a_r. Sparse ECP uses one
fixed complexity prior over r=1,...,s. At each budget we report the empirical regret, every a_r,
the best admissible simultaneous oracle bound, and its coverage. No sparsity level is chosen
after observing optimization outcomes.

### Proposal complexity

For exact-size proposals on the centered cube, we record the total number H_n of candidates
needed for n accepted evaluations. We compare H_n with the expected and high-probability bounds
from Theorem 25 and report proposals per accepted evaluation. We report wall time only as a
secondary machine-dependent quantity.

## Noisy observations

At every accepted design x, the noisy oracle returns b independent Gaussian observations with
standard deviation sigma. The optimizer stores the empirical mean, uses the confidence-adjusted
acceptance rule, and recommends the sampled design with the largest empirical mean. Performance
is recommendation regret evaluated against the latent objective, not the best latent value
among sampled points.

We allocate a total of N noisy calls and use the rate-level oracle balance between distinct
designs and replications. Known-support runs isolate the predicted N^{-1/(s+2)} exponent; the
unknown-support runs test the accompanying support factor. We additionally implement the
dyadic multiscale construction of Theorem 22: half of the budget is divided across replication
levels and the remaining half validates the returned recommendations. We report distinct
designs, replications, calls actually used, recommendation regret, finite-bound coverage, and
log-log slopes. Sparse Random Search with the same oracle-balanced replication allocation is
the primary control.

## Sparse scientific data

### Drug combinations

The NCATS and O'Neil SynergyFinder objects provide end-to-end ingestion checks. The main cancer
study uses NCI-ALMANAC dose-level data in repeated drug panels with d in {8,12,16,20}. A support
is a drug pair or higher-order combination; within-support coordinates are normalized positive
log doses, with zero reserved for absence. Each cell line is a separate task. A robust secondary
objective takes the minimum response across selected cell lines only after exact candidate-grid
intersection; missing responses are never imputed.

### Materials, catalysts, and process conditions

The primary materials application is oxidative coupling of methane. Each candidate jointly
encodes a sparse catalyst formulation/support and four process variables; repeated conditions
within a catalyst create a genuine support-selection plus within-support optimization problem.
We also evaluate experimental steel strength, target experimental band gap, and perovskite
formation energy from Matbench. The latter tables are useful support-discovery controls, but
tables with one row per support cannot validate within-support refinement.

## Real-data endpoints and uncertainty

For every retrospective task we report simple regret against the best measured row, queries to
90%, 95%, and 99% of the observed response range, queries to the measured top 1%, trajectory
area, unique-support coverage, and proposal overhead. Threshold failures are assigned budget+1
for the primary censored query count, and reach rates are always shown. Conditional-on-success
means are secondary and never presented alone.

Trajectories show the median and interquartile range. Endpoint summaries use nonparametric
bootstrap confidence intervals over runs. Comparisons with Sparse ECP use paired task/seed
differences. We report all preregistered algorithms and tasks, including negative results, and
do not select a budget or dataset based on the observed winner.

## Reproducibility

All configurations, seeds, data preparation rules, and output schemas are versioned. Each run
writes per-query trajectories, per-run summaries, bootstrap aggregates, paired comparisons,
and figures. Independent runs may use separate processes, while each worker restricts nested
numerical libraries to one thread. The full commands are:

```bash
sparse-ecp synthetic --config configs/synthetic_benchmark_100.yaml
sparse-ecp theory-experiments --config configs/theory_validation.yaml
sparse-ecp biology --config configs/biology_almanac_d8.yaml
sparse-ecp biology --config configs/biology_almanac_d12.yaml
sparse-ecp biology --config configs/biology_almanac_d16.yaml
sparse-ecp biology --config configs/biology_almanac_d20.yaml
sparse-ecp materials --config configs/materials_ocm.yaml
```

The smaller smoke and pilot configurations test software execution only and are not used for
scientific conclusions.
