# Preregistered experiment protocol

## 1. Experimental families

The repository has three complementary families. They answer different questions and must not
be pooled into one headline number.

1. Theorem validation uses centered sparse cubes, hidden Lipschitz cones, exact ground truth,
   and replicated noisy oracles. It checks rates, finite bounds, filtering, adaptation,
   approximate sparsity, and proposal cost.
2. The main synthetic benchmark uses 30 sparse problems and 100 repetitions, matching the ECP
   paper's scale. An extended factorial uses seven translated landscapes (cone, quadratic,
   Rastrigin, Ackley, Griewank, Levy, and Salomon), four intrinsic sparsities, three ambient
   dimensions, known/unknown-support variants, and 100 paired repetitions.
3. Retrospective sparse-data studies hide measured rows in drug-combination, materials,
   catalyst, and process tables and reveal only the row selected at each round.

## 2. Algorithms and ablations

The synthetic benchmark compares support-first Sparse ECP, Sparse Random Search, maximin
space filling, GP-UCB, dense ECP, dense random search, dense-draw/hard-threshold ECP, and its
random-search control. The hard-threshold variants are empirical ablations and are not covered
by the support-first theorem.

Finite historical tables additionally compare candidate-uniform ECP and candidate-uniform
random search. A structured-support ECP may be included only when weights are fixed before
responses are revealed; its result is reported separately from structure-free Sparse ECP.

All algorithms receive the same expensive-evaluation budget and paired task/seed instance.
Cheap candidate proposals are counted separately. Hyperparameters are held fixed within each
study and recorded in YAML.

A separate one-factor-at-a-time ablation varies epsilon_1, tau, and rejection patience C,
including C=1 as the near-unfiltered control. It is not pooled with the primary comparison.

## 3. Theorem-validation endpoints

Primary noiseless endpoints are simple regret, log-log budget slope, empirical finite-bound
coverage, and the known/unknown-support gap. The target slopes are -1/s in the local regime.

Filtering endpoints are q_t, p_y, Psi_C(q_t), predicted target-hit lower bound, empirical
next-query success, and proposals per accepted evaluation. Histories are eligible only after
epsilon reaches L and before the fixed target is hit.

Approximate-sparsity endpoints are regret, each exact a_r, the simultaneous oracle bound, and
coverage. No r is selected after inspecting outcomes.

Noisy endpoints are recommendation regret, noisy calls, distinct designs, replications,
finite-bound coverage, and the log-log slope against total oracle calls. The target slope is
-1/(s+2). Known- and unknown-support runs are both included, together with the dyadic
multiscale allocation from Theorem 22. Best sampled latent value is not substituted for
recommendation value.

Proposal-complexity endpoints are total candidate proposals and indicators for the expected
and high-probability bounds from Theorem 25.

## 4. Retrospective sparse-data protocol

Every response is hidden initially. At each round the optimizer selects one historical row,
only that response is revealed, and subsequent choices use only revealed history. Replicates
are aggregated before optimization unless the study is explicitly a noisy-replication study.

Report:

- simple regret against the best measured candidate;
- evaluations to 90%, 95%, and 99% of the measured response range;
- evaluations to enter the measured top 1%;
- trajectory area, support coverage, and proposals per expensive query;
- median/IQR trajectories, bootstrap confidence intervals, and paired same-seed differences.

For every threshold, report reach rate and a budget-censored query count. Conditional-on-success
means may be retained but never reported alone.

## 5. Sparse application progression

1. SynergyFinder's NCATS triple-malaria and O'Neil objects are end-to-end illustrations.
2. NCI-ALMANAC supplies repeated d=8,12,16,20 drug panels and a three-cell-line worst-case
   objective on exactly aligned candidate grids.
3. The full NCATS malaria dose matrix and full O'Neil screen may be added if their original
   exports are recovered; pair-level summaries are not substitutes for dose surfaces.
4. CADS oxidative coupling of methane is the strongest materials match: a sparse catalyst
   formulation is selected jointly with repeated process conditions.
5. Matbench steels, experimental band gap, and perovskites are independent sparse-composition
   controls. Single-row supports test discovery, not within-support continuous optimization.

## 6. Reproducibility and multiplicity

The full synthetic/theory configurations use 100 paired repetitions, matching the original
ECP paper. Real studies use the seed counts in their configs and repeated tasks/panels. Report
all configured algorithms and endpoints; do not choose tasks or budgets based on observed
winners. Bootstrap intervals describe repeated-run uncertainty and are not corrected p-values.

Independent runs may execute in separate processes. Each worker limits nested numerical
libraries to one thread. Parallelism may change wall time but not job ordering or seeds.

The preregistered 100-seed theorem battery remains the primary coverage experiment. The
separate `configs/theory_rate_followup.yaml` is a targeted diagnostic motivated by inspecting
that battery: it extends only the tractable regimes whose finite-budget slopes had not
stabilized, uses 50 paired noiseless seeds and 30 paired noisy seeds, and suppresses the large
per-query trajectory export. It must be labeled as a follow-up rather than silently pooled
with the preregistered run.

## 7. Interpretation rules

- Historical-oracle results estimate ranking/query efficiency on existing tables; they are
  not prospective wet-lab validation.
- A discrete candidate grid does not verify a continuous minimax theorem.
- The lower bounds are mathematical worst-case statements and cannot be proven empirically.
- Materials results do not show that a proposed composition is synthesizable, stable, or safe.
- Biological results make no clinical safety or efficacy claim.
- A gain from a fixed support prior is instance-dependent side information, not removal of the
  binomial(d,s) minimax barrier.
