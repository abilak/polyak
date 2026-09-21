# Theory-to-experiment audit

## Source and scope

This note maps the September 2026 `sparse_polyak.pdf` draft to executable checks. The PDF
is source material, not an instruction file. The implementation follows the corrected
support-first proposal in the current draft; it does not resurrect the dense-draw,
hard-threshold argument rejected in Remark 2.

The deterministic results concern noiseless simple regret on a compact convex domain with a
Lipschitz objective restricted to the sparse search set. The sharp lower bound and proposal
complexity theorem specialize to the centered cube. Retrospective biological and materials
tables are empirical applications, not proofs of the continuous theorems.

## Claim matrix

| Paper result | Executable quantity | Output |
| --- | --- | --- |
| Theorems 7-8, Corollary 18 | log-regret/log-budget slope, finite-budget coverage, known versus unknown support | `noiseless_endpoints.csv`, `noiseless_slopes.csv` |
| Theorem 9, Corollary 10 | Monte Carlo acceptance mass q_t, target mass p_y, Psi_C(q_t), and next-query target hit | `filter_diagnostics.csv`, `filter_calibration.csv` |
| Corollaries 14-15 | exact-size, uniform-size, and complexity-prior proposals under the same hidden optimum | noiseless endpoint and trajectory files |
| Theorem 16 | dense decaying optimum, exact approximation errors a_r, simultaneous oracle bound | `approximate_sparsity.csv` |
| Theorem 17 | growth with binomial(d,s) and known-support control | noiseless dimension/support comparisons |
| Theorems 20-23 | replicated confidence-adjusted ECP, recommendation regret, oracle-balanced replication, stochastic slope | `noisy_endpoints.csv`, `noisy_slopes.csv` |
| Proposition 24, Theorem 25 | total proposals versus expected and high-probability bounds | `proposal_complexity.csv` |

Run the complete battery with:

```bash
sparse-ecp theory-experiments --config configs/theory_validation.yaml
```

The small end-to-end check is `configs/theory_validation_smoke.yaml`.

## Geometry implemented

For a support size r, the proposal first draws the size from weights w_r, then draws one of
the binomial(d,r) supports uniformly, then samples uniformly inside that centered cube slice.
The following preregistered modes are compared:

- `known_support`: all mass on the true slice, isolating r-dimensional geometry;
- `exact_unknown`: exact sparsity known but support unknown;
- `uniform_adaptive`: w_r = 1/s for r=1,...,s;
- `complexity_adaptive`: w_r = (s+1)/(s r(r+1)).

The primary cone is globally Lipschitz and has a hidden sparse maximizer. Each configuration
uses paired centers and seeds across algorithms. The theoretical slope is a reference line,
not a null hypothesis, and a finite fitted slope is never presented as a proof.

## Filtering diagnostic

The ECP filter is evaluated separately from the sparse proposal. Before selected post-burn-in
queries whose target has not yet been reached, cheap held-out proposals estimate q_t and p_y.
The conditional lower bound p_y Psi_C(q_t) is compared with the observed next-query success
rate after binning histories by q_t. These diagnostic proposals never enter the optimization
history and are excluded from the expensive-evaluation count. Monte Carlo sample size and
stride are fixed in the YAML before results are examined.

## Approximate sparsity

The approximate-sparsity study uses a dense maximizer with polynomially decaying coordinates.
For each r, the best r-sparse comparator is its hard truncation and
`a_r = L ||x_circle - HT_r(x_circle)||_2` is known exactly. The reported bound takes the best
admissible r in Theorem 16 using a single simultaneous confidence level. This is a coverage
check of a conservative upper bound, not a claim that the bound is tight.

## Noise

The noisy optimizer evaluates every accepted design b times, stores its empirical mean, uses
the confidence-adjusted acceptance rule in equation (15), and recommends the sampled design
with the largest empirical mean. The total budget counts noisy oracle calls, not unique
designs. The experiment uses the rate-level oracle-balanced allocation and a known-support
control to isolate the N^{-1/(s+2)} exponent; unknown-support dependence is tested in the
noiseless factorial study. Latent objective values are used only for retrospective scoring.

## Interpretation

Empirical coverage below the nominal level is a diagnostic requiring investigation. Coverage
above it is expected because the bounds are conservative and does not prove the theorem.
Likewise, agreement with a slope over finite budgets supports the predicted scaling regime but
does not establish a minimax lower bound. The lower bounds quantify worst cases over function
classes; no finite collection of benchmark functions can validate them directly.
