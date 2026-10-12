# 1. Gaussian mixtures written against scikit-learn's parameterisation

Status: accepted

## Context

RAPTOR clusters each layer with a Gaussian mixture model whose component count is picked by BIC,
after UMAP has reduced the embeddings to 10 dimensions. The reference implementation calls
`sklearn.mixture.GaussianMixture`. This project implements the clustering itself (`gmm.py`), so
two questions had to be settled: what should "correct" mean for a home-made EM, and which
variant of EM to write.

An EM loop that looks plausible can still be subtly wrong. Typical faults are a missing
`reg_covar` on the diagonal, a log-determinant with the wrong sign, a convergence test on the
total rather than the mean log-likelihood, or a parameter count in BIC that drops the k - 1
free mixture weights. Each of these changes which k BIC picks, so each changes the tree, and
none of them makes anything crash.

## Decision

`gmm.py` follows scikit-learn's numerical choices one for one: Cholesky factors of the
precision matrices, `reg_covar` (1e-6) added to every covariance diagonal, convergence when the
mean per-sample log-likelihood moves by less than `tol`, and BIC computed as
`-2 * n * mean_ll + p * log(n)` with sklearn's parameter count. Densities and responsibilities
are computed in log space with a log-sum-exp that survives rows of `-inf`. Initialisation is our
own k-means++ followed by Lloyd iterations.

Given that, scikit-learn can serve as an oracle. `tests/test_gmm.py` starts both
implementations from the same weights, means and precisions and requires the same iteration
count, the same converged parameters (relative tolerance 1e-7), the same log-likelihood and BIC
(1e-9) and the same responsibilities, for full and diagonal covariances. scikit-learn is a dev
dependency only and is never imported from `src/`.

Two properties are tested without the oracle. Hypothesis generates mixtures to check that exact
EM (`reg_covar=0`) never lowers the log-likelihood, and planted ring mixtures check that BIC
recovers k = 2, 3 and 5.

The clustering follows RAPTOR's settings: full covariance, threshold 0.1, at most 50 components,
seed 224. It differs in one place: a point whose posteriors are all below the threshold (possible
once k > 10) joins its most likely component. In the reference code such a point gets no cluster
and drops out of the next layer.

## Consequences

- Differences between this tree builder and RAPTOR come from documented choices, not from a
  numerical slip inside EM.
- A full-covariance component in 10 dimensions carries 55 covariance and 10 mean parameters.
  A QuALITY article gives roughly 40 to 80 leaves, so BIC's penalty keeps k small. Building
  stops once a layer has 11 nodes or fewer (RAPTOR's `reduction_dim + 1` rule), and the articles
  inspected while building this project gave trees of 2 or 3 layers. The trees are shallow
  because of these rules, not because of a bug, and the benchmark reports the depth histogram
  for every article.
- BIC selection fits every k from 1 to `min(50, n - 1)`. The tree table's stage totals show what
  clustering costs next to embedding.

## Alternatives rejected

- **Call `sklearn.mixture.GaussianMixture`.** It would be the simplest choice, but the project
  exists to implement the clustering maths. Keeping sklearn as a test oracle gives the same
  confidence without the runtime dependency.
- **Diagonal covariance by default.** It is cheaper and BIC would accept more components. It is
  implemented and tested, but RAPTOR uses full covariance, and changing it would confound the
  comparison with the paper's design.
- **Dirichlet-process (variational) mixtures.** These choose k without fitting every candidate,
  but the selection criterion is different and is not what RAPTOR describes.
