# 5. Article-clustered bootstrap intervals and exact paired sign-flip tests

Status: accepted

## Context

QuALITY dev has 2,086 questions but only 115 articles, about 18 questions per article. Questions
about the same article share one tree, one set of retrieved candidates and one writing style.
Their outcomes are correlated. A plain binomial interval treats them as independent and comes out
too narrow.

All methods answer the same questions, so the comparisons are paired. Unpaired intervals on two
accuracies would overstate the uncertainty in their difference.

## Decision

- Accuracy intervals (`stats.clustered_bootstrap`) resample whole articles with replacement,
  2,000 resamples, percentile 95% intervals. Each replicate's accuracy is total hits over total
  questions in the resampled articles. Hard and easy subsets use the same procedure.
- Method comparisons use a two-sided paired sign-flip permutation test on per-question
  correctness differences. Concordant questions contribute zero under any flip, so the
  permutation distribution is that of 2 * Binomial(m, 1/2) - m over the m discordant questions.
  The p-value is computed exactly from the binomial pmf in log space, not by Monte Carlo.
  Hypothesis checks it against the exact sign test.
- The 15 pairwise tests per reader (five methods plus the no-context control) are
  Holm-adjusted. Tables report raw and adjusted p-values and the discordant counts.

## Consequences

- Intervals are wider than naive ones, and honestly so. Small differences between methods may
  not clear them.
- The paired test is exact and deterministic. Rerunning the benchmark gives identical p-values.
- The sign-flip test is per question. It does not account for article clustering in the paired
  comparison. Its p-values are therefore somewhat optimistic when per-article effects are
  strong, and the README treats them as screening numbers, not confirmatory ones.

## Alternatives rejected

- **McNemar's chi-squared test.** It is the large-sample approximation of the same exact test, and
  the exact version is cheap.
- **A cluster-level permutation test on per-article mean differences.** It is valid under
  clustering, but with 115 clusters and many articles at zero difference it has little power. It
  could be added as a secondary column if the per-question results look borderline.
- **Normal-approximation intervals.** These ignore clustering and can extend outside [0, 1]
  on the small hard and easy subsets.
