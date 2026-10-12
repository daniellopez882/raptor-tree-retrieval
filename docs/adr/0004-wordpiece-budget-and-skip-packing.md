# 4. A WordPiece token budget and packing that skips what does not fit

Status: accepted

## Context

The comparison only holds if every method gets the same context size: 2,000 tokens per question.
Two details decide whether that holds in practice: how tokens are counted, and what packing does
when the next ranked node is larger than the budget that remains.

In RAPTOR's collapsed-tree retrieval the loop stops at the first node that does not fit. Summary
nodes are close to full size (up to 128 tokens) and leaves vary (up to 100). Under
stop-at-overflow, a method can stop more than a hundred tokens short of the budget on one
question and a few tokens short on the next, depending on where a large node lands in the
ranking. That is a difference in context size, not in retrieval quality.

## Decision

- Tokens are counted with the embedder's own WordPiece tokenizer (`tokens.py`), without special
  tokens. WordPiece pre-tokenises on whitespace and punctuation, so the count of joined texts
  equals the sum of their counts. The integration suite checks this on a real article, which lets
  packing add up node counts safely.
- `pack()` walks the ranking and keeps every node that still fits. A node that does not fit is
  skipped and the walk continues. Hypothesis checks that the budget is never exceeded for random
  rankings and sizes.
- Truncation stops at the first leaf that does not fit, because "the first 2,000 tokens" means a
  prefix.
- Traversal visits the beam path from the top layer down, then the remaining leaves under the
  final beam in score order, and packs that order to the same budget. Every method therefore
  fills close to 2,000 tokens.
- The benchmark records mean tokens, distinct-sentence tokens and the share of tokens taken by
  summary nodes for each method, so readers can check the fill for themselves.

## Consequences

- Methods fill the budget to within one small node of 2,000 tokens, so the accuracy differences
  are not explained by context size.
- The collapsed tree sometimes skips a high-ranked summary in favour of a lower-ranked leaf that
  fits. This differs from the reference implementation and is recorded here.
- Token counts are in the embedder's vocabulary, not an LLM's. The LLM-free readers are not
  affected. A future LLM reader run should convert the budget to its own tokenizer, or at least
  report the conversion.

## Alternatives rejected

- **tiktoken counts.** These add a dependency and a vocabulary that nothing in the measured
  pipeline uses.
- **Stop-at-overflow packing everywhere, as in the reference code.** Its fill varies with node
  sizes, which confounds the comparison it is meant to support.
- **Truncating the last node to fill the budget exactly.** Partial sentences would break the
  sentence-level provenance that the evidence selector and the TUI rely on.
