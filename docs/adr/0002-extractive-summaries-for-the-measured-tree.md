# 2. Extractive summaries for the measured tree, LLM summaries behind a provider

Status: accepted

## Context

RAPTOR's upper layers are abstractive summaries that an LLM writes for each cluster. This project
runs on a 6-core CPU with no GPU and no paid API. QuALITY dev has 115 articles, and each one
needs roughly 10 to 20 cluster summaries, so well over a thousand summaries in all. A local CPU
LLM would need hours for that, and a single small model's summaries would mostly measure that
model.

Retrieval results are also only meaningful when you can see what reached the context.
Abstractive text cannot be traced back to the article.

## Decision

The measured configuration uses `ExtractiveSummarizer`: maximal marginal relevance against the
normalised centroid of the cluster's member embeddings, with diversity 0.3 and at most 128
WordPiece tokens. It picks whole sentences from the cluster's segments, emits them in article
order and copies them verbatim. A summary node therefore stores segment ids rather than new text,
and every sentence in any context maps to character offsets in the article. The TUI shows this
as "all copied verbatim from the article".

`LlmSummarizer` is a real implementation. It sends the cluster's member texts with a "do not add
anything" instruction through the `ChatProvider` interface, which has an OpenAI-compatible HTTP
adapter and a deterministic fake. `raptree build --summarizer llm` and
`raptree-bench run --summarizer llm` use it. Its sentences become new `summary` segments with no
character offsets. The fake exercises it in tests. No benchmark run uses a real LLM, and the
results file lists that configuration under `not_run`.

## Consequences

- The benchmark tests the hierarchy and its selection, and only partly tests synthesis. An
  extractive parent cannot say anything its children do not already say. It can only bring
  scattered sentences together. A result where RAPTOR collapsed beats dense leaf retrieval would
  show that clustered selection helps. A null result would not refute RAPTOR's claim about
  abstractive synthesis. The README says this next to the results.
- Summary nodes repeat article sentences that may also be retrieved as part of a leaf. The
  benchmark therefore reports "distinct-sentence tokens" next to raw tokens, and the evidence
  selector removes duplicate sentences.
- Tree building is deterministic and costs seconds per article. That leaves the CPU budget for
  the readers.

## Alternatives rejected

- **A local seq2seq summariser (BART or T5 class) in ONNX.** It would add a 0.5 to 1.5 GB model,
  hours of CPU time across the dev set, and hallucination risk. The output would still not match
  the GPT-3.5 summaries in the RAPTOR paper.
- **The first k tokens of each cluster as its summary.** This is cheaper, but it depends on which
  member happens to come first and ignores the centroid. It is a weaker stand-in for "what this
  cluster is about".
- **Leaving summaries out (leaves only).** That would remove the very thing under test.
