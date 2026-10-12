# 3. LLM-free multiple-choice readers over per-option evidence

Status: accepted

## Context

To compare retrieval methods by QA accuracy, something has to read a 2,000-token context and
choose among four options. RAPTOR used GPT-4 and UnifiedQA for this. Neither is available here.
Small pair models (NLI, cross-encoders) run on a CPU, but they accept at most 512 tokens per pair,
and none of them was trained on QuALITY.

## Decision

Both neural readers share one procedure (`readers.py`):

1. For each option, embed "question + option" and keep the context sentences most similar to it,
   up to 192 tokens. Duplicate sentences are counted once and the chosen ones stay in context
   order.
2. Score the pair (evidence, question + option):
   - `nli`: log P(entailment) - log P(not entailment) from an xtremedistil NLI model;
   - `cross-encoder`: the MS MARCO relevance logit of question + option against the evidence.
3. Predict the option with the highest score.

A third reader, `overlap`, scores the share of content words in question + option that appear
in the evidence. It needs no model, so it serves as the lexical baseline and as the reader in
the offline tests.

All readers see contexts from every retrieval method through the same evidence step. A
`no-context` control (empty evidence) shows how much each reader gets from the options alone.
Identical pairs are scored once.

An `llm` reader with the full packed context exists and goes through the provider interface.
It is opt-in (`--readers ...,llm`) and is listed as not run.

## Consequences

- Absolute accuracies are expected to stay near the 25% chance level. QuALITY was built so that
  skimming fails, and many of its "why" questions need inference that no 192-token evidence
  window holds. The comparison that matters is between contexts for the same reader. The
  paired tests answer exactly that question.
- Per-option evidence selection reduces the gap between retrieval methods. A method that
  brings in more relevant sentences helps only when those sentences beat the others inside the
  192-token window. This makes the test conservative: small retrieval differences can disappear.
- Both reader models are small (23 MB int8 and 51 MB fp32 ONNX files) and run on a CPU, but the
  benchmark still scores up to 50,064 pairs per reader (2,086 questions, six contexts, four
  options). The results file records how long each reader took.

## Alternatives rejected

- **A local LLM reader (for example a 1 to 3 B model via llama.cpp).** It needs about 12,500
  prompts of 2,000+ tokens, which means many hours on this CPU. Accuracy for models that small is
  close to chance on QuALITY-hard anyway.
- **Extractive span QA (SQuAD-style).** QuALITY answers are rarely spans of the article.
- **Feeding the whole context to the pair model in 512-token windows and taking the max.** This
  costs about 4 times more pairs. The max over windows also rewards long contexts that contain a
  lucky lexical match, which biases the result towards whichever method retrieves more varied
  text.
