# Reasoning Router Prompt Templates

**Model**: Qwen/Qwen2.5-7B-Instruct (local checkpoint)
**Generation**: temperature=0, top_p=1.0, do_sample=False, max_new_tokens=300
**Date**: 2026-10-18

## Five-Retriever Capability Description (shared by P0/P1/P2)

```
You will choose the best retrieval method for a user question from five options:

- BM (BM25): Lexical keyword matching retrieval. Strong for exact entity names,
  rare terms, and precise string matches; weak for semantic paraphrase.
- TD (Text-Dense): Single-vector dense retrieval over text. Strong for semantic
  similarity and paraphrased questions; weaker on exact keyword and fine-grained
  multi-condition matching.
- TR (Text-Rerank): Text-dense recall followed by token-level late-interaction
  reranking. Strong for fine-grained multi-condition and relevance discrimination
  within text; does not use visual information.
- MD (MM-Dense): Single-vector dense retrieval over multimodal page representations
  (text + image regions). Strong when a page's visual layout or image content
  carries the answer; pages may also contain figure captions as text.
- MR (MM-Rerank): Multimodal-dense recall followed by visual late-interaction
  reranking. Strongest for queries that require reading figures, tables, or
  layout; text pages with captions are also handled, but the visual rerank step
  adds cost.

Note: Text-only pages may include figure captions. Do not assume a non-MR
retriever cannot find an answer that happens to be near a figure.
```

## P0 — Direct Selection

```
{shared_description}

Question: {query}

Output exactly one line in the form:
FINAL=<BM|TD|TR|MD|MR>
```

## P1 — Free Chain-of-Thought

```
{shared_description}

Question: {query}

Briefly reason in 1-3 sentences about what kind of retrieval this question needs,
then choose. End your answer with exactly one line:
FINAL=<BM|TD|TR|MD|MR>
```

## P2 — Capability Reasoning (structured)

```
{shared_description}

Question: {query}

Analyse the question by filling in the five fields below. Keep it concise
(under 150 words total).

Task: <fact lookup / numerical computation / constrained comparison /
       explicit visual request / other — multiple labels allowed>
Constraints: <exact entities, years, numbers, relations, multi-condition>
Matching Need: <lexical exact match / semantic recall / token-level fine-grained
                match / page visual representation>
Comparison: <name at least two competing retrievers and state their capability
             difference for this query; do NOT claim to know where evidence lives>
FINAL=<BM|TD|TR|MD|MR>
```

## Parsing rule

Extract the last `FINAL=<arm>` occurrence (case-insensitive, arm in {BM,TD,TR,MD,MR}).
If absent or invalid, mark `parse_ok=False`; do NOT default to MR.
