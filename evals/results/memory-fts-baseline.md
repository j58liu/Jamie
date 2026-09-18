# SQLite FTS5 memory retrieval baseline

Evaluated on September 18, 2026 using the five fixed memory retrieval cases.

| Metric | Result |
| --- | ---: |
| Cases | 5 |
| Hits | 5/5 |
| Misses | 0 |
| Recall@k | 100% |
| Mean reciprocal rank | 1.000 |
| Total duration | 0.08 seconds |

Each case seeded a separate temporary SQLite database and issued one
natural-language query. Passing required the uniquely identified expected
memory to appear within that case's configured top-k results. The run used the
FTS5 `porter unicode61` tokenizer and did not call Qwen or an embedding model.

This is a deterministic retrieval check. Its duration is machine-specific, and
the five cases are a starting point rather than evidence that FTS5 will handle
every future memory query. Keep this result as the comparison point when new
cases expose retrieval failures or embeddings are evaluated.
