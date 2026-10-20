# Jamie performance and usability improvements

This work implements the seven improvements requested in the goal objective, in order.
Earlier uncommitted edits remain separate from these changes.

1. Session-safe diff and undo: implemented. Registries are retained by resolved workspace
   until exit. `/cwd` switches registries; returning to a workspace restores its undo
   history. `/new` changes conversation history only. Snapshot storage is cleaned on exit.
   Regression: edit, chat, diff, switch workspace, undo, switch back, undo, diff.
   Initial focused verification: 52 tests passed.
2. Current-model benchmark: implemented. The first live baseline passed 18/18 trials
   on Qwen3.5-2B Q4_K_M (median 2.06 s; p95 3.61 s). Cases cover finding files, document questions,
   verified edits, remembered decisions, missing-file recovery, and denied approval.
   Reports preserve failures, case hash, model file hash, actual server properties and
   build, generation settings, code revision, and hardware. First visible text is timed
   at the event sink; model-internal first-token time is a different quantity.
3. Visible answer streaming: implemented. Barrier-based tests prove text reaches the
   caller before completion, reasoning stays hidden, cancellation excludes partial
   answers from memory, and the next turn recovers. Agent action JSON stays buffered
   until validation; it is never sent as visible deltas.
4. Complete request budgeting: implemented, with answer reservation and automatic
   extractive compaction. An initial free-form compaction experiment failed one of
   three live trials; it is retained under `evals/results/context`. Validated excerpt
   selection passed all three subsequent long-session trials under a 4,096-token
   context (`evals/results/context-excerpts`). Final verification passed 3/3 trials,
   each with 12 turns and one compaction, with a matching 256-token reserve and
   generation limit (`evals/results/context-final`). Early context experiments used a more conservative answer reserve than
   their generation setting; this was corrected before the inference comparison.
5. Modes, scopes and local commands: implemented. `/mode chat|task`, `/scope`, `/tokens`,
   and `/read`, `/ls`, `/find`, `/search` are covered by interaction and policy tests.
   Local reads use zero model calls. The full-tool comparison had median initial
   input of 1,858 tokens, versus 1,060 for files, with 13/18 versus 17/18 passes.
   These are small stochastic samples, not a guarantee for all tasks.
6. Research: bounded concurrent search and retrieval, stable source ordering,
   cooperative cancellation, stage timings, structured synthesis, model-assisted
   claim reviews, verbatim supporting-quote validation and numeric checks are
   implemented. Adversarial tests reject unsupported claims, invented quotes,
   missing passage coverage and numbers absent from quoted evidence. Initial live
   structured-synthesis fixtures passed 2/2. Repeated serial and concurrent runs
   both passed 6/6. With 0.5 seconds of simulated delay per network call, concurrent
   retrieval cut median search/retrieval time from 2.27 to 1.01 seconds and total
   task time from 9.62 to 8.76 seconds.
7. HTTP clients and streaming loops are reused and explicitly closed. Tests cover
   reuse across worker threads and cleanup from an async application. The cached
   model/context/token/cache comparison completed eight profiles serially on an
   isolated server. All used the same six tasks and three runs per task. The 2B
   Q4_K_M 8k/512 profile passed 18/18 at a 2.12-second median; the 4B profile passed
   18/18 at 4.24 seconds. IQ4_XS passed 11/18. Defaults retain the 2B model and the
   32k window for large tool/page inputs. Validated launcher environment settings
   expose the measured alternatives. Agent deadlines cancel cooperative model
   requests before shared clients close; a regression test covers timeout cleanup.

Two overlapping benchmark attempts shared the GPU and were stopped. They are retained
under `evals/results/interrupted` and excluded from performance conclusions.

Benchmarks use synthetic fixtures in temporary workspaces and fresh SQLite databases.
The model server remains warm between trials. Report latency includes failures and
setup. Record these limitations when interpreting before/after comparisons.

The visible-streaming comparison passed 3/3 in each mode. Median time to visible
text was 0.11 seconds streamed versus 5.80 seconds buffered. This measures earlier
reading, not faster generation. See [the full results and report links](../../evals/results/performance-summary.md).

Completion verification: the prepared commit snapshot passed 499 tests; the working
tree, including earlier user edits, passed 501 tests.
`ruff check .`, `ruff format --check .`, `jamie --version`, and `jamie doctor`
all passed. The user's model server on port 8080 remains available, and the owned
test server on port 8081 has stopped. The changes are organized into seven focused commits.
