# Planning and verification

Simple requests use the bounded agent loop. Requests that chain three or more actions ("find…, check…, and prepare…") use the planner. Use `--plan always` or `--plan never` (or `/plan` in the REPL) to choose explicitly. If no valid plan can be made, the request falls back to the agent loop.

## Who decides what

| Job | Done by |
| --- | --- |
| Propose the steps and their dependencies | Model (JSON constrained to the plan schema) |
| Validate the plan: size, ids, dependency order, tools, known arguments | Runtime |
| Fill one step's arguments from earlier results | Model (constrained to that tool's schema) |
| Enforce order, permissions, approvals, retries, and budgets | Runtime |
| Verify writes, and decide whether to retry, replan, ask, or stop | Runtime |
| Write the final reply | Model (falls back to a plain report) |

## Plan schema

Each plan has a `goal` and at most 8 `steps`. Each step has:
- `id`, `description`, and `tool` (`ask_user` for clarification),
- `depends_on`, which may only name earlier steps,
- known `args`, which must exist in the tool's schema,
- `multiple`, which allows up to five calls (for example one event per deadline),
- `expected_output` (`nonempty` for searches),
- `verification`, which the runtime adds for writes,
- `status`.

## Verification and recovery

| Situation | Behavior |
| --- | --- |
| Event or draft created | Read back with the provider id (`calendar_get_event`, `gmail_get_draft`) |
| Rate limit | Retry with backoff, at most 2 times |
| Ambiguous result on an idempotent write | Retry with the same operation key, which returns the first result |
| Ambiguous result on another write (for example sending) | Stop and report it for review |
| Approval denied, or access missing | Stop the plan |
| Search returns nothing | Retry once with feedback, then ask the user |
| Item not found, or the wrong tool chosen | Replan once, then fail |
| Earlier reads are older than 10 minutes before a write | Re-read them first |

Budgets: 8 steps, 24 model calls, 1 replan, 180 s per step, and 600 s per plan.
