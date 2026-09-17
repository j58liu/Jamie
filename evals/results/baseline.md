# Model evaluation baseline

Qwen3.8-9B-Distill Q4_K_M, evaluated on September 16, 2026.

This completed run used a **120-second client read timeout** and a 5-second
connection timeout. The application default was subsequently changed to
600 seconds. These results are retained unchanged and have not been rerun
or combined with attempts using the new timeout.

| Metric | Result |
| --- | ---: |
| Cases | 20 |
| Runs per case | 5 |
| Completed attempts | 100/100 |
| Passes | 38/100 |
| Failures | 62/100 |
| JSON validity | 11.4% (8/70) |
| Instruction following | 38% (38/100) |
| Median latency | 59.23 seconds |
| Reported output tokens | 14,514 |
| Attempts reporting token usage | 77/100 |
| Total duration | 110 minutes 46 seconds |
| Overall result | Fail |

Structured-output cases passed 1/40 attempts, planning cases passed 7/30,
and text instruction-following cases passed 30/30. Of the 62 failures,
39 returned Markdown-fenced text instead of a complete JSON response and
23 timed out. Timeouts count as failures and are included in latency metrics;
their token usage is unknown. JSON validity applies only to the 70 JSON attempts.

Every request used the unchanged case prompt in a fresh conversation,
temperature 0.6, and a maximum of 1,024 generated tokens. The server used an
8,192-token context and one request slot. No prompts were tuned, responses
repaired, or failed requests retried.

The [original report](model-20260916T201410573853Z-b9c7438c.json) contains the
case definitions, their SHA-256 hash, every completed attempt, and aggregate
metrics. The [environment record](model-20260916T201410573853Z-b9c7438c.environment.json)
records the model revision and file hash, server build and settings, and the
timeouts used. The original report predates the explicit timeout field added
to new reports; use the environment record for that setting.
