# Action protocol evaluation baseline

Qwen3.8-9B-Distill Q4_K_M, evaluated on September 17, 2026.

| Metric | Result |
| --- | ---: |
| Cases | 10 |
| Runs per case | 5 |
| Completed attempts | 50/50 |
| Protocol parse success | 100% (50/50) |
| Instruction following | 90% (45/50) |
| Median latency | 4.51 seconds |
| Reported output tokens | 916 |
| Attempts reporting token usage | 50/50 |
| Total duration | 3 minutes 54 seconds |

The protocol parse target of greater than 95% was met. Forty-nine responses
were direct JSON and one was enclosed in a `json` Markdown fence; the parser
accepted both forms. There were no request failures, malformed actions, or
timeouts.

All five instruction-following failures came from `protocol-008`. The model
returned a valid finish action each time, but its message contained literal
backslashes before the quotation marks and omitted the expected final period.
The case and its outputs are retained unchanged.

Every request used the fixed action protocol system instruction and the
unchanged case prompt in a fresh conversation, temperature 0.6, a maximum of
1,024 generated tokens, and a 600-second client read timeout. The server used a
4,096-token context, one request slot, and disabled reasoning. No JSON grammar,
prompt adjustment, output repair, or retry was used.

The [original report](model-20260917T171225957721Z-f30473c9.json) contains every
response and metric. The [environment record](model-20260917T171225957721Z-f30473c9.environment.json)
records the model, server, and client settings used for the run.
