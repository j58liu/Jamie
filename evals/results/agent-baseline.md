# Read-only agent evaluation baseline

Qwen3.8-9B-Distill Q4_K_M, evaluated on September 17, 2026.

| Metric | Result |
| --- | ---: |
| Cases | 4 |
| Runs per case | 5 |
| Completed attempts | 20/20 |
| Passes | 20/20 |
| Expected tool called | 20/20 |
| Correct tool output | 20/20 |
| Answer check | 20/20 |
| Median end-to-end latency | 16.09 seconds |
| Reported output tokens | 1,209 |
| Attempts reporting token usage | 20/20 |
| Total duration | 6 minutes 5 seconds |

The fixed cases exercised `read_file`, `list_directory`, `find_files`, and
`search_files`. Every attempt ran in a new temporary workspace. Passing required
the expected tool call, successful output containing the fixture content, and a
final answer containing the expected result.

Each agent run used the published tool schemas and action protocol, temperature
0.6, a maximum of 1,024 tokens per model response, and a 300-second total agent
timeout. The server used a 4,096-token context, one request slot, and disabled
reasoning.

The [original report](agent-20260917T192322469218Z-9bb3787d.json) contains every
model response, tool call, observation, answer, and metric. The
[environment record](agent-20260917T192322469218Z-9bb3787d.environment.json)
records the model, server, and client settings used for the run.
