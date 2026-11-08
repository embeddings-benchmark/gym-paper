8. Cost and speed, from measured logs only. Anything not measured is left out.

- JEV-27B judging the Qwen3.5-4B queries (data/logs/small_judge.log): grades and wall time from the log's own timestamps.
- Qwen3.5-4B generation (data/logs/small_gen.log): wall time from its timestamps.
- Single-GPU grade rates for JEV-9B, GEV-26B, JEV-35B, d1-3B (data/logs/speed/models-*.out, the measured lines of the run logs): SciFact, 200 grades.
- Chat judges' completion tokens per grade (data/logs/chat_tokens): logged only in the final no-cap passes, which mostly
  regraded the hardest documents, so they overstate a typical grade. A decision model's output is one token.

| what | measured | time | rate | source |
|---|---|---|---|---|
| JEV-27B pointwise judging, Qwen3.5-4B queries, 28 tasks | 207091 grades | 42 min between first and last finished task (starts not logged) | 81 grades/s on 6 GPUs | data/logs/small_judge.log |
| Qwen3.5-4B generation, 28 tasks x 100 queries | 28 tasks logged | 1.1 min between first and last finished task (2 servers; the agent measured about 3 min in all) |  | data/logs/small_gen.log |
| d1, vLLM 0.32, one client | 200 SciFact grades |  | 32.0 grades/s, 1 GPU | models-d1-5317.out |
| d1, vLLM 0.32, one client | 200 SciFact grades |  | 32.0 grades/s, 1 GPU | models-d1-5325.out |
| decider, direct, one at a time | 200 SciFact grades |  | 1.3 grades/s, 1 GPU | models-decider-5285.out |
| decs1, decider.serve, one client | 200 SciFact grades |  | 5.2 grades/s, 1 GPU | models-decs1-5316.out |
| gev, direct, one at a time | 200 SciFact grades |  | 4.1 grades/s, 1 GPU | models-gev-5283.out |
| gev, direct, one at a time | 200 SciFact grades |  | 4.8 grades/s, 1 GPU | models-gev-5289.out |
| gev, patched vLLM, as the stored run was, with its flags, one at a time | 200 SciFact grades |  | 7.7 grades/s, 1 GPU | models-gev-5289.out |
| gev, patched vLLM, as the stored run was, with its flags, 32 at a time | 200 SciFact grades |  | 162.2 grades/s, 1 GPU | models-gev-5289.out |
| gevsh, bash GEV-26B-Decide/serve.sh, one at a time | 200 SciFact grades |  | 12.3 grades/s, 1 GPU | models-gevsh-5298.out |
| gevsh, bash GEV-26B-Decide/serve.sh, 32 at a time | 200 SciFact grades |  | 232.7 grades/s, 1 GPU | models-gevsh-5298.out |
| jev35, direct on one GPU, one at a time | 200 SciFact grades |  | 2.1 grades/s, 1 GPU | models-jev35-5288.out |
| jev35, direct on one GPU, one at a time | 200 SciFact grades |  | 4.2 grades/s, 1 GPU | models-jev35-5297.out |
| jev35, plain vLLM, the error's flags, one at a time | 200 SciFact grades |  | 10.1 grades/s, 1 GPU | models-jev35-5297.out |
| jev35, plain vLLM, the error's flags, 32 at a time | 200 SciFact grades |  | 25.2 grades/s, 1 GPU | models-jev35-5297.out |
| jev35, as the stored run was, one at a time | 200 SciFact grades |  | 9.9 grades/s, 1 GPU | models-jev35-5297.out |
| jev35, as the stored run was, 32 at a time | 200 SciFact grades |  | 32.6 grades/s, 1 GPU | models-jev35-5297.out |
| jev9, direct, one at a time | 200 SciFact grades |  | 5.9 grades/s, 1 GPU | models-jev9-5287.out |
| jev9, the command from the error, one at a time | 200 SciFact grades |  | 21.9 grades/s, 1 GPU | models-jev9-5291.out |
| jev9, the command from the error, 32 at a time | 200 SciFact grades |  | 85.7 grades/s, 1 GPU | models-jev9-5291.out |
| jev9, direct, one at a time | 200 SciFact grades |  | 6.3 grades/s, 1 GPU | models-jev9-5291.out |
| jev9, the command from the error, one at a time | 200 SciFact grades |  | 23.1 grades/s, 1 GPU | models-jev9-5323.out |
| jev9, the command from the error, 32 at a time | 200 SciFact grades |  | 84.3 grades/s, 1 GPU | models-jev9-5323.out |
| jev9, direct, one at a time | 200 SciFact grades |  | 5.1 grades/s, 1 GPU | models-jev9-5323.out |
| jevpin, direct at the pinned revision, one at a time | 200 SciFact grades |  | 5.8 grades/s, 1 GPU | models-jevpin-5315.out |
| strands, strands-decider, one client | 200 SciFact grades |  | 8.6 grades/s, 1 GPU | models-strands-5319.out |
| v11, see log, one at a time | 200 SciFact grades |  | 10.2 grades/s, 1 GPU | models-v11-5286.out |
| v11, see log, 32 at a time | 200 SciFact grades |  | 23.9 grades/s, 1 GPU | models-v11-5286.out |
| v11, see log, one at a time | 200 SciFact grades |  | 10.6 grades/s, 1 GPU | models-v11-5324.out |
| v11, see log, 32 at a time | 200 SciFact grades |  | 24.2 grades/s, 1 GPU | models-v11-5324.out |
| Qwen3.8-27B chat judge, completion tokens per grade | 1447 logged grades (biased to hard ones) |  | median 8048, max 37469 tokens | data/logs/chat_tokens |
| DeepSeek-V4-Flash chat judge, completion tokens per grade | 80 logged grades (biased to hard ones) |  | median 64666, max 65308 tokens | data/logs/chat_tokens |
