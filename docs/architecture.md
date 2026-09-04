# System Architecture

## Component graph

```mermaid
flowchart TD
    CLI[Experiment or baseline CLI] --> CFG[ExperimentConfig]
    CFG --> RUN[EpisodeRunner]
    RUN --> ENV[Task environment]
    RUN --> POP[AgentPopulation]
    POP --> AG[BaseAgent]
    AG --> LLM[OllamaClient]
    RUN --> MEM[Memory policies]
    RUN --> SHARE[MemorySharingEngine]
    SHARE --> TOPO[TopologyGenerator]
    RUN --> POISON[PoisonInjector]
    RUN --> LOG[RunLogger]
    LOG --> ART[Run artifacts]
    ART --> AGG[ResultAggregator]
    AGG --> PLOT[Plotting and Markdown tables]
```

## Runtime ownership

`EpisodeRunner` owns one environment, one population, one policy per agent, one sharing engine, one poison injector, and one run logger. The population shares a single Ollama client when one is supplied. The runner is intended to be used as a context manager so resources are released after a run.

## Main boundaries

### Agent boundary

`BaseAgent.format_prompt()` assembles system prompt, agent ID, memory context, observation, available actions, and an instruction. `BaseAgent.act()` calls Ollama and extracts an action from the response. The last assembled prompt is retained for event logging.

### Memory boundary

Every policy implements the base memory interface: reset, get context, update, serialize, and restore. The runner supplies the current experience and the shared context to update.

### Communication boundary

`MemorySharingEngine.step()` collects each policy's context, truncates the published artifact to the configured word-token proxy budget, and routes snippets using `TopologyGenerator`. The engine stores incoming snippets per agent and exposes formatted shared context.

### Environment boundary

The environment receives a dictionary of parsed actions, updates its state, and returns observations, rewards, a done flag, and diagnostic info. The runner never directly computes task rewards.

### Analysis boundary

`ResultAggregator` first prefers `results_summary.csv`; only when that file is absent does it recursively rehydrate `summary.json` files. This matters when new run folders are added without regenerating the experiment-level CSV.

## Episode data flow

1. Seed Python and NumPy and reset the environment.
2. Inject initial internal poison if configured.
3. Increment the round.
4. Inject channel/gradual poison if configured.
5. Route published memory on the sharing cadence.
6. Combine local and shared context for action prompts.
7. Collect actions from every agent.
8. Step the environment.
9. Update each local memory policy with the experience and shared context.
10. Log the prompt, response/action, reward, memory state, and environment info.
11. Repeat until the environment is done.
12. Compute summary metrics and write `summary.json`.

## Current implementation caveats

- `EpisodeRunner` accepts one seed; `ExperimentConfig.seeds` is metadata/default input, while scripts iterate seeds externally.
- The runner's event `raw_response` field currently receives the parsed action rather than the untouched model response.
- Some metrics described in project-scope documents, such as propagation latency and regret, are not currently included in runner summaries.
