# Memory Policies

All policies implement the memory interface used by `EpisodeRunner`: update from an experience, optionally ingest shared context, expose a prompt context, and serialize state.

## Naive overwrite

**Implementation:** `src/seam/memory/naive_overwrite.py`

The policy replaces its memory after every experience. With a client, it asks the LLM to rewrite a concise reflection from previous memory, the current observation/action/reward, and peer context. Without a client, it creates a deterministic last-action summary and appends peer lines.

The current implementation enforces the configured `max_tokens` as a whitespace-token budget. This controls stored memory length but is not a model tokenizer count.

**Interpretation:** high compression and recency bias; useful as a deliberately simple baseline.

## Raw trajectory buffer

**Implementation:** `src/seam/memory/raw_trajectory.py`

The policy retains a bounded window of recent experience records. It preserves more raw trajectory detail than overwrite memory but can expose repeated observations and actions directly.

**Interpretation:** a persistence/noisy-memory baseline that separates accumulation from LLM summarization.

## Structured incremental

**Implementation:** `src/seam/memory/structured_incremental.py`

The policy maintains a playbook of active rules. LLM responses are parsed for explicit `ADD:` and `DEPRECATE: Rule #N` directives. Deterministic mode creates rules from reward outcomes and ingests peer messages. Deprecated rules are purged during pruning, and conversational LLM output without a recognized directive is ignored rather than promoted to a rule.

**Interpretation:** an ACE-style Generate -> Reflect -> Curate abstraction, represented here as a compact rule store rather than a raw transcript.

## Shared context and local memory

The runner keeps local policy context separate from the shared context when constructing action prompts. It records local context for Self-BLEU and the updated policy state for memory-length/logging. This separation is intended to prevent peer convergence from being mistaken for temporal self-collapse.

## What to report

For every policy comparison, report the policy configuration, max memory limits, model, episode length, sharing settings, and seed count. A policy's observed performance cannot be interpreted independently of the amount and form of context it receives.

## Serialization

The three policies serialize different state shapes. Naive overwrite stores the policy name, configured maximum, and one `memory_text`. Raw trajectory stores a list of experience dictionaries and its window size. Structured incremental stores rule dictionaries, maximum active entries, and the next rule ID. These representations are useful for debugging but are not automatically written as standalone checkpoints by the runner; the active memory state is included in each logged event.

## Update timing

The runner computes the local context before action selection. After the environment step, it updates memory using the just-observed experience and the current shared context. Thus a peer artifact received in a round can affect both that round's action prompt and the subsequent local memory update. This ordering should be stated when discussing whether sharing influences action selection, memory formation, or both.

## Policy comparison limits

Naive overwrite uses an LLM reflection call when a client is present; raw trajectory does not call the LLM for memory updates; structured incremental may call the LLM for rule directives. A comparison therefore changes both memory representation and the number/type of model calls. The manuscript should report this computational asymmetry rather than presenting the policies as equal-cost mechanisms.
