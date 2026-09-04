# Sharing and Poisoning

## Sharing engine

**Implementation:** `src/seam/sharing/engine.py`

The engine maintains one inbox per agent. On each publish round it:

1. reads each policy's current context;
2. truncates the artifact to `max_artifact_tokens` whitespace tokens;
3. asks `TopologyGenerator` for the target's neighbors;
4. formats incoming messages as `[agent_id]: artifact`;
5. applies `consume_mode=top_k` by retaining at most two received chunks;
6. exposes the inbox as `=== Shared Peer Memories ===`.

Publishing starts at round 1 and repeats every `publish_every_n_rounds` rounds. Sharing mode `off` disables routing and context exposure.

## Topologies

`TopologyGenerator` supports `off`, `full_broadcast`, `ring`, `star`, and `cluster`. The standard experiment grid uses:

- `off`: no peer communication;
- `full_broadcast`: each agent receives from all other agents;
- `ring`: each agent receives from its ring neighbors.

The topology controls reachability; the sharing mode controls whether the communication feature is enabled and how it is interpreted.

## Poison payloads

**Implementation:** `src/seam/poisoning/injector.py`

Default payloads are selected by environment. A payload can also be loaded from `poison_file`.

- `internal`: seed the configured agent's local policy before the episode.
- `channel`: place the payload into every peer inbox, excluding the seed agent.
- `gradual`: begin peer inbox injection at round 5.
- `clean`: no injection.

The internal seed uses a reward of `1.0`, aligned with the bounded reward scale used by the task environments rather than an artificially dominant reward.

## Contamination measurement

The runner derives detection phrases from quoted payload phrases and simple payload tokens, then computes the fraction of non-seed agents whose recorded memory states contain a boundary-matched phrase at any point in the episode. The metric is zero when sharing is inactive.

This is a presence/fraction metric. It does not measure first-arrival latency, dosage, semantic similarity, or causal performance loss. Those are legitimate extensions but should not be claimed as implemented outputs.

## Expected sanity pattern

A basic validation should show:

- clean conditions: zero contamination;
- sharing off: zero peer contamination, even under internal poisoning;
- full broadcast: fastest/widest propagation when peers receive the seed's artifact;
- ring: propagation constrained by local neighborhood reachability.

Unexpected violations of this pattern should trigger an artifact and event-log audit before interpreting means.

## Direction of an edge

The adjacency matrix is defined so entry `(i, j) == 1` means receiver `i` receives from sender `j`. For a ring, each agent receives from its two numerical neighbors modulo `n`. For full broadcast, every off-diagonal entry is one. For the built-in cluster topology, agents are split into two halves and communicate only within their half; there is no cross-cluster bridge.

## Inbox replacement versus injection

Normal `step()` routing replaces each inbox with the messages produced by that publish event. Poison injection uses the engine's public inbox method to append a message to peer inboxes. Consequently, the order of injection and routing matters: the runner injects channel poison before calling `step()`, so a later normal routing operation can replace that injected message if the condition does not otherwise preserve it. Internal poisoning avoids this route because it begins in the seed policy's local state.

## Contamination interpretation

Detection is intentionally conservative about word boundaries but still lexical. A peer may contain a phrase because it copied a message, because its policy summarized it, or because the same phrase arose independently. The current metric cannot distinguish those mechanisms or establish that the phrase changed the peer's action policy.
