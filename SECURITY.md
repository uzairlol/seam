# Security Policy

## Reporting a vulnerability

If you find a security issue in SEAM (or a vulnerability introduced by a runtime
prompt, model response, or configuration), please **do not** open a public
issue. Report it privately instead:

- Email: `uarif2093@gmail.com`

You should receive a reply within a few days. If you do not, follow up on the
same thread. Please include:

- The affected file(s) and line(s), if known.
- A minimal reproduction (config + commands) if possible.
- The impact you observed.

## Scope

This project runs local LLM inference via [Ollama](https://ollama.com). The
repository itself never calls remote model endpoints without explicit
configuration, but note that:

- **Prompt injection / poison payloads are a designed feature.** The poisoning
  module intentionally seeds directives into agent memories. Do not run
  untrusted poison or prompt content against agents you rely on.
- **Runtime logging stores model inputs and outputs** under `runs/`. Do not
  commit `runs/` artifacts containing sensitive content.

## Supported versions

Security fixes are applied to the current `master` branch. There is no
backport policy; if you need a fix on an older revision, state the commit/branch
and we will work out a patch.

## No bounty

This project does not offer a bug bounty.