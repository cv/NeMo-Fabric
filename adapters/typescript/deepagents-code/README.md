<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Deep Agents Code Adapter

This package provides the `nvidia.fabric.langchain.deepagents-code` process
adapter. It runs LangChain's
[Deep Agents Code](https://github.com/langchain-ai/deepagents/tree/main/libs/code)
(`dcode`) as an ACP agent: one `dcode --acp` child process and one ACP session
per NeMo Fabric runtime, reused across ordered invocations. It can also write a
launcher for `dcode`'s terminal UI that uses the runtime's configuration.

## Install the Adapter

Install Node.js 22.19.0 or newer and the adapter:

```bash
npm install nemo-fabric-adapters-deepagents-code
```

Install `dcode` separately. It requires Python 3.14 and pins exact dependency
versions, so give it its own environment:

```bash
uv tool install deepagents-code==0.2.2
```

The adapter runs `dcode` from `PATH`, or from `harness.settings.dcode_command`.
It never installs or updates `dcode`. The adapter was tested with
`deepagents-code` 0.2.2.

For source development, run these commands from the repository root:

```bash
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-deepagents-code
```

Register the packaged descriptor at
`node_modules/nemo-fabric-adapters-deepagents-code/deepagents-code.fabric-adapter.json`.
Source checkouts can register
`adapters/typescript/deepagents-code/deepagents-code.fabric-adapter.json`.

## Supported Configuration

- **Model.** The `default` model role, or the only role, uses provider
  `openai`. With `base_url`, `dcode` calls that OpenAI-compatible endpoint
  through Chat Completions; without it, OpenAI's own endpoint. The key comes
  from the variable named by `api_key_env`, `OPENAI_API_KEY` by default. HTTP
  endpoints must be loopback or listed in
  `harness.settings.insecure_http_model_hosts`. The adapter writes the variable's
  name, never its value, to `dcode`'s `config.toml`.
- **`approval_mode`** (required). No person can answer an approval during an
  invocation, so choose `dcode`'s unattended behavior explicitly:
  - `auto` runs `dcode --acp --auto-approve`. `dcode`'s classifier, using the
    default model, reviews gated actions, and the adapter denies any it
    escalates.
  - `yolo` allows every gated action, such as shell commands and file writes.
  `manual` isn't accepted.
- **`dcode_command`** (default `dcode`). An executable name looked up on
  `PATH`, or a path; paths with a directory resolve from the NeMo Fabric base
  directory.
- **`startup_timeout_seconds`** (default 60, at most 75). Time for the ACP
  session to start, within NeMo Fabric's 90-second start limit.
- **`interfaces.terminal`**. See [Open the Terminal UI](#open-the-terminal-ui).

NeMo Fabric system instructions, skills, MCP servers, tool policy, telemetry,
usage reporting, streaming and cancellation aren't mapped yet. `dcode` reports
no token usage over ACP.

## State and Outbound Requests

Each runtime keeps `dcode`'s state, including `config.toml`, sessions and
memory, in its own `DEEPAGENTS_HOME`:
`<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>/deepagents`. The
artifact root is the runtime context's artifact directory, or
`<base_dir>/artifacts`. The configuration is regenerated at each start.

The adapter passes `dcode` only a fixed set of process variables, such as
`HOME`, `PATH`, proxy and certificate settings, the runtime environment's
variables, and the model key's variable. It runs `dcode` with `--no-tracing`
and turns off its update checks, automatic updates, and price and plugin
refreshes through `DEEPAGENTS_CODE_NO_UPDATE_CHECK`,
`DEEPAGENTS_CODE_AUTO_UPDATE`, `DEEPAGENTS_CODE_PRICES_AUTO_UPDATE` and
`DEEPAGENTS_CODE_PLUGIN_AUTO_UPDATE`.

## Open the Terminal UI

With `harness.settings.interfaces.terminal` present, the adapter writes an
executable launcher to
`<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>/interfaces/terminal`.
The runtime handle reports that path as `interfaces.terminal.launcher`, from
`fabric-agent-ctl start` or, later, `fabric-agent-ctl status --agent NAME
--runtime-id ID`. In an OpenShell sandbox, a person runs it with:

```bash
openshell sandbox exec -n SANDBOX --tty -- <launcher path>
```

`openshell sandbox connect` attaches to the sandbox's main process, not a
shell, so it can't run the launcher. On a `noexec` mount, run it as
`sh <launcher path>`.

The launcher opens `dcode`'s terminal UI in a **separate session** with the
runtime's `DEEPAGENTS_HOME`, configuration and working directory. `dcode` can't
attach to another process's agent, so the terminal doesn't see invocations
live, and it can't open the runtime's own thread. The runtime's ACP session is
a `dcode` thread whose ID is in `interfaces/terminal.running`; the ACP session
holds it while the runtime runs, and `dcode` refuses to resume a thread open
elsewhere ("Thread can't be resumed"). The terminal UI can still list it with
the runtime's other sessions:

```bash
openshell sandbox exec -n SANDBOX --tty -- <launcher path> threads list
```

- The launcher holds paths and switches, never secrets. The terminal UI reads
  the model key from the launching environment under the configured variable
  name, unless the endpoint needs no key.
- The terminal UI uses `dcode`'s own approval behavior, not `approval_mode`.
- The terminal UI starts its own `dcode` agent server on `127.0.0.1` without
  authentication, as `dcode` always does; its `THREAT_MODEL.md` lists this as
  T6. Invocations use ACP over stdio and open no listener.
- After the ACP session starts, the adapter writes `interfaces/terminal.running`;
  `stop` removes it before stopping `dcode`. The launcher refuses to start when
  the marker is absent. Terminal sessions already open when the runtime stops
  keep running.
- If the adapter host is killed, the marker can remain, and the launcher then
  opens a session against that runtime's configuration until the sandbox is
  replaced.
- The agent runs as the same user and can modify the launcher.

## Runtime Health

`Fabric.check()` supports `live`, `active` and `ready`. `dcode` runs its agent
inside the ACP process and keeps the runtime's session open in it, so each level
observes that the ACP child process is still running. A busy process passes. A
stopped or exited process fails. Checks don't send prompts.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx)
for shared behavior.
