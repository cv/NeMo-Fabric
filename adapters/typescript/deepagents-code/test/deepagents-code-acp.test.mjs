// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { DeepAgentsCodeAcpSessionFactory } from "../dist/deepagents-code-acp.js";
import { DeepAgentsCodeAdapterRuntime } from "../dist/runtime.js";

const run = promisify(execFile);
const fixture = resolve(fileURLToPath(new URL("../../common/test/fixtures/acp-agent.mjs", import.meta.url)));

// A dcode stand-in: with --acp it records its arguments and environment and
// serves the shared ACP fixture; otherwise it reports the terminal UI's arguments.
async function fakeDcode(directory) {
  const executable = join(directory, "dcode");
  await writeFile(
    executable,
    [
      "#!/bin/sh",
      'if [ "$1" != "--acp" ]; then echo "tui $* home=$DEEPAGENTS_HOME cwd=$(pwd)"; exit 0; fi',
      'printf "%s\\n" "$@" > "$TEST_ARGUMENTS"',
      'printf "home=%s\\nkey=%s\\nupdate=%s\\ndotenv=%s\\n" "$DEEPAGENTS_HOME" "$FABRIC_MODEL_KEY" "$DEEPAGENTS_CODE_NO_UPDATE_CHECK" "$DEEPAGENTS_CODE_READ_PROJECT_DOTENV" > "$TEST_ENVIRONMENT"',
      'exec "$TEST_NODE_EXECUTABLE" "$TEST_ACP_FIXTURE"',
      "",
    ].join("\n"),
  );
  await chmod(executable, 0o755);
  return executable;
}

async function startInput(settings) {
  const directory = await mkdtemp(join(tmpdir(), "fabric-deepagents-code-"));
  const executable = await fakeDcode(directory);
  process.env.FABRIC_MODEL_KEY = "model-key-value";
  return {
    directory,
    input: {
      agentName: "deepagents-code-test",
      baseDir: directory,
      config: {
        models: {
          default: {
            provider: "openai",
            model: "fabric-echo",
            base_url: "http://127.0.0.1:9/v1",
            api_key_env: "FABRIC_MODEL_KEY",
          },
        },
        harness: { settings: { dcode_command: executable, ...settings } },
      },
      runtimeContext: {
        runtime_id: "runtime-1",
        invocation_id: "start",
        request_id: "request",
        artifacts: {},
        environment: {
          environment_id: "environment",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace: directory,
          env: {
            TEST_NODE_EXECUTABLE: process.execPath,
            TEST_ACP_FIXTURE: fixture,
            TEST_ARGUMENTS: join(directory, "arguments"),
            TEST_ENVIRONMENT: join(directory, "environment"),
          },
        },
      },
    },
  };
}

function runtimeDirectory(directory) {
  return join(directory, "artifacts/.fabric/deepagents-code/runtimes/runtime-1");
}

for (const [approvalMode, flags, permission] of [
  ["auto", ["--acp", "--no-tracing", "--auto-approve"], "reject"],
  ["yolo", ["--acp", "--no-tracing"], "allow"],
]) {
  test(`runs dcode over ACP with ${approvalMode} approvals and the runtime's configuration`, async () => {
    const { directory, input } = await startInput({ approval_mode: approvalMode });
    const runtime = new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory());
    await runtime.start(input);
    let first;
    let second;
    let checks;
    try {
      first = await runtime.invoke({ input: "permission" }, input.runtimeContext);
      second = await runtime.invoke({ input: "again" }, input.runtimeContext);
      checks = await Promise.all(["live", "active", "ready"].map((level) => runtime.check(level)));
    } finally {
      await runtime.stop();
    }
    const home = join(runtimeDirectory(directory), "deepagents");

    assert.equal(first.status, "succeeded");
    assert.match(first.output.response, new RegExp(`^permission:1:.*:permission=${permission}:`, "u"));
    assert.match(second.output.response, /^again:2:/u);
    assert.equal(first.extensions.session_id, second.extensions.session_id);
    assert.deepEqual(checks.map((check) => check.outcome), ["passed", "passed", "passed"]);
    assert.deepEqual((await readFile(join(directory, "arguments"), "utf8")).trim().split("\n"), flags);
    assert.equal(
      await readFile(join(directory, "environment"), "utf8"),
      `home=${home}\nkey=model-key-value\nupdate=1\ndotenv=0\n`,
    );
    assert.match(await readFile(join(home, "config.toml"), "utf8"), /api_key_env = "FABRIC_MODEL_KEY"/u);
    assert.doesNotMatch(await readFile(join(home, "config.toml"), "utf8"), /model-key-value/u);
  });
}

test("offers the terminal UI only while the runtime runs", async () => {
  const { directory, input } = await startInput({ approval_mode: "auto", interfaces: { terminal: {} } });
  const interfaces = join(runtimeDirectory(directory), "interfaces");
  const launcher = join(interfaces, "terminal");
  const marker = join(interfaces, "terminal.running");
  const runtime = new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory());
  await runtime.start(input);
  let running;
  let sessionId;
  try {
    sessionId = (await readFile(marker, "utf8")).trim();
    running = await run(launcher, ["-r", sessionId]);
  } finally {
    await runtime.stop();
  }
  const stopped = await run(launcher).then(
    () => assert.fail("the launcher ran after stop"),
    (error) => error,
  );

  assert.ok(sessionId.length > 0);
  assert.equal(
    running.stdout.trim(),
    `tui --no-tracing -r ${sessionId} home=${join(runtimeDirectory(directory), "deepagents")} cwd=${directory}`,
  );
  assert.equal((await stat(launcher)).mode & 0o777, 0o700);
  assert.doesNotMatch(await readFile(launcher, "utf8"), /model-key-value/u);
  assert.equal(stopped.code, 1);
  assert.match(stopped.stderr, /This NeMo Fabric runtime is not running/u);
});

test("writes no terminal marker when the ACP session fails to start", async () => {
  const { directory, input } = await startInput({ approval_mode: "auto", interfaces: { terminal: {} } });
  input.runtimeContext.environment.env.TEST_PROTOCOL_MISMATCH = "1";
  const runtime = new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory());

  await assert.rejects(runtime.start(input), { code: "deepagents_code_acp_protocol_unsupported" });
  await assert.rejects(stat(join(runtimeDirectory(directory), "interfaces/terminal.running")), { code: "ENOENT" });
  await rm(directory, { recursive: true, force: true });
});

test("reports a missing dcode_command before starting anything", async () => {
  const { input } = await startInput({ approval_mode: "auto", dcode_command: "/nonexistent/dcode" });

  await assert.rejects(
    new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory()).start(input),
    { code: "deepagents_code_harness_unavailable" },
  );
});
