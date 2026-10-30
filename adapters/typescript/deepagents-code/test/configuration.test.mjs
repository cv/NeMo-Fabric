// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { configToml, launcherScript, selectModel, selectSettings } from "../dist/configuration.js";

function config(settings, model = { provider: "openai", model: "fabric-echo" }) {
  return { models: { default: model }, harness: { settings } };
}

test("reads Deep Agents Code settings with their defaults", () => {
  assert.deepEqual(selectSettings(config({ approval_mode: "auto" })), {
    approvalMode: "auto",
    dcodeCommand: "dcode",
    startupTimeoutMs: 60_000,
    terminal: false,
  });
  assert.equal(selectSettings(config({ approval_mode: "yolo", interfaces: { terminal: {} } })).terminal, true);
  assert.equal(
    selectSettings(config({ approval_mode: "yolo", interfaces: { terminal: { enabled: false } } })).terminal,
    false,
  );
});

for (const [settings, field] of [
  [{}, "harness.settings.approval_mode"],
  [{ approval_mode: "manual" }, "harness.settings.approval_mode"],
  [{ approval_mode: "auto", dcode_command: "" }, "harness.settings.dcode_command"],
  [{ approval_mode: "auto", startup_timeout_seconds: 76 }, "harness.settings.startup_timeout_seconds"],
  [{ approval_mode: "auto", startup_timeout_seconds: 0 }, "harness.settings.startup_timeout_seconds"],
  [{ approval_mode: "auto", interfaces: { api: {} } }, "harness.settings.interfaces.api"],
  [{ approval_mode: "auto", interfaces: { terminal: { port: 1 } } }, "harness.settings.interfaces.terminal.port"],
  [{ approval_mode: "auto", interfaces: { terminal: { enabled: "yes" } } }, "harness.settings.interfaces.terminal.enabled"],
]) {
  test(`rejects invalid settings at ${field}: ${JSON.stringify(settings)}`, () => {
    assert.throws(() => selectSettings(config(settings)), (error) => {
      assert.equal(error.code, "deepagents_code_settings_invalid");
      assert.deepEqual(error.metadata, { field });
      return true;
    });
  });
}

test("maps an OpenAI-compatible endpoint to Chat Completions", () => {
  const model = selectModel(
    config({ approval_mode: "auto" }, {
      provider: "openai",
      model: "fabric-echo",
      base_url: "http://127.0.0.1:9/v1",
      api_key_env: "FABRIC_MODEL_KEY",
    }),
  );

  assert.equal(
    configToml(model),
    [
      "[models]",
      'default = "openai:fabric-echo"',
      "",
      "[models.providers.openai]",
      'models = ["fabric-echo"]',
      'api_key_env = "FABRIC_MODEL_KEY"',
      'base_url = "http://127.0.0.1:9/v1"',
      "",
      "[models.providers.openai.params]",
      "use_responses_api = false",
      "",
    ].join("\n"),
  );
});

test("uses OpenAI's own endpoint and key variable without a base_url", () => {
  assert.deepEqual(selectModel(config({ approval_mode: "auto" })), { model: "fabric-echo", apiKeyEnv: "OPENAI_API_KEY" });
});

for (const [model, code] of [
  [{ provider: "anthropic", model: "claude" }, "deepagents_code_model_invalid"],
  [{ provider: "openai", model: "m", base_url: "http://models.example.test/v1" }, "deepagents_code_model_endpoint_invalid"],
]) {
  test(`rejects unsupported model ${JSON.stringify(model)}`, () => {
    assert.throws(() => selectModel(config({ approval_mode: "auto" }, model)), { code });
  });
}

test("accepts an HTTP endpoint on a trusted host", () => {
  const settings = { approval_mode: "auto", insecure_http_model_hosts: ["inference.local"] };
  const model = { provider: "openai", model: "m", base_url: "http://inference.local/v1" };

  assert.equal(selectModel(config(settings, model)).baseUrl, "http://inference.local/v1");
});

test("writes a launcher that refuses without the marker and holds no secrets", () => {
  const script = launcherScript({
    home: "/runtime/deepagents",
    workspace: "/work space",
    command: "/opt/dcode/bin/dcode",
    marker: "/runtime/interfaces/terminal.running",
  });

  assert.match(script, /^#!\/bin\/sh\n/u);
  assert.match(script, /if \[ ! -e '\/runtime\/interfaces\/terminal.running' \]; then/u);
  assert.match(script, /export DEEPAGENTS_HOME='\/runtime\/deepagents'/u);
  assert.match(script, /export DEEPAGENTS_CODE_NO_UPDATE_CHECK='1'/u);
  assert.match(script, /cd '\/work space' \|\| exit 1/u);
  assert.match(script, /exec '\/opt\/dcode\/bin\/dcode' --no-tracing "\$@"/u);
});

test("declares its settings, model, and health levels", async () => {
  const descriptor = JSON.parse(await readFile(new URL("../deepagents-code.fabric-adapter.json", import.meta.url), "utf8"));
  const properties = descriptor.settings_schema.properties;

  assert.equal(descriptor.adapter_id, "nvidia.fabric.langchain.deepagents-code");
  assert.deepEqual(descriptor.config.accepts, ["models", "models.base_url"]);
  assert.deepEqual(descriptor.settings_schema.required, ["approval_mode"]);
  assert.deepEqual(properties.approval_mode.enum, ["auto", "yolo"]);
  assert.deepEqual(Object.keys(properties.interfaces.properties), ["terminal"]);
  assert.equal(properties.interfaces.additionalProperties, false);
  assert.equal(properties.interfaces.properties.terminal.additionalProperties, false);
  assert.equal(properties.startup_timeout_seconds.maximum, 75);
  assert.deepEqual(descriptor.model_schema.properties.provider, { const: "openai" });
  assert.deepEqual(descriptor.capabilities, {
    metadata: { health_checks: ["live", "active", "ready"] },
    streaming: false,
    cancellation: false,
    updates: false,
    service: false,
  });
});
