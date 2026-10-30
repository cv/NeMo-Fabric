// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { constants } from "node:fs";
import { access, realpath, stat } from "node:fs/promises";
import { delimiter, isAbsolute, join, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import {
  LifecycleError,
  insecureHttpModelHosts,
  modelEndpointAllowed,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

export const DEFAULT_STARTUP_TIMEOUT_SECONDS = 60;
// Fabric gives the adapter host 90 s to start; leave time for cleanup.
export const MAX_STARTUP_TIMEOUT_SECONDS = 75;

// Deep Agents Code's own switches for its update checks, automatic updates
// and price and plugin refreshes.
export const QUIET_ENVIRONMENT: Readonly<Record<string, string>> = {
  DEEPAGENTS_CODE_NO_UPDATE_CHECK: "1",
  DEEPAGENTS_CODE_AUTO_UPDATE: "0",
  DEEPAGENTS_CODE_PRICES_AUTO_UPDATE: "0",
  DEEPAGENTS_CODE_PLUGIN_AUTO_UPDATE: "0",
};

export interface DeepAgentsCodeSettings {
  approvalMode: "auto" | "yolo";
  dcodeCommand: string;
  startupTimeoutMs: number;
  terminal: boolean;
}

export interface DeepAgentsCodeModel {
  model: string;
  apiKeyEnv: string;
  baseUrl?: string;
}

const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalid(field: string, message: string): LifecycleError {
  return new LifecycleError("deepagents_code_settings_invalid", message, {
    metadata: { field: `harness.settings.${field}` },
  });
}

function selectTerminal(value: unknown): boolean {
  if (value === undefined) return false;
  if (!isRecord(value)) throw invalid("interfaces", "Deep Agents Code interfaces must be an object");
  const unsupported = Object.keys(value).filter((name) => name !== "terminal").sort();
  if (unsupported[0] !== undefined) {
    throw invalid(`interfaces.${unsupported[0]}`, `Deep Agents Code doesn't support the ${unsupported[0]} interface`);
  }
  const terminal = value.terminal;
  if (terminal === undefined) return false;
  if (!isRecord(terminal)) throw invalid("interfaces.terminal", "Deep Agents Code terminal interface must be an object");
  const unknown = Object.keys(terminal).filter((name) => name !== "enabled").sort();
  if (unknown[0] !== undefined) {
    throw invalid(`interfaces.terminal.${unknown[0]}`, `Deep Agents Code terminal interface doesn't accept ${unknown[0]}`);
  }
  const enabled = terminal.enabled ?? true;
  if (typeof enabled !== "boolean") throw invalid("interfaces.terminal.enabled", "Deep Agents Code terminal enabled must be a boolean");
  return enabled;
}

export function selectSettings(config: AgentConfig): DeepAgentsCodeSettings {
  const value = config.harness?.settings ?? {};
  if (!isRecord(value)) {
    throw new LifecycleError("deepagents_code_settings_invalid", "Deep Agents Code harness settings must be an object", {
      metadata: { field: "harness.settings" },
    });
  }
  const approvalMode = value.approval_mode;
  if (approvalMode !== "auto" && approvalMode !== "yolo") {
    throw invalid("approval_mode", "Deep Agents Code approval_mode must be auto or yolo");
  }
  const dcodeCommand = value.dcode_command ?? "dcode";
  if (typeof dcodeCommand !== "string" || dcodeCommand.trim().length === 0) {
    throw invalid("dcode_command", "Deep Agents Code dcode_command must be a non-empty string");
  }
  const startupTimeout = value.startup_timeout_seconds ?? DEFAULT_STARTUP_TIMEOUT_SECONDS;
  if (
    typeof startupTimeout !== "number" ||
    !Number.isFinite(startupTimeout) ||
    startupTimeout <= 0 ||
    startupTimeout > MAX_STARTUP_TIMEOUT_SECONDS
  ) {
    throw invalid(
      "startup_timeout_seconds",
      `Deep Agents Code startup_timeout_seconds must be greater than 0 and at most ${MAX_STARTUP_TIMEOUT_SECONDS}`,
    );
  }
  return {
    approvalMode,
    dcodeCommand,
    startupTimeoutMs: startupTimeout * 1000,
    terminal: selectTerminal(value.interfaces),
  };
}

export function selectModel(config: AgentConfig): DeepAgentsCodeModel {
  const models = Object.entries(config.models ?? {});
  const selected = models.length === 1 ? models[0]?.[1] : config.models?.default;
  if (selected === undefined) {
    throw new LifecycleError("deepagents_code_model_missing", "Deep Agents Code requires a default model or exactly one model");
  }
  if (selected.provider !== "openai" || selected.model.trim().length === 0) {
    throw new LifecycleError("deepagents_code_model_invalid", "Deep Agents Code requires a non-empty model from provider openai");
  }
  const apiKeyEnv = selected.api_key_env ?? "OPENAI_API_KEY";
  if (!ENVIRONMENT_NAME.test(apiKeyEnv)) {
    throw new LifecycleError("deepagents_code_model_invalid", "Deep Agents Code api_key_env must be an environment variable name");
  }
  const baseUrl = selected.base_url ?? undefined;
  const insecureHosts = insecureHttpModelHosts(config.harness?.settings, "deepagents_code_settings_invalid");
  if (baseUrl !== undefined && !modelEndpointAllowed(baseUrl, insecureHosts)) {
    throw new LifecycleError(
      "deepagents_code_model_endpoint_invalid",
      "Deep Agents Code base_url must use HTTPS, loopback HTTP, or a host in insecure_http_model_hosts",
    );
  }
  return { model: selected.model, apiKeyEnv, ...(baseUrl === undefined ? {} : { baseUrl }) };
}

/** Return Deep Agents Code's config.toml for one OpenAI-compatible model. */
export function configToml(model: DeepAgentsCodeModel): string {
  const lines = [
    "[models]",
    `default = ${JSON.stringify(`openai:${model.model}`)}`,
    "",
    "[models.providers.openai]",
    `models = [${JSON.stringify(model.model)}]`,
    `api_key_env = ${JSON.stringify(model.apiKeyEnv)}`,
  ];
  if (model.baseUrl !== undefined) {
    // OpenAI-compatible endpoints serve Chat Completions, not the Responses API.
    lines.push(
      `base_url = ${JSON.stringify(model.baseUrl)}`,
      "",
      "[models.providers.openai.params]",
      "use_responses_api = false",
    );
  }
  return `${lines.join("\n")}\n`;
}

export async function workspacePath(input: AdapterStartInput): Promise<string> {
  const configured = input.runtimeContext.environment.workspace ?? input.baseDir;
  try {
    const workspace = await realpath(isAbsolute(configured) ? configured : resolve(input.baseDir, configured));
    if (!(await stat(workspace)).isDirectory()) throw new Error("not a directory");
    return workspace;
  } catch {
    throw new LifecycleError("deepagents_code_workspace_invalid", "The Deep Agents Code workspace must be an existing directory");
  }
}

/** Return `<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>`. */
export function runtimeDirectory(input: AdapterStartInput): string {
  const root = input.runtimeContext.artifacts.root;
  const artifactRoot =
    root === undefined || root === null
      ? resolve(input.baseDir, "artifacts")
      : resolve(input.baseDir, String(root));
  return join(artifactRoot, ".fabric", "deepagents-code", "runtimes", input.runtimeContext.runtime_id);
}

/** Resolve dcode_command to an absolute executable, as a shell would. */
export async function resolveCommand(command: string, baseDir: string, path: string | undefined): Promise<string> {
  const candidates = command.includes("/")
    ? [resolve(baseDir, command)]
    : (path ?? "").split(delimiter).filter((directory) => directory.length > 0).map((directory) => join(directory, command));
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      if ((await stat(candidate)).isFile()) return candidate;
    } catch {
      // Try the next candidate.
    }
  }
  throw new LifecycleError("deepagents_code_harness_unavailable", "The configured dcode_command was not found or is not executable", {
    metadata: { field: "harness.settings.dcode_command" },
  });
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** Return the terminal launcher, which holds paths and switches but no secrets. */
export function launcherScript(options: { home: string; workspace: string; command: string; marker: string }): string {
  const exports = Object.entries({ DEEPAGENTS_HOME: options.home, ...QUIET_ENVIRONMENT })
    .map(([name, value]) => `export ${name}=${shellQuote(value)}`);
  return [
    "#!/bin/sh",
    "# Opens Deep Agents Code's terminal UI with this NeMo Fabric runtime's configuration.",
    `if [ ! -e ${shellQuote(options.marker)} ]; then`,
    "  echo 'This NeMo Fabric runtime is not running.' >&2",
    "  exit 1",
    "fi",
    ...exports,
    `cd ${shellQuote(options.workspace)} || exit 1`,
    `exec ${shellQuote(options.command)} --no-tracing "$@"`,
    "",
  ].join("\n");
}
