# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool to execute shell commands."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import pathlib
import sys
from typing import Any
from typing import Optional

from google.genai import types

from ..features import experimental
from ..features import FeatureName
from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

_MAX_OUTPUT_LENGTH = 100_000

_SHELL_PROMPT = (
    "Execute a shell command and return stdout, stderr, and the return code."
    " Commands run in `workdir` if provided, otherwise the current working"
    " directory.\n\nThe `command` parameter (string) is required. The `timeout`"
    " parameter (integer, optional) sets a per-command timeout in seconds that"
    " overrides the default.\n\nPrefer using the `workdir` parameter instead of"
    " chaining `cd` commands.\n\nWhen running scripts from a skill, use the"
    " `base_directory` from `load_skill` as the `workdir`.\n\nExamples:\n"
    "  Python scripts: `{sys.executable} scripts/foo.py --arg value`\n"
    "  Shell scripts: `bash scripts/foo.sh --arg value`".format(
        sys=sys,
    )
)


@dataclasses.dataclass(frozen=True)
class ShellToolPolicy:
  """Configuration for allowed shell commands and execution limits.

  Set allowed_command_prefixes to ("*",) to allow all commands (default),
  or explicitly list allowed prefixes.
  """

  allowed_command_prefixes: tuple[str, ...] = ("*",)
  blocked_operators: tuple[str, ...] = ()
  timeout_seconds: int = 30
  confirm_required: bool = True


def _validate_command(command: str, policy: ShellToolPolicy) -> Optional[str]:
  """Validates a shell command against the policy.

  Args:
    command: The command string to validate.
    policy: The policy to validate against.

  Returns:
    An error message string if validation fails, or None if the command is
    allowed.
  """
  stripped = command.strip()
  if not stripped:
    return "Command is required."

  for op in policy.blocked_operators:
    if op in command:
      return f"Command contains blocked operator: {op}"

  if "*" in policy.allowed_command_prefixes:
    return None

  for prefix in policy.allowed_command_prefixes:
    if stripped.startswith(prefix):
      return None

  allowed = ", ".join(policy.allowed_command_prefixes)
  return f"Command blocked. Permitted prefixes are: {allowed}"


def _truncate_output(output: str) -> str:
  """Truncates output to the maximum allowed length."""
  if len(output) > _MAX_OUTPUT_LENGTH:
    return output[:_MAX_OUTPUT_LENGTH] + "\n... [truncated]"
  return output


@experimental(FeatureName.SKILL_TOOLSET)
class ShellTool(BaseTool):
  """Tool to execute a validated shell command within a workspace directory.

  Uses asyncio.create_subprocess_shell for cross-platform shell command
  execution. On Unix, processes are started in a new session so the entire
  process group can be killed on timeout. On Windows, the default process
  creation is used with proc.kill().
  """

  def __init__(
      self,
      workspace: pathlib.Path | None = None,
      policy: Optional[ShellToolPolicy] = None,
  ):
    if workspace is None:
      workspace = pathlib.Path.cwd()
    policy = policy or ShellToolPolicy()
    allowed_hint = (
        "any command"
        if "*" in policy.allowed_command_prefixes
        else (
            "commands matching prefixes:"
            f" {', '.join(policy.allowed_command_prefixes)}"
        )
    )
    super().__init__(
        name="execute_shell",
        description=_SHELL_PROMPT,
    )
    self._workspace = workspace
    self._policy = policy

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "Working directory for the command. Defaults to the"
                        " tool's workspace."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Per-command timeout in seconds. Overrides the default"
                        " policy timeout."
                    ),
                },
            },
            "required": ["command"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    command = args.get("command")
    if not command:
      return {"error": "Command is required."}

    # Static validation.
    error = _validate_command(command, self._policy)
    if error:
      return {"error": error}

    # Confirmation check.
    if self._policy.confirm_required:
      if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Please approve or reject the shell command: {command}",
        )
        tool_context.actions.skip_summarization = True
        return {
            "error": (
                "This tool call requires confirmation, please approve or"
                " reject."
            )
        }
      elif not tool_context.tool_confirmation.confirmed:
        return {"error": "This tool call is rejected."}

    # Determine working directory.
    workdir = args.get("workdir")
    cwd = str(pathlib.Path(workdir)) if workdir else str(self._workspace)

    # Determine timeout.
    timeout = args.get("timeout")
    timeout_seconds = int(timeout) if timeout is not None else self._policy.timeout_seconds

    stdout = None
    stderr = None
    try:
      # Build subprocess kwargs.
      subprocess_kwargs = {
          "cwd": cwd,
          "stdout": asyncio.subprocess.PIPE,
          "stderr": asyncio.subprocess.PIPE,
      }

      is_unix = sys.platform != "win32"
      if is_unix:
        subprocess_kwargs["start_new_session"] = True

      process = await asyncio.create_subprocess_shell(
          command,
          **subprocess_kwargs,
      )

      try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
      except asyncio.TimeoutError:
        # Kill the process (group on Unix, single on Windows).
        try:
          if is_unix and process.pid:
            os.killpg(process.pid, 9)  # SIGKILL
          else:
            process.kill()
        except (ProcessLookupError, OSError):
          pass
        stdout_bytes, stderr_bytes = await process.communicate()

        stdout_str = _truncate_output(
            stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        )
        stderr_str = _truncate_output(
            stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        )

        return {
            "error": (
                f"Command timed out after {timeout_seconds} seconds."
            ),
            "stdout": stdout_str,
            "stderr": stderr_str,
            "returncode": process.returncode,
        }

      stdout_str = _truncate_output(
          stdout_bytes.decode(errors="replace") if stdout_bytes else ""
      )
      stderr_str = _truncate_output(
          stderr_bytes.decode(errors="replace") if stderr_bytes else ""
      )

      return {
          "stdout": stdout_str,
          "stderr": stderr_str,
          "returncode": process.returncode,
      }
    except Exception as e:  # pylint: disable=broad-except
      logger.exception("ShellTool execution failed")

      stdout_res = (
          stdout.decode(errors="replace") if stdout else ""
      )
      stderr_res = (
          stderr.decode(errors="replace") if stderr else ""
      )

      return {
          "error": f"Execution failed: {str(e)}",
          "stdout": stdout_res,
          "stderr": stderr_res,
      }
