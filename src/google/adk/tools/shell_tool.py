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
import signal
import subprocess
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

  On Unix, uses asyncio.create_subprocess_shell with start_new_session so the
  entire process group can be killed on timeout. On Windows, falls back to
  subprocess.run in a thread since asyncio subprocess is not supported there.
  """

  def __init__(
      self,
      workspace: pathlib.Path | None = None,
      policy: Optional[ShellToolPolicy] = None,
  ):
    if workspace is None:
      workspace = pathlib.Path.cwd()
    policy = policy or ShellToolPolicy()
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

    error = _validate_command(command, self._policy)
    if error:
      return {"error": error}

    workdir = args.get("workdir")
    cwd = str(pathlib.Path(workdir)) if workdir else str(self._workspace)

    timeout = args.get("timeout")
    timeout_seconds = (
        int(timeout) if timeout is not None else self._policy.timeout_seconds
    )

    if sys.platform == "win32":
      return await asyncio.to_thread(
          _run_subprocess_win32, command, cwd, timeout_seconds
      )
    return await _run_subprocess_unix(command, cwd, timeout_seconds)


async def _stream_reader(
    stream: asyncio.StreamReader | None, console: Any
) -> str:
  """Reads from a subprocess stream line by line, tee-ing to console."""
  if stream is None:
    return ""
  chunks = []
  while True:
    chunk = await stream.read(4096)
    if not chunk:
      break
    text = chunk.decode(errors="replace")
    chunks.append(text)
    console.write(text)
    console.flush()
  return "".join(chunks)


async def _run_subprocess_unix(
    command: str, cwd: str, timeout_seconds: int
) -> dict[str, Any]:
  """Runs a shell command on Unix via asyncio.create_subprocess_shell."""
  process = await asyncio.create_subprocess_shell(
      command,
      cwd=cwd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      start_new_session=True,
  )

  try:
    stdout_task = asyncio.create_task(
        _stream_reader(process.stdout, sys.stdout)
    )
    stderr_task = asyncio.create_task(
        _stream_reader(process.stderr, sys.stderr)
    )
    await asyncio.wait_for(
        asyncio.gather(stdout_task, stderr_task, process.wait()),
        timeout=timeout_seconds,
    )
    stdout_str = _truncate_output(stdout_task.result())
    stderr_str = _truncate_output(stderr_task.result())
  except asyncio.TimeoutError:
    try:
      if process.pid:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
      process.kill()
    await process.wait()

    # Drain remaining output after kill.
    remaining_out = await process.stdout.read() if process.stdout else b""
    remaining_err = await process.stderr.read() if process.stderr else b""
    stdout_str = _truncate_output(
        (stdout_task.result() if stdout_task.done() else "")
        + remaining_out.decode(errors="replace")
    )
    stderr_str = _truncate_output(
        (stderr_task.result() if stderr_task.done() else "")
        + remaining_err.decode(errors="replace")
    )

    return {
        "error": f"Command timed out after {timeout_seconds} seconds.",
        "stdout": stdout_str,
        "stderr": stderr_str,
        "returncode": process.returncode,
    }

  return {
      "stdout": stdout_str,
      "stderr": stderr_str,
      "returncode": process.returncode,
  }


def _run_subprocess_win32(
    command: str, cwd: str, timeout_seconds: int
) -> dict[str, Any]:
  """Runs a shell command on Windows via subprocess.Popen with live tee."""
  try:
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
      stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
      proc.kill()
      stdout_bytes, stderr_bytes = proc.communicate()

      stdout_str = _truncate_output(stdout_bytes.decode(errors="replace"))
      stderr_str = _truncate_output(stderr_bytes.decode(errors="replace"))

      return {
          "error": f"Command timed out after {timeout_seconds} seconds.",
          "stdout": stdout_str,
          "stderr": stderr_str,
          "returncode": proc.returncode,
      }

    stdout_str = _truncate_output(stdout_bytes.decode(errors="replace"))
    stderr_str = _truncate_output(stderr_bytes.decode(errors="replace"))

    # Tee to console.
    if stdout_str:
      sys.stdout.write(stdout_str)
      sys.stdout.flush()
    if stderr_str:
      sys.stderr.write(stderr_str)
      sys.stderr.flush()

    return {
        "stdout": stdout_str,
        "stderr": stderr_str,
        "returncode": proc.returncode,
    }
  except Exception as e:  # pylint: disable=broad-except
    return {"error": f"Execution failed: {e}"}
