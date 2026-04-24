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

import asyncio
import sys
from unittest import mock

from google.adk.tools import shell_tool
from google.adk.tools import tool_context
from google.adk.tools.tool_confirmation import ToolConfirmation
import pytest


@pytest.fixture
def tool_context_no_confirmation():
  """ToolContext with no confirmation (initial call)."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx.tool_confirmation = None
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_confirmed():
  """ToolContext with confirmation approved."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = True
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_no_confirm_required():
  """ToolContext for tools with confirm_required=False."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx.tool_confirmation = None
  ctx.actions = mock.MagicMock()
  return ctx


class TestShellTool:

  @pytest.mark.asyncio
  async def test_shell_tool_echo(self, tool_context_confirmed):
    """Basic echo command returns expected output."""
    tool = shell_tool.ShellTool()
    if sys.platform == "win32":
      cmd = "cmd /c echo hello"
    else:
      cmd = "echo hello"
    result = await tool.run_async(
        args={"command": cmd},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]

  @pytest.mark.asyncio
  async def test_shell_tool_with_workdir(self, tmp_path, tool_context_confirmed):
    """Run command in a specific working directory."""
    tool = shell_tool.ShellTool()
    if sys.platform == "win32":
      cmd = "cmd /c echo hello"
    else:
      cmd = "echo hello"
    result = await tool.run_async(
        args={"command": cmd, "workdir": str(tmp_path)},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]

  @pytest.mark.asyncio
  async def test_shell_tool_timeout(self, tool_context_confirmed):
    """Command that sleeps should time out."""
    policy = shell_tool.ShellToolPolicy(timeout_seconds=1)
    tool = shell_tool.ShellTool(policy=policy)
    # Use Python for a cross-platform sleep command.
    cmd = f'{sys.executable} -c "import time; time.sleep(30)"'
    result = await tool.run_async(
        args={"command": cmd},
        tool_context=tool_context_confirmed,
    )
    assert "error" in result
    assert "timed out" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_shell_tool_nonzero_exit(self, tool_context_confirmed):
    """Exit with a non-zero return code."""
    tool = shell_tool.ShellTool()
    result = await tool.run_async(
        args={"command": f'{sys.executable} -c "exit(42)"'},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 42

  @pytest.mark.asyncio
  async def test_shell_tool_stderr(self, tool_context_confirmed):
    """Capture stderr output."""
    tool = shell_tool.ShellTool()
    result = await tool.run_async(
        args={
            "command": (
                f'{sys.executable} -c "import sys; sys.stderr.write(\\"err\\")"'
            )
        },
        tool_context=tool_context_confirmed,
    )
    assert "err" in result["stderr"]

  @pytest.mark.asyncio
  async def test_shell_tool_python_script(self, tmp_path, tool_context_confirmed):
    """Run a .py file."""
    script = tmp_path / "hello.py"
    script.write_text("print('hello from script')")
    tool = shell_tool.ShellTool()
    result = await tool.run_async(
        args={
            "command": f"{sys.executable} {script}",
        },
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "hello from script" in result["stdout"]

  @pytest.mark.asyncio
  async def test_shell_tool_empty_command(self, tool_context_confirmed):
    """Empty command returns error."""
    tool = shell_tool.ShellTool()
    result = await tool.run_async(
        args={"command": ""},
        tool_context=tool_context_confirmed,
    )
    assert "error" in result
    assert "required" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_shell_tool_declaration(self):
    """Verify FunctionDeclaration has expected structure."""
    tool = shell_tool.ShellTool()
    declaration = tool._get_declaration()
    assert declaration is not None
    assert declaration.name == "execute_shell"
    schema = declaration.parameters_json_schema
    assert "command" in schema["properties"]
    assert "workdir" in schema["properties"]
    assert "timeout" in schema["properties"]
    assert schema["required"] == ["command"]

  @pytest.mark.asyncio
  async def test_shell_tool_custom_timeout(self, tool_context_confirmed):
    """Per-command timeout override is respected."""
    policy = shell_tool.ShellToolPolicy(timeout_seconds=60)
    tool = shell_tool.ShellTool(policy=policy)
    # Use Python for a cross-platform sleep command.
    cmd = f'{sys.executable} -c "import time; time.sleep(30)"'
    result = await tool.run_async(
        args={"command": cmd, "timeout": 1},
        tool_context=tool_context_confirmed,
    )
    assert "error" in result
    assert "timed out" in result["error"].lower()
    assert "1 second" in result["error"]

  @pytest.mark.asyncio
  async def test_shell_tool_policy_blocks_command(
      self, tool_context_no_confirmation
  ):
    """Policy with limited prefixes blocks disallowed commands."""
    policy = shell_tool.ShellToolPolicy(allowed_command_prefixes=("echo",))
    tool = shell_tool.ShellTool(policy=policy)
    result = await tool.run_async(
        args={"command": "rm -rf ."},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "Permitted prefixes are: echo" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_not_called()
