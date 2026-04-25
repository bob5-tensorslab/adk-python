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

import logging
import pathlib
import sys
from unittest import mock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import llm_request as llm_request_model
from google.adk.skills import models
from google.adk.tools import skill_toolset
from google.adk.tools import tool_context
from google.genai import types
import pytest


@pytest.fixture(name="mock_skill1_frontmatter")
def _mock_skill1_frontmatter():
  """Fixture for skill1 frontmatter."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "skill1"
  frontmatter.description = "Skill 1 description"
  frontmatter.allowed_tools = ["test_tool"]
  frontmatter.model_dump.return_value = {
      "name": "skill1",
      "description": "Skill 1 description",
  }
  return frontmatter


@pytest.fixture(name="mock_skill1")
def _mock_skill1(mock_skill1_frontmatter):
  """Fixture for skill1."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill1"
  skill.description = "Skill 1 description"
  skill.instructions = "instructions for skill1"
  skill.frontmatter = mock_skill1_frontmatter
  skill.resources = mock.MagicMock(
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
  )

  def get_ref(name):
    if name == "ref1.md":
      return "ref content 1"
    if name == "doc.pdf":
      return b"fake pdf content"
    return None

  def get_asset(name):
    if name == "asset1.txt":
      return "asset content 1"
    if name == "image.png":
      return b"fake image content"
    return None

  def get_script(name):
    if name == "setup.sh":
      return models.Script(src="echo setup")
    if name == "run.py":
      return models.Script(src="print('hello')")
    if name == "build.rb":
      return models.Script(src="puts 'hello'")
    return None

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset
  skill.resources.get_script.side_effect = get_script
  skill.resources.list_references.return_value = ["ref1.md", "doc.pdf"]
  skill.resources.list_assets.return_value = ["asset1.txt", "image.png"]
  skill.resources.list_scripts.return_value = [
      "setup.sh",
      "run.py",
      "build.rb",
  ]
  return skill


@pytest.fixture(name="mock_skill2_frontmatter")
def _mock_skill2_frontmatter():
  """Fixture for skill2 frontmatter."""
  frontmatter = mock.create_autospec(models.Frontmatter, instance=True)
  frontmatter.name = "skill2"
  frontmatter.description = "Skill 2 description"
  frontmatter.allowed_tools = []
  frontmatter.model_dump.return_value = {
      "name": "skill2",
      "description": "Skill 2 description",
  }
  return frontmatter


@pytest.fixture(name="mock_skill2")
def _mock_skill2(mock_skill2_frontmatter):
  """Fixture for skill2."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = "skill2"
  skill.description = "Skill 2 description"
  skill.instructions = "instructions for skill2"
  skill.frontmatter = mock_skill2_frontmatter
  skill.resources = mock.MagicMock(
      spec=[
          "get_reference",
          "get_asset",
          "get_script",
          "list_references",
          "list_assets",
          "list_scripts",
      ]
  )

  def get_ref(name):
    if name == "ref2.md":
      return "ref content 2"
    return None

  def get_asset(name):
    if name == "asset2.txt":
      return "asset content 2"
    return None

  skill.resources.get_reference.side_effect = get_ref
  skill.resources.get_asset.side_effect = get_asset
  skill.resources.list_references.return_value = ["ref2.md"]
  skill.resources.list_assets.return_value = ["asset2.txt"]
  skill.resources.list_scripts.return_value = []
  return skill


@pytest.fixture
def tool_context_instance():
  """Fixture for tool context."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx._invocation_context = mock.MagicMock()
  ctx._invocation_context.agent = mock.MagicMock()
  ctx._invocation_context.agent.name = "test_agent"
  ctx._invocation_context.agent_states = {}
  ctx.agent_name = "test_agent"
  return ctx


# SkillToolset tests
def test_get_skill(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  assert toolset._get_skill("skill1") == mock_skill1
  assert toolset._get_skill("nonexistent") is None


def test_list_skills(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  skills = toolset._list_skills()
  assert len(skills) == 2
  assert mock_skill1 in skills
  assert mock_skill2 in skills


@pytest.mark.asyncio
async def test_get_tools(mock_skill1, mock_skill2):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  tools = await toolset.get_tools()
  assert len(tools) == 4
  assert isinstance(tools[0], skill_toolset.ListSkillsTool)
  assert isinstance(tools[1], skill_toolset.LoadSkillTool)
  assert isinstance(tools[2], skill_toolset.LoadSkillResourceTool)
  assert isinstance(tools[3], skill_toolset.ShellTool)


@pytest.mark.asyncio
async def test_resolve_additional_tools_from_state_none(mock_skill1):
  toolset = skill_toolset.SkillToolset([mock_skill1])

  # Mock ReadonlyContext
  readonly_context = mock.create_autospec(ReadonlyContext, instance=True)
  readonly_context.agent_name = "test_agent"
  readonly_context.state.get.return_value = None

  result = await toolset._resolve_additional_tools_from_state(readonly_context)

  assert not result


@pytest.mark.asyncio
async def test_list_skills_tool(
    mock_skill1, mock_skill2, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  tool = skill_toolset.ListSkillsTool(toolset)
  result = await tool.run_async(args={}, tool_context=tool_context_instance)
  assert "<available_skills>" in result
  assert "skill1" in result
  assert "skill2" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            {"skill_name": "skill1"},
            {
                "skill_name": "skill1",
                "instructions": "instructions for skill1",
                "frontmatter": {
                    "name": "skill1",
                    "description": "Skill 1 description",
                },
            },
        ),
        (
            {"skill_name": "nonexistent"},
            {
                "error": "Skill 'nonexistent' not found.",
                "error_code": "SKILL_NOT_FOUND",
            },
        ),
        (
            {},
            {
                "error": "Argument 'skill_name' is required.",
                "error_code": "INVALID_ARGUMENTS",
            },
        ),
    ],
)
async def test_load_skill_run_async(
    mock_skill1, tool_context_instance, args, expected_result
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillTool(toolset)
  result = await tool.run_async(args=args, tool_context=tool_context_instance)
  # Check expected fields (exact match for error cases, subset for success)
  for key, value in expected_result.items():
    assert result[key] == value
  # Success case now also returns base_directory and files
  if "error" not in result:
    assert "base_directory" in result
    assert "files" in result


@pytest.mark.asyncio
async def test_load_skill_run_async_state_none(
    mock_skill1, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillTool(toolset)

  # Mock state to return None for the key
  state_key = "_adk_activated_skill_test_agent"
  tool_context_instance.state.get.return_value = None

  result = await tool.run_async(
      args={"skill_name": "skill1"}, tool_context=tool_context_instance
  )

  assert result["skill_name"] == "skill1"
  # Verify that it correctly set the list in state
  tool_context_instance.state.__setitem__.assert_called_with(
      state_key, ["skill1"]
  )


@pytest.mark.asyncio
async def test_load_skill_materializes_resources(mock_skill1, tmp_path):
    """load_skill should create a temp dir with skill resources and return base_directory."""
    toolset = skill_toolset.SkillToolset([mock_skill1])
    tool = skill_toolset.LoadSkillTool(toolset)
    ctx = _make_tool_context_with_agent()
    result = await tool.run_async(
        args={"skill_name": "skill1"},
        tool_context=ctx,
    )
    assert "base_directory" in result
    assert "files" in result
    file_list = result["files"]
    assert any("ref1.md" in f for f in file_list)
    assert any("setup.sh" in f for f in file_list)


@pytest.mark.asyncio
async def test_load_skill_files_exist_on_disk(mock_skill1):
    """Files returned by load_skill should actually exist on disk."""
    toolset = skill_toolset.SkillToolset([mock_skill1])
    tool = skill_toolset.LoadSkillTool(toolset)
    ctx = _make_tool_context_with_agent()
    result = await tool.run_async(
        args={"skill_name": "skill1"},
        tool_context=ctx,
    )
    import pathlib
    base = pathlib.Path(result["base_directory"])
    assert base.exists()
    assert (base / "references" / "ref1.md").exists()
    assert (base / "references" / "ref1.md").read_text() == "ref content 1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            {"skill_name": "skill1", "file_path": "references/ref1.md"},
            {
                "skill_name": "skill1",
                "file_path": "references/ref1.md",
                "content": "ref content 1",
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "assets/asset1.txt"},
            {
                "skill_name": "skill1",
                "file_path": "assets/asset1.txt",
                "content": "asset content 1",
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "references/doc.pdf"},
            {
                "skill_name": "skill1",
                "file_path": "references/doc.pdf",
                "status": (
                    "Binary file detected. The content has been injected into"
                    " the conversation history for you to analyze."
                ),
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "assets/image.png"},
            {
                "skill_name": "skill1",
                "file_path": "assets/image.png",
                "status": (
                    "Binary file detected. The content has been injected into"
                    " the conversation history for you to analyze."
                ),
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "scripts/setup.sh"},
            {
                "skill_name": "skill1",
                "file_path": "scripts/setup.sh",
                "content": "echo setup",
            },
        ),
        (
            {"skill_name": "nonexistent", "file_path": "references/ref1.md"},
            {
                "error": "Skill 'nonexistent' not found.",
                "error_code": "SKILL_NOT_FOUND",
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "references/other.md"},
            {
                "error": (
                    "Resource 'references/other.md' not found in skill"
                    " 'skill1'."
                ),
                "error_code": "RESOURCE_NOT_FOUND",
            },
        ),
        (
            {"skill_name": "skill1", "file_path": "invalid/path.txt"},
            {
                "error": (
                    "Path must start with 'references/', 'assets/',"
                    " or 'scripts/'."
                ),
                "error_code": "INVALID_RESOURCE_PATH",
            },
        ),
        (
            {"file_path": "references/ref1.md"},
            {
                "error": "Argument 'skill_name' is required.",
                "error_code": "INVALID_ARGUMENTS",
            },
        ),
        (
            {"skill_name": "skill1"},
            {
                "error": "Argument 'file_path' is required.",
                "error_code": "INVALID_ARGUMENTS",
            },
        ),
    ],
)
async def test_load_resource_run_async(
    mock_skill1, tool_context_instance, args, expected_result
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillResourceTool(toolset)
  result = await tool.run_async(args=args, tool_context=tool_context_instance)
  assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource_path, expected_mime, fake_content",
    [
        ("references/doc.pdf", "application/pdf", b"fake pdf content"),
        ("assets/image.png", "image/png", b"fake image content"),
    ],
)
async def test_load_resource_process_llm_request_binary(
    mock_skill1,
    tool_context_instance,
    resource_path,
    expected_mime,
    fake_content,
):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillResourceTool(toolset)

  llm_req = mock.create_autospec(llm_request_model.LlmRequest, instance=True)

  part = types.Part.from_function_response(
      name=tool.name,
      response={
          "skill_name": "skill1",
          "file_path": resource_path,
          "status": (
              "Binary file detected. The content has been injected into the"
              " conversation history for you to analyze."
          ),
      },
  )
  content = types.Content(role="model", parts=[part])
  llm_req.contents = [content]

  await tool.process_llm_request(
      tool_context=tool_context_instance, llm_request=llm_req
  )

  assert len(llm_req.contents) == 2
  injected_content = llm_req.contents[1]
  assert injected_content.role == "user"
  assert len(injected_content.parts) == 2
  assert (
      f"The content of binary file '{resource_path}' is:"
      in injected_content.parts[0].text
  )
  assert injected_content.parts[1].inline_data.data == fake_content
  assert injected_content.parts[1].inline_data.mime_type == expected_mime


@pytest.mark.asyncio
async def test_process_llm_request(
    mock_skill1, mock_skill2, tool_context_instance
):
  toolset = skill_toolset.SkillToolset([mock_skill1, mock_skill2])
  llm_req = mock.create_autospec(llm_request_model.LlmRequest, instance=True)

  await toolset.process_llm_request(
      tool_context=tool_context_instance, llm_request=llm_req
  )

  llm_req.append_instructions.assert_called_once()
  args, _ = llm_req.append_instructions.call_args
  instructions = args[0]
  assert len(instructions) == 2
  assert instructions[0] == skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  assert "<available_skills>" in instructions[1]
  assert "skill1" in instructions[1]
  assert "skill2" in instructions[1]


def test_default_skill_system_instruction_warning():
  with pytest.warns(
      UserWarning, match="DEFAULT_SKILL_SYSTEM_INSTRUCTION is experimental"
  ):
    instruction = skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
    assert "specialized 'skills'" in instruction


def test_duplicate_skill_name_raises(mock_skill1):
  skill_dup = mock.create_autospec(models.Skill, instance=True)
  skill_dup.name = "skill1"
  with pytest.raises(ValueError, match="Duplicate skill name"):
    skill_toolset.SkillToolset([mock_skill1, skill_dup])


@pytest.mark.asyncio
async def test_scripts_resource_not_found(mock_skill1, tool_context_instance):
  toolset = skill_toolset.SkillToolset([mock_skill1])
  tool = skill_toolset.LoadSkillResourceTool(toolset)
  result = await tool.run_async(
      args={"skill_name": "skill1", "file_path": "scripts/nonexistent.sh"},
      tool_context=tool_context_instance,
  )
  assert result["error_code"] == "RESOURCE_NOT_FOUND"


def _make_tool_context_with_agent(agent=None):
  """Creates a mock ToolContext with _invocation_context.agent."""
  ctx = mock.MagicMock(spec=tool_context.ToolContext)
  ctx._invocation_context = mock.MagicMock()
  ctx._invocation_context.agent = agent or mock.MagicMock()
  ctx._invocation_context.agent.name = "test_agent"
  ctx._invocation_context.agent_states = {}
  ctx.agent_name = "test_agent"
  ctx.state = {}
  return ctx


# ── System instruction references correct tool name ──


def test_system_instruction_references_execute_shell():
  """System instruction must reference execute_shell."""
  assert "execute_shell" in skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  assert "run_skill_script" not in skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION


# ── Integration test for shell-based execution ──


def _make_skill_with_script_simple(skill_name, script_name, script):
  """Creates a minimal Skill with a single script."""
  skill = mock.create_autospec(models.Skill, instance=True)
  skill.name = skill_name
  skill.description = f"Test skill {skill_name}"
  skill.instructions = "test instructions"
  fm = mock.create_autospec(models.Frontmatter, instance=True)
  fm.name = skill_name
  fm.description = f"Test skill {skill_name}"
  fm.metadata = {}
  fm.model_dump.return_value = {"name": skill_name, "description": f"Test skill {skill_name}"}
  skill.frontmatter = fm
  skill.resources = mock.MagicMock(
      spec=["get_reference", "get_asset", "get_script",
            "list_references", "list_assets", "list_scripts"])

  def get_script(name):
    if name == script_name:
      return script
    return None

  skill.resources.get_script.side_effect = get_script
  skill.resources.get_reference.return_value = None
  skill.resources.get_asset.return_value = None
  skill.resources.list_references.return_value = []
  skill.resources.list_assets.return_value = []
  skill.resources.list_scripts.return_value = [script_name]
  return skill


@pytest.mark.asyncio
async def test_skill_execution_via_shell():
  """End-to-end: load skill, get base dir, run script via ShellTool."""
  script = models.Script(src="import sys; print('arg:', sys.argv[1])")
  skill = _make_skill_with_script_simple("test_skill", "run.py", script)
  toolset = skill_toolset.SkillToolset([skill])

  # Step 1: Load skill (materializes resources)
  load_tool = skill_toolset.LoadSkillTool(toolset)
  ctx = _make_tool_context_with_agent()
  result = await load_tool.run_async(
      args={"skill_name": "test_skill"},
      tool_context=ctx,
  )
  assert "base_directory" in result
  base_dir = result["base_directory"]
  assert (pathlib.Path(base_dir) / "scripts" / "run.py").exists()

  # Step 2: Execute script via ShellTool
  shell_tool = toolset._tools[3]
  assert shell_tool.name == "execute_shell"
  shell_result = await shell_tool.run_async(
      args={
          "command": f"{sys.executable} scripts/run.py hello",
          "workdir": base_dir,
      },
      tool_context=ctx,
  )
  assert shell_result["returncode"] == 0
  assert "arg: hello" in shell_result["stdout"]


@pytest.mark.asyncio
async def test_skill_toolset_dynamic_tool_resolution(mock_skill1, mock_skill2):
  # Set up skills with additional_tools in metadata
  mock_skill1.frontmatter.metadata = {
      "adk_additional_tools": ["my_custom_tool", "my_func", "shared_tool"]
  }
  mock_skill1.name = "skill1"

  mock_skill2.frontmatter.metadata = {
      "adk_additional_tools": [
          "skill2_tool",
          "shared_tool",
          "prefixed_mock_tool",
      ]
  }
  mock_skill2.name = "skill2"

  # Prepare additional tools
  custom_tool = mock.create_autospec(skill_toolset.BaseTool, instance=True)
  custom_tool.name = "my_custom_tool"

  skill2_tool = mock.create_autospec(skill_toolset.BaseTool, instance=True)
  skill2_tool.name = "skill2_tool"

  shared_tool = mock.create_autospec(skill_toolset.BaseTool, instance=True)
  shared_tool.name = "shared_tool"

  def my_func():
    """My function description."""
    pass

  # Setup prefixed toolset
  mock_tool = mock.create_autospec(skill_toolset.BaseTool, instance=True)
  mock_tool.name = "prefixed_mock_tool"
  prefixed_set = mock.create_autospec(skill_toolset.BaseToolset, instance=True)
  prefixed_set.get_tools_with_prefix.return_value = [mock_tool]

  toolset = skill_toolset.SkillToolset(
      [mock_skill1, mock_skill2],
      additional_tools=[
          custom_tool,
          skill2_tool,
          shared_tool,
          my_func,
          prefixed_set,
      ],
  )

  ctx = _make_tool_context_with_agent()
  # Initial tools (only core)
  tools1 = await toolset.get_tools_with_prefix(readonly_context=ctx)
  assert len(tools1) == 4

  # Activate skills
  load_tool = skill_toolset.LoadSkillTool(toolset)
  await load_tool.run_async(args={"skill_name": "skill1"}, tool_context=ctx)
  await load_tool.run_async(args={"skill_name": "skill2"}, tool_context=ctx)

  # Dynamic tools should now be resolved
  tools = await toolset.get_tools_with_prefix(readonly_context=ctx)
  assert tools is not tools1
  tool_names = {t.name for t in tools}

  # Core tools
  assert "list_skills" in tool_names
  assert "load_skill" in tool_names
  assert "load_skill_resource" in tool_names
  assert "execute_shell" in tool_names

  # Skill 1 tools
  assert "my_custom_tool" in tool_names
  assert "my_func" in tool_names

  # Skill 2 tools
  assert "skill2_tool" in tool_names

  # Shared tool (should only appear once)
  assert "shared_tool" in tool_names
  assert len([t for t in tools if t.name == "shared_tool"]) == 1

  # Prefixed toolset tool
  assert "prefixed_mock_tool" in tool_names

  # Check specific tool resolution details
  my_func_tool = next(t for t in tools if t.name == "my_func")
  assert isinstance(my_func_tool, skill_toolset.FunctionTool)
  assert my_func_tool.description == "My function description."


@pytest.mark.asyncio
async def test_skill_toolset_resolution_error_handling(mock_skill1, caplog):
  mock_skill1.frontmatter.metadata = {
      "adk_additional_tools": ["nonexistent_tool"]
  }
  mock_skill1.name = "skill1"
  toolset = skill_toolset.SkillToolset([mock_skill1])
  ctx = _make_tool_context_with_agent()

  # Activate skill
  load_tool = skill_toolset.LoadSkillTool(toolset)
  await load_tool.run_async(args={"skill_name": "skill1"}, tool_context=ctx)

  with caplog.at_level(logging.WARNING):
    tools = await toolset.get_tools(readonly_context=ctx)

  # Should still return basic skill tools
  assert len(tools) == 4
