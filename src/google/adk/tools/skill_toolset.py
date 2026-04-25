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

# pylint: disable=g-import-not-at-top,protected-access

"""Toolset for discovering, viewing, and executing agent skills."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import tempfile
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from typing_extensions import override

from ..agents.readonly_context import ReadonlyContext
from ..code_executors.base_code_executor import BaseCodeExecutor
from ..features import experimental
from ..features import FeatureName
from ..skills import models
from ..skills import prompt
from .base_tool import BaseTool
from .base_toolset import BaseToolset
from .function_tool import FunctionTool
from .shell_tool import ShellTool
from .shell_tool import ShellToolPolicy
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.llm_agent import ToolUnion
  from ..models.llm_request import LlmRequest

logger = logging.getLogger("google_adk." + __name__)

# Kept for backward compat with deprecated script_timeout param.
_DEFAULT_SCRIPT_TIMEOUT = 300

# Message used for the "Content Injection" pattern.
_BINARY_FILE_DETECTED_MSG = (
    "Binary file detected. The content has been injected into the"
    " conversation history for you to analyze."
)

_DEFAULT_SKILL_SYSTEM_INSTRUCTION = (
    "You can use specialized 'skills' to help you with complex tasks. "
    "You MUST use the skill tools to interact with these skills.\n\n"
    "Skills are folders of instructions and resources that extend your "
    "capabilities for specialized tasks. Each skill folder contains:\n"
    "- **SKILL.md** (required): The main instruction file with skill "
    "metadata and detailed markdown instructions.\n"
    "- **references/** (Optional): Additional documentation or examples for "
    "skill usage.\n"
    "- **assets/** (Optional): Templates, scripts or other resources used by "
    "the skill.\n"
    "- **scripts/** (Optional): Executable scripts (.py, .sh, .bash).\n\n"
    "This is very important:\n\n"
    "1. If a skill seems relevant to the current user query, you MUST use "
    'the `load_skill` tool with `skill_name="<SKILL_NAME>"` to read '
    "its full instructions before proceeding.\n"
    "2. Once you have read the instructions, follow them exactly as "
    "documented before replying to the user.\n"
    "3. The `load_skill` tool returns a `base_directory` path and `files` "
    "list. Scripts are located at `<base_directory>/scripts/<filename>`.\n"
    "4. To execute a script from a skill, use the `execute_shell` tool "
    "with `workdir` set to the skill's `base_directory`:\n"
    "   - Python: `execute_shell(command=\"python scripts/run.py --arg value\", "
    "workdir=\"<base_directory>\")`\n"
    "   - Shell: `execute_shell(command=\"bash scripts/setup.sh\", "
    "workdir=\"<base_directory>\")`\n"
    "5. The `load_skill_resource` tool is for viewing files within a "
    "skill's directory. Do NOT use other tools to access these files.\n"
)


@experimental(FeatureName.SKILL_TOOLSET)
class ListSkillsTool(BaseTool):
  """Tool to list all available skills."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="list_skills",
        description=(
            "Lists all available skills with their names and descriptions."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skills = self._toolset._list_skills()
    return prompt.format_skills_as_xml(skills)


@experimental(FeatureName.SKILL_TOOLSET)
class LoadSkillTool(BaseTool):
  """Tool to load a skill's instructions."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="load_skill",
        description="Loads the SKILL.md instructions for a given skill.",
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                },
            },
            "required": ["skill_name"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("skill_name")
    if not skill_name:
      return {
          "error": "Argument 'skill_name' is required.",
          "error_code": "INVALID_ARGUMENTS",
      }

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    # Record skill activation in agent state for tool resolution.
    agent_name = tool_context.agent_name
    state_key = f"_adk_activated_skill_{agent_name}"

    activated_skills = list(tool_context.state.get(state_key) or [])
    if skill_name not in activated_skills:
      activated_skills.append(skill_name)
      tool_context.state[state_key] = activated_skills

    # Materialize resources to temp dir
    base_directory, files = self._toolset._materialize_skill(skill)

    return {
        "skill_name": skill_name,
        "instructions": skill.instructions,
        "frontmatter": skill.frontmatter.model_dump(),
        "base_directory": base_directory,
        "files": files,
    }


@experimental(FeatureName.SKILL_TOOLSET)
class LoadSkillResourceTool(BaseTool):
  """Tool to load resources (references, assets, or scripts) from a skill."""

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name="load_skill_resource",
        description=(
            "Loads a resource file (from references/, assets/, or"
            " scripts/) from within a skill."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "The relative path to the resource (e.g.,"
                        " 'references/my_doc.md', 'assets/template.txt',"
                        " or 'scripts/setup.sh')."
                    ),
                },
            },
            "required": ["skill_name", "file_path"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name = args.get("skill_name")
    file_path = args.get("file_path")

    errors = []
    if not skill_name:
      errors.append("Argument 'skill_name' is required.")
    if not file_path:
      errors.append("Argument 'file_path' is required.")

    if errors:
      return {
          "error": "\n".join(errors),
          "error_code": "INVALID_ARGUMENTS",
      }

    skill = self._toolset._get_skill(skill_name)
    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    content = None
    if file_path.startswith("references/"):
      ref_name = file_path[len("references/"):]
      content = skill.resources.get_reference(ref_name)
    elif file_path.startswith("assets/"):
      asset_name = file_path[len("assets/"):]
      content = skill.resources.get_asset(asset_name)
    elif file_path.startswith("scripts/"):
      script_name = file_path[len("scripts/"):]
      script = skill.resources.get_script(script_name)
      if script is not None:
        content = script.src
    else:
      return {
          "error": (
              "Path must start with 'references/', 'assets/', or 'scripts/'."
          ),
          "error_code": "INVALID_RESOURCE_PATH",
      }

    if content is None:
      return {
          "error": f"Resource '{file_path}' not found in skill '{skill_name}'.",
          "error_code": "RESOURCE_NOT_FOUND",
      }

    if isinstance(content, bytes):
      return {
          "skill_name": skill_name,
          "file_path": file_path,
          "status": _BINARY_FILE_DETECTED_MSG,
      }

    return {
        "skill_name": skill_name,
        "file_path": file_path,
        "content": content,
    }

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: Any
  ) -> None:
    """Injects binary content into the LLM request if the model viewed it."""
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    if not llm_request.contents:
      return

    # Check for LoadSkillResource calls on binary files in the last turn
    for part in llm_request.contents[-1].parts:
      if not part.function_response or part.function_response.name != self.name:
        continue

      response = part.function_response.response or {}
      if response.get("status") != _BINARY_FILE_DETECTED_MSG:
        continue

      skill_name = response.get("skill_name")
      file_path = response.get("file_path")
      if not skill_name or not file_path:
        continue

      skill = self._toolset._get_skill(skill_name)
      if not skill:
        continue

      # Find the binary content
      content = None
      if file_path.startswith("references/"):
        ref_name = file_path[len("references/"):]
        content = skill.resources.get_reference(ref_name)
      elif file_path.startswith("assets/"):
        asset_name = file_path[len("assets/"):]
        content = skill.resources.get_asset(asset_name)

      if not isinstance(content, bytes):
        continue

      # Determine mime type based on extension
      mime_type, _ = mimetypes.guess_type(file_path)
      if not mime_type:
        mime_type = "application/octet-stream"

      # Append binary content to llm_request
      llm_request.contents.append(
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=f"The content of binary file '{file_path}' is:"
                  ),
                  types.Part(
                      inline_data=types.Blob(
                          data=content,
                          mime_type=mime_type,
                      )
                  ),
              ],
          )
      )


@experimental(FeatureName.SKILL_TOOLSET)
class SkillToolset(BaseToolset):
  """A toolset for managing and interacting with agent skills."""

  def __init__(
      self,
      skills: list[models.Skill],
      *,
      code_executor: Optional[BaseCodeExecutor] = None,
      script_timeout: int = _DEFAULT_SCRIPT_TIMEOUT,
      additional_tools: list[ToolUnion] | None = None,
      shell_policy: Optional[ShellToolPolicy] = None,
  ):
    """Initializes the SkillToolset.

    Args:
      skills: List of skills to register.
      code_executor: Deprecated. Skill script execution now uses ShellTool.
        This parameter will be removed in a future release.
      script_timeout: Deprecated. Timeout is now configured via ShellToolPolicy.
        This parameter will be removed in a future release.
      additional_tools: Optional list of additional tools that skills can
        reference via the ``adk_additional_tools`` frontmatter metadata key.
      shell_policy: Optional policy for the ShellTool used to execute skill
        scripts.
    """
    super().__init__()

    if code_executor is not None:
      warnings.warn(
          "code_executor is deprecated. Skill script execution now uses"
          " ShellTool. The code_executor parameter will be removed in a"
          " future release.",
          DeprecationWarning,
          stacklevel=2,
      )

    # Check for duplicate skill names
    seen: set[str] = set()
    for skill in skills:
      if skill.name in seen:
        raise ValueError(f"Duplicate skill name '{skill.name}'.")
      seen.add(skill.name)

    self._skills = {skill.name: skill for skill in skills}
    self._use_invocation_cache = False
    self._skill_dirs: dict[str, str] = {}

    self._provided_tools_by_name = {}
    self._provided_toolsets = []
    for tool_union in additional_tools or []:
      if isinstance(tool_union, BaseToolset):
        self._provided_toolsets.append(tool_union)
      elif isinstance(tool_union, BaseTool):
        self._provided_tools_by_name[tool_union.name] = tool_union
      elif callable(tool_union):
        ft = FunctionTool(tool_union)
        self._provided_tools_by_name[ft.name] = ft

    # Initialize core skill tools
    self._tools = [
        ListSkillsTool(self),
        LoadSkillTool(self),
        LoadSkillResourceTool(self),
        ShellTool(policy=shell_policy),
    ]

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns the list of tools in this toolset."""
    dynamic_tools = await self._resolve_additional_tools_from_state(
        readonly_context
    )
    return self._tools + dynamic_tools

  async def _resolve_additional_tools_from_state(
      self, readonly_context: ReadonlyContext | None
  ) -> list[BaseTool]:
    """Resolves tools listed in the "adk_additional_tools" metadata of skills."""

    if not readonly_context:
      return []

    agent_name = readonly_context.agent_name
    state_key = f"_adk_activated_skill_{agent_name}"
    activated_skills = readonly_context.state.get(state_key) or []

    if not activated_skills:
      return []

    additional_tool_names = set()
    for skill_name in activated_skills:
      skill = self._skills.get(skill_name)
      if skill:
        additional_tools = skill.frontmatter.metadata.get(
            "adk_additional_tools"
        )
        if additional_tools:
          additional_tool_names.update(additional_tools)

    if not additional_tool_names:
      return []

    # Collect all candidate tools from both individual tools and toolsets
    candidate_tools = self._provided_tools_by_name.copy()
    if self._provided_toolsets:
      ts_results = await asyncio.gather(*(
          ts.get_tools_with_prefix(readonly_context)
          for ts in self._provided_toolsets
      ))
      for ts_tools in ts_results:
        for t in ts_tools:
          candidate_tools[t.name] = t

    resolved_tools = []
    existing_tool_names = {t.name for t in self._tools}
    for name in additional_tool_names:
      if name in candidate_tools:
        tool = candidate_tools[name]
        if tool.name in existing_tool_names:
          logger.error(
              "Tool name collision: tool '%s' already exists.", tool.name
          )
          continue
        resolved_tools.append(tool)
        existing_tool_names.add(tool.name)

    return resolved_tools

  def _materialize_skill(self, skill: models.Skill) -> tuple[str, list[str]]:
    """Writes all skill resources to a temp directory.

    Returns:
        (base_directory, file_list)
    """
    if skill.name in self._skill_dirs:
      base_dir = self._skill_dirs[skill.name]
      file_list = []
      for root, _dirs, fnames in os.walk(base_dir):
        for fname in fnames:
          full = os.path.join(root, fname)
          file_list.append(os.path.relpath(full, base_dir).replace(os.sep, "/"))
      return base_dir, sorted(file_list)

    base_dir = tempfile.mkdtemp(prefix=f"adk_skill_{skill.name}_")
    self._skill_dirs[skill.name] = base_dir

    file_list: list[str] = []

    for ref_name in skill.resources.list_references():
      content = skill.resources.get_reference(ref_name)
      if content is not None:
        rel_path = f"references/{ref_name}"
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(full_path, mode) as f:
          f.write(content)
        file_list.append(rel_path)

    for asset_name in skill.resources.list_assets():
      content = skill.resources.get_asset(asset_name)
      if content is not None:
        rel_path = f"assets/{asset_name}"
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(full_path, mode) as f:
          f.write(content)
        file_list.append(rel_path)

    for scr_name in skill.resources.list_scripts():
      scr = skill.resources.get_script(scr_name)
      if scr is not None and scr.src is not None:
        rel_path = f"scripts/{scr_name}"
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "wb" if isinstance(scr.src, bytes) else "w"
        with open(full_path, mode) as f:
          f.write(scr.src)
        file_list.append(rel_path)

    return base_dir, sorted(file_list)

  def _get_skill(self, skill_name: str) -> models.Skill | None:
    """Retrieves a skill by name."""
    return self._skills.get(skill_name)

  def _list_skills(self) -> list[models.Skill]:
    """Lists all available skills."""
    return list(self._skills.values())

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request to include available skills."""
    skills = self._list_skills()
    skills_xml = prompt.format_skills_as_xml(skills)
    instructions = []
    instructions.append(_DEFAULT_SKILL_SYSTEM_INSTRUCTION)
    instructions.append(skills_xml)
    llm_request.append_instructions(instructions)


def __getattr__(name: str) -> Any:
  if name == "DEFAULT_SKILL_SYSTEM_INSTRUCTION":
    warnings.warn(
        "DEFAULT_SKILL_SYSTEM_INSTRUCTION is experimental. Its content "
        "is internal implementation and will change in minor/patch releases "
        "to tune agent performance.",
        UserWarning,
        stacklevel=2,
    )
    return _DEFAULT_SKILL_SYSTEM_INSTRUCTION
  raise AttributeError(f"module {__name__} has no attribute {name}")
