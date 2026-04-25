# Shell 方式执行 Skill 脚本 - 实施方案

> **给 agentic worker 的说明：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务执行此方案。步骤使用复选框 (`- [x]`) 标记进度。

**目标：** 用借鉴自 OpenCode 的、更简洁的 shell 执行模型替换自解压包装器方案（`run_skill_script` + `_SkillScriptCodeExecutor`）。Skill 资源被物化到磁盘后，模型使用 `ShellTool` 执行脚本。

**架构：** 当 skill 被 `load_skill` 加载时，其资源（references、assets、scripts）被写入临时目录。模型拿到基础目录路径和文件列表，然后通过新的 `ShellTool` 执行 shell 命令（如 `python scripts/foo.py`）。这消除了复杂的自解压包装器模式，降低了模型出错概率。

**技术栈：** Python 3.10+, asyncio subprocess, tempfile, pytest

---

## 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 执行模型 | Shell 方式（`subprocess`） | OpenCode 验证过可行；模型构造命令更自然 |
| 资源物化 | `load_skill` 时写入临时目录 | Skill 可能来自内存/GCS，需要磁盘路径才能执行 |
| 工具命名 | `execute_shell`（非 `execute_bash`） | 跨平台兼容，Windows 也能用 |
| 向后兼容 | 弃用 `code_executor` 参数，但仍接受 | 已有用户只收到警告，不会崩溃 |
| 确认机制 | 不需要确认，直接执行 | 与 OpenCode 行为一致，简化流程 |
| 输出透传 | 实时 tee 到控制台 | 执行过程可看到日志 |

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `src/google/adk/tools/shell_tool.py` | 新的 `ShellTool` - shell 命令执行 |
| 修改 | `src/google/adk/tools/skill_toolset.py` | 用 `ShellTool` 替换 `RunSkillScriptTool`，更新 `LoadSkillTool`，更新系统提示词 |
| 修改 | `tests/unittests/tools/test_skill_toolset.py` | 更新和新增测试 |
| 新建 | `tests/unittests/tools/test_shell_tool.py` | 新 `ShellTool` 的测试 |
| 修改 | `contributing/samples/skills_agent/agent.py` | 移除 `code_executor` 依赖 |

---

### Task 1: 创建 ShellTool

**文件：**
- 新建: `src/google/adk/tools/shell_tool.py`

- [x] **Step 1: 创建 ShellTool 类**

创建 `src/google/adk/tools/shell_tool.py`，包含：

1. **`ShellToolPolicy` 数据类** - 策略配置（允许的命令前缀、超时等）
2. **`_validate_command()`** - 命令验证
3. **`ShellTool` 类**（继承 `BaseTool`）:
   - 工具名: `"execute_shell"`
   - 参数: `command`（必填）, `workdir`（可选）, `timeout`（可选）
   - 跨平台支持：
     - Unix: `asyncio.create_subprocess_shell` + `start_new_session=True` + `os.killpg` 杀进程组
     - Windows: `subprocess.Popen` via `asyncio.to_thread`（避免 Windows 事件循环不支持 asyncio 子进程的 `NotImplementedError`）
   - 无需确认，直接执行
   - 实时输出透传到控制台（tee 模式）

- [x] **Step 2: 创建测试**

创建 `tests/unittests/tools/test_shell_tool.py`，10 个测试覆盖：echo 命令、workdir、超时、非零退出码、stderr、Python 脚本、空命令、声明验证、自定义超时、策略阻止命令。

- [x] **Step 3: 运行测试确认通过**

`python -m pytest tests/unittests/tools/test_shell_tool.py -v` → 10 passed

- [x] **Step 4: 提交**

`git commit -m "feat: add ShellTool for cross-platform shell command execution"`

---

### Task 2: 修改 LoadSkillTool 以物化资源

**文件：**
- 修改: `src/google/adk/tools/skill_toolset.py`

核心变更：当 skill 被加载时，将其资源写入临时目录，返回基础路径和文件列表。

- [x] **Step 1: 添加 `_skill_dirs` 字典和 `_materialize_skill` 方法**

在 `SkillToolset` 中：
- `__init__` 添加 `self._skill_dirs: dict[str, str] = {}`
- 新增 `_materialize_skill(skill)` 方法：创建临时目录，写入 references/assets/scripts，返回 `(base_directory, files)`

- [x] **Step 2: 修改 `LoadSkillTool.run_async`**

调用 `_materialize_skill(skill)` 后在返回值中加入 `"base_directory"` 和 `"files"` 字段。

- [x] **Step 3: 新增测试**

`test_load_skill_materializes_resources` 和 `test_load_skill_files_exist_on_disk`

- [x] **Step 4: 提交**

`git commit -m "feat: add resource materialization to LoadSkillTool"`

---

### Task 3: 用 ShellTool 替换 RunSkillScriptTool

**文件：**
- 修改: `src/google/adk/tools/skill_toolset.py`

这是核心重构任务。

- [x] **Step 1: 删除旧代码**

- 删除 `_SkillScriptCodeExecutor` 类（~270 行自解压包装器代码生成器）
- 删除 `RunSkillScriptTool` 类（~150 行）
- 删除 `_MAX_SKILL_PAYLOAD_BYTES` 常量
- 移除不再需要的 `json`、`CodeExecutionInput` 等导入

- [x] **Step 2: 更新系统提示词**

`_DEFAULT_SKILL_SYSTEM_INSTRUCTION` 改为引用 `execute_shell`，说明：
- `load_skill` 返回 `base_directory` 路径和 `files` 列表
- 用 `execute_shell` + `workdir` 执行脚本
- Python 脚本示例：`execute_shell(command="python scripts/run.py --arg value", workdir="<base_directory>")`
- Shell 脚本示例：`execute_shell(command="bash scripts/setup.sh", workdir="<base_directory>")`

- [x] **Step 3: 更新 SkillToolset.__init__**

- 新增 `shell_policy: Optional[ShellToolPolicy] = None` 参数
- `code_executor` 参数标记为弃用，传入时发出 `DeprecationWarning`
- 工具列表中用 `ShellTool(policy=shell_policy)` 替换 `RunSkillScriptTool(self)`

- [x] **Step 4: 更新测试断言**

`test_get_tools` 改为检查 `ShellTool`；`test_skill_toolset_dynamic_tool_resolution` 中 `run_skill_script` → `execute_shell`

- [x] **Step 5: 提交**

`git commit -m "refactor: replace run_skill_script with shell-based execution"`

**效果：净减 1427 行代码（-454 行 skill_toolset.py，-1010 行测试）**

---

### Task 4: 更新已有测试

**文件：**
- 修改: `tests/unittests/tools/test_skill_toolset.py`

- [x] **Step 1: 删除过时测试**

删除所有 `RunSkillScriptTool` 和 `_SkillScriptCodeExecutor` 相关的测试（~40 个）：
- `test_execute_script_*`（全部）
- `test_shell_json_envelope_*`（全部）
- `test_execute_script_input_files_packaged`
- `test_execute_script_empty_files_mounted`
- `test_execute_script_binary_content_packaged`
- `test_system_instruction_references_run_skill_script`
- 集成测试（`test_integration_python_*`、`test_integration_shell_*`）
- 辅助函数（`_make_mock_executor`、`_make_skill_with_script`、`_make_real_executor_toolset`）

- [x] **Step 2: 新增端到端测试**

`test_skill_execution_via_shell` - 完整流程：load_skill → 物化资源 → execute_shell 执行脚本

- [x] **Step 3: 更新系统提示词测试**

`test_system_instruction_references_execute_shell` - 检查 `execute_shell` 存在、`run_skill_script` 不存在

- [x] **Step 4: 运行测试**

31 passed, 0 failed

---

### Task 5: 更新示例 Agent

**文件：**
- 修改: `contributing/samples/skills_agent/agent.py`

- [x] **Step 1: 移除 code_executor**

删除 `UnsafeLocalCodeExecutor` 导入和 `code_executor=UnsafeLocalCodeExecutor()` 参数。SkillToolset 不再需要 code_executor。

注：`skills_agent_gcs/agent.py` 的 code_executor 在 Agent 级别而非 SkillToolset 级别，无需修改。`local_environment_skill/agent.py` 使用 EnvironmentToolset，不受影响。

- [x] **Step 2: 提交**

`git commit -m "refactor: remove UnsafeLocalCodeExecutor from skills_agent sample"`

---

### Task 6: 检查导出和清理

- [x] **Step 1: 搜索残留引用**

`grep -rn "RunSkillScriptTool\|_SkillScriptCodeExecutor" src/google/adk/` → 无残留

- [x] **Step 2: 确认导出干净**

`src/google/adk/tools/__init__.py` 和 `src/google/adk/skills/__init__.py` 无需修改，所有导入路径通过子模块直接引用。

---

### Task 7: 最终验证

- [x] **运行全部测试**

`python -m pytest tests/unittests/tools/test_skill_toolset.py tests/unittests/tools/test_shell_tool.py -v` → **41 passed, 0 failed**

---

## 实施过程中的调整记录

以下是原始方案中未涉及、在实施过程中发现并修复的问题：

### 调整 1: 移除确认机制

**问题：** 初始实现中 `ShellToolPolicy` 有 `confirm_required: bool = True`，每次执行 shell 命令都需要用户确认，严重影响使用体验。

**修复：** 删除 `confirm_required` 字段和整个确认逻辑。`run_async` 直接执行命令，不再调用 `tool_context.request_confirmation()`。

### 调整 2: Windows 兼容 - asyncio 子进程不支持

**问题：** 在 Windows 上调用 `asyncio.create_subprocess_shell()` 抛出 `NotImplementedError`，因为 Windows 的 SelectorEventLoop 不支持子进程。Python 3.10+ 默认使用 ProactorEventLoop，但某些场景（如 ADK 框架内部）会覆盖事件循环。

**修复：** 按平台分治：
- **Unix**：保持 `asyncio.create_subprocess_shell` + `start_new_session=True` + `os.killpg` 进程组管理
- **Windows**：用 `subprocess.Popen` via `asyncio.to_thread` 在线程中执行，避免事件循环限制

```python
if sys.platform == "win32":
    return await asyncio.to_thread(_run_subprocess_win32, ...)
return await _run_subprocess_unix(...)
```

### 调整 3: 实时输出透传到控制台

**问题：** `capture_output=True` / `PIPE` 截断了 stdout/stderr 流向控制台，执行过程中看不到任何输出。

**修复：** 实现 tee 模式——边捕获边透传：
- **Unix**：`_stream_reader()` 协程逐块（4096 字节）读取 stdout/stderr，每块同时写入 `sys.stdout`/`sys.stderr` 和收集缓冲区
- **Windows**：`subprocess.Popen.communicate()` 是原子等待（无法逐行流式），完成后将完整输出写入 `sys.stdout`/`sys.stderr`

---

## 提交记录

| 提交 | 说明 | 变更量 |
|------|------|--------|
| `cf832a5` | 添加资源物化到 `LoadSkillTool` | 新增 `_materialize_skill()` 写入临时目录 |
| `6b2826d` | 新增 `ShellTool` 跨平台 shell 执行 | 新文件 `shell_tool.py` + 10 个测试 |
| `0af265c` | 用 shell 方式替换 `run_skill_script` | **净减 1427 行** |
| `cc8ed39` | 示例 agent 移除 `code_executor` | 不再需要 UnsafeLocalCodeExecutor |
