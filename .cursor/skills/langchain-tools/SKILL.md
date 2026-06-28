---
name: langchain-tools
description: Assist with creating, updating, and wiring LangChain tools in the tools/ package of this project. Use when adding new tools, modifying existing tools, or integrating tools into agents, models, and tests.
---


## 开发 LangChain 工具的约定

1. **文件位置**
   - 通用工具：放在 `tools/` 目录（如 `weather_tool.py`、`web_tool.py`）。
   - 复杂链路：可以使用子包（如现有的 `tools/text_to_sql/`）。

2. **命名与导出**
   - 单个工具通常以 `<name>_tool` 结尾，例如：`weather_search_tool`。
   - 在对应模块中定义工具对象（例如 LangChain 的 `Tool` 或兼容接口）。
   - 在 `tools/__init__.py` 中显式导出新工具，方便 Agent 工厂集中引用。

3. **输入输出与类型**
   - 明确声明工具函数/调用签名，必要时使用类型注解帮助 IDE 和静态检查。
   - 返回结果应尽量是**结构化数据**（dict/list），由上层 Agent 控制最终回答格式。
   - 需要与外部系统交互时，封装好异常处理与日志（使用本项目的 `core.logger`）。

4. **LangChain 官方用法**
   - 优先使用 LangChain 官方推荐的工具封装方式（如 `tool` 装饰器或 `StructuredTool` 等，视项目当前 LangChain 版本而定）。
   - 保持工具是**无状态或轻状态**的函数式接口，避免在工具内部创建重资源对象（数据库连接、大模型实例等），这类对象应在更外层管理。

## 将工具接入 Agent

1. **工具注册机制**
   - **Function Calling 工具**：使用 `discover_function_calling_tools()` 自动发现并注册，无需手动装饰
   - **MCP 工具**：通过 `register_mcp_tools()` 显式注册到 ToolRegistry
   - 新工具只需继承 `BaseTool` 或使用 `@tool` 装饰器定义，系统会自动识别并注册

2. **获取默认工具列表**
   - 使用 `AgentFactory.get_default_tools()` 或 `AgentFactory.get_default_tools_async()`

2. **系统提示与结构化输出**
   - 如新工具的结果需要特定字段（例如增加一种新查询类型），在 `AgentFactory` 或相关 system prompt 中补充约定字段说明。
   - 避免在工具内部硬编码自然语言模板，更多地依赖上层 Prompt/模型来组织回答。

## 测试工具

1. **测试位置**
   - 在 `tests/` 目录中为每个重要工具添加对应的测试文件，如：`test_weather_tool.py`。
   - 对于复杂链路（如 `text_to_sql`），为每个步骤或整体流程添加测试用例。

2. **测试重点**
   - 输入输出的正确性与边界条件（空输入、格式错误、外部服务异常等）。
   - 对外部服务调用进行适当的 mock 或隔离，避免测试依赖不稳定的网络环境。

## 工作流程（建议）

1. 与用户确认要实现/修改的工具功能和数据结构。
2. 在 `tools/` 中创建或修改对应模块，实现工具逻辑并添加类型注解与日志：
   - 继承 `BaseTool` 并定义 `name`、`description`、`args_schema`
   - 或使用 LangChain 的 `@tool` 装饰器
3. 工具会通过 `discover_function_calling_tools()` 自动注册到 ToolRegistry（无需手动注册）
4. 在 `tests/` 目录添加或更新测试，确保核心行为被覆盖。
5. 运行测试（例如使用 `uv run python -m pytest`）验证改动。

