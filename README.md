# x-langchain

`x-langchain` 是一个生产级的 LangChain 学习与实践项目，旨在帮助开发者系统学习和掌握 LangChain 框架的核心概念与应用方法。本项目通过实际案例展示如何使用 LangChain 构建大语言模型应用，包括模型集成、工具调用、上下文管理等关键功能，为 LangChain 初学者和进阶开发者提供实践参考。

**核心价值**：开箱即用的多模型支持、插件化工具系统、完整的 TextToSQL 解决方案

**适用场景**：智能客服、数据查询助手、企业知识库问答、LLM 应用原型开发

---

## 什么是 LangChain？

- **官方定义**："LangChain is a framework for developing applications powered by large language models"
- **Gartner 描述**："LLM application development frameworks like LangChain"

简单来说，`LangChain` 是一个帮助开发者快速构建基于大语言模型（LLM）应用的开发框架。

---

## 核心特性

- **Agent 五大基础能力** - 支持模型、计划（未抽象体现）、行动（未抽象体现）、工具、记忆五大核心能力
- **多模型兼容** - 支持 DeepSeek、豆包、阿里云通义千问等主流 LLM 后端
- **工具调用（Function Calling）** - 通过声明式接口集成外部 API 与业务系统
- **TextToSQL 功能** - 支持自然语言到 SQL 的转换，包括问题重写、Schema 解析、SQL 生成、验证和执行
- **MCP 协议支持** - 集成 Model Context Protocol，支持 MCP 工具调用
- **上下文管理** - 内置对话历史管理，支持连续对话
- **安全合规** - API 密钥管理（从环境变量加载，避免硬编码）、输入/输出内容过滤
- **可观测性** - 集成结构化日志系统，便于监控和调试
- **插件化架构** - 基于装饰器的工具自动注册，支持热插拔

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **核心框架** | LangChain, LangGraph |
| **模型集成** | langchain-openai, langchain-community, dashscope |
| **配置管理** | pydantic-settings, python-dotenv |
| **工具库** | duckduckgo-search, sqlalchemy, pymysql |
| **MCP 协议** | langchain-mcp-adapters |
| **日志系统** | loguru |
| **包管理** | uv |
| **部署** | Docker |

---

## 项目结构

```
x-langchain/
├── src/                                # 源代码目录
│   ├── main.py                         # 项目主入口，CLI 交互接口
│   ├── __init__.py                     # 包初始化
│   ├── core/                           # 核心模块
│   │   ├── __init__.py               # 核心模块导出
│   │   ├── config.py                  # 配置管理（pydantic-settings）
│   │   ├── logger.py                  # 日志系统（loguru）
│   │   ├── container.py              # 依赖注入容器
│   │   ├── middleware.py             # 中间件
│   │   └── exceptions.py             # 自定义异常
│   ├── constants/                      # 常量模块
│   │   ├── __init__.py               # 常量导出
│   │   ├── base.py                   # 基础常量
│   │   ├── develop.py                 # 开发相关常量
│   │   └── streaming_modes.py         # 流式传输模式
│   ├── llms/                          # LLM 模块（模型提供者）
│   │   ├── __init__.py               # 模块导出
│   │   └── providers.py               # 多模型提供者（DeepSeek/豆包/通义千问/Mock）
│   ├── memories/                        # Memories 模块（记忆管理）
│   │   ├── __init__.py               # 模块导出
│   │   └── memory.py                 # 对话记忆（基于 LangChain）
│   ├── agent/                        # Agent 模块（核心）
│   │   ├── __init__.py               # 模块导出
│   │   └── lc_agent.py               # LangChain Agent 实现（基于 LangGraph）
│   ├── tools/                        # Tools 模块（工具系统）
│   │   ├── __init__.py               # 工具模块导出
│   │   ├── registry.py               # 工具注册表
│   │   ├── weather_tool.py           # 天气查询
│   │   ├── calendar_tool.py          # 日历查询
│   │   ├── web_tool.py               # 网络搜索
│   │   ├── exchange_rate_tool.py     # 汇率查询
│   │   ├── qiuchi_mcp/               # 秋池 MCP 工具包
│   │   └── text_to_sql/              # TextToSQL 工具链
│   │       ├── __init__.py
│   │       ├── question_rewrite_tool.py    # 问题重写
│   │       ├── get_schema_tool.py         # Schema 解析
│   │       ├── generate_sql_tool.py       # SQL 生成
│   │       ├── validate_sql_tool.py       # SQL 验证
│   │       ├── execute_sql_tool.py        # SQL 执行
│   │       └── convert_to_natural_language_tool.py  # 结果转换
│   └── infras/                        # 基础设施层
│       └── mysql/                    # MySQL 数据库模块
│           ├── __init__.py
│           ├── models.py              # ORM 模型
│           ├── mysql.py               # 数据库连接
│           └── operations.py          # 数据库操作
├── tests/                             # 测试模块
├── docs/                              # 文档目录
├── examples/                          # 示例代码
├── logs/                              # 日志目录
├── .env.example                       # 环境变量配置示例
├── pyproject.toml                     # 项目元数据和依赖
├── Dockerfile                         # Docker 构建文件
└── README.md                          # 项目文档
```

> **Note**: Agent 的"计划"和"行动"能力由 LangChain/LangGraph 的 ReAct Agent 统一提供，已集成在 `agent/lc_agent.py` 中，不作为独立模块存在。

---

## 系统架构

### 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          Agent (协调器)                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│  │    LLM     │   Memory    │  ReAct     │   Tools    │    │
│  │  (大脑)    │   (记忆)    │ (推理执行) │   (执行)   │    │
│  └─────────────┴─────────────┴─────────────┴─────────────┘    │
└─────────────────────────────────────────────────────────────────┘

技术实现：基于 LangChain / LangGraph 构建，ReAct 范式统一推理与执行
```

### 组件职责

| 组件 | 目录 | 职责 |
|------|------|------|
| LLM | `llms/` | 统一封装多种模型提供者（DeepSeek/豆包/通义千问/Mock） |
| Memory | `memories/` | 对话历史记忆（基于 LangChain ChatMessageHistory） |
| ReAct Agent | `agent/` | 推理-行动循环，基于 LangGraph 实现 |
| Tools | `tools/` | 插件化工具系统（天气/搜索/数据库/MCP） |

### 分层架构图

```mermaid
graph TB
    subgraph 用户层
        CLI[命令行接口<br/>main.py]
    end

    subgraph 应用层
        AG[Agent 实例<br/>lc_agent.py]
    end

    subgraph 核心组件层
        CFG[配置管理<br/>config.py]
        LOG[日志系统<br/>logger.py]
        CTN[依赖注入容器<br/>container.py]
    end

    subgraph 记忆层
        MM[对话记忆<br/>memories/memory.py]
    end

    subgraph 模型层
        MF[模型提供者<br/>providers.py]
        DS[DeepSeek]
        DJ[豆包]
        TY[通义千问]
        MK[Mock 模型]
    end

    subgraph 工具层
        subgraph 本地工具
            WT[天气工具]
            CT[日历工具]
            WS[网络搜索]
            ER[汇率查询]
        end
        subgraph MCP 工具
            WMCP[天气 MCP]
            QMCP[秋池 MCP]
        end
        subgraph TextToSQL
            QR[问题重写]
            GS[获取 Schema]
            SG[生成 SQL]
            VS[验证 SQL]
            ES[执行 SQL]
            CN[结果转换]
        end
    end

    subgraph 存储层
        DB[(MySQL 数据库<br/>infras/mysql)]
    end

    CLI --> AG
    AG --> MM
    AG --> MF
    AG --> CFG & LOG & CTN
    MM --> CFG
    MF --> DS & DJ & TY & MK

    AG -.-> QR & GS & SG & VS & ES & CN
    GS -.-> DB
    ES -.-> DB
```

> **说明**：
> - 核心组件层：配置、日志、依赖注入容器等基础设施支撑
> - 记忆层：对话历史管理（基于 LangChain ChatMessageHistory）
> - 工具层：TextToSQL 工具链内，问题重写(QR) → 获取Schema → 生成SQL → 验证 → 执行 → 结果转换
> - 存储层：MySQL 数据库，Schema 解析和 SQL 执行依赖此层

### 核心业务流程图（ReAct 执行循环）

```mermaid
flowchart TD
    subgraph 入口
        U[用户输入]
        CLI[命令行接口]
    end

    subgraph 执行循环
        subgraph 循环头
            MM[加载历史上下文]
            M1[拼接上下文]
        end

        M2[推理思考 Think]
        M3{是否需要工具?}

        subgraph 分支判断
            M4[直接回复]
            M5[执行工具]
        end

        M6[观测结果 Observe]
        M7{继续循环?}

        subgraph 工具执行
            T[工具层]
            DB[(MySQL)]
            T --> DB
        end
    end

    M8[输出最终答案]

    U --> CLI
    CLI --> MM
    MM --> M1
    M1 --> M2
    M2 --> M3

    M3 -->|否| M4
    M4 --> M8

    M3 -->|是| M5
    M5 --> T
    T --> M6
    M6 --> M7

    M7 -->|是| M2
    M7 -->|否| M8

    M8 --> CLI
    CLI --> U

    style M2 fill:#e1f5fe
    style M5 fill:#fff3e0
    style M6 fill:#e8f5e9
    style M8 fill:#f3e5f5
```

> **技术栈**：基于 LangGraph 实现状态机管理，支持多轮 ReAct 循环

> **说明**：
> - Think：LLM 推理决策，判断是否需要调用工具
> - Act：执行工具调用（如 TextToSQL 查询）
> - Observe：获取工具结果，反馈给 LLM 继续推理

```mermaid
sequenceDiagram
    participant U as 用户
    participant CLI as CLI
    participant MM as Memory
    participant G as LangGraph
    participant L as LLM
    participant T as Tools
    participant DB as MySQL

    U->>CLI: 用户输入
    CLI->>MM: 加载历史上下文
    MM-->>CLI: 返回历史消息
    CLI->>G: 创建状态
    G->>L: Think: 推理决策
    L-->>G: 返回推理结果
    alt 不需要工具
        G-->>CLI: 直接回复
        CLI-->>U: 返回答案
    else 需要工具
        G->>T: Act: 执行工具
        T->>DB: 查询/写入数据
        DB-->>T: 返回结果
        T-->>G: Observe: 返回结果
        G->>MM: 更新记忆
        G->>L: 继续推理
        loop ReAct 循环
            L-->>G: 思考
            alt 继续调用工具
                G->>T: Act
                T-->>G: Observe
            end
        end
        G-->>CLI: 输出最终答案
        CLI-->>U: 返回答案
    end
```

### 模块依赖关系

```mermaid
graph LR
    subgraph 入口
        M[main.py]
    end

    subgraph 核心组件
        CC[core<br/>config/logger/container]
    end

    subgraph 核心模块
        AG[agent<br/>lc_agent.py]
        LL[llms]
        MM[memories]
    end

    subgraph 存储层
        DB[(MySQL)]
    end

    M --> AG
    M --> CC
    AG --> LL
    AG --> MM
    AG --> CC
    LL --> CC
    MM --> CC
```

> **说明**: Agent 的推理和行动调度由 LangChain/LangGraph 统一实现，不再需要独立的 Planning/Action 模块。

---

## 快速开始

### 环境要求

| 环境 | 要求 |
|------|------|
| **Windows** | Python 3.11+, 推荐使用 PowerShell 或 Git Bash |
| **Linux/macOS** | Python 3.11+, 任意 Shell |

> 推荐使用 [`uv`](https://github.com/astral-sh/uv) 作为包管理器，亦可兼容 `pip`

### 项目克隆

```bash
git clone https://github.com/chain-engine/x-langchain.git
cd x-langchain
```

### 依赖安装

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 配置文件创建

```bash
# 复制配置模板
cp .env.example .env
```

编辑 `.env` 文件，配置必要的 API 密钥：

```env
# DeepSeek API（推荐）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL_NAME=deepseek-chat

# 或豆包 API
DOUBAO_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DOUBAO_API_BASE=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL_NAME=ep-xxxxxxxxxxxxxx

# 或阿里云通义千问
ALIYUN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALIYUN_MODEL_NAME=qwen-plus

# 数据库配置（TextToSQL 功能需要）
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_database
```

### 本地启动

```bash
# 使用默认模型（DeepSeek）
uv run src/main.py

# 或使用已安装的命令行入口
uv run x-langchain

# 通过环境变量指定模型
MODEL_NAME=deepseek uv run src/main.py
MODEL_NAME=doubao uv run src/main.py
MODEL_NAME=tongyi uv run src/main.py
```

### Docker 启动

```bash
# 构建镜像
docker build -t x-langchain:latest .

# 运行容器（挂载配置和日志目录）
docker run -it --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/logs:/app/logs \
  x-langchain:latest

# Windows PowerShell
docker run -it --rm `
  -v ${PWD}/.env:/app/.env:ro `
  -v ${PWD}/logs:/app/logs `
  x-langchain:latest
```

### 常用命令

```bash
# 运行测试
uv run python -m pytest

# 运行特定测试模块
uv run python -m pytest tests/test_providers.py -v

# 代码格式化（如果安装了 ruff）
uv run ruff format .

# 类型检查（如果安装了 pyright）
uv run pyright
```

---

## 使用方法

### 交互式对话模式

启动程序后，进入交互式对话模式：

```bash
$ uv run src/main.py

==================================================
欢迎使用智能助手！输入 'exit'、'quit' 或 '退出' 结束对话
==================================================

你: 上海天气怎么样？

2026-03-05 10:00:00,123 - INFO - 查询结果:
上海今天多云，气温 18°C，东风 3级，湿度 65%，空气质量良好。

你: 帮我查询数据库中的用户数量

2026-03-05 10:01:00,456 - INFO - 查询结果:
根据数据库查询，当前系统中有 150 个用户。

你: exit

感谢使用，再见！
```

### 模型选择

通过环境变量 `MODEL_NAME` 指定模型：

| 模型 | 环境变量 | 说明 |
|------|---------|------|
| DeepSeek | `MODEL_NAME=deepseek` | 默认模型，性价比高 |
| 豆包 | `MODEL_NAME=doubao` | 字节跳动出品 |
| 通义千问 | `MODEL_NAME=tongyi` | 阿里云出品 |

```bash
# 使用 DeepSeek（默认）
uv run src/main.py

# 使用豆包
MODEL_NAME=doubao uv run src/main.py

# 使用通义千问
MODEL_NAME=tongyi uv run src/main.py
```

---

## 插件化工具系统

x-langchain 提供插件化的工具管理系统，开发者可轻松添加新工具，无需修改核心代码。

### 核心特性

- **自动注册** - 使用 `@register_tool` 装饰器自动注册工具
- **工具发现** - 自动扫描 `tools/` 目录，发现新工具
- **类别管理** - 按类别组织工具，便于管理和过滤
- **向后兼容** - 完全兼容现有工具代码

### 快速开始

创建新工具只需三步：

```python
# 1. 在 tools/ 目录下创建新文件
# tools/my_tool.py

# 2. 使用装饰器注册工具
from tools.registry import register_tool

@register_tool(name="my_tool", category="custom", description="我的工具")
class MyTool:
    def __init__(self):
        self.name = "my_tool"
        self.description = "我的自定义工具"

    def run(self, param: str) -> str:
        return f"处理参数: {param}"

# 3. 完成！工具会在模块导入时自动注册
```

### 工具查询

```python
from tools import ToolRegistry

# 检查工具是否存在
if ToolRegistry.contains("my_tool"):
    tool = ToolRegistry.get("my_tool")

# 获取所有工具
all_tools = ToolRegistry.get_all()

# 按类别获取工具
custom_tools = ToolRegistry.get_all(category="custom")

# 获取统计信息
stats = ToolRegistry.get_stats()
print(f"总工具数: {stats['total_tools']}")
```

> 详细文档请参考 [插件开发指南](docs/插件开发指南.md)

---

## 存储配置

### 数据库配置（TextToSQL）

项目支持 MySQL 数据库连接，用于 TextToSQL 功能：

```env
# 分项配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_database

# 或使用完整 URL（优先级更高）
DB_URL=mysql+pymysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
```

---

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangChain 中文文档](https://langchain-doc.cn/v1/python/langchain/overview.html)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Python 官方文档](https://docs.python.org/3/)
- [uv 包管理器](https://github.com/astral-sh/uv)
- [DeepSeek 官方文档](https://platform.deepseek.com/docs/api)
- [豆包官方文档](https://www.doubao.com/)
- [阿里云通义千问](https://help.aliyun.com/product/1081203.html)

---

## 联系方式

| 项目 | 信息 |
|------|------|
| **作者** | John Young（夜雨诗来） |
| **邮箱** | john.young@foxmail.com |
| **Gitee** | https://gitee.com/yeyushilai |
| **GitHub** | https://github.com/yeyushilai |
| **项目地址** | https://github.com/chain-engine/x-langchain |
