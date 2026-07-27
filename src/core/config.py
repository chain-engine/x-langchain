# -*- coding: utf-8 -*-
"""
项目统一配置入口

支持从环境变量和 YAML 配置文件读取配置。
优先级：环境变量 > YAML配置文件 > 默认配置

使用方式：
    from core.config import settings

统一管理：
- LLM Provider 配置
- Agent 运行配置
- 数据库/TextToSQL 配置
- 工具配置
- 日志配置
- 中间件配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml
from dotenv import load_dotenv

from constants.agent import AgentMode


# =============================================================================
# 默认值常量
# =============================================================================

DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_MAX_TOKENS: Final[int] = 4096
DEFAULT_TIMEOUT: Final[int] = 60
DEFAULT_MAX_ITERATIONS: Final[int] = 10
DEFAULT_TEXT_TO_SQL_MAX_ROWS: Final[int] = 100
DEFAULT_TEXT_TO_SQL_TIMEOUT: Final[int] = 30


# =============================================================================
# Agent 配置
# =============================================================================

@dataclass
class AgentConfig:
    """Agent 运行配置"""
    model_provider: str = "deepseek"
    model_name: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    timeout: int = DEFAULT_TIMEOUT
    system_prompt: str | None = None
    enable_memory: bool = True
    enable_planning: bool = False
    enable_tools: bool = True
    mode: AgentMode | str = AgentMode.REACT
    debug: bool = False


# =============================================================================
# 数据库 / TextToSQL 配置
# =============================================================================

@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_url: str = ""
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = ""
    text_to_sql_max_rows: int = DEFAULT_TEXT_TO_SQL_MAX_ROWS
    text_to_sql_timeout: int = DEFAULT_TEXT_TO_SQL_TIMEOUT
    text_to_sql_model_name: str | None = None


# =============================================================================
# 工具配置
# =============================================================================

@dataclass
class ToolsConfig:
    """工具配置"""
    amap_api_key: str = ""
    mcp_enabled: bool = False


# =============================================================================
# 日志配置
# =============================================================================

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "./logs"
    rotation: str = "1 day"
    retention: str = "7 days"
    console_level: str = "DEBUG"


# =============================================================================
# 中间件配置
# =============================================================================

@dataclass
class MiddlewareConfig:
    """中间件配置"""
    max_iterations: int = 20
    warn_threshold: float = 0.8


# =============================================================================
# LLM Provider 配置
# =============================================================================

@dataclass
class LLMProvidersConfig:
    """LLM 提供商配置"""
    # 模型选择
    model_name: str = "deepseek"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model_name: str = "deepseek-v4-pro"

    # 豆包
    doubao_api_key: str = ""
    doubao_api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    doubao_model_name: str = ""

    # 阿里云通义
    aliyun_api_key: str = ""
    aliyun_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    aliyun_model_name: str = "qwen-plus"


# =============================================================================
# 主配置类
# =============================================================================

class Settings:
    """
    应用配置类

    支持从环境变量和 YAML 配置文件读取配置
    优先级：环境变量 > YAML配置文件 > 默认配置
    """

    CONFIG_FILE_PATH: Final[str] = "config.yaml"

    def __init__(self) -> None:
        """初始化配置"""
        load_dotenv()
        self._config: dict[str, Any] = self._load_config()
        self._parse_config()

    def _get_default_config(self) -> dict[str, Any]:
        """获取默认配置"""
        return {
            "agent": {
                "model_provider": "deepseek",
                "model_name": None,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "max_iterations": DEFAULT_MAX_ITERATIONS,
                "timeout": DEFAULT_TIMEOUT,
                "system_prompt": None,
                "enable_memory": True,
                "enable_planning": False,
                "enable_tools": True,
                "mode": AgentMode.REACT.value,
                "debug": False,
            },
            "database": {
                "db_url": "",
                "db_host": "localhost",
                "db_port": 3306,
                "db_user": "root",
                "db_password": "",
                "db_name": "",
                "text_to_sql_max_rows": DEFAULT_TEXT_TO_SQL_MAX_ROWS,
                "text_to_sql_timeout": DEFAULT_TEXT_TO_SQL_TIMEOUT,
                "text_to_sql_model_name": None,
            },
            "tools": {
                "amap_api_key": "",
                "mcp_enabled": False,
            },
            "logging": {
                "level": "INFO",
                "log_dir": "./logs",
                "rotation": "1 day",
                "retention": "7 days",
                "console_level": "DEBUG",
            },
            "middleware": {
                "max_iterations": 20,
                "warn_threshold": 0.8,
            },
            "llm_providers": {
                "model_name": "deepseek",
                "deepseek_api_key": "",
                "deepseek_api_base": "https://api.deepseek.com/v1",
                "deepseek_model_name": "deepseek-v4-pro",
                "doubao_api_key": "",
                "doubao_api_base": "https://ark.cn-beijing.volces.com/api/v3",
                "doubao_model_name": "",
                "aliyun_api_key": "",
                "aliyun_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "aliyun_model_name": "qwen-plus",
            },
        }

    def _load_config(self) -> dict[str, Any]:
        """加载配置"""
        config: dict[str, Any] = self._get_default_config()
        self._load_from_file(config)
        self._load_from_env(config)
        return config

    def _load_from_file(self, config: dict[str, Any]) -> None:
        """从 YAML 文件加载配置"""
        config_file: Path = Path(self.CONFIG_FILE_PATH)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config: dict[str, Any] = yaml.safe_load(f) or {}
                self._merge_config(config, file_config)
            except Exception as e:
                print(f"Warning: Cannot load config file {self.CONFIG_FILE_PATH}: {e}")

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _load_from_env(self, config: dict[str, Any]) -> None:
        """从环境变量加载配置"""
        env_mappings = {
            # Agent
            "AGENT_MODEL_PROVIDER": ("agent", "model_provider", str),
            "AGENT_MODEL_NAME": ("agent", "model_name", str),
            "AGENT_TEMPERATURE": ("agent", "temperature", float),
            "AGENT_MAX_TOKENS": ("agent", "max_tokens", int),
            "AGENT_MAX_ITERATIONS": ("agent", "max_iterations", int),
            "AGENT_TIMEOUT": ("agent", "timeout", int),
            "AGENT_DEBUG": ("agent", "debug", lambda v: v.lower() == "true"),
            "AGENT_ENABLE_PLANNING": ("agent", "enable_planning", lambda v: v.lower() == "true"),
            # Database
            "DB_URL": ("database", "db_url", str),
            "DB_HOST": ("database", "db_host", str),
            "DB_PORT": ("database", "db_port", int),
            "DB_USER": ("database", "db_user", str),
            "DB_PASSWORD": ("database", "db_password", str),
            "DB_NAME": ("database", "db_name", str),
            "TEXT_TO_SQL_MAX_ROWS": ("database", "text_to_sql_max_rows", int),
            "TEXT_TO_SQL_TIMEOUT": ("database", "text_to_sql_timeout", int),
            # Tools
            "AMAP_API_KEY": ("tools", "amap_api_key", str),
            "MCP_ENABLED": ("tools", "mcp_enabled", lambda v: v.lower() == "true"),
            # Logging
            "LOG_LEVEL": ("logging", "level", str),
            "LOG_DIR": ("logging", "log_dir", str),
            "LOG_ROTATION": ("logging", "rotation", str),
            "LOG_RETENTION": ("logging", "retention", str),
            # Middleware
            "MIDDLEWARE_MAX_ITERATIONS": ("middleware", "max_iterations", int),
            # LLM Providers
            "MODEL_NAME": ("llm_providers", "model_name", str),
            "DEEPSEEK_API_KEY": ("llm_providers", "deepseek_api_key", str),
            "DEEPSEEK_API_BASE": ("llm_providers", "deepseek_api_base", str),
            "DEEPSEEK_MODEL_NAME": ("llm_providers", "deepseek_model_name", str),
            "DOUBAO_API_KEY": ("llm_providers", "doubao_api_key", str),
            "DOUBAO_API_BASE": ("llm_providers", "doubao_api_base", str),
            "DOUBAO_MODEL_NAME": ("llm_providers", "doubao_model_name", str),
            "ALIYUN_API_KEY": ("llm_providers", "aliyun_api_key", str),
            "ALIYUN_API_BASE": ("llm_providers", "aliyun_api_base", str),
            "ALIYUN_MODEL_NAME": ("llm_providers", "aliyun_model_name", str),
        }

        for env_key, (section, field_key, converter) in env_mappings.items():
            env_value = os.environ.get(env_key)
            if env_value is not None:
                try:
                    config[section][field_key] = converter(env_value)
                except (ValueError, TypeError):
                    pass

    def _parse_config(self) -> None:
        """解析配置到具体配置对象"""
        agent_config_dict = self._config["agent"].copy()
        agent_config_dict["mode"] = AgentMode.from_value(agent_config_dict["mode"])
        # 自动填充 model_name（当未指定时）
        if not agent_config_dict.get("model_name"):
            provider = agent_config_dict.get("model_provider", "").lower()
            if provider == "deepseek":
                agent_config_dict["model_name"] = self._config["llm_providers"]["deepseek_model_name"]
            elif provider == "doubao":
                agent_config_dict["model_name"] = self._config["llm_providers"]["doubao_model_name"]
            elif provider in {"tongyi", "aliyun"}:
                agent_config_dict["model_name"] = self._config["llm_providers"]["aliyun_model_name"]
        self.agent = AgentConfig(**agent_config_dict)
        self.database = DatabaseConfig(**self._config["database"])
        self.tools = ToolsConfig(**self._config["tools"])
        self.logging = LoggingConfig(**self._config["logging"])
        self.middleware = MiddlewareConfig(**self._config["middleware"])
        self.llm_providers = LLMProvidersConfig(**self._config["llm_providers"])

    def reload(self) -> None:
        """重新加载配置"""
        self._config = self._load_config()
        self._parse_config()

    # =============================================================================
    # 属性快捷方式（兼容旧代码）
    # =============================================================================

    # Agent
    @property
    def DEBUG(self) -> bool:
        return self.agent.debug

    @property
    def STRUCTURED(self) -> bool:
        return False

    @property
    def TEMPERATURE(self) -> float:
        """返回 Agent 配置的 temperature 值"""
        return self.agent.temperature

    # Database
    @property
    def DB_URL(self) -> str:
        return self.database.db_url

    @property
    def DB_HOST(self) -> str:
        return self.database.db_host

    @property
    def DB_PORT(self) -> int:
        return self.database.db_port

    @property
    def DB_USER(self) -> str:
        return self.database.db_user

    @property
    def DB_PASSWORD(self) -> str:
        return self.database.db_password

    @property
    def DB_NAME(self) -> str:
        return self.database.db_name

    @property
    def TEXT_TO_SQL_MAX_ROWS(self) -> int:
        return self.database.text_to_sql_max_rows

    @property
    def TEXT_TO_SQL_QUERY_TIMEOUT(self) -> int:
        return self.database.text_to_sql_timeout

    # Tools
    @property
    def AMAP_API_KEY(self) -> str:
        return self.tools.amap_api_key

    @property
    def MCP_ENABLED(self) -> bool:
        return self.tools.mcp_enabled

    # Logging
    @property
    def LOG_LEVEL(self) -> str:
        return self.logging.level

    # LLM Providers
    @property
    def MODEL_NAME(self) -> str:
        return self.llm_providers.model_name

    @property
    def DEEPSEEK_API_KEY(self) -> str:
        return self.llm_providers.deepseek_api_key

    @property
    def DEEPSEEK_API_BASE(self) -> str:
        return self.llm_providers.deepseek_api_base

    @property
    def DEEPSEEK_MODEL_NAME(self) -> str:
        return self.llm_providers.deepseek_model_name

    @property
    def DOUBAO_API_KEY(self) -> str:
        return self.llm_providers.doubao_api_key

    @property
    def DOUBAO_API_BASE(self) -> str:
        return self.llm_providers.doubao_api_base

    @property
    def DOUBAO_MODEL_NAME(self) -> str:
        return self.llm_providers.doubao_model_name

    @property
    def ALIYUN_API_KEY(self) -> str:
        return self.llm_providers.aliyun_api_key

    @property
    def ALIYUN_API_BASE(self) -> str:
        return self.llm_providers.aliyun_api_base

    @property
    def ALIYUN_MODEL_NAME(self) -> str:
        return self.llm_providers.aliyun_model_name

    # =============================================================================
    # 便捷方法
    # =============================================================================

    def get_db_url(self) -> str:
        """返回数据库连接地址"""
        if self.DB_URL:
            return self.DB_URL
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    def validate_model_config(self, model_name: str) -> bool:
        """检查 Provider 是否具备足够的运行配置"""
        provider = model_name.lower()
        if provider == "deepseek":
            return bool(self.DEEPSEEK_API_KEY and self.DEEPSEEK_MODEL_NAME)
        if provider == "doubao":
            return bool(self.DOUBAO_API_KEY and self.DOUBAO_MODEL_NAME)
        if provider in {"tongyi", "aliyun"}:
            return bool(self.ALIYUN_API_KEY and self.ALIYUN_MODEL_NAME)
        if provider == "mock":
            return True
        return False


# =============================================================================
# 全局配置实例
# =============================================================================

settings: Final[Settings] = Settings()

__all__ = [
    # 常量
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TEXT_TO_SQL_MAX_ROWS",
    "DEFAULT_TEXT_TO_SQL_TIMEOUT",
    # 配置类
    "AgentConfig",
    "DatabaseConfig",
    "ToolsConfig",
    "LoggingConfig",
    "MiddlewareConfig",
    "LLMProvidersConfig",
    # 主类
    "Settings",
    "settings",
    # AgentMode
    "AgentMode",
]
