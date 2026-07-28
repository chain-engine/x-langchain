# -*- coding: utf-8 -*-
"""
提示词模板加载器

统一管理所有提示词模板，支持：
- 从 YAML 文件加载提示词
- 模板变量替换（{{ variable }} 语法）
- 模板缓存，避免重复读取文件

使用方式：

    # 方法1: 直接加载（最常用） ===
    from prompts import load_prompt

    prompt = load_prompt("agent_system")                    # 无变量
    prompt = load_prompt("generate_sql", schema="...")     # 带变量渲染

    # 方法2: 获取模板对象（需要额外操作时用） ===
    from prompts import get_template

    template = get_template("agent_system")
    rendered = template.render(session_id="xxx")           # 手动渲染
    missing = template.validate(session_id="xxx")           # 验证变量
    print(template.version)                                # 查看元数据
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from core.logger import logger

# 全局配置
_TEMPLATES_DIR: str | None = None
_ENABLE_CACHE: bool = True
_CACHE: dict[str, "PromptTemplate"] = {}
_CACHE_LOCK = threading.Lock()


def _get_templates_dir() -> Path:
    """获取模板目录路径"""
    global _TEMPLATES_DIR
    if _TEMPLATES_DIR:
        return Path(_TEMPLATES_DIR)

    # 优先从配置读取
    try:
        from core.config import settings
        if settings.prompts.templates_dir:
            _TEMPLATES_DIR = settings.prompts.templates_dir
            return Path(_TEMPLATES_DIR)
    except Exception:
        pass

    # 优先从环境变量读取
    env_path = os.environ.get("X_LANGCHAIN_PROMPTS_DIR")
    if env_path:
        _TEMPLATES_DIR = env_path
        return Path(env_path)

    # 默认使用 src/prompts/templates
    base_dir = Path(__file__).parent
    default_path = base_dir / "templates"
    if default_path.exists():
        _TEMPLATES_DIR = str(default_path)
        return default_path

    # 回退到当前目录
    current = Path(__file__).parent
    _TEMPLATES_DIR = str(current)
    return current


def configure(
    templates_dir: str | None = None,
    enable_cache: bool = True,
    clear_cache: bool = False,
) -> None:
    """
    配置提示词加载器

    Args:
        templates_dir: 模板目录路径，默认使用 src/prompts/templates
        enable_cache: 是否启用缓存，默认 True
        clear_cache: 是否清除现有缓存，默认 False
    """
    global _ENABLE_CACHE, _CACHE

    if templates_dir:
        global _TEMPLATES_DIR
        _TEMPLATES_DIR = templates_dir

    _ENABLE_CACHE = enable_cache

    if clear_cache:
        with _CACHE_LOCK:
            _CACHE.clear()


@dataclass
class PromptTemplate:
    """
    提示词模板类

    支持简单的变量替换：
    - {{ variable }} - 简单变量替换
    - {{ variable | default("xxx") }} - 带默认值的变量
    - {{ variable | upper }} - 支持过滤器
    """

    name: str
    description: str
    version: str
    content: str
    variables: list[str] = field(default_factory=list)

    # 内置过滤器
    FILTERS: ClassVar[dict[str, Any]] = {
        "upper": lambda x: str(x).upper() if x else "",
        "lower": lambda x: str(x).lower() if x else "",
        "title": lambda x: str(x).title() if x else "",
        "strip": lambda x: str(x).strip() if x else "",
        "default": lambda x, d="": str(x) if x else d,
        "int": lambda x: int(x) if x else 0,
        "float": lambda x: float(x) if x else 0.0,
        "bool": lambda x: bool(x) if x else False,
        "len": lambda x: len(x) if x else 0,
        "str": lambda x: str(x) if x else "",
        "first": lambda x: x[0] if x and len(x) > 0 else None,
        "last": lambda x: x[-1] if x and len(x) > 0 else None,
        "join": lambda x, sep=", ": sep.join(str(v) for v in x) if x else "",
    }

    # 变量提取正则 - 支持 {{ var }} 和 {{ var | filter }}
    _VAR_PATTERN: ClassVar[re.Pattern] = re.compile(r"\{\{\s*(\w+)(?:\s*\|\s*[\w\(\'\"]+)?\s*\}\}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """从字典创建模板"""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            content=data.get("content", ""),
        )

    def render(self, **kwargs: Any) -> str:
        """
        渲染模板，替换变量

        Args:
            **kwargs: 模板变量

        Returns:
            渲染后的提示词
        """
        content = self.content

        # 替换 {{ variable }} 格式的变量
        def replace_var(match: re.Match) -> str:
            full_match = match.group(0)
            var_name = match.group(1)
            filter_part = match.group(2) if match.lastindex >= 2 else None

            # 获取变量值
            value = kwargs.get(var_name, "")

            # 应用过滤器
            if filter_part:
                value = self._apply_filters(value, filter_part)

            return str(value)

        # 匹配 {{ var }} 或 {{ var | filter }} 格式
        pattern = r"\{\{\s*(\w+)(?:\s*\|\s*([\w(\"\',\s]+?))?\s*\}\}"
        content = re.sub(pattern, replace_var, content)

        return content

    def _apply_filters(self, value: Any, filter_str: str) -> str:
        """应用过滤器链"""
        value = str(value) if value else ""

        # 解析过滤器链
        filters = filter_str.split("|")
        for f in filters:
            f = f.strip()
            if not f:
                continue

            # 处理带参数的过滤器
            if "(" in f:
                filter_name, args_str = f.split("(", 1)
                filter_name = filter_name.strip()
                args_str = args_str.rstrip(")").strip()

                # 处理 default("xxx") 格式
                if args_str.startswith('"') or args_str.startswith("'"):
                    default_val = args_str.strip('"\'')
                    value = value or default_val
                else:
                    # 尝试调用带参数的过滤器
                    filter_func = self.FILTERS.get(filter_name)
                    if filter_func:
                        try:
                            value = filter_func(value, **eval(f"dict({args_str})"))
                        except Exception:
                            pass
            else:
                # 简单过滤器
                filter_func = self.FILTERS.get(f)
                if filter_func:
                    try:
                        value = filter_func(value)
                    except Exception:
                        pass

        return value

    def extract_variables(self) -> list[str]:
        """提取模板中的所有变量名"""
        if self.variables:
            return self.variables

        variables: set[str] = set()
        for match in self._VAR_PATTERN.finditer(self.content):
            variables.add(match.group(1))

        self.variables = list(variables)
        return self.variables

    def validate_variables(self, **kwargs: Any) -> list[str]:
        """
        验证提供的变量是否覆盖所有必需变量

        Returns:
            缺失的变量列表
        """
        required = set(self.extract_variables())
        provided = set(kwargs.keys())
        return list(required - provided)


@dataclass
class PromptMetadata:
    """提示词元数据"""
    name: str
    description: str
    version: str
    last_updated: str
    file_path: str


class PromptLoader:
    """
    提示词加载器

    负责从文件系统加载 YAML 模板文件，支持缓存。
    """

    def __init__(
        self,
        templates_dir: str | Path | None = None,
        enable_cache: bool = True,
    ):
        """
        初始化加载器

        Args:
            templates_dir: 模板目录，默认使用 src/prompts/templates
            enable_cache: 是否启用缓存
        """
        self._templates_dir = templates_dir or _get_templates_dir()
        # 优先使用构造参数，否则从配置读取
        if enable_cache is None:
            try:
                from core.config import settings
                self._enable_cache = settings.prompts.enable_cache
            except Exception:
                self._enable_cache = _ENABLE_CACHE
        else:
            self._enable_cache = enable_cache
        self._cache: dict[str, PromptTemplate] = {}
        self._metadata_cache: dict[str, PromptMetadata] = {}
        self._lock = threading.Lock()

    @property
    def templates_dir(self) -> Path:
        """获取模板目录"""
        if isinstance(self._templates_dir, str):
            self._templates_dir = Path(self._templates_dir)
        return self._templates_dir

    def _get_cache_key(self, name: str) -> str:
        """获取缓存键"""
        return f"{self.templates_dir}:{name}"

    def _load_yaml(self, name: str) -> dict[str, Any]:
        """加载 YAML 文件"""
        yaml_path = self.templates_dir / f"{name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"提示词模板不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"提示词模板为空: {yaml_path}")

        return data

    def load(self, name: str, use_cache: bool = True) -> PromptTemplate:
        """
        加载提示词模板

        Args:
            name: 模板名称（不含 .yaml 后缀）
            use_cache: 是否使用缓存

        Returns:
            PromptTemplate 实例
        """
        cache_key = self._get_cache_key(name)

        # 检查缓存
        if use_cache and self._enable_cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug(f"从缓存加载提示词: {name}")
                return cached

        # 加载模板
        data = self._load_yaml(name)
        template = PromptTemplate.from_dict(data)

        # 更新缓存
        if self._enable_cache:
            with self._lock:
                self._cache[cache_key] = template

        logger.debug(f"加载提示词模板: {name} (v{template.version})")
        return template

    def get_metadata(self, name: str) -> PromptMetadata:
        """获取模板元数据"""
        cache_key = self._get_cache_key(name)

        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        data = self._load_yaml(name)
        metadata = PromptMetadata(
            name=data.get("name", name),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            last_updated=data.get("last_updated", ""),
            file_path=str(self.templates_dir / f"{name}.yaml"),
        )

        self._metadata_cache[cache_key] = metadata
        return metadata

    def list_templates(self) -> list[str]:
        """列出所有可用的模板"""
        if not self.templates_dir.exists():
            return []

        return [
            p.stem
            for p in self.templates_dir.glob("*.yaml")
            if p.is_file() and not p.name.startswith("_")
        ]

    def reload(self, name: str) -> PromptTemplate:
        """重新加载指定模板（清除缓存后加载）"""
        cache_key = self._get_cache_key(name)

        with self._lock:
            self._cache.pop(cache_key, None)
            self._metadata_cache.pop(cache_key, None)

        return self.load(name, use_cache=False)

    def reload_all(self) -> list[PromptTemplate]:
        """重新加载所有模板"""
        templates = []
        for name in self.list_templates():
            try:
                templates.append(self.reload(name))
            except Exception as e:
                logger.warning(f"重新加载模板失败 {name}: {e}")
        return templates

    def clear_cache(self) -> None:
        """清除所有缓存"""
        with self._lock:
            self._cache.clear()
            self._metadata_cache.clear()


# 全局加载器实例
_GLOBAL_LOADER: PromptLoader | None = None


def _get_global_loader() -> PromptLoader:
    """获取全局加载器"""
    global _GLOBAL_LOADER
    if _GLOBAL_LOADER is None:
        _GLOBAL_LOADER = PromptLoader(enable_cache=_ENABLE_CACHE)
    return _GLOBAL_LOADER


def load_prompt(name: str, use_cache: bool = True, **kwargs: Any) -> str:
    """
    快捷函数：加载并渲染提示词

    Args:
        name: 模板名称
        use_cache: 是否使用缓存
        **kwargs: 模板变量

    Returns:
        渲染后的提示词字符串

    Raises:
        ValueError: 缺少必需变量时抛出

    Example:
        >>> prompt = load_prompt("generate_sql", schema_description="...")
        >>> prompt = load_prompt("question_rewrite")
    """
    loader = _get_global_loader()
    template = loader.load(name, use_cache=use_cache)

    # 验证必需变量
    missing = template.validate_variables(**kwargs)
    if missing:
        raise ValueError(
            f"模板 '{name}' 缺少必需变量: {missing}。"
            f"请检查是否传入了所有必需参数。"
        )

    return template.render(**kwargs)


def get_template(name: str, use_cache: bool = True) -> PromptTemplate:
    """
    快捷函数：获取模板对象

    Args:
        name: 模板名称
        use_cache: 是否使用缓存

    Returns:
        PromptTemplate 实例

    Example:
        >>> template = get_template("agent_system")
        >>> rendered = template.render(session_id="xxx")
    """
    loader = _get_global_loader()
    return loader.load(name, use_cache=use_cache)


def list_prompts() -> list[str]:
    """快捷函数：列出所有可用模板"""
    loader = _get_global_loader()
    return loader.list_templates()


def reload_prompt(name: str) -> PromptTemplate:
    """快捷函数：重新加载指定模板"""
    loader = _get_global_loader()
    return loader.reload(name)


def configure_loader(
    templates_dir: str | None = None,
    enable_cache: bool | None = None,
    clear_cache: bool = False,
) -> None:
    """
    配置全局加载器

    Args:
        templates_dir: 模板目录
        enable_cache: 是否启用缓存（None 表示使用配置值）
        clear_cache: 是否清除缓存
    """
    global _ENABLE_CACHE, _GLOBAL_LOADER

    if clear_cache:
        _CACHE.clear()

    if templates_dir or clear_cache or _GLOBAL_LOADER is None:
        _GLOBAL_LOADER = PromptLoader(
            templates_dir=templates_dir,
            enable_cache=enable_cache,
        )

    if enable_cache is not None and enable_cache != _ENABLE_CACHE:
        _ENABLE_CACHE = enable_cache


__all__ = [
    "PromptTemplate",
    "PromptMetadata",
    "PromptLoader",
    "load_prompt",
    "get_template",
    "list_prompts",
    "reload_prompt",
    "configure",
    "configure_loader",
]
