# -*- coding: utf-8 -*-
"""
少样本提示模板模块

在 langchain_core.prompts.FewShotPromptTemplate 基础上提供：
- FewShotPromptTemplate: 静态少样本提示模板
- DynamicFewShotPromptTemplate: 动态少样本提示模板（基于语义相似度）
- create_few_shot_prompt: 静态少样本提示工厂
- create_dynamic_few_shot_prompt: 动态少样本提示工厂

少样本提示通过在 prompt 中插入若干示例（example）来引导大模型
按照预期格式与风格输出，适合：
- 任务格式化输出（SQL、JSON、代码）
- 风格 / 语气控制
- 复杂推理任务
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate as LCFewShotPromptTemplate
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_core.vectorstores import VectorStore

from core.logger import logger


# =============================================================================
# 静态少样本提示模板
# =============================================================================

class FewShotPromptTemplate:
    """
    静态少样本提示模板

    在 langchain_core.prompts.FewShotPromptTemplate 基础上提供：
    - get_examples(): 获取示例列表
    - add_example(): 追加单个示例
    - from_examples(): 便捷工厂方法
    - format(**kwargs): 渲染最终提示词

    使用方式：
        ```python
        template = FewShotPromptTemplate(
            examples=[{"q": "1+1", "a": "2"}],
            example_template=PromptTemplate.from_template("Q: {q}\\nA: {a}"),
            prefix="回答下列问题：",
            suffix="Q: {input}\\nA:",
        )
        print(template.format(input="2+2"))
        ```
    """

    def __init__(
        self,
        examples: List[dict],
        example_template: LCPromptTemplate,
        prefix: str = "",
        suffix: str = "",
        example_separator: str = "\n\n",
        **kwargs: Any,
    ):
        """
        初始化少样本提示模板

        Args:
            examples: 示例字典列表，每个字典的 key 必须能在 example_template 中找到对应变量
            example_template: 单个示例的渲染模板（langchain PromptTemplate 或本模块的 PromptTemplate）
            prefix: 示例前的引导文本
            suffix: 示例后的实际提问文本
            example_separator: 示例之间的分隔符，默认两个换行
            **kwargs: 透传给 langchain FewShotPromptTemplate 的其他参数（如 input_variables）
        """
        # 适配本模块的 PromptTemplate 包装
        if isinstance(example_template, LCPromptTemplate):
            lc_example_template = example_template
        else:
            raise TypeError(
                f"example_template 类型错误，期望 PromptTemplate，实际 {type(example_template)}"
            )

        self.examples: List[dict] = list(examples)
        self.example_template = lc_example_template
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator

        self._template = LCFewShotPromptTemplate(
            examples=self.examples,
            example_prompt=lc_example_template,
            prefix=prefix,
            suffix=suffix,
            example_separator=example_separator,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def format(self, **kwargs: Any) -> str:
        """
        渲染最终提示词

        Args:
            **kwargs: 模板中的占位变量

        Returns:
            渲染好的提示词字符串
        """
        return self._template.format(**kwargs)

    # ------------------------------------------------------------------ #
    # 示例管理
    # ------------------------------------------------------------------ #
    def get_examples(self) -> List[dict]:
        """
        获取示例列表（拷贝，避免外部修改）

        Returns:
            示例字典列表
        """
        return list(self.examples)

    def add_example(self, example: dict) -> None:
        """
        追加一个示例到末尾

        注意：该方法仅更新本对象的 examples 列表与内部缓存，
        不会持久化到外部 langchain FewShotPromptTemplate 实例。
        如需重新生成底层模板，请结合业务自行处理。

        Args:
            example: 示例字典
        """
        if not isinstance(example, dict):
            raise TypeError(f"example 必须是 dict 类型，实际 {type(example)}")

        self.examples.append(example)
        # 同步到底层 langchain 模板，使后续 format 使用最新 examples
        self._template.examples = self.examples
        logger.debug(f"已添加示例，当前共 {len(self.examples)} 条")

    # ------------------------------------------------------------------ #
    # 工厂
    # ------------------------------------------------------------------ #
    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        input_variables: List[str],
        **kwargs: Any,
    ) -> "FewShotPromptTemplate":
        """
        便捷工厂：传入示例与变量列表，自动构建 example_template

        要求：
        - examples 中每个 dict 的 key 集合必须一致
        - input_variables 表示 suffix 中出现的变量（example 字段变量除外）

        Args:
            examples: 示例字典列表
            input_variables: suffix 中需要使用的变量名列表
            **kwargs: 透传给构造函数的 prefix / suffix / example_separator 等

        Returns:
            FewShotPromptTemplate 实例
        """
        if not examples:
            raise ValueError("examples 不能为空")

        # 从第一个 example 推断 example_template 的变量
        example_keys = list(examples[0].keys())
        example_template_str = "\n".join(f"{k}: {{{k}}}" for k in example_keys)
        example_template = LCPromptTemplate(
            input_variables=example_keys,
            template=example_template_str,
        )

        # 默认 suffix：把 input_variables 列出来
        suffix = kwargs.pop("suffix", None)
        if suffix is None:
            suffix_lines = [f"输入: {{{v}}}" for v in input_variables]
            suffix = "\n".join(suffix_lines) + "\n输出:"

        return cls(
            examples=examples,
            example_template=example_template,
            suffix=suffix,
            input_variables=input_variables,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # 透传属性
    # ------------------------------------------------------------------ #
    @property
    def input_variables(self) -> List[str]:
        """底层模板的输入变量列表"""
        return list(self._template.input_variables)

    def to_langchain_template(self) -> LCFewShotPromptTemplate:
        """
        导出原始 langchain FewShotPromptTemplate

        Returns:
            langchain 官方 FewShotPromptTemplate 实例
        """
        return self._template


# =============================================================================
# 动态少样本提示模板
# =============================================================================

class DynamicFewShotPromptTemplate:
    """
    动态少样本提示模板

    使用 SemanticSimilarityExampleSelector 根据输入动态选择最相关的示例，
    适用于：
    - 示例数量较多，全部塞入 prompt 会超出 token 限制
    - 不同输入需要不同示例才能更好引导模型

    使用方式：
        ```python
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_openai import OpenAIEmbeddings

        vs = InMemoryVectorStore(embedding=OpenAIEmbeddings())
        template = DynamicFewShotPromptTemplate.from_vectorstore(
            vectorstore=vs,
            examples=[{"q": "1+1", "a": "2"}],
            input_variables=["input"],
            k=3,
        )
        print(template.format(input="2+2"))
        ```
    """

    def __init__(
        self,
        example_selector: SemanticSimilarityExampleSelector,
        example_template: LCPromptTemplate,
        input_variables: List[str],
        prefix: str = "",
        suffix: str = "",
        example_separator: str = "\n\n",
    ):
        """
        初始化动态少样本提示模板

        Args:
            example_selector: SemanticSimilarityExampleSelector 实例
            example_template: 单个示例渲染模板（langchain PromptTemplate）
            input_variables: suffix 中的变量列表
            prefix: 示例前的引导文本
            suffix: 示例后的实际提问文本
            example_separator: 示例之间的分隔符
        """
        if not isinstance(example_selector, SemanticSimilarityExampleSelector):
            raise TypeError(
                "example_selector 必须是 SemanticSimilarityExampleSelector 实例，"
                f"实际 {type(example_selector)}"
            )

        self.example_selector = example_selector
        self.example_template = example_template
        self.input_variables = list(input_variables)
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator

        # 底层使用 langchain FewShotPromptTemplate，把 examples 替换为可调用对象
        # langchain 支持传入 BaseExampleSelector，会在每次调用时动态选择
        self._template = LCFewShotPromptTemplate(
            examples=[],  # 使用 example_selector 时这里可为空
            example_prompt=example_template,
            example_selector=example_selector,
            prefix=prefix,
            suffix=suffix,
            example_separator=example_separator,
            input_variables=input_variables,
        )

    # ------------------------------------------------------------------ #
    # 动态选择 / 渲染
    # ------------------------------------------------------------------ #
    def select_examples(self, input: str, k: int = 4) -> List[dict]:
        """
        根据输入选择最相关的 k 个示例

        Args:
            input: 用于相似度匹配的查询字符串
            k: 返回示例数量

        Returns:
            选中的示例字典列表
        """
        try:
            return list(self.example_selector.select_examples({"input": input}, k=k))
        except TypeError:
            # 旧版本 selector 不支持 k 参数，使用 selector 自身的 k
            return list(self.example_selector.select_examples({"input": input}))

    def format(self, **kwargs: Any) -> str:
        """
        渲染最终提示词（按 kwargs 选择示例）

        Args:
            **kwargs: 模板变量，通常包含 input 字段

        Returns:
            渲染好的提示词字符串
        """
        return self._template.format(**kwargs)

    # ------------------------------------------------------------------ #
    # 工厂
    # ------------------------------------------------------------------ #
    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: VectorStore,
        examples: List[dict],
        input_variables: List[str],
        k: int = 4,
        prefix: str = "",
        suffix: Optional[str] = None,
        example_separator: str = "\n\n",
        **kwargs: Any,
    ) -> "DynamicFewShotPromptTemplate":
        """
        从 VectorStore 构建 DynamicFewShotPromptTemplate

        Args:
            vectorstore: 已经存在（或新建）的 langchain VectorStore
            examples: 初始示例字典列表，会写入向量库
            input_variables: suffix 中的变量列表
            k: 单次选择示例数量
            prefix: 示例前的引导文本
            suffix: 示例后的实际提问文本，None 时自动生成
            example_separator: 示例之间的分隔符
            **kwargs: 透传给 SemanticSimilarityExampleSelector.from_examples

        Returns:
            DynamicFewShotPromptTemplate 实例
        """
        if not examples:
            raise ValueError("examples 不能为空")

        # 从第一个 example 推断 example_template
        example_keys = list(examples[0].keys())
        example_template_str = "\n".join(f"{k_}: {{{k_}}}" for k_ in example_keys)
        example_template = LCPromptTemplate(
            input_variables=example_keys,
            template=example_template_str,
        )

        # 自动生成 suffix
        if suffix is None:
            suffix_lines = [f"输入: {{{v}}}" for v in input_variables]
            suffix = "\n".join(suffix_lines) + "\n输出:"

        # 从 vectorstore 创建 selector
        # 注意：from_examples 会把 vectorstore 内部的 examples 写入到 vectorstore
        selector = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,
            example_template=example_template,
            vectorstore_cls=lambda **kw: vectorstore,
            k=k,
            **kwargs,
        )

        return cls(
            example_selector=selector,
            example_template=example_template,
            input_variables=input_variables,
            prefix=prefix,
            suffix=suffix,
            example_separator=example_separator,
        )

    def to_langchain_template(self) -> LCFewShotPromptTemplate:
        """
        导出原始 langchain FewShotPromptTemplate（含 example_selector）

        Returns:
            langchain 官方 FewShotPromptTemplate 实例
        """
        return self._template


# =============================================================================
# 工厂函数
# =============================================================================

def create_few_shot_prompt(
    examples: List[dict],
    input_variables: List[str],
    example_template_str: str,
    **kwargs: Any,
) -> FewShotPromptTemplate:
    """
    工厂函数：从示例列表和模板字符串构建少样本提示模板

    Args:
        examples: 示例字典列表
        input_variables: suffix 中使用的变量名列表
        example_template_str: 单个示例渲染模板字符串（如 "Q: {q}\\nA: {a}"）
        **kwargs: 透传给 FewShotPromptTemplate（prefix/suffix/example_separator 等）

    Returns:
        FewShotPromptTemplate 实例

    使用示例：
        ```python
        prompt = create_few_shot_prompt(
            examples=[
                {"q": "1+1", "a": "2"},
                {"q": "2+2", "a": "4"},
            ],
            input_variables=["q"],
            example_template_str="Q: {q}\\nA: {a}",
            prefix="回答算术题：",
        )
        print(prompt.format(q="3+3"))
        ```
    """
    # 从 example_template_str 中提取变量名
    import re

    template_vars = set(re.findall(r"\{([^{}]+)\}", example_template_str))

    if not template_vars:
        raise ValueError(
            f"example_template_str 中必须至少包含一个 {{变量}}，实际: {example_template_str!r}"
        )

    # 检查 examples 是否覆盖所有变量
    if examples:
        first_keys = set(examples[0].keys())
        missing = template_vars - first_keys
        if missing:
            raise ValueError(
                f"examples[0] 缺少模板所需的变量: {missing}，"
                f"实际 keys: {first_keys}"
            )

    example_template = LCPromptTemplate(
        input_variables=sorted(template_vars),
        template=example_template_str,
    )

    # 如果用户没有指定 suffix，则自动生成
    if "suffix" not in kwargs:
        suffix_lines = [f"输入: {{{v}}}" for v in input_variables]
        kwargs["suffix"] = "\n".join(suffix_lines) + "\n输出:"

    kwargs.setdefault("input_variables", input_variables)

    return FewShotPromptTemplate(
        examples=examples,
        example_template=example_template,
        **kwargs,
    )


def create_dynamic_few_shot_prompt(
    vectorstore: VectorStore,
    example_template: LCPromptTemplate,
    input_variables: List[str],
    k: int = 4,
    **kwargs: Any,
) -> DynamicFewShotPromptTemplate:
    """
    工厂函数：从已有 VectorStore 创建动态少样本提示模板

    Args:
        vectorstore: langchain VectorStore 实例
        example_template: 单个示例渲染模板（langchain PromptTemplate）
        input_variables: suffix 中使用的变量名列表
        k: 单次选择示例数量，默认 4
        **kwargs: 透传给 DynamicFewShotPromptTemplate（prefix/suffix/example_separator 等）

    Returns:
        DynamicFewShotPromptTemplate 实例

    使用示例：
        ```python
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_core.embeddings import DeterministicFakeEmbedding

        vs = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=8))
        example_template = PromptTemplate.from_template("Q: {q}\\nA: {a}")
        prompt = create_dynamic_few_shot_prompt(
            vectorstore=vs,
            example_template=example_template,
            input_variables=["q"],
            examples=[{"q": "1+1", "a": "2"}],
        )
        ```
    """
    examples = kwargs.pop("examples", [])
    return DynamicFewShotPromptTemplate.from_vectorstore(
        vectorstore=vectorstore,
        examples=examples,
        input_variables=input_variables,
        k=k,
        example_template=example_template,
        **kwargs,
    )


__all__ = [
    "FewShotPromptTemplate",
    "DynamicFewShotPromptTemplate",
    "create_few_shot_prompt",
    "create_dynamic_few_shot_prompt",
]