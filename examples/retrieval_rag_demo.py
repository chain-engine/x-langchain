# -*- coding: utf-8 -*-
"""
Retrieval / RAG 示例

演示如何使用 retrieval 模块构建语义记忆和 RAG 系统。
"""

from retrieval import (
    MockEmbedding,
    InMemoryVectorStore,
    SemanticMemory,
    SemanticMemoryConfig,
    DocumentLoader,
    create_text_splitter,
)


def demo_vector_search():
    """向量检索基础"""
    print("[向量检索] 使用 Mock Embedding")

    embed = MockEmbedding(dimensions=8)
    vs = InMemoryVectorStore(embedding=embed)

    # 添加文档
    docs = [
        "Python 是一种高级编程语言",
        "LangChain 是一个 LLM 应用开发框架",
        "向量数据库用于存储和检索 embeddings",
        "RAG 是检索增强生成技术",
    ]
    vs.add_texts(docs)
    print(f"[向量检索] 添加了 {len(docs)} 条文档")

    # 检索
    queries = ["编程语言有哪些", "什么是 RAG", "框架工具推荐"]
    for q in queries:
        results = vs.similarity_search(q, k=2)
        print(f"\n[查询] {q}")
        for r in results:
            print(f"  -> {r.page_content}")


def demo_semantic_memory():
    """语义记忆"""
    print("\n" + "=" * 60)
    print("[语义记忆] Agent 长期记忆示例")

    config = SemanticMemoryConfig(
        embedding_provider="mock",
        vectorstore_provider="memory",
        chunk_size=100,
        chunk_overlap=20,
    )
    memory = SemanticMemory(config)

    # 添加记忆
    memories = [
        ("用户叫张三，是项目经理", {"type": "user_info"}),
        ("项目使用 LangChain 框架", {"type": "project"}),
        ("数据库使用 MySQL", {"type": "tech_stack"}),
        ("每周五下午有周会", {"type": "meeting"}),
    ]
    for content, meta in memories:
        memory.add(content, metadata=meta)

    print(f"[语义记忆] 共 {len(memory)} 条记忆")

    # 检索
    questions = [
        "用户叫什么名字？",
        "项目用的什么技术？",
        "什么时候开会？",
    ]
    for q in questions:
        results = memory.search(q, k=2)
        print(f"\n[查询] {q}")
        for r in results:
            meta = getattr(r, "metadata", {})
            print(f"  [{meta.get('type', 'unknown')}] {r.page_content}")

    # 获取上下文字符串
    ctx = memory.get_context("用户的项目信息", k=2)
    print(f"\n[上下文字符串]\n{ctx}")


def demo_document_loader():
    """文档加载"""
    print("\n" + "=" * 60)
    print("[文档加载] 加载 README.md")

    docs = DocumentLoader.load("README.md")
    print(f"加载了 {len(docs)} 个文档")
    if docs:
        print(f"前 100 字符: {docs[0].page_content[:100]}...")


def demo_text_splitter():
    """文档分块"""
    print("\n" + "=" * 60)
    print("[文档分块] 分块示例")

    splitter = create_text_splitter("recursive", chunk_size=50, chunk_overlap=10)
    text = (
        "第一章：介绍。\n\n"
        "这是第一章的内容，讲述了基础知识。\n\n"
        "第二章：进阶。\n\n"
        "这是第二章的内容，深入讲解了高级特性。"
    )
    chunks = splitter.split_text(text)
    print(f"原始文本长度: {len(text)}")
    print(f"分块数量: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  块 {i+1}: {c}")


if __name__ == "__main__":
    demo_vector_search()
    demo_semantic_memory()
    demo_document_loader()
    demo_text_splitter()
