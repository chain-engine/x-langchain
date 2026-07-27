# -*- coding: utf-8 -*-
"""
Memories 模块 - 对话记忆管理

提供 LangChain Memory 组件的封装：
|- ChatMessageHistory: 基础聊天消息历史
|- ConversationMemory: 对话记忆（支持上下文窗口）
|- BufferMemory: 缓冲区记忆
|- AdvancedMemory: 高级记忆（摘要/窗口/实体）
|- RedisChatHistory: Redis 存储
|- FileChatHistory: 文件存储
|- PostgresChatHistory: PostgreSQL 存储
|- MongoDBChatHistory: MongoDB 存储
|- create_chat_history: Chat History 工厂函数
|- create_conversation_memory: 创建对话记忆
|- create_buffer_memory: 创建缓冲区记忆
|- create_advanced_memory: 创建高级记忆

使用方式：
    ```python
    from memories import (
        create_conversation_memory,
        create_buffer_memory,
        create_advanced_memory,
        create_chat_history,
    )

    # 基础记忆
    memory = create_buffer_memory()

    # 高级记忆（带摘要）
    memory = create_advanced_memory(memory_type="summary")

    # 持久化存储
    history = create_chat_history("user-123", backend="redis")
    ```
"""

from .memory import (
    ChatMessageHistory,
    ConversationMemory,
    BufferMemory,
    create_conversation_memory,
    create_buffer_memory,
)
from .advanced_memory import (
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
    CombinedMemory,
    AdvancedMemory,
    create_advanced_memory,
)
from .chat_history import (
    ChatHistoryAdapter,
    RedisChatHistory,
    FileChatHistory,
    PostgresChatHistory,
    MongoDBChatHistory,
    create_chat_history,
    create_redis_history,
    create_file_history,
)

__all__ = [
    # Base Memory
    "ChatMessageHistory",
    "ConversationMemory",
    "BufferMemory",
    "create_conversation_memory",
    "create_buffer_memory",
    # Advanced Memory
    "ConversationSummaryMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "CombinedMemory",
    "AdvancedMemory",
    "create_advanced_memory",
    # Chat History
    "ChatHistoryAdapter",
    "RedisChatHistory",
    "FileChatHistory",
    "PostgresChatHistory",
    "MongoDBChatHistory",
    "create_chat_history",
    "create_redis_history",
    "create_file_history",
]
