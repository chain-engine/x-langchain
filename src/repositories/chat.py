# -*- coding: utf-8 -*-
"""
Chat Repository

会话与消息的组合仓储，封装多表联查、联合操作。
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Message
from repositories.base import Repository
from repositories.conversation import ConversationRepository
from repositories.message import MessageRepository


class ChatRepository(Repository):
    """
    会话仓储

    封装 Conversation 和 Message 的联合操作，
    提供会话级别的完整业务 CRUD。
    """

    def __init__(self, session: AsyncSession):
        self._session = session
        self._conversation_repo = ConversationRepository(session)
        self._message_repo = MessageRepository(session)

    @property
    def conversation_repo(self) -> ConversationRepository:
        """获取会话仓储"""
        return self._conversation_repo

    @property
    def message_repo(self) -> MessageRepository:
        """获取消息仓储"""
        return self._message_repo

    async def get_by_id(self, entity_id: str) -> Optional[Conversation]:
        """根据ID获取会话"""
        return await self._conversation_repo.get_by_id(entity_id)

    async def create(self, data: dict[str, Any]) -> Conversation:
        """创建会话"""
        return await self._conversation_repo.create(data)

    async def update(self, entity_id: str, data: dict[str, Any]) -> Optional[Conversation]:
        """更新会话"""
        return await self._conversation_repo.update(entity_id, data)

    async def delete(self, entity_id: str) -> bool:
        """删除会话及其所有消息"""
        await self._message_repo.delete_by_conversation(entity_id)
        return await self._conversation_repo.delete(entity_id)

    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> dict[str, Any]:
        """分页查询所有会话"""
        return await self._conversation_repo.list_all(
            page=page,
            page_size=page_size,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def get_or_create(
        self,
        session_id: str,
        model_provider: str = "tongyi",
        title: Optional[str] = None,
    ) -> Conversation:
        """获取或创建会话"""
        conv = await self.get_by_id(session_id)
        if conv is None:
            conv = await self.create({
                "id": session_id,
                "title": title or "新对话",
                "model_provider": model_provider,
            })
        return conv

    async def get_with_messages(self, session_id: str) -> Optional[Conversation]:
        """获取会话及其所有消息（预加载）"""
        return await self._conversation_repo.get_with_messages(session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_provider: str = "tongyi",
    ) -> tuple[Conversation, Message]:
        """添加消息到会话（自动创建会话）"""
        conv = await self.get_or_create(session_id, model_provider)
        msg = await self._message_repo.create({
            "conversation_id": conv.id,
            "role": role,
            "content": content,
        })
        conv.updated_at = datetime.now()
        await self._session.flush()
        return conv, msg

    async def add_messages_batch(
        self,
        session_id: str,
        messages: list[tuple[str, str]],
        model_provider: str = "tongyi",
    ) -> tuple[Conversation, list[Message]]:
        """批量添加消息"""
        conv = await self.get_or_create(session_id, model_provider)
        msg_objects = await self._message_repo.bulk_create(
            conversation_id=conv.id,
            messages=[{"role": r, "content": c} for r, c in messages],
        )
        conv.updated_at = datetime.now()
        await self._session.flush()
        return conv, msg_objects

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[Message]:
        """获取会话消息"""
        return await self._message_repo.get_by_conversation(session_id, limit=limit)

    async def get_recent_messages(self, session_id: str, limit: int = 10) -> list[Message]:
        """获取最近的消息"""
        return await self._message_repo.get_recent_by_conversation(session_id, limit)

    async def search_messages(
        self,
        session_id: str,
        keyword: str,
        limit: int = 20,
    ) -> list[Message]:
        """搜索会话消息"""
        return await self._message_repo.search_in_conversation(session_id, keyword, limit)

    async def delete_conversation_with_messages(self, session_id: str) -> bool:
        """删除会话及其所有消息"""
        return await self.delete(session_id)

    async def clear_conversation_messages(self, session_id: str) -> bool:
        """清空会话消息（保留会话）"""
        if await self.get_by_id(session_id) is None:
            return False
        await self._message_repo.delete_by_conversation(session_id)
        return True

    async def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """获取会话摘要统计"""
        conv = await self.get_by_id(session_id)
        if conv is None:
            return {}

        msg_count = await self._message_repo.count_by_conversation(session_id)
        user_count = await self._message_repo.count_by_conversation_and_role(session_id, "user")

        return {
            "session_id": session_id,
            "title": conv.title,
            "model_provider": conv.model_provider,
            "total_messages": msg_count,
            "user_messages": user_count,
            "assistant_messages": msg_count - user_count,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "summary": conv.summary,
        }

    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
        model_provider: Optional[str] = None,
    ) -> list[Conversation]:
        """列出会话（支持过滤）"""
        filters = {}
        if model_provider:
            filters["model_provider"] = model_provider
        page = offset // limit + 1
        result = await self.list_all(
            page=page,
            page_size=limit,
            filters=filters if filters else None,
            sort_by="updated_at",
            sort_order="desc",
        )
        return result["items"]

    async def search_conversations(self, keyword: str, limit: int = 20) -> list[Conversation]:
        """搜索会话"""
        return await self._conversation_repo.search_by_title(keyword, limit)


@asynccontextmanager
async def chat_repository(auto_commit: bool = True) -> AsyncGenerator["ChatRepository", None]:
    """
    创建 ChatRepository 的异步上下文管理器

    用法:
        async with chat_repository() as repo:
            conv, msg = await repo.add_message(session_id, "user", "你好")
    """
    from infras.mysql import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        repo = ChatRepository(session)
        try:
            yield repo
            if auto_commit:
                await session.commit()
        except Exception:
            await session.rollback()
            raise


def generate_session_id() -> str:
    """生成新的会话ID（UUID4）"""
    return str(uuid.uuid4())
