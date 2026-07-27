# -*- coding: utf-8 -*-
"""
Message Repository

消息数据访问层，封装业务 CRUD、多表联查、分页、条件查询。
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from models import Message
from repositories.base import Repository


class MessageRepository(Repository):
    """消息仓储"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, entity_id: str) -> Optional[Message]:
        """根据ID获取消息"""
        stmt = select(Message).where(Message.id == int(entity_id))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, data: dict[str, Any]) -> Message:
        """创建消息"""
        instance = Message(
            conversation_id=data["conversation_id"],
            role=data["role"],
            content=data["content"],
        )
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def update(self, entity_id: str, data: dict[str, Any]) -> Optional[Message]:
        """更新消息"""
        instance = await self.get_by_id(entity_id)
        if instance is None:
            return None
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await self._session.flush()
        return instance

    async def delete(self, entity_id: str) -> bool:
        """删除消息"""
        instance = await self.get_by_id(entity_id)
        if instance is None:
            return False
        await self._session.delete(instance)
        await self._session.flush()
        return True

    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> dict[str, Any]:
        """分页查询所有消息"""
        stmt = select(Message)
        if filters:
            for key, value in filters.items():
                if hasattr(Message, key):
                    stmt = stmt.where(getattr(Message, key) == value)

        order_col = getattr(Message, sort_by or "created_at", Message.created_at)
        if sort_order == "asc":
            stmt = stmt.order_by(order_col.asc())
        else:
            stmt = stmt.order_by(order_col.desc())

        # 总数
        count_stmt = select(func.count()).select_from(Message)
        if filters:
            for key, value in filters.items():
                if hasattr(Message, key):
                    count_stmt = count_stmt.where(getattr(Message, key) == value)
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        offset = (page - 1) * page_size
        stmt = stmt.offset(offset).limit(page_size)
        result = await self._session.execute(stmt)
        items = list(result.scalars().all())

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
        }

    async def get_by_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Message]:
        """获取指定会话的消息"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_by_conversation(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """获取指定会话最近的N条消息"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())
        return list(reversed(messages))

    async def search_in_conversation(
        self,
        conversation_id: str,
        keyword: str,
        limit: int = 20,
    ) -> list[Message]:
        """在指定会话中搜索消息"""
        escaped_keyword = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.content.ilike(f"%{escaped_keyword}%"))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_conversation(self, conversation_id: str) -> int:
        """统计指定会话的消息数量"""
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def count_by_conversation_and_role(
        self,
        conversation_id: str,
        role: str,
    ) -> int:
        """统计指定会话中指定角色的消息数量"""
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.role == role)
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def get_with_conversation(self, message_id: int) -> Optional[Message]:
        """获取消息及其关联的会话（预加载）"""
        stmt = (
            select(Message)
            .options(joinedload(Message.conversation))
            .where(Message.id == message_id)
        )
        result = await self._session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """删除指定会话的所有消息"""
        from sqlalchemy import delete

        stmt = delete(Message).where(Message.conversation_id == conversation_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0

    async def bulk_create(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
    ) -> list[Message]:
        """批量创建消息"""
        objects = [
            Message(conversation_id=conversation_id, **msg)
            for msg in messages
        ]
        self._session.add_all(objects)
        await self._session.flush()
        return objects

    def _model_to_dict(self, model: Message) -> dict[str, Any]:
        """将 ORM 模型转换为字典"""
        return {
            "id": model.id,
            "conversation_id": model.conversation_id,
            "role": model.role,
            "content": model.content,
            "created_at": model.created_at.isoformat() if model.created_at else None,
        }
