# -*- coding: utf-8 -*-
"""
Conversation Repository

会话数据访问层，封装业务 CRUD、多表联查、分页、条件查询。
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Conversation, Message
from repositories.base import Repository


class ConversationRepository(Repository):
    """会话仓储"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, entity_id: str) -> Optional[Conversation]:
        """根据ID获取会话"""
        stmt = select(Conversation).where(Conversation.id == entity_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_title(self, title: str) -> Optional[Conversation]:
        """根据标题获取会话"""
        stmt = select(Conversation).where(Conversation.title == title)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, data: dict[str, Any]) -> Conversation:
        """创建会话"""
        instance = Conversation(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "新对话"),
            model_provider=data.get("model_provider", "tongyi"),
            summary=data.get("summary"),
        )
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def update(self, entity_id: str, data: dict[str, Any]) -> Optional[Conversation]:
        """更新会话"""
        instance = await self.get_by_id(entity_id)
        if instance is None:
            return None
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        instance.updated_at = datetime.now()
        await self._session.flush()
        return instance

    async def delete(self, entity_id: str) -> bool:
        """删除会话"""
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
        """分页查询所有会话"""
        # 构建查询
        stmt = select(Conversation)
        if filters:
            for key, value in filters.items():
                if hasattr(Conversation, key):
                    stmt = stmt.where(getattr(Conversation, key) == value)

        # 排序
        order_col = getattr(Conversation, sort_by or "updated_at", Conversation.updated_at)
        if sort_order == "asc":
            stmt = stmt.order_by(order_col.asc())
        else:
            stmt = stmt.order_by(order_col.desc())

        # 总数
        count_stmt = select(func.count()).select_from(Conversation)
        if filters:
            for key, value in filters.items():
                if hasattr(Conversation, key):
                    count_stmt = count_stmt.where(getattr(Conversation, key) == value)
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        # 分页
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

    async def get_with_messages(self, entity_id: str) -> Optional[Conversation]:
        """获取会话及其消息（预加载）"""
        stmt = (
            select(Conversation)
            .where(Conversation.id == entity_id)
            .options(selectinload(Conversation.messages))
        )
        result = await self._session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def get_with_message_count(self) -> list[dict[str, Any]]:
        """获取会话及其消息数量"""
        stmt = (
            select(
                Conversation.id,
                Conversation.title,
                Conversation.summary,
                Conversation.model_provider,
                Conversation.created_at,
                Conversation.updated_at,
                func.count(Message.id).label("message_count"),
            )
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return [row._asdict() for row in result.fetchall()]

    async def search_by_title(self, keyword: str, limit: int = 20) -> list[Conversation]:
        """搜索标题包含关键词的会话"""
        stmt = (
            select(Conversation)
            .where(Conversation.title.ilike(f"%{keyword}%"))
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_statistics(self) -> dict[str, Any]:
        """获取会话统计信息"""
        total_count = await self.count()
        stmt = (
            select(
                Conversation.model_provider,
                func.count(Conversation.id).label("count"),
            )
            .group_by(Conversation.model_provider)
        )
        result = await self._session.execute(stmt)
        by_provider = {row.model_provider: row.count for row in result.fetchall()}
        return {
            "total": total_count,
            "by_model_provider": by_provider,
        }

    async def count(self) -> int:
        """统计总数"""
        stmt = select(func.count()).select_from(Conversation)
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    def _model_to_dict(self, model: Conversation) -> dict[str, Any]:
        """将 ORM 模型转换为字典"""
        return {
            "id": model.id,
            "title": model.title,
            "summary": model.summary,
            "model_provider": model.model_provider,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
        }
