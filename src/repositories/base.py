# -*- coding: utf-8 -*-
"""
Repository基类

定义数据访问层的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Repository(ABC):
    """
    仓库基类

    定义数据访问层的通用接口。
    """

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[Any]:
        """根据ID获取实体"""
        pass

    @abstractmethod
    async def create(self, data: dict[str, Any]) -> Any:
        """创建实体"""
        pass

    @abstractmethod
    async def update(self, entity_id: str, data: dict[str, Any]) -> Any:
        """更新实体"""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """删除实体"""
        pass

    @abstractmethod
    async def list_all(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> dict[str, Any]:
        """分页查询所有实体"""
        pass
