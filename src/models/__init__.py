# -*- coding: utf-8 -*-
"""
ORM 实体层

纯数据表映射模型，仅定义字段和表关联关系，无任何查询、业务逻辑。
"""

from .base import Base
from .conversation import Conversation
from .message import Message

__all__ = [
    "Base",
    "Conversation",
    "Message",
]
