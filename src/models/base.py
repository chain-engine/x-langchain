# -*- coding: utf-8 -*-
"""
SQLAlchemy 声明式基类
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类，所有模型必须继承此类"""
    pass
