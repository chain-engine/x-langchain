# -*- coding: utf-8 -*-
"""
枚举基类

提供可描述枚举类的基类封装
"""

from enum import Enum


class BaseEnum(Enum):
    """
    可描述的枚举类基建
    mark: int | str    唯一标识
    desc: str          描述信息
    """

    def __init__(self, mark: int | str, desc: str) -> None:
        self._mark = mark
        self._desc = desc

    @property
    def mark(self) -> int | str:
        return self._mark

    @property
    def value(self) -> int | str:
        """重写 value，使枚举可直接赋值给 int 或 str 类型字段（如 Pydantic BaseModel）"""
        return self._mark

    @property
    def desc(self) -> str:
        return self._desc

    def __str__(self) -> str:
        return str(self._mark)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            return super().__eq__(other)
        if isinstance(other, str) and isinstance(self._mark, str):
            return self._mark == other
        if isinstance(other, int) and isinstance(self._mark, int):
            return self._mark == other
        if isinstance(other, str):
            return str(self._mark) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._mark)

    @classmethod
    def get_all_marks(cls) -> list[int | str]:
        return [described_enum.mark for described_enum in cls]

    @classmethod
    def get_all_descs(cls) -> list[str]:
        return [described_enum.desc for described_enum in cls]

    @classmethod
    def get_choices(cls) -> tuple[tuple[int, str], ...]:
        return tuple((described_enum.mark, described_enum.desc) for described_enum in cls)
