"""TextToSQL 工具使用的数据库辅助能力。"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings
from core.logger import logger

if TYPE_CHECKING:
    pass

_MUTATING_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|merge|grant|revoke)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


def get_db_url() -> str:
    """返回当前配置的 SQLAlchemy 数据库连接地址。"""
    return settings.get_db_url()


def is_safe_select_sql(sql: str) -> bool:
    """仅当 SQL 是单条只读 SELECT 语句时返回 True。"""
    normalized = sql.strip()
    if not normalized:
        return False

    body = normalized[:-1].strip() if normalized.endswith(";") else normalized
    if ";" in body:
        return False
    if not body.lower().startswith("select"):
        return False
    if _MUTATING_SQL_RE.search(body):
        return False
    return True


def apply_default_limit(sql: str, max_rows: int | None = None) -> str:
    """当安全 SELECT 查询没有 LIMIT 时，自动追加默认 LIMIT。"""
    if not is_safe_select_sql(sql):
        raise ValueError("只允许执行单条只读 SELECT 语句")

    limit = max_rows or settings.TEXT_TO_SQL_MAX_ROWS
    body = sql.strip()
    trailing_semicolon = body.endswith(";")
    body = body[:-1].strip() if trailing_semicolon else body

    if not _LIMIT_RE.search(body):
        body = f"{body} LIMIT {limit}"

    return f"{body};"


class DBOperations:
    """TextToSQL 工具使用的数据库操作封装。

    支持上下文管理器协议，可安全地用于 `with` 语句自动释放资源。

    Example:
        with DBOperations() as db:
            schema = db.get_schema_info()
            results = db.execute_sql("SELECT * FROM users LIMIT 10")
    """

    def __init__(self, db_url: str | None = None) -> None:
        self.db_url: str = db_url or get_db_url()
        self.engine: Engine | None = None
        self.Session: sessionmaker[Session] | None = None
        self.metadata: MetaData | None = None
        self._schema_cache: dict[str, Any] | None = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库连接和元数据。"""
        try:
            connect_args: dict[str, Any] = {}
            if self.db_url.startswith("mysql"):
                connect_args["connect_timeout"] = settings.TEXT_TO_SQL_QUERY_TIMEOUT

            self.engine = create_engine(
                self.db_url,
                pool_pre_ping=True,
                connect_args=connect_args,
            )
            self.Session = sessionmaker(bind=self.engine)
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            logger.info("Connected to database: %s", self.db_url.split("@")[-1] if "@" in self.db_url else "unknown")
        except SQLAlchemyError as exc:
            logger.error("Database connection failed: %s", exc)
            raise

    def get_schema_info(self, force_refresh: bool = False) -> dict[str, Any]:
        """返回表、字段、主键、外键等元数据信息。

        Args:
            force_refresh: 是否强制刷新缓存，默认为 False（使用缓存）

        Returns:
            包含所有表结构的字典，键为表名，值为表结构信息
        """
        if not force_refresh and self._schema_cache is not None:
            logger.debug("Returning cached schema for %d tables", len(self._schema_cache))
            return self._schema_cache

        if not self.metadata:
            logger.error("Metadata is not initialized")
            return {}

        schema_info: dict[str, Any] = {}
        try:
            for table_name, table in self.metadata.tables.items():
                columns = [
                    {
                        "name": column.name,
                        "type": str(column.type),
                        "primary_key": column.primary_key,
                    }
                    for column in table.columns
                ]
                schema_info[table_name] = {
                    "columns": columns,
                    "primary_keys": [col.name for col in table.primary_key],
                    "foreign_keys": [
                        {
                            "column": fk.parent.name,
                            "references": fk.target_fullname,
                        }
                        for fk in table.foreign_keys
                    ],
                }
            self._schema_cache = schema_info
            logger.info("Loaded database schema for %d tables", len(schema_info))
        except SQLAlchemyError as exc:
            logger.error("Failed to load database schema: %s", exc)
        return schema_info

    def execute_sql(self, sql: str) -> list[dict[str, Any]]:
        """执行安全 SELECT 查询，并以字典列表返回结果。

        Args:
            sql: SELECT 查询语句

        Returns:
            查询结果列表，每行数据为一个字典

        Raises:
            RuntimeError: 数据库引擎未初始化
            ValueError: SQL 不是安全的 SELECT 语句
            SQLAlchemyError: SQL 执行失败
        """
        if not self.engine:
            raise RuntimeError("Database engine is not initialized")

        safe_sql = apply_default_limit(sql)
        results: list[dict[str, Any]] = []

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(safe_sql))
                columns = list(result.keys())
                for row in result.fetchmany(settings.TEXT_TO_SQL_MAX_ROWS):
                    results.append(dict(zip(columns, row)))
            logger.info("SQL executed successfully, returned %d rows", len(results))
        except SQLAlchemyError as exc:
            logger.error("SQL execution failed [%s]: %s", safe_sql[:100], exc)
            raise
        return results

    def validate_sql(self, sql: str) -> bool:
        """在不真正执行查询的前提下校验 SQL 语法。

        Args:
            sql: 待校验的 SQL 语句

        Returns:
            SQL 语法有效返回 True，否则返回 False
        """
        if not self.engine:
            logger.error("Database engine is not initialized")
            return False
        if not is_safe_select_sql(sql):
            return False

        try:
            safe_sql = apply_default_limit(sql)
            with self.engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {safe_sql}"))
            return True
        except SQLAlchemyError as exc:
            logger.warning("SQL validation failed [%s]: %s", sql[:100], exc)
            return False

    def invalidate_cache(self) -> None:
        """清除 Schema 缓存，下次调用会重新加载。"""
        self._schema_cache = None
        logger.debug("Schema cache invalidated")

    def close(self) -> None:
        """关闭数据库引擎，释放资源。"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None
            self.metadata = None
            self._schema_cache = None
            logger.info("Database connection closed")

    def __enter__(self) -> "DBOperations":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出，自动关闭连接。"""
        self.close()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """获取数据库会话的上下文管理器。

        Yields:
            SQLAlchemy Session 对象
        """
        if not self.Session:
            raise RuntimeError("Database Session is not initialized")
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
