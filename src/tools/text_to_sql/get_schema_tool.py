# -*- coding: utf-8 -*-
"""
获取数据库结构工具

获取数据库的表结构信息，包括表名、字段、主键、外键等。
"""

from typing import Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tools.base import BaseXTool
from core.logger import logger


class GetSchemaArgs(BaseModel):
    """获取数据库结构参数"""


class GetSchemaTool(BaseXTool):
    """获取数据库结构工具"""

    name: str = "get_schema"
    description: str = "获取数据库的表结构信息，包括表名、字段、主键、外键等"
    args_schema: type[GetSchemaArgs] = GetSchemaArgs
    retry_count: int = 1

    def _run(self) -> Dict[str, Any]:
        """
        获取数据库结构

        Returns:
            数据库结构信息
        """
        try:
            logger.info("获取数据库结构")

            # 使用单例 DBOperations，避免重复创建连接
            from infras.mysql.operations import get_db_operations

            db_ops = get_db_operations()
            schema_info: Dict[str, Any] = db_ops.get_schema_info()

            logger.info(f"成功获取数据库结构，包含 {len(schema_info)} 个表")

            return {"schema_info": schema_info, "success": True}
        except Exception as e:
            logger.error(f"获取数据库结构失败: {e}")
            return {"error": "获取数据库结构失败，请稍后重试", "success": False}
