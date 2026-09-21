# -*- coding: utf-8 -*-
"""
通用常量定义

存放项目中跨模块使用的常量。
"""

# =============================================================================
# 秋池 MCP 服务器常量
# =============================================================================
# 不从环境变量读取 QIUCHI_MCP_PATH / QIUCHI_MCP_MODE，原因：
# Git Bash (MSYS) 会把 os.environ 中以 "/" 开头的值自动转为 Windows 路径
# （如 /mcp → C:/Program Files/Git/mcp），导致 URL 拼接错误。
# 如需自定义，实例化 QiuChiMCPClient 时通过参数传入。

QIUCHI_MCP_BASE_URL: str = "http://localhost:8000"
QIUCHI_MCP_PATH: str = "/mcp"
QIUCHI_MCP_MODE: str = "http"  # http 或 stdio
