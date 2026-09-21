# -*- coding: utf-8 -*-
"""
枚举常量

合并自 agent.py、develop.py、streaming_modes.py，统一管理项目中所有枚举类型。
"""

from __future__ import annotations

from .base import BaseEnum


# =============================================================================
# Agent 相关
# =============================================================================

class AgentMode(BaseEnum):
    """
    Agent 运行模式

    - CHAIN: 简单链式调用，直接调用 LLM
    - REACT: ReAct 推理模式（Thought → Action → Observation 循环）
    - PLAN:  规划模式（预留扩展）
    """
    CHAIN = ("chain", "简单链式调用")
    REACT = ("react", "ReAct 推理模式（工具调用循环）")
    PLAN = ("plan", "规划模式（预留扩展）")


# =============================================================================
# 流式传输模式
# =============================================================================

class StreamMode(BaseEnum):
    UPDATES = ("updates", "增量更新模式")
    CUSTOM = ("custom", "自定义模式")
    MESSAGES = ("messages", "消息模式")


DEFAULT_STREAM_MODES: list[StreamMode] = [StreamMode.UPDATES, StreamMode.CUSTOM, StreamMode.MESSAGES]


# =============================================================================
# 开发相关
# =============================================================================

class HttpContentType(BaseEnum):
    JSON = ("application/json", "JSON 数据")
    FILE = ("application/octet-stream", "文件流")
    FORM_URL_ENCODED = ("application/x-www-form-urlencoded", "URL 编码表单")
    MULTIPART_FORM_DATA = ("multipart/form-data", "多部分表单")


class HashAlgorithm(BaseEnum):
    MD5 = ("MD5", "MD5 哈希")
    SHA1 = ("SHA1", "SHA1 哈希")
    SHA256 = ("SHA256", "SHA256 哈希")
    SHA512 = ("SHA512", "SHA512 哈希")
    SM3 = ("SM3", "国密 SM3 哈希")


class SymmetricCipher(BaseEnum):
    AES = ("AES", "高级加密标准")
    SM4 = ("SM4", "国密 SM4")
    DES = ("DES", "数据加密标准")
    THREE_DES = ("3DES", "三重 DES")
    CHACHA20 = ("ChaCha20", "ChaCha20 流密码")
    RC4 = ("RC4", "RC4 流密码")


class AsymmetricCipher(BaseEnum):
    RSA = ("RSA", "RSA 非对称加密")
    ECC = ("ECC", "椭圆曲线密码学")
    DSA = ("DSA", "数字签名算法")
    SM2 = ("SM2", "国密 SM2")


class CipherMode(BaseEnum):
    ECB = ("ECB", "电子密码本模式")
    CBC = ("CBC", "密码块链接模式")
    GCM = ("GCM", "伽罗瓦计数器模式")


class SymmetricPadding(BaseEnum):
    PKCS7 = ("PKCS7", "PKCS7 填充")
    ISO10126 = ("ISO10126", "ISO10126 填充")
    NO_PADDING = ("NoPadding", "无填充")
    ZERO_PADDING = ("ZeroPadding", "零填充")


class AsymmetricPadding(BaseEnum):
    PKCS1V15 = ("PKCS1v15", "PKCS1v15 填充")
    OAEP = ("OAEP", "最优非对称加密填充")


class EncodingType(BaseEnum):
    BASE64 = ("base64", "Base64 编码")
    HEX = ("hex", "十六进制编码")
