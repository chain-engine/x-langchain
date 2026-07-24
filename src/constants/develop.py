# -*- coding: utf-8 -*-
"""
开发相关常量

本模块定义了开发过程中常用的常量，包括：
- HTTP 内容类型
- 加密算法
- 工作模式
- 填充方式
- 编码方式
"""

from .base import BaseEnum


# HTTP 内容类型
class HttpContentType(BaseEnum):
    JSON = ("application/json", "JSON 数据")
    FILE = ("application/octet-stream", "文件流")
    FORM_URL_ENCODED = ("application/x-www-form-urlencoded", "URL 编码表单")
    MULTIPART_FORM_DATA = ("multipart/form-data", "多部分表单")


# 单向加密算法（哈希算法）
class HashAlgorithm(BaseEnum):
    MD5 = ("MD5", "MD5 哈希")
    SHA1 = ("SHA1", "SHA1 哈希")
    SHA256 = ("SHA256", "SHA256 哈希")
    SHA512 = ("SHA512", "SHA512 哈希")
    SM3 = ("SM3", "国密 SM3 哈希")


# 对称加密算法
class SymmetricCipher(BaseEnum):
    AES = ("AES", "高级加密标准")
    SM4 = ("SM4", "国密 SM4")
    DES = ("DES", "数据加密标准")
    THREE_DES = ("3DES", "三重 DES")
    CHACHA20 = ("ChaCha20", "ChaCha20 流密码")
    RC4 = ("RC4", "RC4 流密码")


# 非对称加密算法
class AsymmetricCipher(BaseEnum):
    RSA = ("RSA", "RSA 非对称加密")
    ECC = ("ECC", "椭圆曲线密码学")
    DSA = ("DSA", "数字签名算法")
    SM2 = ("SM2", "国密 SM2")


# 工作模式
class CipherMode(BaseEnum):
    ECB = ("ECB", "电子密码本模式")
    CBC = ("CBC", "密码块链接模式")
    GCM = ("GCM", "伽罗瓦计数器模式")


# 对称加密块填充方式
class SymmetricPadding(BaseEnum):
    PKCS7 = ("PKCS7", "PKCS7 填充")
    ISO10126 = ("ISO10126", "ISO10126 填充")
    NO_PADDING = ("NoPadding", "无填充")
    ZERO_PADDING = ("ZeroPadding", "零填充")


# 非对称加密填充方式
class AsymmetricPadding(BaseEnum):
    PKCS1V15 = ("PKCS1v15", "PKCS1v15 填充")
    OAEP = ("OAEP", "最优非对称加密填充")


# 编码方式
class EncodingType(BaseEnum):
    BASE64 = ("base64", "Base64 编码")
    HEX = ("hex", "十六进制编码")
