"""企业微信（WeCom）回调消息加解密。

实现企业微信 / 微信客服「回调通知」的 AES-256-CBC 加解密与 SHA1 签名校验，
对齐官方 WXBizMsgCrypt 规范，但**自实现、不依赖腾讯样例代码**（规避再分发的
许可负担），底层走 cryptography（milu 环境已具备，作为 `milu[wechat]` 可选依赖声明）。

参考：回调配置 https://developer.work.weixin.qq.com/document/path/90930

加密报文内层结构（解密 PKCS7 去填充后）：
    [16 字节随机前缀][4 字节大端消息长度][消息明文][receive_id]
其中 receive_id 对企业自建应用 / 微信客服即为 corpid。
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# 企业微信 PKCS7 以 32 字节为填充块（注意：不是 AES 的 16 字节）
_BLOCK_SIZE = 32


class WeComCryptError(Exception):
    """加解密 / 签名校验失败。"""


def _pkcs7_pad(data: bytes) -> bytes:
    """PKCS7 填充到 32 字节块的整数倍（满块时再补一整块）。"""
    pad = _BLOCK_SIZE - (len(data) % _BLOCK_SIZE)
    if pad == 0:
        pad = _BLOCK_SIZE
    return data + bytes([pad]) * pad


def _pkcs7_unpad(data: bytes) -> bytes:
    """去除 PKCS7 填充；填充字节越界时按官方实现容错为不去填充。"""
    if not data:
        return data
    pad = data[-1]
    if pad < 1 or pad > _BLOCK_SIZE:
        return data
    return data[:-pad]


def sha1_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    """官方签名算法：对 [token, timestamp, nonce, encrypt] 字典序排序后拼接取 SHA1。"""
    items = sorted([token, timestamp, nonce, encrypt])
    sha = hashlib.sha1()
    sha.update("".join(items).encode())
    return sha.hexdigest()


@dataclass(frozen=True)
class WeComCrypto:
    """企业微信回调加解密器。

    :param token: 回调配置的 Token
    :param encoding_aes_key: 回调配置的 EncodingAESKey（43 位 base64）
    :param receive_id: 接收方 ID（企业自建应用 / 微信客服为 corpid）；
                       传空串则解密时跳过 receive_id 校验。
    """

    token: str
    encoding_aes_key: str
    receive_id: str

    # ── 内部：AES 密钥 / cipher ───────────────────────────────────────
    @property
    def _key(self) -> bytes:
        # EncodingAESKey 为 43 位，补一个 '=' 后 base64 解码得 32 字节密钥
        return base64.b64decode(self.encoding_aes_key + "=")

    def _cipher(self) -> Cipher:
        key = self._key
        # IV 取密钥前 16 字节（官方约定）
        return Cipher(algorithms.AES(key), modes.CBC(key[:16]))

    # ── 解密 ──────────────────────────────────────────────────────────
    def decrypt(self, encrypt_b64: str) -> str:
        """解密 base64 密文，返回内层消息明文（XML 或随机串）。"""
        try:
            decryptor = self._cipher().decryptor()
            raw = decryptor.update(base64.b64decode(encrypt_b64)) + decryptor.finalize()
            raw = _pkcs7_unpad(raw)
            content = raw[16:]  # 去掉 16 字节随机前缀
            msg_len = struct.unpack(">I", content[:4])[0]
            msg = content[4 : 4 + msg_len]
            receive_id = content[4 + msg_len :].decode(errors="ignore")
        except WeComCryptError:
            raise
        except Exception as e:  # noqa: BLE001 —— 任何解析异常统一归一化
            raise WeComCryptError(f"解密失败：{e}") from e
        if self.receive_id and receive_id != self.receive_id:
            raise WeComCryptError(
                f"receive_id 不匹配：{receive_id!r} != {self.receive_id!r}"
            )
        return msg.decode()

    # ── 加密 ──────────────────────────────────────────────────────────
    def encrypt(self, plaintext: str) -> str:
        """加密明文，返回 base64 密文（用于被动响应，或本地自测构造请求）。"""
        msg = plaintext.encode()
        payload = (
            os.urandom(16)
            + struct.pack(">I", len(msg))
            + msg
            + self.receive_id.encode()
        )
        encryptor = self._cipher().encryptor()
        ct = encryptor.update(_pkcs7_pad(payload)) + encryptor.finalize()
        return base64.b64encode(ct).decode()

    # ── URL 验证（配置回调时的 GET 请求）──────────────────────────────
    def verify_url(
        self, msg_signature: str, timestamp: str, nonce: str, echostr: str
    ) -> str:
        """校验签名并返回解密后的 echostr 明文（原样回给企业微信即验证通过）。"""
        if sha1_signature(self.token, timestamp, nonce, echostr) != msg_signature:
            raise WeComCryptError("URL 验证签名不匹配")
        return self.decrypt(echostr)

    # ── 解密回调消息（用户发消息时的 POST 请求）───────────────────────
    def decrypt_message(
        self, body: str, msg_signature: str, timestamp: str, nonce: str
    ) -> str:
        """从加密回调 XML 体中校验签名并解出明文 XML。"""
        try:
            encrypt = ET.fromstring(body).findtext("Encrypt") or ""
        except ET.ParseError as e:
            raise WeComCryptError(f"回调体非合法 XML：{e}") from e
        if not encrypt:
            raise WeComCryptError("回调体缺少 Encrypt 字段")
        if sha1_signature(self.token, timestamp, nonce, encrypt) != msg_signature:
            raise WeComCryptError("回调消息签名不匹配")
        return self.decrypt(encrypt)

    # ── 加密被动响应（微信客服走主动 send_msg，一般用不到；保留完整性）──
    def encrypt_message(self, reply_xml: str, timestamp: str, nonce: str) -> str:
        """把明文 XML 加密封装成可回给企业微信的密文 XML。"""
        encrypt = self.encrypt(reply_xml)
        signature = sha1_signature(self.token, timestamp, nonce, encrypt)
        return (
            "<xml>"
            f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            "</xml>"
        )
