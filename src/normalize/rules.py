"""语义归一化层。

设计原则（很重要）：**只统一书写形式，绝不动语义。**
- 可以合并：全角/半角、空格、大小写、位号前缀
- 绝不能合并：DN100 与 DN150、1250A 与 1600A

因为下游要判断"参数是否一致"，一旦把参数值抹平，就会造成误配。
"""
from __future__ import annotations

import re
import unicodedata

# 设备位号：1-3 个大写字母 + 1-4 位数字，后面紧跟中文或括号
# 白名单排除：DN（公称直径）、PN（公称压力）、MPa 等口径/压力标识
_TAG_HEAD = re.compile(r"^(?![Dd][Nn])(?![Pp][Nn])(?![Mm][Pp][Aa])[A-Z]{1,3}\d{1,4}(?=[\u4e00-\u9fa5(（])")
_TAG_TAIL = re.compile(r"(?<=[\u4e00-\u9fa5)）])[A-Z]{1,3}\d{2,4}$")
_TAG_PAREN = re.compile(r"[（(][A-Z]{1,3}\d{1,4}[)）]")
# 罗马数字 / 中文序号前缀，如「一、」「Ⅰ」
_SEQ_HEAD = re.compile(r"^(?:[一二三四五六七八九十]+[、.]|[ⅠⅡⅢⅣⅤ]\s*)")

# 无信息占位符
PLACEHOLDERS = [
    "详见技术规格书", "详见技术附件", "详见技术协议", "详见规格书", "详见清单",
    "详见设备表", "详见招标文件", "详见技术规格", "详见图纸", "详见附图",
    "详见附件", "见图纸", "见附图", "见图", "同上", "甲供", "待定", "无",
]

_PUNCT = re.compile(r"[\s，,。;；、·\.\(\)（）\[\]【】:：/\\\-—–_*#\"'“”‘’]+")


class Normalizer:
    """可配置的归一化器。所有规则集中在 __init__，便于版本化管理。"""

    def __init__(self, strip_tags: bool = True, drop_placeholders: bool = True):
        self.strip_tags = strip_tags
        self.drop_placeholders = drop_placeholders
        self._ph_norm = [self._basic(p) for p in PLACEHOLDERS]

    # ---------- 基础清洗 ----------
    @staticmethod
    def _basic(s: str) -> str:
        """全角转半角 + 去换行 + 压空白。不改变语义。"""
        if s is None:
            return ""
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("\u3000", " ").replace("\xa0", " ")
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    # ---------- 位号剥离 ----------
    def strip_tag(self, s: str) -> str:
        if not self.strip_tags or not s:
            return s
        s = _SEQ_HEAD.sub("", s)
        s = _TAG_PAREN.sub("", s)
        s = _TAG_HEAD.sub("", s)
        s = _TAG_TAIL.sub("", s)
        return s.strip()

    # ---------- 占位符 ----------
    def is_placeholder(self, s: str) -> bool:
        """判断规格是否为「详见技术规格书」这类无信息文本。"""
        t = self._basic(s)
        if not t:
            return True
        neg = _PUNCT.sub("", t.lower())
        if not neg:
            return True
        return any(neg == p or (p and neg in p) for p in self._ph_norm)

    def drop_ph(self, s: str) -> str:
        if not self.drop_placeholders:
            return s
        t = self._basic(s)
        low = _PUNCT.sub("", t.lower())
        for p in self._ph_norm:
            if p and low == p:
                return ""
        return t

    # ---------- 对外主入口 ----------
    def name(self, s: str) -> str:
        """名称归一化：清洗 + 去位号 + 去标点 + 小写。"""
        t = self._basic(s)
        t = self.strip_tag(t)
        t = self.drop_ph(t)
        return _PUNCT.sub("", t.lower())

    def spec(self, s: str) -> str:
        """规格归一化：只做清洗与占位符剔除，保留所有参数值与符号。"""
        return _PUNCT.sub("", self.drop_ph(s).lower())

    def display(self, s: str) -> str:
        """给人看的形态：清洗 + 去位号，保留可读性。"""
        return self.strip_tag(self._basic(s))


# ---------- 参数抽取（为二期"参数级比对"预留） ----------
# 只抽最通用的几类，单位与符号并存时统一到符号形式
_PARAM_PATTERNS = {
    "DN": re.compile(r"(?:DN|dn|Dg)\s*([0-9]{2,4})"),
    "PN": re.compile(r"(?:PN|pn)\s*([0-9]+(?:\.[0-9]+)?)"),
    "MPa": re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MPa|Mpa|mpa)"),
    "kW": re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:kW|KW|kw|千瓦)"),
    "mm": re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*mm(?![a-zA-Z])"),
    "t": re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:t(?![a-zA-Z])|吨)"),
    "m3": re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m3|m³|立方米)"),
    "C": re.compile(r"([0-9]+)\s*(?:℃|°C)"),
    "Q": re.compile(r"(?:Q|q)\s*[=＝]\s*([0-9]+(?:\.[0-9]+)?)"),
}


def extract_params(spec: str) -> dict[str, str]:
    """从规格文本抽结构化参数。二期做参数比对时用得上。"""
    t = Normalizer._basic(spec)
    out: dict[str, str] = {}
    for k, pat in _PARAM_PATTERNS.items():
        m = pat.search(t)
        if m:
            out[k] = m.group(1)
    return out
