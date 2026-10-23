"""统一数据结构。所有层之间只通过这里的类型传递，保证接口稳定。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Item:
    """一条待分类的清单物料。"""

    idx: int                      # 在源表中的行号（1-based，便于回写）
    name: str                     # 原始名称
    spec: str                     # 原始规格
    name_norm: str = ""           # 归一化名称（去位号）
    spec_norm: str = ""           # 归一化规格
    sheet: str = ""               # 来源 sheet
    raw: dict[str, Any] = field(default_factory=dict)  # 原始整行，便于回写


@dataclass
class Candidate:
    """召回层给出的一个候选类目。"""

    b: str                        # 大类
    m: str                        # 中类
    s: str                        # 小类
    score: float                  # 相似度 / 投票权重
    src_idx: int = -1             # 知识库中的行号，便于溯源
    src_name: str = ""            # 命中的知识库物料名，便于解释
    src_spec: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.b, self.m, self.s)


@dataclass
class Prediction:
    """判定层的输出。"""

    b: str = ""
    m: str = ""
    s: str = ""
    confidence: float = 0.0       # 0~1
    strategy: str = ""            # 产出该结果的策略名
    reason: str = ""              # 可读的判定理由
    candidates: list[Candidate] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def band(self) -> str:
        if self.confidence >= 0.6:
            return "高"
        if self.confidence >= 0.4:
            return "中"
        return "低"


@dataclass
class TableLayout:
    """一个 Excel 页签的解析布局。由配置或自动探测给出。"""

    sheet: str
    header_row: int               # 表头所在行（1-based）
    data_start_row: int           # 数据起始行（1-based）
    name_col: int                 # 名称列（1-based）
    spec_col: int                 # 规格列（1-based）
    skip_keywords: list[str] = field(default_factory=lambda: ["小计", "合计", "总计", "合价"])
    layout_name: str = "auto"


@dataclass
class RunStats:
    """一次运行的统计，用于 CLI 汇报与评测。"""

    total: int = 0
    ok: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    by_strategy: dict[str, int] = field(default_factory=dict)
    by_band: dict[str, int] = field(default_factory=dict)
