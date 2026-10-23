"""判定层的统一接口。

这是整个系统**唯一允许接入 LLM 的地方**。
所有策略必须实现同一个签名，这样「换策略」和「换模型」是两个正交维度，
实验才能做成 策略 × 模型 的矩阵。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import Candidate, Item, Prediction


@runtime_checkable
class Classifier(Protocol):
    """判定器协议。批量入、批量出，便于 LLM 并发。"""

    name: str

    def classify(
        self,
        items: list[Item],
        exact_cands: list[list[Candidate]],
        knn_cands: list[list[Candidate]],
    ) -> list[Prediction]:
        ...


class BaseClassifier:
    """策略基类，提供默认的兜底与计时。"""

    name = "base"

    def __init__(self, kb=None, config: dict | None = None):
        self.kb = kb
        self.config = config or {}

    def fallback(self, item: Item, cands: list[Candidate], note: str = "") -> Prediction:
        """当策略无法给出结果时，退回到最近邻 Top-1。"""
        if not cands:
            return Prediction(strategy=self.name, reason=f"无候选 {note}".strip())
        c = cands[0]
        return Prediction(
            b=c.b, m=c.m, s=c.s,
            confidence=min(c.score, 0.3),
            strategy=self.name,
            reason=f"兜底：取最近邻 {c.src_name[:20]} {note}".strip(),
            candidates=cands[:5],
        )

    def classify(self, items, exact_cands, knn_cands) -> list[Prediction]:  # pragma: no cover
        raise NotImplementedError
