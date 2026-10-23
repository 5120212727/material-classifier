"""策略 A：纯规则判定（不依赖任何模型）。

两条路径：
1. 名称已对齐 → 在同名记录中用规格相似度挑最接近的一条，分类直接继承
2. 名称未对齐 → Top-K 近邻加权投票

置信度设计（这是上一版最大的缺陷，此处已修正）：
- 名称对齐路径：置信度 = 校准系数 × 同名记录的**分类一致率**
  （旧版错用了规格相似度，导致分类正确但置信度只有 0.1）
- 近邻路径：置信度 = 大类投票的**加权一致度**
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..schema import Candidate, Item, Prediction
from .base import BaseClassifier


def _weighted_mode(cands: list[Candidate]) -> tuple[tuple[str, str, str], float]:
    """按 score 加权投票，返回 (众数分类, 一致度)。"""
    w: dict[tuple[str, str, str], float] = defaultdict(float)
    tot = 0.0
    for c in cands:
        ww = max(c.score, 1e-6)
        w[c.key] += ww
        tot += ww
    if not w:
        return ("", "", ""), 0.0
    best = max(w, key=w.get)
    return best, (w[best] / tot if tot else 0.0)


class KnnVoteClassifier(BaseClassifier):
    """规则基线。无需 GPU、无需 API，单机秒级。"""

    name = "knn_vote"

    def __init__(self, kb=None, config: dict | None = None):
        super().__init__(kb, config)
        self.topk = int(self.config.get("topk", 5))
        # 名称对齐路径的置信度上限，应对应其真实准确率（约 87.5%）
        self.exact_conf_cap = float(self.config.get("exact_conf_cap", 0.88))
        self.knn_topk = int(self.config.get("knn_topk", 5))
        # 绝对相似度的校准区间：低于 sim_low 视为库内无相近物料
        self.sim_low = float(self.config.get("sim_low", 0.30))
        self.sim_high = float(self.config.get("sim_high", 0.70))
        # 低于该相似度直接拒识，避免"库里没有却硬归一个最近类"
        self.reject_sim = float(self.config.get("reject_sim", 0.28))

    def classify(self, items, exact_cands, knn_cands) -> list[Prediction]:
        out: list[Prediction] = []
        for it, ec, kc in zip(items, exact_cands, knn_cands):
            if ec:
                out.append(self._by_exact(it, ec, kc))
            else:
                out.append(self._by_knn(it, kc))
        return out

    # ---------- 路径一：名称已对齐 ----------
    def _by_exact(self, it: Item, ec: list[Candidate], kc: list[Candidate]) -> Prediction:
        # 同名记录的标签分布（用全部同名记录算一致率，避免被 Top-K 截断影响）
        all_ids = self.kb.name_index.get(it.name_norm, []) if self.kb else []
        labels = [
            (self.kb.records[i]["b"], self.kb.records[i]["m"], self.kb.records[i]["s"])
            for i in all_ids
        ] if all_ids else []
        if labels:
            cnt = Counter(labels)
            agree = cnt.most_common(1)[0][1] / len(labels)
            n_same = len(labels)
        else:
            cnt = Counter(c.key for c in ec)
            agree = cnt.most_common(1)[0][1] / max(1, len(ec))
            n_same = len(ec)

        # 规格消歧：ec 已按规格相似度降序，取第一条
        pick = ec[0]
        conf = self.exact_conf_cap * (0.5 + 0.5 * agree)

        top_cls, top_n = cnt.most_common(1)[0]
        reason = (
            f"名称对齐：同名历史记录 {n_same} 条，其中 {top_n} 条属「{top_cls[0]}/{top_cls[1]}/{top_cls[2]}」；"
            f"按规格挑最接近的「{pick.src_name[:18]} / {pick.src_spec[:24]}」(相似度 {pick.score:.2f})"
        )
        return Prediction(
            b=pick.b, m=pick.m, s=pick.s,
            confidence=conf, strategy=self.name, reason=reason, candidates=ec[:5],
        )

    # ---------- 路径二：近邻投票 ----------
    def _by_knn(self, it: Item, kc: list[Candidate]) -> Prediction:
        if not kc:
            return Prediction(strategy=self.name, reason="无候选")
        top = kc[: self.knn_topk]
        (b, m, s), agree_b = _weighted_mode(top)
        (_, m2, s2), _ = _weighted_mode([c for c in top if c.b == b] or top)

        top1_sim = float(top[0].score)
        # 一致性 × 绝对相似度：只有邻居足够近，投票才有意义
        span = max(self.sim_high - self.sim_low, 1e-6)
        sim_factor = min(1.0, max(0.0, (top1_sim - self.sim_low) / span))
        conf = agree_b * (0.55 + 0.45 * sim_factor)

        n_same = sum(1 for c in top if c.b == b)
        if top1_sim < self.reject_sim:
            # 拒识：库里没有足够相近的物料，不硬归
            conf = min(conf, 0.18)
            reason = (f"拒识：最近邻相似度仅 {top1_sim:.2f}，低于阈值 {self.reject_sim:.2f}，"
                      f"知识库无相近物料；参考类目「{b}」仅供参考")
        else:
            reason = (
                f"近邻投票：Top{len(top)} 中 {n_same} 条属「{b}」，加权一致度 {agree_b:.2f}，"
                f"最近邻相似度 {top1_sim:.2f}（「{top[0].src_name[:18]}」）"
            )
        return Prediction(
            b=b, m=m2, s=s2,
            confidence=round(conf, 3), strategy=self.name, reason=reason, candidates=top,
        )
