"""策略 C：LLM 直选（对照组，不做检索召回）。

用途：证明「检索 + 重排」相对「纯 LLM 分类」的价值。
流程：LLM 先选大类 → 在该大类下用近邻找最相似记录 → 取其中类小类。
"""
from __future__ import annotations

from ..llm.client import LLMClient, parse_json
from ..schema import Candidate, Item, Prediction
from .base import BaseClassifier

SYSTEM = "你是钢铁冶金工程采购物料分类专家。只能从给定的大类列表中选择一个，只输出 JSON。"

USER_TMPL = """物料名称：{name}
规格型号：{spec}

可选大类（共 {n} 个）：
{blist}

请输出 JSON：{{"choice": 序号, "confidence": 0到1之间的小数, "reason": "不超过30字"}}"""


class LLMDirectClassifier(BaseClassifier):
    name = "llm_direct"

    def __init__(self, kb=None, config: dict | None = None, client: LLMClient | None = None):
        super().__init__(kb, config)
        self.client = client

    def classify(self, items, exact_cands, knn_cands) -> list[Prediction]:
        from .knn import KnnVoteClassifier

        rule = KnnVoteClassifier(self.kb, self.config)
        if self.client is None or not self.client.available or self.kb is None:
            preds = rule.classify(items, exact_cands, knn_cands)
            for p in preds:
                p.strategy = f"{self.name}(降级:{rule.name})"
            return preds

        big = sorted({r["b"] for r in self.kb.records})
        blist = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(big))
        pairs = [(SYSTEM, USER_TMPL.format(name=it.name[:60], spec=it.spec[:180], n=len(big), blist=blist))
                 for it in items]
        replies = self.client.batch_chat(pairs)

        rule_preds = rule.classify(items, exact_cands, knn_cands)
        out: list[Prediction] = []
        for i, (it, kc) in enumerate(zip(items, knn_cands)):
            d = parse_json(replies[i])
            if not d:
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:解析失败)"
                out.append(p)
                continue
            try:
                b = big[int(d["choice"]) - 1]
            except Exception:
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:序号非法)"
                out.append(p)
                continue
            # 在该大类内挑最近的候选，补齐中类/小类
            in_b = [c for c in kc if c.b == b]
            if in_b:
                c = in_b[0]
                out.append(Prediction(
                    b=b, m=c.m, s=c.s,
                    confidence=min(float(d.get("confidence", 0.6) or 0.6), 0.9),
                    strategy=self.name,
                    reason=f"LLM 直选大类：{str(d.get('reason', ''))[:40]}；中类小类取该大类内最近邻",
                    candidates=kc[:5],
                ))
            else:
                out.append(Prediction(
                    b=b, m="", s="",
                    confidence=0.35, strategy=self.name,
                    reason=f"LLM 直选大类：{str(d.get('reason', ''))[:40]}；库内无该类目记录，中类小类留空",
                    candidates=kc[:5],
                ))
        return out
