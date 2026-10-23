"""策略 B：LLM 重排（RAG + Rerank）。

正确的用法是让 LLM 做「从候选里选」，而不是「从 2641 个类里裸造」：

1. 检索层先召回 Top-K 历史记录
2. 聚合成 6~8 个互不相同的候选类目（每个类目附一条历史依据）
3. LLM 只在这几个里选，输出结构化 JSON
4. 解析失败 / 超时 → 自动降级到规则策略

这样每条约 1 次调用，成本可控，准确率可测，还能拿到可读的判定理由。
"""
from __future__ import annotations

from collections import OrderedDict

from ..llm.client import LLMClient, parse_json
from ..schema import Candidate, Item, Prediction
from .base import BaseClassifier

SYSTEM = (
    "你是钢铁冶金工程采购物料分类专家。"
    "任务：把采购清单里的一条物料，归类到给定的候选类目之一。"
    "必须严格从候选中选择，不得自创类目。"
    "重要原则：候选里附带的「历史依据」是本项目的权威分类口径。"
    "如果某个候选的历史依据与该物料的名称/规格高度重合（近乎同一条设备），"
    "必须选择该候选，即使它的类目看起来不符合通用工程知识。"
    "只输出 JSON，不要任何解释性文字。"
)

USER_TMPL = """物料名称：{name}
规格型号：{spec}

候选类目（只能从中选一个）：
{cands}

请输出 JSON：{{"choice": 序号, "confidence": 0到1之间的小数, "reason": "不超过30字的判断依据"}}"""


def _build_candidates(ec: list[Candidate], kc: list[Candidate], cap: int) -> list[Candidate]:
    """把候选记录聚合成「互不相同的类目」，每个类目保留分数最高的那条作为依据。"""
    pool: "OrderedDict[tuple[str, str, str], Candidate]" = OrderedDict()
    for c in ec + kc:
        if c.key in pool:
            if c.score > pool[c.key].score:
                pool[c.key] = c
        else:
            pool[c.key] = c
    ranked = sorted(pool.values(), key=lambda c: -c.score)[:cap]
    return ranked


class LLMRerankClassifier(BaseClassifier):
    name = "llm_rerank"

    def __init__(self, kb=None, config: dict | None = None, client: LLMClient | None = None):
        super().__init__(kb, config)
        self.client = client
        self.cap = int(self.config.get("candidate_cap", 6))
        self.max_items = int(self.config.get("max_llm_items", 0))  # 0 = 不限，用于省钱调试

    def classify(self, items, exact_cands, knn_cands) -> list[Prediction]:
        if self.client is None or not self.client.available:
            # 未配置模型 → 直接交给规则策略，保证系统永远能出结果
            from .knn import KnnVoteClassifier

            sub = KnnVoteClassifier(self.kb, self.config)
            preds = sub.classify(items, exact_cands, knn_cands)
            for p in preds:
                p.strategy = f"{self.name}(降级:{sub.name})"
            return preds

        cand_box = [_build_candidates(ec, kc, self.cap) for ec, kc in zip(exact_cands, knn_cands)]
        limit = self.max_items if self.max_items > 0 else len(items)
        pairs = []
        for i, it in enumerate(items):
            if i >= limit or not cand_box[i]:
                pairs.append(("", ""))
                continue
            lines = []
            for j, c in enumerate(cand_box[i], 1):
                lines.append(f"{j}. {c.b} / {c.m} / {c.s}\n   历史依据：{c.src_name[:26]}｜{c.src_spec[:40]}")
            pairs.append((SYSTEM, USER_TMPL.format(
                name=it.name[:60], spec=it.spec[:180], cands="\n".join(lines))))

        active = [(i, p) for i, p in enumerate(pairs) if p[0]]
        replies = self.client.batch_chat([p for _, p in active]) if active else []
        resp_map = {active[k][0]: replies[k] for k in range(len(active))} if active else {}

        from .knn import KnnVoteClassifier

        rule = KnnVoteClassifier(self.kb, self.config)
        rule_preds = rule.classify(items, exact_cands, knn_cands)

        out: list[Prediction] = []
        for i, (it, cands) in enumerate(zip(items, cand_box)):
            if i not in resp_map or not cands:
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:无响应)"
                out.append(p)
                continue
            d = parse_json(resp_map[i])
            if not d or not isinstance(d.get("choice"), (int, float, str)):
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:解析失败)"
                out.append(p)
                continue
            try:
                k = int(d["choice"]) - 1
            except Exception:
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:序号非法)"
                out.append(p)
                continue
            if not (0 <= k < len(cands)):
                p = rule_preds[i]
                p.strategy = f"{self.name}(降级:越界)"
                out.append(p)
                continue
            pick = cands[k]
            self_conf = float(d.get("confidence", 0.6) or 0.6)
            self_conf = min(max(self_conf, 0.05), 0.98)
            # 融合：模型自评 + 检索证据
            conf = round(0.55 * self_conf + 0.45 * max(pick.score, rule_preds[i].confidence * 0.9), 4)
            out.append(Prediction(
                b=pick.b, m=pick.m, s=pick.s,
                confidence=min(conf, 0.98),
                strategy=self.name,
                reason=f"LLM 选第{k + 1}项（自评{self_conf:.2f}）：{str(d.get('reason', ''))[:40]}",
                candidates=cands,
            ))
        return out
