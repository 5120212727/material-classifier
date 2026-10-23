"""流水线编排：把五层串起来。

这一层不含任何算法，只负责装配。换策略 / 换模型都不需要改这里。
"""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from .decide.base import BaseClassifier
from .decide.knn import KnnVoteClassifier
from .decide.llm_direct import LLMDirectClassifier
from .decide.llm_rerank import LLMRerankClassifier
from .ingest.reader import read_workbook
from .llm.client import load_client
from .normalize.rules import Normalizer
from .postprocess.writer import write_overlay, write_standard
from .retrieve.index import KnowledgeBase
from .schema import Item, Prediction, RunStats

STRATEGIES = {
    "knn_vote": KnnVoteClassifier,
    "llm_rerank": LLMRerankClassifier,
    "llm_direct": LLMDirectClassifier,
}


class Pipeline:
    def __init__(self, config: dict):
        self.cfg = config
        nz = config.get("normalize", {})
        self.nz = Normalizer(
            strip_tags=bool(nz.get("strip_tags", True)),
            drop_placeholders=bool(nz.get("drop_placeholders", True)),
        )
        self.kb: KnowledgeBase | None = None
        self.clf: BaseClassifier | None = None
        self.strategy_name = ""

    # ---------- 装配 ----------
    def build(self, force_rebuild: bool = False) -> "Pipeline":
        rc = self.cfg.get("retrieve", {})
        self.kb = KnowledgeBase(
            self.cfg["knowledge_base"],
            self.nz,
            cache_dir=self.cfg.get("cache_dir", "cache"),
            name_weight=float(rc.get("name_weight", 0.75)),
            spec_weight=float(rc.get("spec_weight", 0.25)),
            min_df=int(rc.get("min_df", 2)),
            dirty_patterns=tuple(rc.get("dirty_b_patterns", ("工程量",))),
        ).build(force=force_rebuild)

        dc = self.cfg.get("decide", {})
        name = dc.get("strategy", "knn_vote")
        if name not in STRATEGIES:
            raise KeyError(f"未知策略 '{name}'，可选：{list(STRATEGIES)}")
        params = dict(dc.get(name, {}))
        params.setdefault("topk", rc.get("topk", 10))

        if name.startswith("llm_"):
            model = dc.get("model")
            if not model:
                raise KeyError("使用 LLM 策略时必须在 decide.model 指定模型名")
            client = load_client(model, self.cfg.get("models_path", "config/models.yaml"))
            self.clf = STRATEGIES[name](self.kb, params, client=client)
        else:
            self.clf = STRATEGIES[name](self.kb, params)
        self.strategy_name = name
        return self

    # ---------- 执行 ----------
    def run(
        self,
        input_path: str | Path,
        sheets: list[str] | None = None,
        out_path: str | Path | None = None,
        save_layout: bool = False,
        overlay: bool = False,
    ) -> tuple[list[Item], list[Prediction], RunStats]:
        t0 = time.time()
        if self.kb is None or self.clf is None:
            self.build()

        layout_dir = self.cfg.get("layout_dir", "config/layouts")
        sc = self.cfg.get("sheets", {}) or {}
        exclude = sc.get("exclude", []) or []
        if sheets is None and sc.get("include"):
            sheets = list(sc["include"])
        items, layouts = read_workbook(
            input_path, sheets=sheets, layout_dir=layout_dir,
            save_detected=save_layout, exclude_sheets=exclude,
        )
        stats = RunStats(total=len(items))
        if not items:
            stats.elapsed_s = time.time() - t0
            return [], [], stats

        for it in items:
            it.name_norm = self.nz.name(it.name)
            it.spec_norm = self.nz.spec(it.spec)

        topk = int(self.cfg.get("retrieve", {}).get("topk", 10))
        exact, knn = self.kb.search(items, topk=topk)

        # 分块，避免 LLM 策略一次排太多
        preds = self.clf.classify(items, exact, knn)

        stats.ok = sum(1 for p in preds if p.b)
        stats.failed = stats.total - stats.ok
        for p in preds:
            stats.by_strategy[p.strategy] = stats.by_strategy.get(p.strategy, 0) + 1
            stats.by_band[p.band] = stats.by_band.get(p.band, 0) + 1
        stats.elapsed_s = time.time() - t0

        if out_path:
            if overlay:
                write_overlay(input_path, items, preds, out_path)
            else:
                write_standard(items, preds, out_path)
        return items, preds, stats


def load_config(path: str | Path) -> dict:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # 相对路径按配置文件所在目录解析
    base = p.resolve().parent.parent if p.parent.name == "config" else p.resolve().parent
    for k in ("knowledge_base", "cache_dir", "layout_dir", "models_path"):
        if k in cfg and cfg[k] and not Path(cfg[k]).is_absolute():
            cfg[k] = str(base / cfg[k])
        elif k not in cfg:
            defaults = {"cache_dir": "cache", "layout_dir": "config/layouts", "models_path": "config/models.yaml"}
            if k in defaults:
                cfg[k] = str(base / defaults[k])
    return cfg
