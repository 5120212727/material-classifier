"""召回层：从历史库中找候选类目。

两条独立通道，互不干扰：
- exact 通道：归一化名称完全相等的历史记录（强信号）
- knn   通道：名称+规格的 TF-IDF 近邻（弱信号，但覆盖面广）

索引会缓存到 cache/，历史库不变则不重建。
"""
from __future__ import annotations

import hashlib
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..normalize.rules import Normalizer
from ..schema import Candidate, Item

DIRTY_B_PATTERNS = ("工程量",)


def _is_dirty(b: str, m: str, s: str, patterns: tuple[str, ...] = ()) -> bool:
    """过滤分类体系里的脏标签（自引用、把口径混进分类）。patterns 为空则不过滤。"""
    if b == m == s:
        return True
    return any(p in b or p in m for p in patterns)


class KnowledgeBase:
    """历史物料知识库。"""

    def __init__(
        self,
        path: str | Path,
        normalizer: Normalizer,
        cache_dir: str | Path = "cache",
        name_weight: float = 0.75,
        spec_weight: float = 0.25,
        min_df: int = 2,
        exclude_keys: set[tuple[str, str]] | None = None,
        dirty_patterns: tuple[str, ...] = DIRTY_B_PATTERNS,
    ):
        self.path = Path(path)
        self.nz = normalizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name_weight = name_weight
        self.spec_weight = spec_weight
        self.min_df = min_df
        self.dirty_patterns = tuple(dirty_patterns)
        # 评测时用于剔除留出集，避免训练/测试泄漏
        self.exclude_keys = exclude_keys or set()

        self.records: list[dict] = []
        self.name_index: dict[str, list[int]] = {}
        self._vn: TfidfVectorizer | None = None
        self._vs: TfidfVectorizer | None = None
        self._Xs = None
        self._Xa = None
        self._nn: NearestNeighbors | None = None
        self.stats: dict = {}

    # ---------- 构建 ----------
    def _cache_key(self) -> str:
        st = self.path.stat()
        ex = hashlib.md5("|".join(sorted(f"{a}\x00{b}" for a, b in self.exclude_keys)).encode()).hexdigest()[:8] if self.exclude_keys else "none"
        dp = hashlib.md5("|".join(self.dirty_patterns).encode()).hexdigest()[:6]
        raw = f"{self.path.name}|{st.st_size}|{int(st.st_mtime)}|{self.name_weight}|{self.spec_weight}|{self.min_df}|{ex}|{dp}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def build(self, force: bool = False) -> "KnowledgeBase":
        cache = self.cache_dir / f"kb_{self._cache_key()}.pkl"
        if cache.exists() and not force:
            with open(cache, "rb") as f:
                d = pickle.load(f)
            self.__dict__.update(d)
            return self
        self._load_rows()
        self._fit()
        with open(cache, "wb") as f:
            pickle.dump(
                {
                    "records": self.records,
                    "name_index": self.name_index,
                    "_vn": self._vn,
                    "_vs": self._vs,
                    "_Xs": self._Xs,
                    "_Xa": self._Xa,
                    "_nn": self._nn,
                    "stats": self.stats,
                },
                f,
                protocol=4,
            )
        return self

    def _load_rows(self) -> None:
        wb = openpyxl.load_workbook(self.path, data_only=True, read_only=True)
        ws = wb.worksheets[0]
        raw: list[tuple[str, str, str, str, str]] = []
        for r in ws.iter_rows(min_row=2, max_col=5, values_only=True):
            if not r:
                continue
            r = tuple(r) + (None,) * 5
            nm, sp, b, m, s = [(str(x).strip() if x is not None else "") for x in r[:5]]
            if b and m and s:
                raw.append((nm, sp, b, m, s))
        wb.close()

        n_all = len(raw)
        # 按 (名称, 规格) 归一化后聚合，标签取众数
        grp: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for nm, sp, b, m, s in raw:
            key = (self.nz.name(nm), self.nz.spec(sp))
            if self.exclude_keys and key in self.exclude_keys:
                continue
            grp[key].append((b, m, s))

        records = []
        dirty = 0
        conflict = 0
        for (k1, k2), labs in grp.items():
            c = Counter(labs)
            (b, m, s), _ = c.most_common(1)[0]
            if _is_dirty(b, m, s, self.dirty_patterns):
                dirty += 1
                continue
            if len(c) > 1:
                conflict += 1
            records.append({"nk": k1, "sk": k2, "b": b, "m": m, "s": s})
        self.records = records

        ni: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            ni[rec["nk"]].append(i)
        self.name_index = dict(ni)

        self.stats = {
            "rows_labeled": n_all,
            "unique_records": len(grp),
            "kept": len(records),
            "dropped_dirty": dirty,
            "label_conflicts": conflict,
            "names": len(self.name_index),
        }

    def _fit(self) -> None:
        self._vn = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=self.min_df, sublinear_tf=True)
        Xn = self._vn.fit_transform([r["nk"] for r in self.records])
        self._vs = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=self.min_df, sublinear_tf=True)
        self._Xs = self._vs.fit_transform([r["sk"] for r in self.records])
        self._Xa = hstack([Xn * self.name_weight, self._Xs * self.spec_weight]).tocsr()

    # ---------- 检索 ----------
    def search(
        self,
        items: list[Item],
        topk: int = 10,
        exact_cap: int = 50,
    ) -> tuple[list[list[Candidate]], list[list[Candidate]]]:
        """返回 (exact_cands, knn_cands)，与 items 等长。"""
        if not self.records:
            raise RuntimeError("知识库为空，请先 build()")
        if self._nn is None:
            k = min(max(topk, 10), len(self.records))
            self._nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1).fit(self._Xa)

        q_n = self._vn.transform([it.name_norm for it in items])
        q_s = self._vs.transform([it.spec_norm for it in items])
        q_a = hstack([q_n * self.name_weight, q_s * self.spec_weight]).tocsr()

        k = min(topk, len(self.records))
        dist, ind = self._nn.kneighbors(q_a, n_neighbors=k)
        sim = 1.0 - dist

        exact_out: list[list[Candidate]] = []
        knn_out: list[list[Candidate]] = []
        for qi, it in enumerate(items):
            # exact 通道：同名记录，按规格相似度排序（规格相似度用稀疏点积算）
            ids = self.name_index.get(it.name_norm, [])
            ec: list[Candidate] = []
            if ids:
                ids = ids[:exact_cap]
                ss = (self._Xs[ids] @ q_s[qi].T).toarray().ravel()
                order = np.argsort(-ss)
                for j in order[:topk]:
                    rid = ids[int(j)]
                    rec = self.records[rid]
                    ec.append(Candidate(rec["b"], rec["m"], rec["s"], float(ss[int(j)]), rid, rec["nk"], rec["sk"]))
            # knn 通道
            kc = []
            for j in range(k):
                rid = int(ind[qi][j])
                rec = self.records[rid]
                kc.append(Candidate(rec["b"], rec["m"], rec["s"], float(sim[qi][j]), rid, rec["nk"], rec["sk"]))
            exact_out.append(ec)
            knn_out.append(kc)
        return exact_out, knn_out

    def class_vocab(self) -> dict[str, list[str]]:
        """返回大类 -> 中类列表、中类 -> 小类列表，供 LLM 提示词使用。"""
        bm: dict[str, set[str]] = defaultdict(set)
        ms: dict[str, set[str]] = defaultdict(set)
        for r in self.records:
            bm[r["b"]].add(r["m"])
            ms[r["m"]].add(r["s"])
        return {"b2m": {k: sorted(v) for k, v in bm.items()}, "m2s": {k: sorted(v) for k, v in ms.items()}}
