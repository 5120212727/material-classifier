# -*- coding: utf-8 -*-
"""调试驱动：LLM 策略带进度打点 + 扩大连接池（不改 agent 本体）."""
import sys
import threading
import time

import requests

sys.path.insert(0, r"D:\Projects\material-classifier")
from src.llm import client as lc  # noqa: E402
from src.pipeline import Pipeline, load_config  # noqa: E402

counter = {"done": 0, "fail": 0, "lock": threading.Lock()}
_orig_chat = lc.LLMClient.chat


def patched_chat(self, system, user):
    t0 = time.time()
    r = _orig_chat(self, system, user)
    with counter["lock"]:
        counter["done"] += 1
        if r is None:
            counter["fail"] += 1
        n = counter["done"]
        if n % 10 == 0 or counter["fail"] > 0 and n % 5 == 0:
            print(f"  [{n}] ok={counter['done']-counter['fail']} fail={counter['fail']} last={time.time()-t0:.1f}s", flush=True)
    return r


lc.LLMClient.chat = patched_chat


def patched_init(self, cfg):
    self.cfg = cfg
    self.key = cfg.resolve_key()
    s = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
    s.mount("http://", ad)
    s.mount("https://", ad)
    self._session = s


lc.LLMClient.__init__ = patched_init

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
inp = sys.argv[2] if len(sys.argv) > 2 else r"D:\Projects\material-classifier\out\副本汇总报价样例1_清理版.xlsx"
outp = sys.argv[3] if len(sys.argv) > 3 else r"D:\Projects\material-classifier\out\炼铁_llm_debug.xlsx"
model = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-27b"

cfg = load_config(r"D:\Projects\material-classifier\config\default.yaml")
cfg["decide"]["strategy"] = "llm_rerank"
cfg["decide"]["model"] = model
cfg.setdefault("decide", {}).setdefault("llm_rerank", {})["max_llm_items"] = limit

pipe = Pipeline(cfg).build()
t = time.time()
items, preds, st = pipe.run(inp, sheets=["炼铁"], out_path=outp)
print(f"stats: total={st.total} ok={st.ok} elapsed={st.elapsed_s:.1f}s")
print("bands:", dict(st.by_band))
print("strategies:", dict(st.by_strategy))
print(f"wall={time.time()-t:.1f}s")
