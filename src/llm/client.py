"""统一 LLM 客户端。

设计要点：
1. **只依赖 requests**，走 OpenAI 兼容协议 —— DeepSeek / Qwen / Moonshot / GLM / vLLM 全部适配
2. 换模型只改 config/models.yaml，不碰代码
3. 并发 + 超时 + 重试 + 失败返回 None，由调用方决定降级策略
4. 没有 api_key 时 available=False，上层自动退回规则策略
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests


@dataclass
class LLMConfig:
    name: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    timeout: int = 20
    max_retries: int = 2
    max_concurrency: int = 8
    max_tokens: int = 512
    extra: dict = field(default_factory=dict)

    def resolve_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.key = cfg.resolve_key()
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.key)

    def chat(self, system: str, user: str) -> str | None:
        """单次调用，失败返回 None（不抛异常）。"""
        if not self.available:
            return None
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        payload.update(self.cfg.extra or {})
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
                if resp.status_code == 200:
                    d = resp.json()
                    return d["choices"][0]["message"]["content"]
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.2 * (attempt + 1))
                    continue
                return None
            except Exception:
                if attempt >= self.cfg.max_retries:
                    return None
                time.sleep(0.8 * (attempt + 1))
        return None

    def batch_chat(self, pairs: list[tuple[str, str]]) -> list[str | None]:
        """并发批量调用，保持输入顺序。"""
        if not self.available:
            return [None] * len(pairs)
        out: list[str | None] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as pool:
            futs = {pool.submit(self.chat, s, u): i for i, (s, u) in enumerate(pairs)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    out[i] = f.result()
                except Exception:
                    out[i] = None
        return out


def parse_json(text: str | None) -> dict | None:
    """从模型输出里抠出 JSON。容忍 ```json 包裹和前后废话。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = _JSON_BLOCK.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def load_client(name: str, models_path: str = "config/models.yaml") -> LLMClient:
    """按名字从注册表加载客户端。"""
    import yaml

    p = os.path.abspath(models_path)
    if not os.path.exists(p):
        return LLMClient(LLMConfig(name=name, model=name))
    d = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("models", {})
    if name not in d:
        raise KeyError(f"models.yaml 中没有模型 '{name}'，已注册：{list(d)}")
    return LLMClient(LLMConfig(**{"name": name, **d[name]}))
