#!/usr/bin/env python3
"""CLI 入口。

用法：
  # 跑分类（默认规则策略）
  python run.py --input 清单.xlsx --out out/result.xlsx

  # 换策略 / 换模型（一行参数的事）
  python run.py --input 清单.xlsx --strategy llm_rerank --model deepseek-chat --out out/r2.xlsx

  # 只看表头能不能识别
  python run.py --inspect 清单.xlsx

  # 重建知识库索引（历史库变了才需要）
  python run.py --input 清单.xlsx --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingest.reader import inspect as inspect_xlsx  # noqa: E402
from src.pipeline import Pipeline, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="物料分类 Agent")
    ap.add_argument("--input", "-i", help="待分类的清单 Excel")
    ap.add_argument("--config", "-c", default="config/default.yaml")
    ap.add_argument("--strategy", "-s", help="覆盖配置中的策略：knn_vote | llm_rerank | llm_direct")
    ap.add_argument("--model", "-m", help="覆盖配置中的模型名（见 config/models.yaml）")
    ap.add_argument("--sheets", help="只处理指定页签，逗号分隔")
    ap.add_argument("--out", "-o", help="输出路径，默认 out/<输入名>_<策略>.xlsx")
    ap.add_argument("--rebuild", action="store_true", help="强制重建知识库索引")
    ap.add_argument("--save-layout", action="store_true", help="把自动探测到的布局写回 config/layouts")
    ap.add_argument("--overlay", action="store_true", help="输出为原清单副本 + 结果页签")
    ap.add_argument("--inspect", help="只探测表头结构，不执行分类")
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect_xlsx(args.inspect)
        return 0
    if not args.input:
        ap.error("需要 --input 或 --inspect")

    cfg = load_config(args.config)
    if args.strategy:
        cfg.setdefault("decide", {})["strategy"] = args.strategy
    if args.model:
        cfg.setdefault("decide", {})["model"] = args.model

    pipe = Pipeline(cfg)
    if not args.quiet:
        print(f"· 加载知识库 {cfg['knowledge_base']} ...", end="", flush=True)
    pipe.build(force_rebuild=args.rebuild)
    s = pipe.kb.stats
    if not args.quiet:
        print(f"\r· 知识库：保留 {s['kept']:,} 条唯一记录 / 剔除脏标签 {s['dropped_dirty']:,} 条 / "
              f"标签冲突 {s['label_conflicts']:,} 条 / 不同名称 {s['names']:,} 个")

    sheets = [x.strip() for x in args.sheets.split(",")] if args.sheets else None
    out = args.out
    if not out:
        stem = Path(args.input).stem
        out = f"out/{stem}_{pipe.strategy_name}.xlsx"

    items, preds, st = pipe.run(
        args.input, sheets=sheets, out_path=out, save_layout=args.save_layout, overlay=args.overlay
    )

    if not args.quiet:
        print(f"· 清单：{st.total} 条（{len({i.sheet for i in items})} 个页签）")
        print(f"· 策略：{pipe.strategy_name}")
        print(f"· 已给出分类：{st.ok} / {st.total}"
              + (f"　未给出：{st.failed}" if st.failed else ""))
        print("· 置信度分布：" + "　".join(f"{k} {v}" for k, v in sorted(st.by_band.items())))
        print("· 判定路径：" + "　".join(f"{k} {v}" for k, v in sorted(st.by_strategy.items(), key=lambda x: -x[1])))
        print(f"· 耗时：{st.elapsed_s:.1f}s　吞吐：{st.total / max(st.elapsed_s, 0.01):.0f} 条/秒")
        print(f"· 输出：{out}")

    if preds:
        band = {k: 0 for k in ("高", "中", "低")}
        for p in preds:
            band[p.band] = band.get(p.band, 0) + 1
        hi = band["高"] / len(preds) * 100
        if not args.quiet:
            print(f"· 高置信占比 {hi:.1f}%（这批可直接自动确认）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
