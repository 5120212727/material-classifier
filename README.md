# 物料分类 Agent

把采购清单 Excel 自动归到 CPM 大 / 中 / 小类。**换 LLM 只改一行配置，其余全部确定。**

---

## 快速开始

```bash
pip install -r requirements.txt

# 1. 看表头能不能自动识别
python run.py --inspect 清单.xlsx

# 2. 跑分类（规则策略，无需任何 API）
python run.py --input 清单.xlsx --out out/result.xlsx

# 3. 只跑某个页签
python run.py --input 清单.xlsx --sheets 炼铁 --out out/炼铁.xlsx
```

输出：`out/result.xlsx` + 同名 `.csv`。列含预测大中小类、置信度、置信档、判定路径、判定依据。

---

## 换模型

```bash
export DEEPSEEK_API_KEY=sk-xxx
python run.py --input 清单.xlsx --strategy llm_rerank --model deepseek-chat --out out/r1.xlsx
python run.py --input 清单.xlsx --strategy llm_rerank --model qwen-max     --out out/r2.xlsx
python run.py --input 清单.xlsx --strategy llm_rerank --model gpt-4o-mini  --out out/r3.xlsx
```

新增模型只改 `config/models.yaml`，代码零改动。没配 key 会自动降级到规则策略，不会中断。

---

## 三种判定策略

| 策略 | 说明 | 速度 | 依赖 |
|---|---|---|---|
| `knn_vote` | 名称对齐 + 近邻加权投票 | ~1 ms/条 | 无 |
| `llm_rerank` | 检索召回 Top-6 → LLM 选一个 | ~1–3 s/条 | 模型 API |
| `llm_direct` | LLM 直接选大类（对照组，不做检索） | ~1–2 s/条 | 模型 API |

**为什么 LLM 只做 Rerank 不做分类**：2,641 个小类塞不进 prompt，让它裸选准确率没保证。正确姿势是检索先召回 6 个候选，LLM 只做「6 选 1 + 说理由」。每条约 1 次调用，成本可控，增益可测。

---

## 评测（做多模型对比的前提）

```bash
# 单策略
python -m eval.runner -s knn_vote

# 多策略 / 多模型
python -m eval.runner -s knn_vote llm_rerank -m deepseek-chat
python -m eval.runner -s llm_direct llm_rerank -m qwen-max --holdout 0.15
```

评测的**关键设计**：留出集在切分**之前**先按 (名称, 规格) 去重，所以测试样本在知识库里绝对不存在完全相同的记录 —— 测的是「对没见过的物料」的能力，不是「能不能查到自己」。

当前基线（留出 10%，6,064 条测试）：

| 指标 | 值 |
|---|---|
| 大类准确率 | **79.5%** |
| 中类准确率 | 77.6% |
| 小类准确率 | 74.6% |
| 三级全对 | 72.8% |
| 覆盖 80.6% 时大类准确率 | **86.6%** |
| 覆盖 58.1% 时大类准确率 | 92.1% |

最后两行是验收该用的口径：**别谈"准确率必须 90%"，谈"覆盖 X% 时准确率 Y%"**。

---

## 架构

```
Layer 0  ingest/      输入适配     Excel → Item，布局用 yaml 描述
Layer 1  normalize/   语义归一化   全角半角、位号剥离、占位符识别
Layer 2  retrieve/    候选召回     exact（同名）+ knn（TF-IDF）双通道
Layer 3  decide/      ★ 判定插槽 ★  唯一允许接 LLM 的地方
Layer 4  postprocess/ 输出落盘     xlsx / csv / 回写副本
```

`src/pipeline.py` 只做装配，不含算法。换策略、换模型都不碰它。

### 归一化的红线

**只统一书写形式，绝不动语义。**

- 可以合并：`DN100` / `dn100` / `ＤＮ１００`、`Z41H-10C` / `Z41H-10C型`
- 绝不能合并：`DN100` 与 `DN150`、`1250A` 与 `1600A`

同义扩展（手动闸阀 ≈ 手动闸板阀）要放别名层，不要污染主键 —— 否则会造成误配，而误配比漏配严重。

---

## 新增清单格式

首次遇到新格式，自动探测表头：

```bash
python run.py --input 新清单.xlsx --save-layout --out out/tmp.xlsx
```

探测结果会写进 `config/layouts/<页签名>.yaml`，下次直接复用：

```yaml
header_row: 2
data_start_row: 4
name_col: 6        # 工程或费用名称
spec_col: 7        # 型号、规格及技术性能
skip_keywords: [小计, 合计, 总计, 合价]
```

探测失败时（例如汇总表而非明细表）手工补一个 yaml 即可。

---

## 目录

```
material-classifier/
├── run.py                    CLI 入口
├── config/
│   ├── default.yaml          全局配置
│   ├── models.yaml           模型注册表 ← 换模型只改这里
│   └── layouts/              各页签解析布局
├── src/
│   ├── schema.py             Item / Candidate / Prediction
│   ├── pipeline.py           装配与编排
│   ├── ingest/reader.py      布局探测 + 明细抽取
│   ├── normalize/rules.py    归一化 + 参数抽取（二期用）
│   ├── retrieve/index.py     知识库 + 双通道召回
│   ├── decide/               策略：knn / llm_rerank / llm_direct
│   ├── llm/client.py         统一客户端（OpenAI 兼容）
│   └── postprocess/writer.py 输出
├── eval/runner.py            留出集评测
├── data/history.xlsx         知识库
└── out/                      结果
```

---

## 已知边界

- **知识库覆盖决定上限**。炼铁设备仅 3,296 条（占已标注 2.8%），该专业准确率天然低于给排水 / 供电 / 传动。
- **小类长尾**。2,641 个小类中 1,246 个样本不足 5 条，长尾类预测靠近邻运气，建议对样本数 <5 的类降级到「大类+中类」输出。
- **不做价格**。知识库本身不含价格字段，输出也不含。若后续要回到"出价格"口径，需先做参数结构化（见 `normalize/rules.py::extract_params` 预留接口）。
- **Excel 解析比 LLM 更容易崩**。表头跨行、合并单元格、分组行混排都会影响抽取，`--inspect` 先看清楚再跑。
