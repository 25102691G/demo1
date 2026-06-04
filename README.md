# Guideline Skill Engine Workflow

本项目用于把医学指南 PDF 转为可执行的 guideline skill pack，并通过通用 `SkillEngine`
对用户病例输入执行 workflow，最终生成符合 schema 的 JSON 输出。

当前主流程只有三步：

1. PDF -> recommendation cards JSONL
2. JSONL -> `skill.yaml` + `cards.jsonl`
3. 运行 `scripts/run_skill_engine.py` 生成 `workflow_output` JSON

本项目输出仅用于指南知识组织和临床决策支持，不替代医生诊断、处方或急诊处理。

## 环境准备

建议使用 Python 3.10+。

```powershell
pip install -r requirements.txt
```

运行测试：

```powershell
pytest
```

## 1. PDF 到 JSONL

推荐使用 `guideline_skill.cli` 从指南 PDF 抽取 `result.jsonl`。这是当前和后续
`scripts/build_skill_pack.py` 对接的主路径。

输入：

- 指南 PDF

输出：

- JSONL 文件，每行一个抽取单元或推荐卡片
- `summary.json`，记录分类、抽取数量和人工复核数量等信息

示例：

```powershell
python -m guideline_skill.cli extract --input data/guides/example.pdf
```

默认输出到：

```text
data/extractor/<PDF文件名>/
  result.jsonl
  summary.json
```

批量抽取：

```powershell
python -m guideline_skill.cli batch --input-dir data/guides
```

## 2. JSONL 到 skill.yaml

使用 `scripts/build_skill_pack.py` 将抽取后的 JSONL 构建为 skill pack。

输入：

- `data/extractor/<指南名>/result.jsonl`

输出：

```text
data/skill/<指南名>/
  skill.yaml
  cards.jsonl
```

单文件构建示例：

```powershell
python scripts/build_skill_pack.py --cards data/extractor/中国克罗恩病诊治指南（2023年·广州）/result.jsonl --force
```

批量构建示例：

```powershell
python scripts/build_skill_pack.py --cards data/extractor --force
```

`--cards` 支持：

- 单个 `.jsonl` 文件
- 一个目录，脚本会递归处理其中的 `*.jsonl`

构建时会执行：

- `schema/recommendation_card.schema.json` 校验
- `schema/skill_pack.schema.json` 校验
- workflow entrypoint 校验
- workflow transition 目标校验
- workflow subskill 引用校验
- output template 引用校验
- subskill 中引用的 card_id 存在性校验

只校验不写文件：

```powershell
python scripts/build_skill_pack.py --cards data/extractor --out-dir data/skill --skill-schema schema/skill_pack.schema.json --card-schema schema/recommendation_card.schema.json --schema-version 0.3 --dry-run
```

## 3. 运行 SkillEngine

使用 `scripts/run_skill_engine.py` 输入病例并执行 skill workflow。

支持三种输入方式：

- `--input-text`：直接传入病例文本
- `--input-file`：读取病例文本文件
- `--case-json`：读取已经部分结构化的病例 JSON

示例：

```powershell
python scripts/run_skill_engine.py --input-text "腹痛腹泻半年，体重下降，肛瘘，粪便钙卫蛋白升高" --debug
```

## 输出文件

`data/skill/<指南名>/skill.yaml`

- skill pack 的声明式配置
- 包含 metadata、routing_profile、workflow、subskills、output_templates 等
- 不内嵌全部推荐卡片正文，只引用 `cards.jsonl`

`data/skill/<指南名>/cards.jsonl`

- recommendation cards
- 每行一个 card
- workflow 执行时按 card_id 引用

`data/runs/YYYYMMDD_HH_MM.json`

- SkillEngine 的最终输出
- 符合 `schema/workflow_output.schema.json`
- 包含 top candidates、workflow 执行步骤、最终响应、安全信息、引用和 debug trace

## Schema 校验

主流程中会真实执行 schema 校验。

PDF 到 JSONL 后，构建 skill pack 时校验：

```text
schema/recommendation_card.schema.json
schema/skill_pack.schema.json
```

运行 SkillEngine 时校验：

```text
schema/canonical_case.schema.json
schema/workflow_output.schema.json
```

如果校验失败，脚本会报错退出，并输出包含 json path 和 schema path 的错误信息。

## 当前限制

- 病例规范化是规则版，抽取粒度较粗。
- 路由是规则匹配，不是诊断概率。
- 当前不调用 LLM。
- 当前不做向量检索或 embedding 语义检索。
- PDF 抽取结果需要人工审核。
- 管理建议只组织指南卡片内容，不生成个体化处方或剂量。
- 有安全红旗时，workflow 会优先停止并提示临床安全评估。
