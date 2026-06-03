# Medical Guideline Skill Pack Demo

## 最新 CLI 输出路径

单文件抽取不需要手动传 `--output` 和 `--summary`：

```bash
python -m guideline_skill.cli extract `
  --input "/path/to/file.pdf"
```

默认输出到：

```text
data/extractor/<PDF 文件名不含扩展名>/
  result.jsonl
  summary.json
```

批量抽取不需要手动传 `--output-dir`：

```bash
python -m guideline_skill.cli batch `
  --input-dir "data/guides"
```

批量模式默认非递归扫描目录下的 `.pdf`、`.txt`、`.md` 文件，并为每个输入文件分别创建对应子目录。`batch` 仍兼容 `--inputs "/path/a.pdf" "/path/b.pdf"` 手动指定文件，也兼容 `--output-dir` 作为输出根目录覆盖参数。

## 指南分类与抽取 CLI

通用指南分类与抽取入口：

```bash
python -m guideline_skill.cli extract `
  --input "/path/to/file.pdf"
```

批量抽取：

```bash
python -m guideline_skill.cli batch `
  --inputs "/path/a.pdf" "/path/b.pdf"
```

默认输出目录为 `data/extractor/`：

- `data/extractor/result.jsonl`：每行一个抽取单元。
- `data/extractor/summary.json`：分类分数、单元数量、人工复核数量、LLM 模型等汇总信息。

### 分类逻辑

文档先由 `AnchorRegistry` 和 `GuidelineClassifier` 分类：

- `structured_guideline`：命中足够多结构化主锚点，例如“推荐意见”“共识意见”“陈述”“Recommendation”。
- `narrative_guideline`：未命中足够结构化主锚点时，默认归为叙述性指南。

分类只依赖结构化主锚点命中情况。字段锚点，例如“证据等级”“推荐强度”“实施建议”，不会单独决定文档为结构化指南。

### 配置文件

锚点规则配置在 `configs/anchor_rules.yaml`：

- `unit_anchors` 用于结构化指南分类、主切分、statement type 推断。
- `field_anchors` 只用于结构化指南 raw 字段抽取，不作为切分边界。

标题规则配置在 `configs/heading_rules.yaml`。`HeadingSegmenter` 使用该配置识别“一、诊断”“1.1 实验室检查”“II.6.1 化疗药物”等标题层级。

### 结构化指南流程

`structured_guideline` 的处理流程：

1. 使用 `unit_anchors` 切分 `StatementUnit`。
2. 使用 `field_anchors` 抽取 raw 字段，例如 `evidence_quality_raw`、`strength_raw`、`implementation_advice`、`rationale`。
3. 使用 DeepSeek LLM 归一化 `evidence_quality_normalized` 和 `strength_normalized`。
4. 使用 Pydantic schema 和 validators 标记需要人工复核的单元。

结构化指南只对 raw 字段做规则抽取；证据等级和推荐强度归一化必须走 DeepSeek LLM。

### 叙述性指南流程

`narrative_guideline` 的处理流程：

1. 使用 `HeadingSegmenter` 按配置化标题规则切分章节。
2. 过长 segment 会先按段落或长度切成更小 chunk。
3. 每个 chunk 都交给 `ClinicalInfoExtractor`。
4. `ClinicalInfoExtractor` 全部使用 DeepSeek LLM 抽取 `ClinicalInfoUnit`，不做规则字段抽取。
5. LLM 失败或返回非法 JSON 时保留 `raw_text`，输出兜底单元并标记 `needs_human_review=true`。

### DeepSeek 环境变量

运行真实 LLM 抽取前需要配置：

```bash
export DEEPSEEK_MODEL="..."
export DEEPSEEK_BASE_URL="..."
export DEEPSEEK_API_KEY="..."
```

PowerShell：

```powershell
$env:DEEPSEEK_MODEL="..."
$env:DEEPSEEK_BASE_URL="..."
$env:DEEPSEEK_API_KEY="..."
```

缺失任一变量时，`DeepSeekClient` 会报出清晰错误：

```text
Missing DEEPSEEK_MODEL / DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY
```

### 输出示例

结构化指南 JSONL 单行示例：

```json
{"record_type":"statement_unit","guideline_meta":{"title":"...","source_file":"...","doc_type":"structured_guideline"},"unit":{"id":"statement_unit_001_xxxxxxxx","original_label":"推荐意见1：","statement_type":"recommendation","statement_text":"...","clinical_question":null,"evidence_quality_raw":"2","evidence_quality_normalized":"moderate","strength_raw":"强","strength_normalized":"strong","consensus_level":null,"implementation_advice":"...","rationale":"...","source_location":{"page_start":1,"page_end":2,"section":null},"confidence":0.9,"needs_human_review":false,"review_reasons":[]}}
```

叙述性指南 JSONL 单行示例：

```json
{"record_type":"clinical_info_unit","guideline_meta":{"title":"...","source_file":"...","doc_type":"narrative_guideline"},"unit":{"id":"clinical_info_unit_xxxxxxxxxxxx","section_path":["诊断","实验室检查"],"title":"实验室检查","raw_text":"...","unit_type":"test_order","clinical_topic":"diagnosis","action":"完善相关检查","condition":null,"indication":[],"contraindication":[],"diagnostic_criteria":[],"differential_diagnosis":[],"drug":null,"dose":null,"route":null,"frequency":null,"duration":null,"source_location":{"page_start":1,"page_end":1,"section":"诊断 / 实验室检查"},"confidence":0.86,"needs_human_review":false,"review_reasons":[]}}
```

### 当前限制

- PDF 双栏解析可能影响标题和段落顺序。
- LLM 结果必须抽样复核，不能直接作为最终医学结论。
- 标题规则需要随新文档不断扩充。
- 结构化指南的 raw 字段依赖锚点匹配质量。

这是一个“医疗诊治指南 PDF -> Guideline Skill Pack -> Agent 调用”的 Python demo 项目。

第一阶段以《中国克罗恩病诊治指南（2023 年·广州）》为示例，实现了一个最小可运行闭环：

1. 从用户自然语言中抽取 `PatientCase`
2. 根据病例信息召回候选疾病 skill pack
3. 执行克罗恩病规则版 skill executor
4. 聚合输出疑似程度、支持证据、反对证据、缺失信息、建议检查、鉴别诊断和安全提示
5. 提供 skill 质量检查、评估用例，以及 PDF 推荐意见半自动抽取工具

本项目不输出最终确诊，不替代医生面诊、检查解读或治疗决策。

## 当前能力

- 核心数据结构：`DiseaseSkillPack`、`SubSkill`、`RecommendationCard`、`PatientCase`、`SkillExecutionResult` 等
- 克罗恩病 seed skill pack：`data/skills/crohn_disease_2023_guangzhou.yaml`
- 规则版 `PatientCaseExtractor`
- 规则版 `DiseaseSkillRouter`
- 规则版 `CrohnDiseaseSkillExecutor`
- 最小 Agent 编排器：`MedicalGuidelineAgentOrchestrator`
- 本地 YAML/JSON recommendation retriever
- 预留 GraphRAG / Knowledge Graph retriever 接口
- skill quality check
- eval cases 评估框架
- PDF / 已抽取文本 到 draft recommendation cards 的半自动抽取工具

## 项目结构

```text
data/
  eval/
    crohn_cases.yaml
  skills/
    crohn_disease_2023_guangzhou.yaml

scripts/
  check_skill_quality.py
  evaluate_skill.py
  extract_patient_case.py
  extract_recommendations_from_pdf.py
  route_skill.py
  run_agent.py
  run_skill.py
  validate_skill_pack.py

src/
  agent_orchestrator.py
  patient_case_extractor.py
  pdf_guideline_parser.py
  recommendation_extractor.py
  skill_executor.py
  skill_quality_check.py
  skill_router.py
  guideline_skill/
    schema.py
  retrievers/
    base.py
    graph_retriever.py
    local_recommendation_retriever.py

tests/
```

## 安装与测试

建议使用 Python 3.10+。

```bash
python -m pip install -e .
```

如果需要直接解析 PDF，请确保安装 PyMuPDF：

```bash
python -m pip install pymupdf
```

运行测试：

```bash
pytest
```

当前测试覆盖：

```text
schema
seed skill pack
patient case extraction
skill routing
recommendation retrieval
retriever interface
Crohn executor
agent orchestrator
quality check
evaluation
PDF/text recommendation extraction
```

## 快速运行

### 1. 校验 skill pack

```bash
python scripts/validate_skill_pack.py --input data/skills/crohn_disease_2023_guangzhou.yaml
```

### 2. 从用户文本抽取 PatientCase

```bash
python scripts/extract_patient_case.py --text "我腹痛腹泻三个月，体重下降，肠镜提示回盲部溃疡和狭窄"
```

### 3. 召回候选 disease skill

```bash
python scripts/route_skill.py `
  --skills data/skills `
  --text "腹痛腹泻三个月，体重下降，肠镜提示回盲部溃疡和狭窄"
```

### 4. 执行克罗恩病 skill

```bash
python scripts/run_skill.py `
  --skill data/skills/crohn_disease_2023_guangzhou.yaml `
  --text "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄"
```

### 5. 跑完整 Agent 闭环

```bash
python scripts/run_agent.py `
  --skills data/skills `
  --text "腹痛腹泻三个月，体重下降，肠镜提示回盲部及回肠末端多发溃疡并狭窄"
```

输出包含：

- `patient_case_summary`
- `candidate_diseases`
- `skill_results`
- `final_assessment`
- `recommended_next_steps`
- `safety_warnings`
- `disclaimer`
- `readable_summary`

## 质量检查与评估

### Skill 质量检查

```bash
python scripts/check_skill_quality.py --skill data/skills/crohn_disease_2023_guangzhou.yaml
```

检查内容包括：

- 是否包含 `routing_profile`
- 是否包含 `subskills`
- 是否包含 `recommendation_cards`
- recommendation card 是否包含必要字段
- 是否包含 `safety_constraints`
- 是否包含鉴别诊断
- 是否存在禁止项，例如自动确诊、信息不足时直接治疗
- recommendation ID 是否唯一
- source reference 是否能对应到实际 recommendation card

### 评估用例

```bash
python scripts/evaluate_skill.py `
  --skill data/skills/crohn_disease_2023_guangzhou.yaml `
  --cases data/eval/crohn_cases.yaml
```

`data/eval/crohn_cases.yaml` 当前包含：

- 轻微信息不足病例
- 高度疑似 CD 病例
- 有病理支持病例
- 需要鉴别肠结核病例
- 有红旗征象病例
- 检查结果不支持 CD 病例

评估会检查：

- 是否符合 `expected_suspicion_level`
- 是否包含预期缺失信息
- 是否包含预期安全提示
- 是否误输出 `confirmed_by_doctor_only`

## PDF 推荐意见半自动抽取

工具目标是辅助生成 draft recommendation cards，而不是一次性完美解析所有 PDF。生成结果必须人工审核。

```bash
python scripts/extract_recommendations_from_pdf.py `
  --pdf data/raw/crohn_guideline.pdf `
  --disease "Crohn's disease" `
  --guideline-name "中国克罗恩病诊治指南" `
  --guideline-version "2023 Guangzhou" `
  --output data/drafts/crohn_recommendation_cards_draft.yaml
```

输出：

- `draft_recommendation_cards.yaml`：可被 `load_skill_pack()` 加载的 draft skill pack
- `extracted_sections.json`：PDF 文本解析和章节切分结果

每张自动抽取的 draft card 都会标记：

```yaml
review_status: needs_human_review
```

## 核心设计

### 一份指南不是一个巨大 prompt

本项目把一份疾病指南组织成：

- 一个 `DiseaseSkillPack`
- 多个 `SubSkill`
- 多张 `RecommendationCard`
- 一个可替换的 retriever 层
- 一个规则/状态机执行层

### Router 不做诊断

`DiseaseSkillRouter` 只做候选 skill 召回，分数只是规则匹配分，不是确诊概率。

### Executor 不输出最终确诊

`CrohnDiseaseSkillExecutor` 只输出：

- `unlikely`
- `possible`
- `suspected`
- `probable`
- `confirmed_by_doctor_only`

其中 `confirmed_by_doctor_only` 只能在输入明确包含“医生已确诊”这类信息时使用。执行器本身不会自动确诊。

### Retriever 可替换

当前默认使用：

```text
LocalRecommendationRetriever
```

它从本地 YAML/JSON 的 `recommendation_cards` 中检索。

后续可以替换为：

```text
GraphRetriever
```

目前 `GraphRetriever` 只预留 Neo4j / GraphRAG 接口，并抛出 `NotImplementedError`。

## 如何新增一个疾病指南

建议流程：

1. 准备指南 PDF 或已抽取文本
2. 使用 `extract_recommendations_from_pdf.py` 生成 draft cards
3. 人工审核和修改 draft cards
4. 编写新的 `data/skills/<disease>.yaml`
5. 增加 disease-specific executor，或抽象通用 executor
6. 在 orchestrator 中注册 executor
7. 添加 eval cases
8. 运行 quality check 和 pytest

## 医疗安全边界

本项目所有输出都应遵守：

- 不输出最终确诊
- 不在信息不足时直接给治疗方案
- 有红旗征象时优先安全提醒
- 必须输出支持证据、反对证据、缺失信息和鉴别诊断
- 治疗建议必须建立在诊断阶段、病变范围、疾病活动度、并发症和禁忌信息充分的基础上

## 当前限制

- 第一阶段是规则版，不接入 LLM
- PDF 解析是半自动辅助工具，复杂版式需要人工审核
- 当前只有克罗恩病 executor
- GraphRAG / Neo4j 接口已预留，但尚未实现
- 推荐意见抽取结果默认需要人工审核
