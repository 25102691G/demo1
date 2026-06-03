# Medical Guideline Skill Pack Demo

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
