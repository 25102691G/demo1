# Schema 字段说明

本目录包含技能包、病例标准化、指南推荐卡和工作流输出相关的 JSON Schema。本文档按文件汇总各 schema 中字段名的含义，方便开发、数据生成和校验时对照使用。

## 通用 JSON Schema 元字段

| 字段名 | 含义 |
| --- | --- |
| `$schema` | JSON Schema 规范版本地址。 |
| `$id` | 当前 schema 的唯一标识，通常为文件名。 |
| `title` | schema 的标题或用途说明。 |
| `type` | 当前节点允许的数据类型。 |
| `required` | 当前对象必须包含的字段列表。 |
| `properties` | 当前对象可包含的字段定义。 |
| `$defs` | 可复用的子结构定义。 |
| `$ref` | 引用 `$defs` 或其他 schema 中的结构。 |
| `enum` | 字段允许的固定取值列表。 |
| `const` | 字段必须等于的固定值。 |
| `default` | 字段默认值。 |
| `pattern` | 字符串必须匹配的正则表达式。 |
| `minimum` | 数值允许的最小值。 |
| `minItems` | 数组允许的最少元素数量。 |
| `additionalProperties` | 是否允许 schema 未显式声明的额外字段。 |
| `$comment` | schema 作者留下的备注，不参与数据校验。 |

## canonical_case.schema.json

当前顶层必填字段仅为 `case_id`、`raw_input`、`symptoms`；标准化流程默认仅生成这三个字段，其他顶层字段仅在输入数据显式提供时出现。

用于描述从用户输入或病历文本中抽取、归一化后的标准临床病例。

### 顶层字段

| 字段名 | 含义 |
| --- | --- |
| `case_id` | 病例唯一标识。 |
| `raw_input` | 原始输入文本。 |
| `input_language` | 原始输入语言，默认 `zh-CN`。 |
| `demographics` | 患者人口学信息。 |
| `chief_complaint` | 主诉。 |
| `history_of_present_illness` | 现病史。 |
| `symptoms` | 症状列表。 |
| `signs` | 体征列表。 |
| `vitals` | 生命体征。 |
| `labs` | 实验室检查结果集合。 |
| `imaging` | 影像学检查结果集合。 |
| `endoscopy` | 内镜检查结果集合。 |
| `pathology` | 病理检查结果集合。 |
| `diagnoses` | 诊断或鉴别诊断列表。 |
| `medications` | 用药信息列表。 |
| `procedures` | 操作、手术或治疗措施列表。 |
| `allergies` | 过敏史列表。 |
| `comorbidities` | 合并症或既往疾病列表。 |
| `family_history` | 家族史列表。 |
| `extra_manifestations` | 肠外表现或其他系统表现列表。 |
| `scores` | 量表、评分或疾病活动度指标列表。 |
| `red_flags` | 危险信号列表。 |
| `patient_goal` | 患者本次咨询目标或关注点。 |
| `extraction_quality` | 信息抽取质量、缺失项和归一化说明。 |

### `demographics`

| 字段名 | 含义 |
| --- | --- |
| `age` | 年龄。 |
| `sex` | 性别，可为 `male`、`female`、`unknown`。 |
| `pregnancy_status` | 妊娠状态，可为 `pregnant`、`not_pregnant`、`unknown`。 |

### `vitals`

| 字段名 | 含义 |
| --- | --- |
| `temperature` | 体温测量值。 |
| `heart_rate` | 心率测量值。 |
| `respiratory_rate` | 呼吸频率测量值。 |
| `blood_pressure` | 血压。 |
| `oxygen_saturation` | 血氧饱和度测量值。 |
| `systolic` | 收缩压。 |
| `diastolic` | 舒张压。 |
| `unit` | 血压单位，默认 `mmHg`。 |
| `interpretation` | 对血压或测量结果的解释。 |

### 检查结果集合

| 字段名 | 含义 |
| --- | --- |
| `items` | 检查项目列表，用于 `labs`、`imaging`、`endoscopy`、`pathology`。 |

### `extraction_quality`

| 字段名 | 含义 |
| --- | --- |
| `confidence` | 信息抽取整体置信度，范围 0 到 1。 |
| `missing_or_uncertain` | 缺失或不确定的信息列表。 |
| `normalization_notes` | 字段归一化、标准化过程中的说明。 |

### 复用结构

| 结构/字段名 | 含义 |
| --- | --- |
| `interpretation` | 结果解释，可为 `low`、`normal`、`high`、`positive`、`negative`、`abnormal`、`elevated`、`decreased`、`unknown`。 |
| `clinicalItem.name` | 症状、体征或表现名称。 |
| `clinicalItem.standard_name` | 标准化后的名称。 |
| `clinicalItem.status` | 是否存在，可为 `present`、`absent`、`unknown`。 |
| `clinicalItem.duration` | 持续时间。 |
| `clinicalItem.severity` | 严重程度，可为 `mild`、`moderate`、`severe`、`unknown`。 |
| `clinicalItem.value` | 相关数值或文本值。 |
| `clinicalItem.source_text` | 支持该字段的原文片段。 |
| `measurement.value` | 测量数值。 |
| `measurement.unit` | 测量单位。 |
| `measurement.interpretation` | 测量结果解释。 |
| `labItem.name` | 实验室检查名称。 |
| `labItem.standard_name` | 标准化实验室检查名称。 |
| `labItem.value` | 检查结果值。 |
| `labItem.unit` | 检查单位。 |
| `labItem.reference_range` | 参考范围。 |
| `labItem.interpretation` | 检查结果解释。 |
| `labItem.date` | 检查日期。 |
| `labItem.source_text` | 原文依据。 |
| `imagingItem.modality` | 影像检查类型，如 CT、MRI、超声。 |
| `imagingItem.body_part` | 检查部位。 |
| `imagingItem.findings` | 影像所见列表。 |
| `imagingItem.impression` | 影像结论。 |
| `imagingItem.date` | 检查日期。 |
| `imagingItem.source_text` | 原文依据。 |
| `endoscopyItem.type` | 内镜类型。 |
| `endoscopyItem.findings` | 内镜所见列表。 |
| `endoscopyItem.biopsy_taken` | 是否取活检，可为 `true`、`false`、`unknown` 字符串。 |
| `endoscopyItem.date` | 检查日期。 |
| `endoscopyItem.source_text` | 原文依据。 |
| `pathologyItem.specimen` | 病理标本来源。 |
| `pathologyItem.findings` | 病理所见列表。 |
| `pathologyItem.diagnosis` | 病理诊断。 |
| `pathologyItem.date` | 检查日期。 |
| `pathologyItem.source_text` | 原文依据。 |
| `diagnosisItem.name` | 诊断名称。 |
| `diagnosisItem.status` | 诊断状态，可为 `suspected`、`highly_suspected`、`confirmed`、`ruled_out`、`history`、`unknown`。 |
| `diagnosisItem.date` | 诊断日期。 |
| `diagnosisItem.source_text` | 原文依据。 |
| `medicationItem.name` | 药物名称。 |
| `medicationItem.status` | 用药状态，可为 `current`、`past`、`planned`、`unknown`。 |
| `medicationItem.dose` | 剂量。 |
| `medicationItem.frequency` | 用药频率。 |
| `medicationItem.start_date` | 开始用药日期。 |
| `medicationItem.source_text` | 原文依据。 |
| `procedureItem.name` | 操作、手术或治疗措施名称。 |
| `procedureItem.date` | 实施日期。 |
| `procedureItem.result` | 操作或治疗结果。 |
| `procedureItem.source_text` | 原文依据。 |
| `scoreItem.name` | 评分或量表名称。 |
| `scoreItem.value` | 评分值。 |
| `scoreItem.interpretation` | 评分解释。 |
| `scoreItem.date` | 评分日期。 |
| `redFlagItem.name` | 危险信号名称。 |
| `redFlagItem.source` | 来源，可为用户报告、模型推断或规则检测。 |
| `redFlagItem.severity` | 严重程度，可为 `low`、`medium`、`high`、`critical`。 |
| `redFlagItem.source_text` | 原文依据。 |

## recommendation_card.schema.json

用于描述从指南中抽取出的单条推荐意见卡片。

### 顶层字段

| 字段名 | 含义 |
| --- | --- |
| `record_type` | 记录类型，固定为 `recommendation_card`。 |
| `card_id` | 推荐卡唯一标识。 |
| `source_statement_id` | 原指南推荐语句或声明的标识。 |
| `disease` | 推荐意见适用疾病。 |
| `guideline` | 指南来源信息。 |
| `clinical_stage` | 临床阶段或场景。 |
| `clinical_task` | 临床任务，如诊断、评估、治疗、随访。 |
| `population` | 推荐适用人群。 |
| `condition` | 触发该推荐的条件或病情上下文。 |
| `raw_chunk_text` | PDF 文本清洗并分块后，该推荐对应的完整原始分块文本。 |
| `action` | 推荐执行的操作。 |
| `required_inputs` | 应用推荐前需要的输入信息。 |
| `safety_notes` | 安全注意事项。 |
| `evidence` | 证据质量、推荐强度和评级体系信息。 |
| `source_location` | 推荐在原始文档中的位置和引用。 |

### `guideline`

| 字段名 | 含义 |
| --- | --- |
| `title` | 指南标题。 |
| `source_file` | 指南来源文件。 |
| `doc_type` | 文档类型。 |

### `evidence`

| 字段名 | 含义 |
| --- | --- |
| `evidence_quality_raw` | 从原文中提取的证据等级、推荐强度、共识等级、专家共识比例等可信程度表述；没有则为 `null`。 |
| `evidence_quality_normalized` | 在单个指南的全部 `evidence_quality_raw` 生成后统一标准化得到的 `0.6-1.0` 证据加权系数；无法判断或无显式可信程度时默认 `0.75`。 |
| `evidence_quality_normalized_info` | `evidence_quality_normalized` 的来源或默认原因，如 `llm_normalized`、`bps_normalized`、`default_no_raw_evidence`、`default_unmapped_raw_evidence`、`default_missing_llm_score`。 |

### `source_location`

| 字段名 | 含义 |
| --- | --- |
| `pdf` | 来源 PDF 文件名或路径。 |
| `page_start` | 引文起始页码。 |
| `page_end` | 引文结束页码。 |
| `quote` | 原文引用。 |
| `source_span` | 原文范围、段落或字符跨度标识。 |

## skill_pack.schema.json

用于描述单个疾病技能包，包括元数据、路由规则、知识库、工作流、子技能、安全约束和输出模板。

### 顶层字段

| 字段名 | 含义 |
| --- | --- |
| `schema_version` | 技能包 schema 版本，格式如 `0.x`。 |
| `metadata` | 疾病技能包的基础元数据。 |
| `runtime` | 技能运行时配置。 |
| `case_schema` | 输入病例 schema 和字段映射配置。 |
| `routing_profile` | 疾病路由和候选技能匹配配置。 |
| `knowledge_base` | 指南推荐卡和检索索引配置。 |
| `workflow` | 技能执行工作流。 |
| `subskills` | 旧格式兼容字段；新生成技能包不再输出。 |
| `safety_constraints` | 安全约束和急症处理规则。 |
| `output_templates` | 不同输出模板定义。 |
| `validation` | 校验、测试和人工审核策略。 |

### `metadata`

| 字段名 | 含义 |
| --- | --- |
| `skill_id` | 技能包唯一标识。 |
| `disease_name` | 疾病标准名称。 |
| `disease_code` | 疾病编码集合。 |
| `icd10` | ICD-10 疾病编码。 |
| `icd11` | ICD-11 疾病编码。 |
| `disease_aliases` | 疾病别名列表。 |
| `body_system` | 所属身体系统。 |
| `specialty` | 相关医学专科。 |
| `guideline` | 技能包依据的指南信息。 |
| `target_users` | 目标用户群体。 |
| `intended_use` | 预期用途列表。 |
| `scope` | 技能适用范围。 |
| `out_of_scope` | 不适用范围。 |
| `disclaimer` | 免责声明。 |

### `metadata.guideline`

| 字段名 | 含义 |
| --- | --- |
| `name` | 指南名称。 |
| `version` | 指南版本。 |
| `source_pdf` | 指南 PDF 来源路径或文件名。 |
| `publication_year` | 出版年份。 |
| `issuing_organization` | 发布组织。 |
| `language` | 指南语言，默认 `zh-CN`。 |

### `runtime`

| 字段名 | 含义 |
| --- | --- |
| `engine_min_version` | 所需技能引擎最低版本。 |
| `execution_mode` | 执行模式，目前限定为 `declarative_workflow`。 |
| `llm_usage` | 大模型使用权限和用途配置。 |
| `allowed` | 是否允许使用大模型。 |
| `roles` | 允许大模型承担的角色，如病例归一化、语义匹配、解释生成、卡片结构化。 |
| `forbidden_roles` | 禁止大模型承担的角色。 |
| `default_language` | 默认输出语言。 |
| `output_mode` | 输出模式配置。 |
| `default` | 默认输出模式。 |
| `allow_patient_friendly_summary` | 是否允许生成面向患者的易懂摘要。 |

### `case_schema`

| 字段名 | 含义 |
| --- | --- |
| `canonical_case_ref` | 标准病例 schema 引用。 |
| `required_minimum_fields` | 运行技能所需的最低病例字段。 |
| `field_mapping_hints` | 输入字段到标准病例字段的映射提示。 |

### `routing_profile`

| 字段名 | 含义 |
| --- | --- |
| `routing_version` | 路由规则版本。 |
| `disease_identity` | 疾病身份识别信息。 |
| `primary_name` | 疾病主名称。 |
| `aliases` | 疾病别名。 |
| `abbreviations` | 疾病缩写。 |
| `positive_features` | 支持匹配该疾病的阳性特征集合。 |
| `negative_features` | 降低匹配分或提示其他疾病的阴性特征。 |
| `red_flags` | 路由阶段识别的危险信号。 |
| `scoring` | 路由评分配置。 |
| `symptoms` | 症状类阳性特征。 |
| `signs` | 体征类阳性特征。 |
| `labs` | 实验室检查类阳性特征。 |
| `imaging` | 影像类阳性特征。 |
| `endoscopy` | 内镜类阳性特征。 |
| `pathology` | 病理类阳性特征。 |
| `findings` | 其他发现类阳性特征。 |

### `scoring`

| 字段名 | 含义 |
| --- | --- |
| `method` | 评分方法，如加权特征求和或加权特征加语义混合。 |
| `normalization` | 分数归一化方法。 |
| `thresholds` | 候选判定阈值。 |
| `candidate` | 达到候选疾病的最低分。 |
| `strong_candidate` | 达到强候选疾病的最低分。 |
| `very_likely` | 达到高度可能的最低分。 |
| `top_k_default` | 默认返回的候选技能数量。 |
| `safety_override` | 是否允许安全规则覆盖常规路由结果。 |

### `knowledge_base`

| 字段名 | 含义 |
| --- | --- |
| `cards_path` | 推荐卡数据文件路径。 |
| `source_map_path` | 来源映射文件路径。 |
| `embedding_index_path` | 向量索引文件路径。 |
| `card_schema_ref` | 推荐卡 schema 引用。 |
| `retrieval` | 知识库检索配置。 |
| `default_mode` | 默认检索模式。 |
| `modes` | 支持的检索模式列表。 |
| `default_top_k` | 默认返回结果数量。 |
| `filters` | 默认或可用的元数据过滤条件。 |
| `citation_required` | 输出是否必须包含引用。 |

### `workflow`

| 字段名 | 含义 |
| --- | --- |
| `workflow_version` | 工作流版本。 |
| `entrypoint` | 工作流入口步骤 ID。 |
| `global_policies` | 作用于整个工作流的全局策略。 |
| `steps` | 工作流步骤列表。 |

### `safety_constraints`

| 字段名 | 含义 |
| --- | --- |
| `safety_version` | 安全规则版本。 |
| `general` | 通用安全规则列表。 |
| `treatment_policy` | 治疗建议相关限制。 |
| `diagnosis_required` | 给出治疗建议前对诊断确认程度的要求。 |
| `allow_general_management_advice` | 是否允许一般性管理建议。 |
| `allow_specific_drug_dosing` | 是否允许具体药物剂量建议。 |
| `allow_prescription` | 是否允许处方建议。 |
| `require_physician_review` | 是否要求医生审核。 |
| `emergency_policy` | 急症或危险信号处理策略。 |
| `trigger_from` | 触发急症策略的数据来源。 |
| `output_template` | 触发后使用的输出模板。 |
| `stop_workflow` | 是否停止后续工作流。 |

### `validation`

| 字段名 | 含义 |
| --- | --- |
| `required_checks` | 必须执行的校验项。 |
| `review_policy` | 人工审核或质量复核策略。 |
| `test_cases_path` | 测试用例路径。 |

### 复用结构

| 结构/字段名 | 含义 |
| --- | --- |
| `feature.name` | 匹配特征名称。 |
| `feature.synonyms` | 特征同义词。 |
| `feature.path` | 在标准病例中的字段路径。 |
| `feature.operator` | 匹配运算符，如存在、等于、大于、包含、语义匹配。 |
| `feature.threshold` | 匹配阈值。 |
| `feature.unit` | 阈值或检查结果单位。 |
| `feature.weight` | 特征权重。 |
| `feature.match_type` | 匹配方式，如精确、语义、结构化或文本匹配。 |
| `negativeFeature.name` | 阴性特征名称。 |
| `negativeFeature.suggests` | 该特征提示的其他可能疾病。 |
| `negativeFeature.penalty` | 匹配分惩罚值。 |
| `negativeFeature.match_type` | 阴性特征匹配方式。 |
| `redFlag.name` | 危险信号名称。 |
| `redFlag.severity` | 危险信号严重程度。 |
| `redFlag.action` | 识别危险信号后建议动作。 |
| `redFlag.synonyms` | 危险信号同义词。 |
| `differentialDisease.disease_name` | 需要鉴别的疾病名称。 |
| `differentialDisease.reason` | 需要鉴别的原因。 |
| `differentialDisease.key_distinguishing_features` | 关键鉴别特征。 |
| `workflowStep.step_id` | 工作流步骤 ID。 |
| `workflowStep.type` | 步骤类型，如安全分诊、证据检查、鉴别诊断、方案生成。 |
| `workflowStep.description` | 步骤说明。 |
| `workflowStep.config` | 步骤配置，可通过 `card_filter` 按 recommendation card 的 `clinical_stage` / `clinical_task` 筛选卡片，通过 `input_requirements` 声明缺失信息检查，也兼容 `subskill_ref`。 |
| `workflowStep.transitions` | 步骤转移规则。 |
| `transition.when` | 转移条件。 |
| `transition.to` | 目标步骤 ID。 |
| `condition.op` | 条件运算符。 |
| `condition.path` | 条件读取的字段路径。 |
| `condition.left` | 条件左值。 |
| `condition.right` | 条件右值。 |
| `condition.right_ref` | 条件右值引用路径。 |
| `subskill.subskill_id` | 旧格式兼容字段，子技能唯一标识。 |
| `subskill.name` | 子技能名称。 |
| `subskill.type` | 子技能类型。 |
| `subskill.description` | 子技能说明。 |
| `subskill.input_requirements` | 子技能输入要求。 |
| `subskill.card_selection` | 推荐卡选择策略。 |
| `card_selection.mode` | 推荐卡选择模式。 |
| `card_selection.required` | 必须选取或必须满足的推荐卡字段。 |
| `card_selection.optional` | 可选推荐卡字段。 |
| `card_selection.filters` | 推荐卡筛选条件。 |
| `card_selection.semantic_query_from` | 构造语义检索查询的输入字段。 |
| `subskill.output_schema_ref` | 子技能输出 schema 引用。 |
| `subskill.output_fields` | 子技能输出字段列表。 |
| `subskill.policies` | 子技能专属策略。 |
| `safetyRule.id` | 安全规则 ID。 |
| `safetyRule.rule` | 安全规则内容。 |
| `safetyRule.severity` | 安全规则严重程度。 |
| `outputTemplate.audience` | 输出模板面向的受众。 |
| `outputTemplate.structure` | 输出模板结构列表。 |

## workflow_output.schema.json

用于描述技能引擎执行一次病例工作流后的结果。

### 顶层字段

| 字段名 | 含义 |
| --- | --- |
| `run_id` | 本次工作流运行唯一标识。 |
| `case_id` | 对应的病例 ID。 |
| `canonical_case` | 标准化后的完整病例对象，用于调试。 |
| `status` | 工作流整体状态，如完成、安全停止、信息缺失、低置信度、需人工审核或错误。 |
| `top_candidates` | 路由得到的候选疾病技能列表。 |
| `selected_skill_outputs` | 实际执行的技能输出列表。 |
| `final_response` | 最终返回给用户或系统的结果。 |
| `safety` | 安全检查结果。 |
| `citations` | 引用来源列表。 |
| `debug_trace` | 调试追踪信息。 |

### `final_response`

| 字段名 | 含义 |
| --- | --- |
| `audience` | 输出受众，可为 `patient`、`clinician`、`system`、`mixed`。 |
| `summary` | 最终结果摘要。 |
| `structured_result` | 结构化结果。 |
| `disclaimer` | 免责声明。 |
| `diagnosis_status` | 诊断状态或疾病可能性。 |
| `supporting_evidence` | 支持当前判断的证据。 |
| `opposing_evidence` | 反对当前判断的证据。 |
| `missing_information` | 仍缺失的信息。 |
| `recommended_next_steps` | 建议下一步。 |
| `differentials_to_consider` | 需要考虑的鉴别诊断。 |
| `management_recommendations` | 管理或治疗建议。 |
| `monitoring_or_follow_up` | 监测和随访建议。 |
| `safety_notes` | 安全提醒。 |

### `safety`

| 字段名 | 含义 |
| --- | --- |
| `has_red_flags` | 是否存在危险信号。 |
| `red_flags` | 识别出的危险信号列表。 |
| `workflow_stopped` | 工作流是否因安全原因停止。 |
| `safety_message` | 安全提示文本。 |

### `debug_trace`

| 字段名 | 含义 |
| --- | --- |
| `enabled` | 是否启用调试追踪。 |
| `route_scores` | 路由评分详情。 |
| `executed_steps` | 已执行步骤列表。 |
| `errors` | 运行中的错误信息列表。 |

### 复用结构

| 结构/字段名 | 含义 |
| --- | --- |
| `candidateSkill.skill_id` | 候选技能 ID。 |
| `candidateSkill.disease_name` | 候选疾病名称。 |
| `candidateSkill.score` | 路由匹配分数。 |
| `candidateSkill.rank` | 候选排序名次。 |
| `candidateSkill.matched_positive_features` | 命中的正向特征列表。 |
| `candidateSkill.matched_negative_features` | 命中的负向特征列表。 |
| `candidateSkill.missing_key_evidence` | 缺失的关键证据。 |
| `candidateSkill.reasoning_summary` | 路由理由摘要。 |
| `matchedFeature.name` | 命中特征名称。 |
| `matchedFeature.source` | 特征来源，如症状、体征、实验室、影像、内镜、病理、诊断或文本。 |
| `matchedFeature.weight` | 特征权重。 |
| `matchedFeature.evidence_text` | 支持命中的证据文本。 |
| `skillOutput.skill_id` | 已执行技能 ID。 |
| `skillOutput.disease_name` | 已执行技能对应疾病名称。 |
| `skillOutput.workflow_status` | 单个技能工作流状态。 |
| `skillOutput.executed_steps` | 该技能执行过的步骤。 |
| `skillOutput.result` | 该技能产生的结构化结果。 |
| `result.current_status` | 当前判断或处理状态。 |
| `result.suspicion_level` | 疾病怀疑程度。 |
| `result.supporting_evidence` | 支持证据。 |
| `result.opposing_evidence` | 反对证据。 |
| `result.missing_information` | 缺失信息。 |
| `result.recommended_next_steps` | 建议下一步。 |
| `result.used_cards` | 使用过的推荐卡 ID 列表。 |
| `redFlagOutput.name` | 危险信号名称。 |
| `redFlagOutput.severity` | 危险信号严重程度。 |
| `redFlagOutput.source` | 危险信号来源。 |
| `redFlagOutput.recommended_action` | 建议处理动作。 |
| `citation.card_id` | 引用对应的推荐卡 ID。 |
| `citation.recommendation_label` | 推荐标签或名称。 |
| `citation.source` | 引用来源信息。 |
| `source.pdf` | 来源 PDF。 |
| `source.page_start` | 引用起始页。 |
| `source.page_end` | 引用结束页。 |
| `source.quote` | 引用原文。 |
| `routeScore.skill_id` | 被评分技能 ID。 |
| `routeScore.score` | 路由分数。 |
| `routeScore.matched_positive_features` | 命中的正向特征名称列表。 |
| `routeScore.matched_negative_features` | 命中的负向特征名称列表。 |
| `executedStep.step_id` | 已执行步骤 ID。 |
| `executedStep.type` | 已执行步骤类型。 |
| `executedStep.status` | 步骤状态，可为完成、跳过、停止或错误。 |
| `executedStep.result_summary` | 步骤结果摘要。 |
| `executedStep.next_step` | 下一步 ID。 |
