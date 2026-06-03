from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib import error, request

import yaml

from guideline_skill.schema import (
    DiseaseSkillPack,
    RecommendationCard,
    RoutingProfile,
    SubSkill,
    save_skill_pack,
    validate_skill_pack,
)

DEFAULT_SAFETY_NOTE = "本条内容不能替代医生诊断或处方决策。"
LLM_LIST_ITEM_ALIASES = {
    "CTE": "CT小肠成像（CTE）",
    "MRE": "磁共振小肠成像（MRE）",
    "MRI": "磁共振成像（MRI）",
    "IFX": "英夫利西单抗（IFX）",
}

# TODO: 这里目前也是写死，后续需要由推荐意见结合LLM生成
@dataclass(frozen=True)
class SkillPackMetadata:
    skill_name: str = "克罗恩病二零二三广州指南技能包"
    disease_name: str = "克罗恩病"
    disease_aliases: tuple[str, ...] = ("克罗恩病", "克隆氏病", "节段性肠炎")
    guideline_name: str = "中国克罗恩病诊治指南"
    guideline_version: str = "二零二三年广州版"
    source_pdf: str = "中国克罗恩病诊治指南（二零二三年广州）.pdf"
    target_users: tuple[str, ...] = ("消化专科医生", "普通临床医生", "临床决策支持系统")
    scope: str = (
        "根据指南推荐意见支持克罗恩病相关的初筛、诊断证据整合、鉴别诊断、"
        "病情评估、治疗前信息核对、肛周病变处理、围手术期管理、术后复发预防"
        "以及长期随访监测；本技能包不得替代医生作出最终诊断或处方决策。"
    )


@dataclass(frozen=True)
class CsvRecommendation:
    recommendation_id: str
    number: str
    title: str
    text: str
    evidence_grade: str = ""
    recommendation_strength: str = ""
    is_bps: bool = False
    reason: str = ""
    implementation_advice: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    source_text: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "CsvRecommendation":
        return cls(
            recommendation_id=_clean_cell(row.get("recommendation_id", "")),
            number=_clean_cell(row.get("number", "")),
            title=_clean_cell(row.get("title", "")),
            text=_clean_cell(row.get("text", "")),
            evidence_grade=_clean_cell(row.get("evidence_grade", "")),
            recommendation_strength=_clean_cell(row.get("recommendation_strength", "")),
            is_bps=_parse_bool(row.get("is_bps", "")),
            reason=_clean_cell(row.get("reason", "")),
            implementation_advice=_clean_cell(row.get("implementation_advice", "")),
            section=_clean_cell(row.get("section", "")),
            page_start=_parse_int(row.get("page_start", "")),
            page_end=_parse_int(row.get("page_end", "")),
            source_text=_clean_cell(row.get("source_text", "")),
        )


class SemanticEnricher(Protocol):
    def enrich(self, recommendation: CsvRecommendation) -> Mapping[str, Any]:
        """Return semantic fields for one recommendation card."""


class LlmClient(Protocol):
    def complete_json(self, prompt: str) -> Mapping[str, Any]:
        """Return a parsed JSON object generated from a prompt."""


class RuleBasedSemanticEnricher:
    """Deterministic semantic field generator used when no LLM override is supplied."""

    def enrich(self, recommendation: CsvRecommendation) -> Mapping[str, Any]:
        stage = infer_clinical_stage(recommendation)
        return {
            "clinical_stage": stage,
            "clinical_task": infer_clinical_task(recommendation),
            "population": infer_population(recommendation, stage),
            "condition": infer_condition(recommendation, stage),
            "required_inputs": infer_required_inputs(recommendation, stage),
            "safety_notes": infer_safety_notes(recommendation, stage),
        }


@dataclass(frozen=True)
class OpenAICompatibleChatClient:
    """Small OpenAI-compatible chat client using only the Python standard library."""

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
    ) -> "OpenAICompatibleChatClient":
        selected_model = model or os.getenv("OPENAI_MODEL")
        if not selected_model:
            raise ValueError("启用 LLM 语义增强时必须提供 --llm-model 或设置 OPENAI_MODEL。")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("启用 LLM 语义增强时必须设置 DEEPSEEK_API_KEY。")
        return cls(
            model=selected_model,
            api_key=api_key,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    def complete_json(self, prompt: str) -> Mapping[str, Any]:
        if not self.api_key:
            raise ValueError("OpenAI-compatible LLM client requires an API key.")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你只输出一个 JSON object。字段名使用英文，字段值使用中文；"
                        "不要输出 Markdown、解释文字或指南之外的医学事实。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        http_request = request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM 请求失败：HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM 请求失败：{exc.reason}") from exc

        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
        return _parse_json_object(content)


@dataclass(frozen=True)
class LlmSemanticEnricher:
    """Semantic enricher backed by an LLM, with optional rule-based fallback."""

    client: LlmClient
    fallback_enricher: SemanticEnricher | None = None

    def enrich(self, recommendation: CsvRecommendation) -> Mapping[str, Any]:
        try:
            payload = self.client.complete_json(build_llm_enrichment_prompt(recommendation))
            payload = fill_empty_llm_condition(payload, recommendation)
            return validate_llm_enrichment_payload(payload)
        except Exception as exc:
            if self.fallback_enricher is None:
                label = recommendation_card_id(recommendation)
                raise ValueError(f"{label} 的 LLM 语义增强失败：{exc}") from exc
            return self.fallback_enricher.enrich(recommendation)


def build_skill_pack_from_csv(
    csv_path: str | Path,
    *,
    metadata: SkillPackMetadata | None = None,
    semantic_enricher: SemanticEnricher | None = None,
    semantic_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    require_chinese_values: bool = True,
) -> DiseaseSkillPack:
    """Build a validated DiseaseSkillPack from a recommendation CSV file.

    The converter keeps YAML field names as schema-defined English names while
    generating human-readable values in Chinese by default.
    """

    meta = metadata or SkillPackMetadata()
    records = read_recommendations_csv(csv_path)
    if not records:
        raise ValueError("CSV 文件中没有可转换的推荐意见。")

    enricher = semantic_enricher or RuleBasedSemanticEnricher()
    cards = [
        build_recommendation_card(
            recommendation,
            semantic_fields=enricher.enrich(recommendation),
            semantic_overrides=semantic_overrides or {},
        )
        for recommendation in records
    ]

    card_ids_by_number = {
        recommendation.number: recommendation_card_id(recommendation)
        for recommendation in records
    }
    skill_pack = DiseaseSkillPack(
        skill_name=meta.skill_name,
        disease_name=meta.disease_name,
        disease_aliases=list(meta.disease_aliases),
        guideline_name=meta.guideline_name,
        guideline_version=meta.guideline_version,
        source_pdf=meta.source_pdf,
        target_users=list(meta.target_users),
        scope=meta.scope,
        routing_profile=build_routing_profile(meta),
        subskills=build_subskills(records, card_ids_by_number),
        recommendation_cards=cards,
        safety_constraints=build_safety_constraints(),
    )

    validated = validate_skill_pack(skill_pack)
    return validated


def write_skill_pack_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    metadata: SkillPackMetadata | None = None,
    semantic_enricher: SemanticEnricher | None = None,
    semantic_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    require_chinese_values: bool = True,
) -> DiseaseSkillPack:
    skill_pack = build_skill_pack_from_csv(
        csv_path,
        metadata=metadata,
        semantic_enricher=semantic_enricher,
        semantic_overrides=semantic_overrides,
        require_chinese_values=require_chinese_values,
    )
    save_skill_pack(skill_pack, output_path)
    return skill_pack


def read_recommendations_csv(csv_path: str | Path) -> list[CsvRecommendation]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CsvRecommendation.from_row(row)
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def build_recommendation_card(
    recommendation: CsvRecommendation,
    *,
    semantic_fields: Mapping[str, Any],
    semantic_overrides: Mapping[str, Mapping[str, Any]],
) -> RecommendationCard:
    recommendation_id = recommendation_card_id(recommendation)
    override = _semantic_override_for(recommendation, recommendation_id, semantic_overrides)
    merged = {
        **semantic_fields,
        **override,
    }

    return RecommendationCard(
        recommendation_id=recommendation_id,
        source_section=recommendation.section or "未分章节",
        source_section_cn=recommendation.section or "未分章节",
        clinical_stage=str(merged["clinical_stage"]),
        clinical_task=str(merged["clinical_task"]),
        population=str(merged["population"]),
        condition=str(merged["condition"]),
        action=str(merged.get("action") or _action_text(recommendation)),
        evidence_level=evidence_level_text(recommendation),
        recommendation_strength=recommendation_strength_text(recommendation),
        rationale=str(merged.get("rationale") or rationale_text(recommendation)),
        required_inputs=list(merged["required_inputs"]),
        safety_notes=list(merged["safety_notes"]),
        source_span=source_span_text(recommendation),
        source_quote=_source_quote(recommendation),
    )

# TODO: 生成routing_profile（这里目前是写死，后续需要由推荐意见结合LLM生成）
def build_routing_profile(metadata: SkillPackMetadata) -> RoutingProfile:
    return RoutingProfile(
        body_system="消化系统",
        key_symptoms=[
            "慢性腹泻",
            "腹痛",
            "体重下降",
            "发热",
            "乏力",
            "贫血",
            "生长发育迟缓",
            "肛周疼痛",
            "肛瘘",
            "肛周脓肿",
            "反复口腔溃疡",
            "关节炎或关节痛",
            "结节性红斑",
        ],
        key_tests=[
            "血常规",
            "白蛋白",
            "C反应蛋白",
            "红细胞沉降率",
            "粪便钙卫蛋白",
            "结肠镜",
            "回肠末段检查",
            "胃十二指肠镜",
            "小肠胶囊内镜",
            "病理活检",
            "CT小肠成像",
            "磁共振小肠成像",
            "肠道超声",
            "肛周磁共振",
            "粪便病原学检查",
            "结核相关检查",
        ],
        key_findings=[
            "节段性炎症",
            "纵行溃疡",
            "铺路石样改变",
            "末段回肠受累",
            "透壁性炎症",
            "肉芽肿",
            "肠腔狭窄",
            "瘘管",
            "脓肿",
            "肠壁增厚",
            "肠系膜梳样征",
            "粪便钙卫蛋白升高",
        ],
        red_flags=[
            "急腹症",
            "疑似肠穿孔",
            "肠梗阻",
            "大量消化道出血",
            "高热伴剧烈腹痛",
            "脓毒症风险",
            "严重脱水",
            "肛周脓肿快速加重",
        ],
        must_differentiate=[
            "肠结核",
            "肠白塞病",
            "肠道淋巴瘤",
            "感染性肠炎",
            "药物性肠炎",
            "溃疡性结肠炎",
            "缺血性肠炎",
        ],
        disease_aliases=list(metadata.disease_aliases),
    )


def build_subskills(
    recommendations: list[CsvRecommendation],
    card_ids_by_number: Mapping[str, str],
) -> list[SubSkill]:
    return [
        SubSkill(
            subskill_id="初筛与安全分诊",
            name="克罗恩病初筛与安全分诊",
            description="识别疑似克罗恩病线索、急症红旗征象和需要尽快补充的基础信息。",
            clinical_tasks=["初筛", "红旗征象分诊", "疑似程度表达"],
            required_inputs=["人口学信息", "症状与体征", "既往病史", "急症红旗征象"],
            recommendation_ids=_ids_for(recommendations, card_ids_by_number, {"1", "2", "10"}),
            output_fields=["疑似程度", "支持证据", "反对证据", "缺失信息", "安全提醒"],
        ),
        SubSkill(
            subskill_id="诊断证据整合",
            name="克罗恩病诊断证据整合",
            description="整合临床表现、实验室、影像、内镜和病理证据，规划符合指南的诊断资料集。",
            clinical_tasks=["诊断评估", "检查选择", "缺失信息核对"],
            required_inputs=["症状与体征", "实验室检查", "影像学检查", "内镜检查", "病理结果"],
            recommendation_ids=_ids_for(
                recommendations,
                card_ids_by_number,
                {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
            ),
            output_fields=["缺失信息", "建议补充检查", "依据来源", "安全提醒"],
        ),
        SubSkill(
            subskill_id="鉴别诊断",
            name="克罗恩病鉴别诊断",
            description="提示需要排除的相似疾病，并列出区分这些疾病所需的关键证据。",
            clinical_tasks=["鉴别诊断", "排除性证据核对"],
            required_inputs=["症状与体征", "感染暴露史", "用药史", "影像学检查", "内镜检查", "病理结果"],
            recommendation_ids=_ids_for(recommendations, card_ids_by_number, {"8", "9", "10"}),
            output_fields=["鉴别诊断", "支持证据", "反对证据", "缺失信息", "安全提醒"],
        ),
        SubSkill(
            subskill_id="分型活动度与并发症评估",
            name="克罗恩病分型、活动度与并发症评估",
            description="在疑似或已诊断克罗恩病时评估病变范围、活动度、并发症和高危因素。",
            clinical_tasks=["疾病分型", "活动度评估", "并发症评估", "高危因素评估"],
            required_inputs=["发病年龄", "吸烟情况", "病变范围", "活动度", "并发症", "肛周病变"],
            recommendation_ids=_ids_for(recommendations, card_ids_by_number, {"2", "7", "8", "11", "12"}),
            output_fields=["支持证据", "缺失信息", "建议下一步", "依据来源"],
        ),
        SubSkill(
            subskill_id="药物与营养治疗选择",
            name="克罗恩病药物与营养治疗选择",
            description="在医生已确认诊断并完成治疗前信息核对后，辅助梳理诱导缓解和维持缓解方案。",
            clinical_tasks=["治疗前信息核对", "诱导缓解方案梳理", "维持缓解方案梳理", "营养治疗"],
            required_inputs=["确诊状态", "病变范围", "活动度", "并发症", "禁忌证", "感染风险", "既往治疗反应"],
            recommendation_ids=_ids_in_section(recommendations, card_ids_by_number, "治疗"),
            output_fields=["缺失信息", "可讨论方案", "安全提醒", "依据来源"],
        ),
        SubSkill(
            subskill_id="肛周病变与瘘管处理",
            name="克罗恩病肛周病变与瘘管处理",
            description="围绕肛周病变、复杂瘘管和相关感染风险，整理评估和处理要点。",
            clinical_tasks=["肛周病变评估", "复杂瘘管处理", "感染风险处理"],
            required_inputs=["肛周症状", "肛周体格检查", "肛周磁共振", "脓肿评估", "感染风险"],
            recommendation_ids=_ids_in_section(recommendations, card_ids_by_number, "CD合并肛周病变/瘘管型CD处理原则"),
            output_fields=["建议下一步", "缺失信息", "安全提醒", "依据来源"],
        ),
        SubSkill(
            subskill_id="围手术期与术后复发预防",
            name="克罗恩病围手术期管理与术后复发预防",
            description="辅助核对手术指征、围手术期风险、术后复发风险和预防复发策略。",
            clinical_tasks=["手术指征核对", "围手术期管理", "术后复发风险评估", "复发预防"],
            required_inputs=["手术指征", "营养状态", "感染或脓肿情况", "用药史", "术后内镜或影像监测"],
            recommendation_ids=_ids_in_section(recommendations, card_ids_by_number, "CD围手术期管理及预防术后复发"),
            output_fields=["缺失信息", "建议下一步", "安全提醒", "依据来源"],
        ),
        SubSkill(
            subskill_id="治疗监测与长期管理",
            name="克罗恩病治疗监测与长期管理",
            description="支持治疗反应、药物安全、营养状态、心理健康和长期随访的结构化监测。",
            clinical_tasks=["治疗反应监测", "药物安全监测", "营养评估", "心理健康评估", "长期随访"],
            required_inputs=["当前症状", "炎症指标", "既往内镜或影像", "治疗史", "营养状态", "心理健康状态"],
            recommendation_ids=_ids_in_section(recommendations, card_ids_by_number, "治疗监测及患者管理"),
            output_fields=["建议下一步", "缺失信息", "安全提醒", "依据来源"],
        ),
    ]


def build_safety_constraints() -> list[str]:
    return [
        "本技能包不得自动作出最终诊断；信息不足时只能表达疑似程度、支持证据、反对证据、缺失信息和建议补充检查。",
        "出现急腹症、疑似肠穿孔、肠梗阻、大量消化道出血、脓毒症风险、严重脱水或肛周脓肿快速加重时，应提示立即急诊或及时就医。",
        "疑似克罗恩病时必须同步考虑肠结核、肠白塞病、肠道淋巴瘤、感染性肠炎、药物性肠炎、溃疡性结肠炎和缺血性肠炎等鉴别诊断。",
        "治疗相关输出必须建立在确诊状态、病变范围、活动度、并发症、禁忌证、感染风险、既往治疗反应和合并症等信息充分的基础上。",
        "所有药物启动、剂量调整、停药、手术和侵入性检查建议均需由具备资质的医生结合患者完整情况决定。",
    ]


def infer_clinical_stage(recommendation: CsvRecommendation) -> str:
    text = _semantic_text(recommendation)
    section = recommendation.section

    if section == "治疗监测及患者管理":
        return "治疗监测与长期管理"
    if section == "CD围手术期管理及预防术后复发":
        return "围手术期管理与术后复发预防"
    if section == "CD合并肛周病变/瘘管型CD处理原则":
        return "肛周病变与瘘管处理"
    if section == "治疗":
        return "药物与营养治疗选择"
    if "鉴别" in text or recommendation.number in {"8", "9", "10"}:
        return "鉴别诊断"
    if recommendation.number in {"11", "12"} or any(term in text for term in ("分型", "活动度", "高危", "并发症")):
        return "分型活动度与并发症评估"
    return "诊断证据整合"


def infer_clinical_task(recommendation: CsvRecommendation) -> str:
    text = _semantic_text(recommendation)
    if "粪便钙卫蛋白" in text or "FC" in text:
        return "肠道炎症水平评估"
    if "结肠镜" in text:
        return "结肠镜和多肠段活检"
    if "胃十二指肠镜" in text or "上消化道" in text:
        return "上消化道受累评估"
    if "胶囊内镜" in text:
        return "小肠胶囊内镜选择"
    if "CTE" in text or "MRE" in text or "小肠成像" in text:
        return "横断面影像评估"
    if "肛周" in text and "MRI" in text:
        return "肛周病变影像评估"
    if "鉴别" in text or recommendation.number in {"8", "9", "10"}:
        return "相似疾病鉴别"
    if "分型" in text or "活动度" in text or "高危" in text:
        return "疾病分型、活动度和高危因素评估"
    if recommendation.section == "CD合并肛周病变/瘘管型CD处理原则":
        return "肛周病变和瘘管处理"
    if recommendation.section == "CD围手术期管理及预防术后复发":
        return "围手术期管理和术后复发预防"
    if recommendation.section == "治疗监测及患者管理":
        return "治疗监测和长期管理"
    if recommendation.section == "治疗":
        if "诱导缓解" in text:
            return "诱导缓解治疗选择"
        if "维持" in text:
            return "维持缓解治疗选择"
        if "营养" in text:
            return "营养治疗"
        return "治疗方案梳理"
    return "综合诊断评估"


def infer_population(recommendation: CsvRecommendation, stage: str) -> str:
    text = _semantic_text(recommendation)
    if "儿童" in text or "青少年" in text:
        return "疑似或已诊断克罗恩病的儿童及青少年患者"
    if "中重度" in text:
        return "中重度活动期克罗恩病患者"
    if "轻度" in text:
        return "轻度活动期克罗恩病患者"
    if "术后" in text:
        return "接受肠切除术后需要预防复发的克罗恩病患者"
    if "肛周" in text or "瘘管" in text:
        return "合并肛周病变或瘘管的克罗恩病患者"
    if "诊断" in stage or "鉴别" in stage:
        return "疑似克罗恩病或需要完善诊断证据的患者"
    return "疑似或已诊断克罗恩病并需要指南化管理的患者"


def infer_condition(recommendation: CsvRecommendation, stage: str) -> str:
    if stage == "诊断证据整合":
        return "当患者存在疑似克罗恩病相关症状、检查异常，或需要完善诊断证据时。"
    if stage == "鉴别诊断":
        return "当克罗恩病尚未最终确认，或存在与克罗恩病表现相似的疾病需要排除时。"
    if stage == "分型活动度与并发症评估":
        return "当患者已疑似或已诊断克罗恩病，并需要明确病变范围、活动度、并发症或高危因素时。"
    if stage == "药物与营养治疗选择":
        return "当医生已考虑克罗恩病治疗，并需要结合病情严重程度和安全前提选择方案时。"
    if stage == "肛周病变与瘘管处理":
        return "当患者存在肛周症状、肛瘘、肛周脓肿或复杂瘘管风险时。"
    if stage == "围手术期管理与术后复发预防":
        return "当患者需要评估手术相关问题、围手术期风险或术后复发预防策略时。"
    if stage == "治疗监测与长期管理":
        return "当患者需要治疗反应、疾病复发、营养状态、药物安全或心理健康随访时。"
    return "当需要根据指南推荐意见补充评估和管理信息时。"


def infer_required_inputs(recommendation: CsvRecommendation, stage: str) -> list[str]:
    text = _semantic_text(recommendation)
    inputs: list[str]
    if stage in {"诊断证据整合", "鉴别诊断"}:
        inputs = ["症状与体征", "实验室检查", "影像学检查", "内镜检查", "病理结果"]
        if stage == "鉴别诊断":
            inputs.extend(["感染暴露史", "结核相关检查", "用药史"])
    elif stage == "分型活动度与并发症评估":
        inputs = ["病变范围", "活动度", "并发症", "肛周病变", "高危因素"]
    elif stage == "药物与营养治疗选择":
        inputs = ["确诊状态", "病变范围", "活动度", "并发症", "禁忌证", "感染风险", "既往治疗反应", "合并症"]
    elif stage == "肛周病变与瘘管处理":
        inputs = ["肛周症状", "肛周体格检查", "肛周磁共振", "脓肿评估", "感染风险"]
    elif stage == "围手术期管理与术后复发预防":
        inputs = ["手术指征", "营养状态", "感染或脓肿情况", "用药史", "术后复发风险"]
    elif stage == "治疗监测与长期管理":
        inputs = ["当前症状", "炎症指标", "既往内镜或影像", "治疗史", "营养状态", "心理健康状态"]
    else:
        inputs = ["症状与体征", "既往病史", "急症红旗征象"]

    keyword_inputs = [
        ("粪便钙卫蛋白", "粪便钙卫蛋白结果"),
        ("FC", "粪便钙卫蛋白结果"),
        ("结肠镜", "结肠镜报告"),
        ("回肠末段", "回肠末段评估"),
        ("活检", "活检部位和病理结果"),
        ("胃十二指肠镜", "胃十二指肠镜报告"),
        ("胶囊内镜", "狭窄或梗阻风险评估"),
        ("CTE", "CT小肠成像报告"),
        ("MRE", "磁共振小肠成像报告"),
        ("肛周", "肛周评估"),
        ("营养", "营养状态"),
        ("手术", "手术相关资料"),
    ]
    for keyword, required_input in keyword_inputs:
        if keyword in text:
            inputs.append(required_input)
    return _dedupe(inputs)


def infer_safety_notes(recommendation: CsvRecommendation, stage: str) -> list[str]:
    text = _semantic_text(recommendation)
    notes = [DEFAULT_SAFETY_NOTE]
    if stage in {"诊断证据整合", "鉴别诊断"}:
        notes.append("信息不足时不得输出最终诊断，应列出支持证据、反对证据和缺失信息。")
    if stage == "药物与营养治疗选择":
        notes.append("药物启动、联合用药、停药或调整剂量必须由医生结合感染风险、禁忌证和患者偏好决定。")
    if stage == "肛周病变与瘘管处理":
        notes.append("肛周脓肿快速加重、发热或全身感染表现需要及时就医。")
    if stage == "围手术期管理与术后复发预防":
        notes.append("手术时机和术后预防方案需要外科、消化科和营养支持团队共同评估。")
    if stage == "治疗监测与长期管理":
        notes.append("症状加重、营养不良或心理危机应及时转由医生或相应专科处理。")
    if "胶囊内镜" in text:
        notes.append("疑似狭窄或梗阻时，应先评估胶囊滞留风险。")
    if "感染" in text or "结核" in text:
        notes.append("使用免疫抑制或生物制剂前，应由医生评估感染和结核风险。")
    return _dedupe(notes)


def evidence_level_text(recommendation: CsvRecommendation) -> str:
    if recommendation.is_bps:
        return "最佳实践声明"
    if recommendation.evidence_grade:
        return f"证据等级{recommendation.evidence_grade}"
    return "证据等级未标明"


def recommendation_strength_text(recommendation: CsvRecommendation) -> str:
    if recommendation.is_bps:
        return "最佳实践声明"
    strength = recommendation.recommendation_strength.strip()
    if strength == "强":
        return "强推荐"
    if strength == "弱":
        return "弱推荐"
    if strength:
        return f"{strength}推荐"
    return "推荐强度未标明"


def rationale_text(recommendation: CsvRecommendation) -> str:
    parts = []
    if recommendation.reason:
        parts.append(f"推荐理由：{_clip(clean_guideline_text(recommendation.reason), 520)}")
    if recommendation.implementation_advice:
        parts.append(f"实施要点：{_clip(clean_guideline_text(recommendation.implementation_advice), 420)}")
    if not parts:
        return "该条为指南推荐意见，使用时需要结合患者完整临床资料和医生判断。"
    return " ".join(parts)


def source_span_text(recommendation: CsvRecommendation) -> str:
    label = recommendation_card_id(recommendation)
    page_text = _page_span_text(recommendation.page_start, recommendation.page_end)
    return f"{label}，{page_text}"


def recommendation_card_id(recommendation: CsvRecommendation) -> str:
    number = recommendation.number or _fallback_number_from_id(recommendation.recommendation_id)
    return f"推荐意见{number}"


def clean_guideline_text(text: str) -> str:
    cleaned = _clean_cell(text)
    for marker in ("——中华消化杂志", "中华消化杂志"):
        index = cleaned.find(marker)
        if index > 0:
            cleaned = cleaned[:index].rstrip("。；; ，,")
    return cleaned


def load_semantic_overrides(path: str | Path) -> dict[str, Mapping[str, Any]]:
    override_path = Path(path)
    text = override_path.read_text(encoding="utf-8")
    if override_path.suffix.lower() == ".json":
        import json

        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)

    if not isinstance(payload, Mapping):
        raise ValueError("语义增强文件顶层必须是映射对象。")
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, Mapping)
    }


def build_llm_enrichment_prompt(recommendation: CsvRecommendation) -> str:
    """Build a Chinese prompt for external LLM semantic enrichment."""

    return (
        "你是临床指南技能包结构化助手。请只基于给定推荐意见，输出 JSON；"
        "JSON 字段名必须使用英文，字段值必须使用中文，不得扩展指南之外的医学事实。\n\n"
        "需要输出的字段：clinical_stage, clinical_task, population, condition, "
        "required_inputs, safety_notes。\n"
        "其中 clinical_stage、clinical_task、population、condition 必须是非空字符串；"
        "condition 必须用一句中文描述本推荐适用的临床情境，不得为空；"
        "required_inputs、safety_notes 必须是 JSON 字符串数组，且都至少包含一个字符串；"
        f"如果没有特殊安全注意事项，safety_notes 必须输出“{DEFAULT_SAFETY_NOTE}”。\n\n"
        f"推荐编号：{recommendation_card_id(recommendation)}\n"
        f"章节：{recommendation.section or '未分章节'}\n"
        f"推荐正文：{clean_guideline_text(recommendation.text)}\n"
        f"证据等级：{evidence_level_text(recommendation)}\n"
        f"推荐强度：{recommendation_strength_text(recommendation)}\n"
        f"推荐理由：{clean_guideline_text(recommendation.reason) or '未提供'}\n"
        f"实施建议：{clean_guideline_text(recommendation.implementation_advice) or '未提供'}"
    )


def fill_empty_llm_condition(
    payload: Mapping[str, Any],
    recommendation: CsvRecommendation,
) -> Mapping[str, Any]:
    if "condition" not in payload:
        return payload
    condition = payload["condition"]
    if isinstance(condition, str) and condition.strip():
        return payload

    fallback_payload = dict(payload)
    stage = payload.get("clinical_stage")
    if not isinstance(stage, str) or not stage.strip():
        stage = infer_clinical_stage(recommendation)
    fallback_payload["condition"] = infer_condition(recommendation, _clean_cell(stage))
    return fallback_payload


def validate_llm_enrichment_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = {
        "clinical_stage",
        "clinical_task",
        "population",
        "condition",
        "required_inputs",
        "safety_notes",
    }
    missing = sorted(required_fields - set(payload))
    if missing:
        raise ValueError("LLM 语义增强结果缺少字段：" + ", ".join(missing))

    enriched = {
        "clinical_stage": _require_text(payload["clinical_stage"], "clinical_stage"),
        "clinical_task": _require_text(payload["clinical_task"], "clinical_task"),
        "population": _require_text(payload["population"], "population"),
        "condition": _require_text(payload["condition"], "condition"),
        "required_inputs": _require_text_list(payload["required_inputs"], "required_inputs"),
        "safety_notes": _require_text_list(
            payload["safety_notes"],
            "safety_notes",
            default_if_empty=DEFAULT_SAFETY_NOTE,
        ),
    }
    return enriched


def _semantic_override_for(
    recommendation: CsvRecommendation,
    recommendation_id: str,
    semantic_overrides: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for key in (
        recommendation_id,
        recommendation.recommendation_id,
        recommendation.number,
        recommendation.title,
    ):
        if key and key in semantic_overrides:
            return semantic_overrides[key]
    return {}


def _action_text(recommendation: CsvRecommendation) -> str:
    return _clip(clean_guideline_text(recommendation.text), 280)


def _source_quote(recommendation: CsvRecommendation) -> str:
    source = recommendation.text or recommendation.source_text
    return _clip(clean_guideline_text(source), 360)


def _parse_json_object(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, Mapping):
        raise ValueError("LLM 语义增强结果必须是 JSON object。")
    return data


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM 语义增强字段 {field_name} 必须是非空字符串。")
    return _clean_cell(value)


def _require_text_list(
    value: Any,
    field_name: str,
    *,
    default_if_empty: str | None = None,
) -> list[str]:
    values = _extract_text_list_items(value)
    if values is None:
        raise ValueError(f"LLM 语义增强字段 {field_name} 必须是字符串列表。")

    cleaned = [
        _normalize_llm_list_item(_require_text(item, field_name))
        for item in values
        if item.strip()
    ]
    if not cleaned:
        if default_if_empty is not None:
            return [default_if_empty]
        raise ValueError(f"LLM 语义增强字段 {field_name} 必须至少包含一个字符串。")
    return _dedupe(cleaned)


def _extract_text_list_items(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return _split_text_list(value)
    if isinstance(value, (list, tuple)):
        values: list[str] = []
        for item in value:
            item_values = _extract_text_list_items(item)
            if item_values is None:
                return None
            values.extend(item_values)
        return values
    if isinstance(value, Mapping):
        values = []
        for item in value.values():
            item_values = _extract_text_list_items(item)
            if item_values is None:
                return None
            values.extend(item_values)
        return values
    return None


def _split_text_list(value: str) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            parsed_values = _extract_text_list_items(parsed)
            if parsed_values is not None:
                return parsed_values
    return [
        item.strip(" -•\t")
        for item in re.split(r"[；;、\n\r]+", text)
        if item.strip(" -•\t")
    ]


def _normalize_llm_list_item(value: str) -> str:
    return LLM_LIST_ITEM_ALIASES.get(value, value)


def _semantic_text(recommendation: CsvRecommendation) -> str:
    return " ".join(
        [
            recommendation.text,
            recommendation.reason,
            recommendation.section,
        ]
    )


def _ids_for(
    recommendations: list[CsvRecommendation],
    card_ids_by_number: Mapping[str, str],
    numbers: set[str],
) -> list[str]:
    return [
        card_ids_by_number[recommendation.number]
        for recommendation in recommendations
        if recommendation.number in numbers
    ]


def _ids_in_section(
    recommendations: list[CsvRecommendation],
    card_ids_by_number: Mapping[str, str],
    section: str,
) -> list[str]:
    return [
        card_ids_by_number[recommendation.number]
        for recommendation in recommendations
        if recommendation.section == section
    ]


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\u3000", " ")
    text = text.replace("", "-").replace("", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes", "y", "是", "真"}


def _parse_int(value: str | None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _fallback_number_from_id(value: str) -> str:
    match = re.search(r"(\d+[A-Za-z]?)", value)
    return match.group(1) if match else "0"


def _page_span_text(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return "页码未标明"
    if page_start is None:
        return f"第{page_end or 0}页"
    if page_end is None or page_end == page_start:
        return f"第{page_start}页"
    return f"第{page_start}至{page_end}页"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ，,；;。") + "……"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _collect_non_chinese_values(value: Any, path: str, issues: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _collect_non_chinese_values(nested, f"{path}.{key}", issues)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _collect_non_chinese_values(nested, f"{path}[{index}]", issues)
        return
    if isinstance(value, str) and value and not _contains_cjk(value):
        issues.append(f"{path}: {value!r}")


def _contains_cjk(value: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", value) is not None
