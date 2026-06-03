from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guideline_skill.csv_to_skill_pack import (  # noqa: E402
    LlmSemanticEnricher,
    OpenAICompatibleChatClient,
    RuleBasedSemanticEnricher,
    SkillPackMetadata,
    load_semantic_overrides,
    write_skill_pack_from_csv,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="将指南推荐意见 CSV 转换为技能包 YAML。")
    parser.add_argument("--csv", required=True, help="推荐意见 CSV 文件路径。")
    parser.add_argument("--output", required=True, help="输出技能包 YAML 文件路径。")
    parser.add_argument("--skill-name", default=SkillPackMetadata.skill_name, help="技能包名称。")
    parser.add_argument("--disease-name", default=SkillPackMetadata.disease_name, help="疾病名称。")
    parser.add_argument("--guideline-name", default=SkillPackMetadata.guideline_name, help="指南名称。")
    parser.add_argument("--guideline-version", default=SkillPackMetadata.guideline_version, help="指南版本。")
    parser.add_argument("--source-pdf", default=SkillPackMetadata.source_pdf, help="来源 PDF 名称。")
    parser.add_argument("--alias", action="append", dest="aliases", help="疾病别名，可重复提供。")
    parser.add_argument("--target-user", action="append", dest="target_users", help="目标用户，可重复提供。")
    parser.add_argument("--scope", default=SkillPackMetadata.scope, help="技能包适用范围说明。")
    parser.add_argument(
        "--semantic-overrides",
        help="可选的 JSON/YAML 语义增强文件；键可为推荐意见编号、CSV recommendation_id 或中文推荐意见 ID。",
    )
    parser.add_argument(
        "--llm-enrich",
        action="store_true",
        help="使用 OpenAI-compatible LLM 生成推荐卡片语义字段；默认使用规则生成。",
    )
    parser.add_argument("--llm-model", help="LLM 模型名；也可使用 OPENAI_MODEL。")
    parser.add_argument(
        "--llm-base-url",
        help="OpenAI-compatible API base URL；默认使用 OPENAI_BASE_URL 或 https://api.openai.com/v1。",
    )
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="LLM 请求超时时间，单位秒。")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM temperature。")
    parser.add_argument(
        "--llm-fallback-rule-based",
        action="store_true",
        help="LLM 请求或 JSON 校验失败时回退到规则版语义增强；默认直接报错。",
    )
    parser.add_argument(
        "--allow-non-chinese-values",
        action="store_true",
        help="允许字段值不含中文；默认会校验除字段名外的字符串值必须含中文。",
    )
    args = parser.parse_args()

    metadata = SkillPackMetadata(
        skill_name=args.skill_name,
        disease_name=args.disease_name,
        disease_aliases=tuple(args.aliases or SkillPackMetadata.disease_aliases),
        guideline_name=args.guideline_name,
        guideline_version=args.guideline_version,
        source_pdf=args.source_pdf,
        target_users=tuple(args.target_users or SkillPackMetadata.target_users),
        scope=args.scope,
    )
    overrides = load_semantic_overrides(args.semantic_overrides) if args.semantic_overrides else None
    semantic_enricher = None
    if args.llm_enrich:
        semantic_enricher = LlmSemanticEnricher(
            client=OpenAICompatibleChatClient.from_env(
                model=args.llm_model,
                base_url=args.llm_base_url,
                timeout_seconds=args.llm_timeout,
                temperature=args.llm_temperature,
            ),
            fallback_enricher=RuleBasedSemanticEnricher() if args.llm_fallback_rule_based else None,
        )
    skill_pack = write_skill_pack_from_csv(
        args.csv,
        args.output,
        metadata=metadata,
        semantic_enricher=semantic_enricher,
        semantic_overrides=overrides,
        require_chinese_values=not args.allow_non_chinese_values,
    )

    summary = {
        "输出文件": str(Path(args.output)),
        "技能包名称": skill_pack.skill_name,
        "疾病名称": skill_pack.disease_name,
        "子技能数量": len(skill_pack.subskills),
        "推荐卡片数量": len(skill_pack.recommendation_cards),
        # "中文内容检查": "已通过" if not args.allow_non_chinese_values else "已跳过",
        "中文内容检查": "已移除"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
