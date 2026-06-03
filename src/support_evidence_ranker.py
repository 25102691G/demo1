from __future__ import annotations

from guideline_skill.schema import PatientCase


def rank_support_evidence(
    patient_case: PatientCase,
    support_evidence: list[str],
) -> list[str]:
    """Normalize and order visible support evidence for the current case."""

    if not support_evidence:
        return []

    ranked: list[str] = []
    raw_text = patient_case.raw_text

    symptom_order = ["腹痛", "腹泻", "体重下降", "发热", "便血", "口腔溃疡", "乏力", "肛瘘", "肛周脓肿"]
    symptoms = [symptom for symptom in symptom_order if symptom in patient_case.symptoms]
    if symptoms:
        ranked.append(f"出现症状：{'、'.join(symptoms)}")

    if "回盲部" in patient_case.endoscopy and "回肠末端" in patient_case.endoscopy:
        ranked.append("内镜提示：回盲部及回肠末端受累")
    elif "回盲部" in patient_case.endoscopy:
        ranked.append("内镜提示：回盲部受累")
    elif "回肠末端" in patient_case.endoscopy:
        ranked.append("内镜提示：回肠末端受累")

    if ("多发" in raw_text or "多处" in raw_text) and "溃疡" in patient_case.endoscopy:
        ranked.append("内镜提示：多发溃疡")
    elif "纵行溃疡" in patient_case.endoscopy:
        ranked.append("内镜提示：纵行溃疡")
    elif "溃疡" in patient_case.endoscopy:
        ranked.append("内镜提示：溃疡")

    if "狭窄" in patient_case.endoscopy or "狭窄" in raw_text:
        ranked.append("内镜或影像提示：狭窄")
    if "瘘管" in patient_case.endoscopy or "瘘管" in raw_text:
        ranked.append("内镜或影像提示：瘘管")

    if patient_case.imaging:
        ranked.extend(
            evidence for evidence in support_evidence if evidence.startswith("影像学检查")
        )

    pathology_terms = [
        term
        for term in ["慢性炎症", "肉芽肿", "透壁性炎症"]
        if term in patient_case.pathology or term in raw_text
    ]
    if pathology_terms:
        ranked.append(f"病理支持信息：{'、'.join(pathology_terms)}")

    ranked.extend(
        evidence
        for evidence in support_evidence
        if not _covered_by_normalized_evidence(evidence, ranked)
    )
    return _dedupe(ranked)


def _covered_by_normalized_evidence(evidence: str, ranked: list[str]) -> bool:
    normalized_text = " ".join(ranked)
    if evidence.startswith("出现症状：") and any(symptom in normalized_text for symptom in ["腹痛", "腹泻", "体重下降", "发热"]):
        return True
    if "回盲部" in evidence and "回盲部" in normalized_text:
        return True
    if "回肠末端" in evidence and "回肠末端" in normalized_text:
        return True
    if "多发溃疡" in evidence and "多发溃疡" in normalized_text:
        return True
    if "内镜/肠腔表现支持：溃疡" in evidence and "溃疡" in normalized_text:
        return True
    if "狭窄" in evidence and "狭窄" in normalized_text:
        return True
    if evidence.startswith("病理支持信息") and "病理支持信息" in normalized_text:
        return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
