from __future__ import annotations

from guideline_skill.schema import DifferentialDiagnosisItem, PatientCase


def enrich_differential_diagnoses(
    patient_case: PatientCase,
    diagnoses: list[DifferentialDiagnosisItem],
) -> list[DifferentialDiagnosisItem]:
    return [_enrich_one(patient_case, diagnosis) for diagnosis in diagnoses]


def _enrich_one(
    patient_case: PatientCase,
    diagnosis: DifferentialDiagnosisItem,
) -> DifferentialDiagnosisItem:
    name = diagnosis.disease_name
    if "肠结核" in name:
        return _with_features(diagnosis, *_intestinal_tb_features(patient_case))
    if "溃疡性结肠炎" in name:
        return _with_features(diagnosis, *_uc_features(patient_case))
    if "白塞" in name:
        return _with_features(diagnosis, *_behcet_features(patient_case))
    if "感染性肠炎" in name:
        return _with_features(diagnosis, *_infectious_features(patient_case))
    if "药物性肠炎" in name:
        return _with_features(diagnosis, *_drug_induced_features(patient_case))
    if "淋巴瘤" in name:
        return _with_features(diagnosis, *_lymphoma_features(patient_case))
    return diagnosis


def _intestinal_tb_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has(patient_case, "回盲部"):
        supporting.append("回盲部受累")
    if _has(patient_case, "溃疡"):
        supporting.append("溃疡")
    if _has(patient_case, "狭窄"):
        supporting.append("狭窄")
    if _has_any(patient_case, ("低热", "发热", "消瘦", "体重下降")):
        supporting.append("全身症状或消耗表现")

    against = []
    if not _has_any(patient_case, ("IGRA", "T-SPOT", "PPD", "结核感染", "TB阳性", "结核阳性")):
        against.append("未提供结核感染证据")
    if not _has_any(patient_case, ("胸片", "胸部CT", "胸部影像", "肺结核")):
        against.append("未提供胸部影像")
    if not _has_any(patient_case, ("抗酸染色", "分枝杆菌", "结核培养", "TB-PCR")):
        against.append("未提供抗酸染色/分枝杆菌检测")

    missing = ["IGRA/T-SPOT 或 PPD", "胸部影像", "病理抗酸染色或分枝杆菌检测"]
    return supporting, against, missing


def _uc_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has(patient_case, "便血"):
        supporting.append("便血")
    if _has_any(patient_case, ("连续", "直肠受累", "结肠炎")):
        supporting.append("连续性结肠炎症或直肠受累线索")
    if _has(patient_case, "腹泻"):
        supporting.append("腹泻")

    against = []
    if _has_any(patient_case, ("回肠末端", "回盲部")):
        against.append("存在回肠末端或回盲部受累，更需与 CD 鉴别")
    if _has_any(patient_case, ("狭窄", "瘘管", "肛瘘", "肛周脓肿")):
        against.append("狭窄、瘘管或肛周病变不典型于普通 UC")

    missing = ["结肠镜病变连续性和直肠受累描述", "病理评估", "小肠影像"]
    return supporting, against, missing


def _behcet_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has(patient_case, "口腔溃疡"):
        supporting.append("口腔溃疡")
    if _has_any(patient_case, ("外阴溃疡", "生殖器溃疡")):
        supporting.append("外阴/生殖器溃疡")
    if _has_any(patient_case, ("眼炎", "葡萄膜炎", "皮疹", "结节红斑")):
        supporting.append("眼部或皮肤系统表现")
    if _has_any(patient_case, ("回盲部", "溃疡")):
        supporting.append("回盲部或肠道溃疡")

    against = []
    if not _has_any(patient_case, ("口腔溃疡", "外阴溃疡", "生殖器溃疡", "眼炎", "皮疹")):
        against.append("未提供复发性口腔/外阴溃疡或眼皮肤表现")

    missing = ["白塞病系统症状评估", "眼科/皮肤科评估", "内镜溃疡形态与病理"]
    return supporting, against, missing


def _infectious_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has(patient_case, "腹泻"):
        supporting.append("腹泻")
    if _has(patient_case, "发热"):
        supporting.append("发热")
    if _has_any(patient_case, ("急性", "不洁饮食", "旅行", "集体发病")):
        supporting.append("急性起病或感染暴露线索")

    against = []
    if _has_any(patient_case, ("三个月", "半年", "多年", "慢性")):
        against.append("病程较长，单纯急性感染性肠炎解释不足")
    if not _has_any(patient_case, ("粪便培养", "病原", "艰难梭菌", "寄生虫")):
        against.append("未提供粪便病原学证据")

    missing = ["粪便培养或病原学", "艰难梭菌检测", "寄生虫或特殊感染评估"]
    return supporting, against, missing


def _drug_induced_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has_any(patient_case, ("NSAID", "非甾体", "止痛药", "抗生素", "免疫治疗")):
        supporting.append("存在可疑用药史")
    if _has(patient_case, "溃疡"):
        supporting.append("小肠或肠道溃疡")

    against = []
    if not _has_any(patient_case, ("NSAID", "非甾体", "止痛药", "抗生素", "免疫治疗", "用药")):
        against.append("未提供可解释肠道损伤的用药时间线")

    missing = ["详细用药史", "停药后变化", "内镜和病理排除其他病因"]
    return supporting, against, missing


def _lymphoma_features(patient_case: PatientCase) -> tuple[list[str], list[str], list[str]]:
    supporting = []
    if _has_any(patient_case, ("淋巴瘤", "肿块", "淋巴结")):
        supporting.append("病理或影像提示淋巴瘤/肿块/淋巴结线索")
    if _has(patient_case, "体重下降"):
        supporting.append("体重下降")
    if _has_any(patient_case, ("狭窄", "溃疡")):
        supporting.append("肠道溃疡或狭窄")

    against = []
    if not _has_any(patient_case, ("免疫组化", "肿块", "淋巴结", "淋巴瘤")):
        against.append("未提供肿块、淋巴结或免疫组化证据")

    missing = ["深取材或重复活检", "免疫组化", "影像评估肿块/淋巴结"]
    return supporting, against, missing


def _with_features(
    diagnosis: DifferentialDiagnosisItem,
    supporting_features: list[str],
    against_features: list[str],
    missing_tests: list[str],
) -> DifferentialDiagnosisItem:
    return diagnosis.model_copy(
        update={
            "supporting_features": _dedupe([*diagnosis.supporting_features, *supporting_features]),
            "against_features": _dedupe([*diagnosis.against_features, *against_features]),
            "missing_tests": _dedupe([*diagnosis.missing_tests, *missing_tests]),
        }
    )


def _has(patient_case: PatientCase, term: str) -> bool:
    return _has_any(patient_case, (term,))


def _has_any(patient_case: PatientCase, terms: tuple[str, ...]) -> bool:
    text = " ".join(
        [
            patient_case.raw_text,
            *patient_case.symptoms,
            *patient_case.endoscopy,
            *patient_case.imaging,
            *patient_case.pathology,
            *patient_case.unknowns,
        ]
    )
    return any(term.casefold() in text.casefold() for term in terms)


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
