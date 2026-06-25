# Guideline Skill Engine Workflow

## ICD10 数据转换

将医保 ICD10 Excel 第二个 sheet `完整分类与代码` 转换为 JSON：

```powershell
python scripts/extract_icd10_xlsx_to_json.py
```

默认输入文件：

```text
data/ICD10/医保ICD10_v2.0_0122.xlsx
```

默认输出文件：

```text
data/ICD10/ICD10.json
```

输出 JSON 中 key 会转换为英文，value 保持 Excel 原文不变。

## ICD10 向量生成

将 `ICD10.json` 中每一项的 `diagnosis_name` 生成 embedding：

```powershell
python scripts/build_icd10_embeddings.py
```

默认输入文件：

```text
data/ICD10/ICD10.json
```

默认模型路径：

```text
data/bge-large-zh-v1.5
```

默认输出文件：

```text
data/ICD10/ICD10_embeddings.pt
```

脚本会自动跳过 `diagnosis_name` 为空的记录。输出 `.pt` 文件为 torch tensor，结构类似 `data/ontology/definition_embeddings.pt`。运行该命令需要当前 Python 环境已安装 `torch` 和 `transformers`。

## HPO 向量生成

将 `data/ontology/hpo.json` 生成对应的 HPO embedding：

```bash
python scripts/build_embeddings.py --input data/ontology/hpo.json
```

默认输出文件：

```text
data/ontology/hpo_embeddings.pt
```

本项目用于把医学指南 PDF 转为可执行的 guideline skill pack，并通过通用 `SkillEngine` 对病例输入执行 workflow，最终生成符合 schema 的 JSON 输出。

当前主流程：

1. PDF -> OCR `*.parse_result.json`
2. PDF 或 OCR `*.parse_result.json` -> recommendation cards JSONL
3. JSONL -> 同目录 `skill.yaml`
4. 运行 `scripts/run_skill_engine.py` 生成 `workflow_output` JSON

本项目输出仅用于指南知识组织和临床决策支持，不替代医生诊断、处方或急诊处理。

## 环境准备

建议使用 `uv` 创建 Python 3.11 环境。`.venv` 是本机环境目录，不应提交到 git。

### 首次 clone 后初始化环境

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 后续使用已有 uv 环境

进入项目目录后激活已有虚拟环境：

```bash
source .venv/bin/activate
```

如果 `pyproject.toml` 中的依赖有变化，再重新安装项目依赖：

```bash
uv pip install -e .
```

验证 Python 与 CUDA 版本：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 1. OCR 提取 PDF

`scripts/preview_pdf_ocr.py` 用于直接调用百度智能云文档解析接口，查看 PDF 经过 OCR/文档解析后的原始结果。该脚本不会运行后续 guideline 抽取流程，成功时控制台不输出信息。

运行前需要配置环境变量：

```powershell
$env:BAIDU_API_KEY="你的百度 API Key"
$env:BAIDU_SECRET_KEY="你的百度 Secret Key"
```

单文档提取：

```bash
python scripts/preview_pdf_ocr.py --input /data/skills/肠结核的诊断与治疗/肠结核的诊断与治疗.pdf
```

批量提取：

```bash
python scripts/preview_pdf_ocr.py --input-dir /data/skills
```

`--input` 和 `--input-dir` 必须且只能指定一个。`--input-dir` 会递归扫描目录下所有 `.pdf` 文件。

每个 PDF 的输出文件保存在 PDF 同目录：

```text
<PDF文件名>.parse_result.json
<PDF文件名>.markdown.md
```

## 2. 转换 OCR 结果为 recommendation_card

如果已经有 OCR 文档解析结果，可以使用 `scripts/ocr_to_cards.py` 将 `*.parse_result.json` 转换为 `recommendation_card.jsonl`。脚本会读取 `pages[*].layouts`，保留 `title` 和 `text`，通过 title 的 `parent` 链构建 `section_path`，按双栏阅读顺序排序，并调用 DeepSeek LLM 过滤元信息和参考文献内容。

运行前需要配置环境变量：

```powershell
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

单文件转换：

```bash
python scripts/ocr_to_cards.py --input data/skills/肠结核的诊断与治疗/肠结核的诊断与治疗.parse_result.json
```

批量转换：

```bash
python scripts/ocr_to_cards.py --input-dir data/skills
```

`--input` 和 `--input-dir` 必须且只能指定一个。

脚本还会在每个输入文件所在目录生成 `recommendation_card_summary.json`，记录原始 layouts 数量、各 type 数量、被丢弃的 layout 明细、被丢弃的 unit 明细和发生合并的 layout 组。输出的每行都是一个 `recommendation_card`，并保留 `section_path` 和 `source_location.source_span` 便于后续溯源。

## 3. recommendation_card 到 skill.yaml

使用 `scripts/build_skill_pack.py` 将抽取后的 JSONL 构建为 skill pack。

输入：

- `data/skills/<指南名>/result.jsonl`

输出：

```text
data/skills/<指南名>/
  result.jsonl
  skill.yaml
```

单文件构建示例：

```bash
python scripts/build_skill_pack.py --cards data/skills/肠结核的诊断与治疗//recommendation_card.jsonl --force --similarity-threshold 0.7 --hpo
```

批量构建示例：

```bash
python scripts/build_skill_pack.py --cards data/skills --force --similarity-threshold 0.7 --hpo
```

`--similarity-threshold` 表示特征匹配相似度门槛值，默认为0.8。

## 4. 运行 SkillEngine

使用 `scripts/run_skill_engine.py` 输入病例并执行 skill workflow。

```bash
python scripts/run_skill_engine.py \
  --name "张三" \
  --sex "male" \
  --age 25 \
  --clinical-presentation "腹痛" \
  --endoscopy "进镜至回肠末端，回盲瓣变形，回盲瓣口及回肠末端四壁散在形态不规则溃疡，部分呈纵形分布，表面覆大量浓稠白色粘液，回肠末端四壁皱襞纠集，肠腔狭窄，内镜无法通过。于回肠末端溃疡周边活检4块，质软。所见升结肠、横结肠、降结肠、乙状结肠粘膜光滑。距肛门15cm以下直肠四壁散在点状糜烂及浅溃疡，血管纹理模糊。于回盲部、升结肠、横结肠、降结肠、乙状结肠、直肠各活检2块，组织软。" \
  --debug \
  --similarity-threshold 0.8 \
  --output output/zhangsan_hpo.json \
  --hpo
```

`--similarity-threshold` 表示特征匹配相似度门槛值，默认为0.8。

### 批量运行病例数据集

病例数据集放在 `test/patients/`，每个病人一个 JSON 文件：

```json
{
  "case_id": "case_001",
  "name": "张三",
  "sex": "male",
  "age": 25,
  "clinical_presentation": "腹痛",
  "lab_tests": "",
  "imaging_tests": "",
  "endoscopy": "内镜检查文本",
  "pathology": ""
}
```

批量运行：

```bash
python test/run_batch.py \
  --patients-dir test/patients \
  --output-dir test/outputs \
  --debug \
  --similarity-threshold 0.8 \
  --hpo
```

每个病人的 workflow 输出会单独写入 `test/outputs/`。

## 输出文件

`data/skills/<指南名>/skill.yaml`

- skill pack 的声明式配置
- 包含 metadata、routing_profile、workflow、output_templates 等
- 不内嵌全部推荐卡片正文，只引用同目录 `result.jsonl`

`data/skills/<指南名>/result.jsonl`

- recommendation cards
- 每行一个 card
- workflow 可按 `clinical_stage` / `clinical_task` 过滤引用，也兼容按 card_id 引用

`data/runs/YYYYMMDD_HH_MM_hpo.json` 或 `--output` 指定的 JSON 文件

- SkillEngine 的最终输出
- 符合 `schema/workflow_output.schema.json`
- 包含 `input_text` 和 `canonical_case`，方便调试原始输入和标准病例结构
- 包含 top candidates、workflow 执行步骤、最终响应、安全信息、引用和 debug trace

## Schema 校验

主流程中会真实执行 schema 校验。

构建 skill pack 时校验：

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
- 当前不做向量检索或 embedding 语义检索。
- PDF/OCR 抽取结果需要人工审核。
- 管理建议只组织指南卡片内容，不生成个体化处方或剂量。
- 有安全红旗时，workflow 会优先停止并提示临床安全评估。
