# 病例数据集批量运行

`test/patients/` 下每个 JSON 文件表示一个病人，批量脚本会逐个读取并调用 `scripts/run_skill_engine.py`。

## 病人 JSON 格式

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

必填字段是 `name`、`sex`、`age`。病例文本字段中至少填写一个：`clinical_presentation`、`lab_tests`、`imaging_tests`、`endoscopy`、`pathology`。

## 批量运行

```bash
python test/run_batch.py \
  --patients-dir test/patients \
  --output-dir test/outputs \
  --debug \
  --similarity-threshold 0.8 \
  --hpo
```

运行结果会写入 `test/outputs/`，每个病人对应一个输出 JSON。
