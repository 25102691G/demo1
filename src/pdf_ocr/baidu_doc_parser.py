from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import TextPage


logger = logging.getLogger(__name__)


class BaiduDocParserError(RuntimeError):
    """百度智能云文档解析失败。"""


@dataclass(frozen=True)
class BaiduDocParserConfig:
    api_key_env: str = "BAIDU_API_KEY"
    secret_key_env: str = "BAIDU_SECRET_KEY"
    token_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    parser_url: str = "https://aip.baidubce.com/rest/2.0/brain/online/v2/parser/task"
    query_url: str = "https://aip.baidubce.com/rest/2.0/brain/online/v2/parser/task/query"
    timeout_seconds: int = 300
    poll_interval_seconds: float = 5.0


@dataclass(frozen=True)
class BaiduDocParserArtifacts:
    task_id: str
    parse_result_url: str
    markdown_url: str
    parse_result_text: str
    markdown_text: str


class BaiduDocParserClient:
    def __init__(self, config: BaiduDocParserConfig | None = None) -> None:
        self.config = config or BaiduDocParserConfig()

    def parse_pdf(self, pdf_path: str | Path) -> list[TextPage]:
        artifacts = self.parse_pdf_artifacts(pdf_path)
        if artifacts.parse_result_text:
            pages = _pages_from_parse_result(_loads_json_object(artifacts.parse_result_text))
        elif artifacts.markdown_text.strip():
            pages = [TextPage(page_number=1, text=artifacts.markdown_text)]
        else:
            pages = []

        if not any(page.text.strip() for page in pages):
            raise BaiduDocParserError("百度文档解析未返回可用文本")
        return pages

    def parse_pdf_artifacts(self, pdf_path: str | Path) -> BaiduDocParserArtifacts:
        path = Path(pdf_path)
        if not path.exists():
            raise BaiduDocParserError(f"PDF 文件不存在: {path}")

        access_token = self._get_access_token()
        submit_url = os.getenv("BAIDU_DOC_PARSER_URL", self.config.parser_url)
        query_url = os.getenv("BAIDU_DOC_PARSER_QUERY_URL", self.config.query_url)
        task_id = self._submit_task(submit_url, access_token, path)
        query_result = self._wait_for_task(query_url, access_token, task_id)
        return self._download_artifacts(task_id, query_result)

    def _submit_task(self, submit_url: str, access_token: str, path: Path) -> str:
        payload = {
            "file_data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "file_name": path.name,
            "recognize_formula": "True",
            "analysis_chart": "True",
            "angle_adjust": "True",
            "parse_image_layout": "True",
            "language_type": "CHN_ENG",
            "switch_digital_width": "half",
            "html_table_format": "True",
        }

        result = self._post_json(f"{submit_url}?access_token={access_token}", payload)
        task_id = _find_task_id(result)
        if not task_id:
            raise BaiduDocParserError(f"百度文档解析提交任务未返回 task_id: {result}")
        _debug("submit task_id=%s keys=%s", task_id, list(result.keys()))
        return task_id

    def _wait_for_task(self, query_url: str, access_token: str, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            result = self._post_json(
                f"{query_url}?access_token={access_token}",
                {"task_id": task_id},
            )
            payload = result.get("result")
            if not isinstance(payload, dict):
                raise BaiduDocParserError(f"百度文档解析查询返回格式异常: {result}")

            status = str(payload.get("status") or "").lower()
            _debug(
                "query task_id=%s status=%s parse_result_url=%s markdown_url=%s",
                task_id,
                status,
                bool(payload.get("parse_result_url")),
                bool(payload.get("markdown_url")),
            )
            if status == "success":
                return payload
            if status == "failed":
                raise BaiduDocParserError(
                    f"百度文档解析任务失败: {payload.get('task_error') or payload}"
                )
            if status not in {"pending", "running", ""}:
                raise BaiduDocParserError(f"百度文档解析任务状态异常: {payload}")
            time.sleep(self.config.poll_interval_seconds)

        raise BaiduDocParserError(f"百度文档解析任务超时: {task_id}")

    def _download_artifacts(
        self,
        task_id: str,
        query_result: dict[str, Any],
    ) -> BaiduDocParserArtifacts:
        parse_result_url = query_result.get("parse_result_url")
        markdown_url = query_result.get("markdown_url")
        parse_result_text = ""
        markdown_text = ""

        if isinstance(parse_result_url, str) and parse_result_url:
            parse_result_text = self._get_text(parse_result_url)
            pages = _pages_from_parse_result(_loads_json_object(parse_result_text))
            _debug(
                "download parse_result pages=%s text_chars=%s",
                len(pages),
                sum(len(page.text) for page in pages),
            )

        if isinstance(markdown_url, str) and markdown_url:
            markdown_text = self._get_text(markdown_url)
            _debug("download markdown text_chars=%s", len(markdown_text))

        if not parse_result_text and not markdown_text:
            raise BaiduDocParserError(f"百度文档解析结果缺少可下载文本: {query_result}")

        return BaiduDocParserArtifacts(
            task_id=task_id,
            parse_result_url=parse_result_url if isinstance(parse_result_url, str) else "",
            markdown_url=markdown_url if isinstance(markdown_url, str) else "",
            parse_result_text=parse_result_text,
            markdown_text=markdown_text,
        )

    def _get_access_token(self) -> str:
        api_key = os.getenv(self.config.api_key_env)
        secret_key = os.getenv(self.config.secret_key_env)
        if not api_key or not secret_key:
            raise BaiduDocParserError(
                f"缺少百度智能云密钥环境变量: {self.config.api_key_env}, {self.config.secret_key_env}"
            )

        token_url = os.getenv("BAIDU_TOKEN_URL", self.config.token_url)
        result = self._post_json(
            token_url,
            {
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            },
        )
        access_token = result.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise BaiduDocParserError(f"百度 access_token 获取失败: {result}")
        return access_token

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=urlencode(payload).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        return self._request_json(request)

    def _get_text(self, url: str) -> str:
        request = Request(url, method="GET")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BaiduDocParserError(f"百度文档解析下载 HTTP 错误: {exc.code} {detail}") from exc
        except URLError as exc:
            raise BaiduDocParserError(f"百度文档解析下载网络错误: {exc}") from exc

    def _request_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BaiduDocParserError(f"百度文档解析 HTTP 错误: {exc.code} {detail}") from exc
        except URLError as exc:
            raise BaiduDocParserError(f"百度文档解析网络错误: {exc}") from exc

        try:
            result = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BaiduDocParserError(f"百度文档解析返回非 JSON: {data[:200]}") from exc
        if not isinstance(result, dict):
            raise BaiduDocParserError(f"百度文档解析返回格式异常: {result}")
        error_code = result.get("error_code")
        if error_code not in (None, 0):
            raise BaiduDocParserError(f"百度文档解析错误: {result}")
        return result


def _find_task_id(payload: dict[str, Any]) -> str | None:
    result = payload.get("result")
    if isinstance(result, dict):
        task_id = result.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
    task_id = payload.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    return None


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaiduDocParserError(f"百度文档解析下载结果不是 JSON: {text[:200]}") from exc
    if not isinstance(result, dict):
        raise BaiduDocParserError(f"百度文档解析下载结果格式异常: {result}")
    return result


def _pages_from_parse_result(payload: dict[str, Any]) -> list[TextPage]:
    pages_payload = payload.get("pages")
    if not isinstance(pages_payload, list):
        return []

    pages: list[TextPage] = []
    for index, item in enumerate(pages_payload, 1):
        if not isinstance(item, dict):
            continue
        page_number = _page_number(item, index)
        text = _page_text(item)
        if text.strip():
            pages.append(TextPage(page_number=page_number, text=text))
    pages.sort(key=lambda page: page.page_number)
    return pages


def _page_number(item: dict[str, Any], fallback: int) -> int:
    value = item.get("page_num")
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str) and value.isdigit():
        return int(value) + 1
    return fallback


def _page_text(item: dict[str, Any]) -> str:
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text

    fragments: list[str] = []
    for key in ("layouts", "tables", "images"):
        value = item.get(key)
        if value is not None:
            fragments.extend(_collect_text_fragments(value))
    return "\n".join(_dedupe_fragments(fragments))


def _collect_text_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(value, dict):
        for key in ("text", "markdown", "content", "image_description"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                fragments.append(item.strip())
        for item in value.values():
            fragments.extend(_collect_text_fragments(item))
    elif isinstance(value, list):
        for item in value:
            fragments.extend(_collect_text_fragments(item))
    return _dedupe_fragments(fragments)


def _dedupe_fragments(fragments: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment in seen:
            continue
        seen.add(fragment)
        result.append(fragment)
    return result


def _debug(message: str, *args: Any) -> None:
    if os.getenv("BAIDU_DOC_PARSER_DEBUG") == "1":
        logger.info("Baidu doc parser " + message, *args)
