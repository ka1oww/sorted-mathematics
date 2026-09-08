"""End-to-end stdio protocol tests against a real FastMCP subprocess."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp_service as service


ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "src" / "mcp_server.py"
SUPPORT = ROOT / "tests" / "support"


def _support_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SUPPORT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy(path: Path, model: Path, threshold: float) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "confidence_threshold": threshold,
        "selection_rule": "lowest validation threshold reaching 0.95 selective accuracy, else 0.00",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "validation_split_sha256": "0" * 64,
    }))
    return path


def _payload(result):
    if not result.isError and result.structuredContent is not None:
        return result.structuredContent
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}


class ProtocolClient:
    def __init__(self, model: Path | None = None, policy: Path | None = None):
        self.stderr = tempfile.TemporaryFile(mode="w+")
        environment = {
            key: value for key, value in os.environ.items() if key != "PYTHONPATH"
        }
        if model is not None and policy is not None:
            environment.update({
                "SORTED_MATH_MODEL_PATH": str(model),
                "SORTED_MATH_POLICY_PATH": str(policy),
            })
        self.parameters = StdioServerParameters(
            command=os.environ.get("TEST_PYTHON", os.sys.executable),
            args=[str(SERVER)], env=environment, cwd=ROOT,
        )

    async def __aenter__(self):
        self.transport = stdio_client(self.parameters, errlog=self.stderr)
        self.read_stream, self.write_stream = await self.transport.__aenter__()
        self.session_context = ClientSession(self.read_stream, self.write_stream)
        self.session = await self.session_context.__aenter__()
        self.initialized = await self.session.initialize()
        return self

    async def __aexit__(self, *details):
        await self.session_context.__aexit__(*details)
        await self.transport.__aexit__(*details)

    def stderr_text(self) -> str:
        self.stderr.seek(0)
        return self.stderr.read()


def test_stdio_server_protocol_and_sanitized_failures(tmp_path):
    fake_model = _support_module("fake_model")
    pdfs = _support_module("build_protocol_pdf")
    model = fake_model.write_fake_model(tmp_path / "model.joblib")
    policy = _policy(tmp_path / "policy.json", model, 0.0)
    paper = pdfs.write_protocol_pdf(tmp_path / "paper.pdf")
    untrusted = pdfs.write_untrusted_pdf(tmp_path / "untrusted.pdf")
    scanned = pdfs.write_image_only_pdf(tmp_path / "scanned.pdf")

    async def exercise():
        async with ProtocolClient(model, policy) as client:
            assert client.initialized.serverInfo.name == "sorted-mathematics"
            tools = await client.session.list_tools()
            assert [tool.name for tool in tools.tools] == ["classify_question", "read_paper"]
            descriptions = {tool.name: tool.description for tool in tools.tools}
            assert descriptions["classify_question"] == service.CLASSIFY_QUESTION_DOC
            assert descriptions["read_paper"] == service.READ_PAPER_DOC
            assert "not solve" in descriptions["classify_question"]
            assert "refuses" in descriptions["classify_question"]
            assert "pointers only" in descriptions["read_paper"]
            assert all(tool.inputSchema for tool in tools.tools)

            resources = await client.session.list_resources()
            assert [str(item.uri) for item in resources.resources] == [
                "sorted-mathematics://chapters"
            ]
            chapter_resource = await client.session.read_resource(
                "sorted-mathematics://chapters"
            )
            chapters = json.loads(chapter_resource.contents[0].text)
            assert len(chapters) == 21
            assert len({row["slug"] for row in chapters}) == 21

            answered = await client.session.call_tool(
                "classify_question", {"question_text": "synthetic client input"}
            )
            assert not answered.isError
            answered_payload = _payload(answered)
            assert answered_payload["answered"] is True
            assert answered_payload["refusal"] is None

            paper_result = await client.session.call_tool(
                "read_paper", {"pdf_path": str(paper)}
            )
            assert not paper_result.isError, paper_result.content[0].text
            paper_payload = _payload(paper_result)
            assert paper_payload["disclosure_mode"] == "pointers"
            assert set(paper_payload["questions"][0]) == {
                "number", "start_page", "pages"
            }

            for tool, arguments, code in (
                ("classify_question", {"question_text": "synthetic client input", "how_many": 22}, "invalid_input"),
                ("read_paper", {"pdf_path": str(tmp_path / "missing.pdf")}, "paper_unreadable"),
                ("read_paper", {"pdf_path": str(untrusted)}, "paper_untrusted"),
                ("read_paper", {"pdf_path": str(scanned)}, "ocr_unavailable"),
            ):
                failure = await client.session.call_tool(tool, arguments)
                assert failure.isError
                body = _payload(failure)
                assert body["code"] == code
                assert "message" in body
                assert "synthetic client input" not in json.dumps(body)

            after_failure = await client.session.read_resource(
                "sorted-mathematics://chapters"
            )
            assert len(json.loads(after_failure.contents[0].text)) == 21
            stderr = client.stderr_text()
            assert "synthetic client input" not in stderr
            assert "Synthetic protocol fixture" not in stderr

    asyncio.run(exercise())


def test_stdio_server_reports_an_explicit_refusal(tmp_path):
    fake_model = _support_module("fake_model")
    model = fake_model.write_fake_model(tmp_path / "model.joblib")
    policy = _policy(tmp_path / "policy.json", model, 0.1)

    async def exercise():
        async with ProtocolClient(model, policy) as client:
            result = await client.session.call_tool(
                "classify_question", {"question_text": "synthetic client input"}
            )
            assert not result.isError
            payload = _payload(result)
            assert payload["answered"] is False
            assert payload["refusal"]["code"] == "below_calibrated_threshold"

    asyncio.run(exercise())


def test_stdio_server_keeps_resources_available_without_a_model():
    async def exercise():
        async with ProtocolClient() as client:
            failure = await client.session.call_tool(
                "classify_question", {"question_text": "synthetic client input"}
            )
            assert failure.isError
            payload = _payload(failure)
            assert payload["code"] == "model_unavailable"
            chapters = await client.session.read_resource(
                "sorted-mathematics://chapters"
            )
            assert len(json.loads(chapters.contents[0].text)) == 21

    asyncio.run(exercise())
