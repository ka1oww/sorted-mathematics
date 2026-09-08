"""Stdio MCP transport for the local sorted-mathematics service."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

import mcp_service as service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
mcp = FastMCP("sorted-mathematics")


def _error_result(error: service.ServiceError) -> CallToolResult:
    payload: dict[str, object] = {
        "code": error.code,
        "message": error.safe_message,
    }
    if error.data:
        payload["data"] = error.data
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, sort_keys=True))],
        structuredContent={},
        isError=True,
    )


def _call_service(
    callable_: Callable[..., dict[str, object]], *args: object
) -> dict[str, object] | CallToolResult:
    """Translate only sanitized service failures into a valid MCP tool error."""
    try:
        return callable_(*args)
    except service.ServiceError as error:
        return _error_result(error)
    # The transport boundary must turn every unexpected service failure into a
    # sanitized MCP error instead of leaking local paths or question content.
    except Exception:  # noqa: BLE001
        incident = uuid.uuid4().hex
        LOGGER.error("MCP service failure incident=%s", incident)
        error = service.ServiceError(
            "internal_error", f"Internal service error (incident {incident})."
        )
        return _error_result(error)


def classify_question(question_text: str, how_many: int = 3) -> dict[str, object]:
    return _call_service(service.classify_question_payload, question_text, how_many)


def read_paper(pdf_path: str) -> dict[str, object]:
    return _call_service(service.read_paper_payload, pdf_path)


def list_chapters() -> str:
    return json.dumps(service.list_chapters_payload())


# The service owns the product wording, including the disclosure policy.  Set
# docs before registration because FastMCP reads the handler metadata then.
classify_question.__doc__ = service.CLASSIFY_QUESTION_DOC
read_paper.__doc__ = service.READ_PAPER_DOC
mcp.tool()(classify_question)
mcp.tool()(read_paper)
mcp.resource("sorted-mathematics://chapters", mime_type="application/json")(
    list_chapters
)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
