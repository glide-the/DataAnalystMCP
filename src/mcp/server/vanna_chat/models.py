import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from vanna.core.user.request_context import RequestContext


class ChatRequest(BaseModel):
    message: str = Field(description="User message")
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatStreamChunk(BaseModel):
    rich: Dict[str, Any]
    simple: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def from_component(
        cls, component: Any, conversation_id: Optional[str], request_id: Optional[str]
    ) -> "ChatStreamChunk":
        rich = _normalize_component(component)
        return cls(
            rich=rich,
            simple=_extract_simple_payload(rich),
            conversation_id=conversation_id,
            request_id=request_id,
        )

    @classmethod
    def from_component_update(
        cls,
        component_update: Any,
        conversation_id: Optional[str],
        request_id: Optional[str],
    ) -> "ChatStreamChunk":
        rich = _normalize_component(component_update)
        return cls(
            rich=rich,
            simple=_extract_simple_payload(rich),
            conversation_id=conversation_id,
            request_id=request_id,
        )


class ChatResponse(BaseModel):
    chunks: List[ChatStreamChunk]
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    total_chunks: int = 0

    @classmethod
    def from_chunks(cls, chunks: List[ChatStreamChunk]) -> "ChatResponse":
        conversation_id = chunks[-1].conversation_id if chunks else None
        request_id = chunks[-1].request_id if chunks else None
        return cls(
            chunks=chunks,
            conversation_id=conversation_id,
            request_id=request_id,
            total_chunks=len(chunks),
        )


def _normalize_component(component: Any) -> Dict[str, Any]:
    if component is None:
        return {}
    if isinstance(component, dict):
        return component
    if hasattr(component, "model_dump"):
        return component.model_dump()
    if hasattr(component, "dict") and callable(getattr(component, "dict")):
        return component.dict()
    if hasattr(component, "__dict__"):
        return dict(component.__dict__)
    return {"value": component}


def _extract_simple_payload(rich: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event_type = rich.get("type")
    if event_type == "text" and rich.get("text"):
        return {"text": rich.get("text")}
    if event_type == "error" and rich.get("error"):
        return {"error": rich.get("error")}
    if event_type == "sql" and rich.get("query"):
        return {"sql": rich.get("query")}
    if event_type == "link" and rich.get("url"):
        return {"title": rich.get("title"), "url": rich.get("url")}
    if event_type == "image" and rich.get("image_url"):
        return {"image_url": rich.get("image_url"), "caption": rich.get("caption")}
    if event_type == "dataframe" and rich.get("json_table"):
        return {"json_table": rich.get("json_table")}
    if event_type == "plotly" and rich.get("json_plotly"):
        return {"json_plotly": rich.get("json_plotly")}
    if event_type == "buttons" and rich.get("buttons"):
        return {"text": rich.get("text"), "buttons": rich.get("buttons")}
    return None
