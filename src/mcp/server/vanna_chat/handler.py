import uuid
from typing import AsyncGenerator, List

from vanna import Agent

from .models import ChatRequest, ChatResponse, ChatStreamChunk


class ChatHandler:
    def __init__(self, agent: Agent):
        self.agent = agent

    async def handle_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        conversation_id = request.conversation_id or self._generate_conversation_id()
        request_id = request.request_id or str(uuid.uuid4())

        async for component in self.agent.send_message(
            request_context=request.request_context,
            message=request.message,
            conversation_id=conversation_id,
        ):
            yield ChatStreamChunk.from_component(component, conversation_id, request_id)

    async def handle_poll(self, request: ChatRequest) -> ChatResponse:
        chunks: List[ChatStreamChunk] = []
        async for chunk in self.handle_stream(request):
            chunks.append(chunk)
        return ChatResponse.from_chunks(chunks)

    def _generate_conversation_id(self) -> str:
        return f"conv_{uuid.uuid4().hex[:8]}"
