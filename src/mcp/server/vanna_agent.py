import os
from typing import Optional

from chromadb.utils import embedding_functions
from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import RequestContext, User, UserResolver
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.postgres import PostgresRunner
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SaveTextMemoryTool,
    SearchSavedCorrectToolUsesTool,
)


class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email == "admin@example.com" else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])


def _build_agent() -> Agent:
    llm = OpenAILlmService(
        model=os.getenv("VANNA_LLM_MODEL", "deepseek-chat"),
        api_key=os.environ["VANNA_LLM_API_KEY"],
        base_url=os.getenv("VANNA_LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )

    db_conn_str = os.environ["VANNA_PG_CONN_STR"]
    db_tool = RunSqlTool(sql_runner=PostgresRunner(connection_string=db_conn_str))

    agent_memory = ChromaAgentMemory(
        collection_name=os.getenv("VANNA_MEMORY_COLLECTION", "vanna_memory"),
        persist_directory=os.getenv("VANNA_CHROMA_DIR", "./chroma_db"),
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_base=os.getenv("VANNA_EMBED_BASE_URL"),
            api_key=os.environ["VANNA_EMBED_API_KEY"],
            model_name=os.getenv("VANNA_EMBED_MODEL", "qwen3-emb-0.6b"),
        ),
    )

    tools = ToolRegistry()
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=["admin"])
    tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=["admin", "user"])
    tools.register_local_tool(SaveTextMemoryTool(), access_groups=["admin", "user"])
    tools.register_local_tool(VisualizeDataTool(), access_groups=["admin", "user"])

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=agent_memory,
    )


_agent: Optional[Agent] = None


def get_vanna_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


agent = get_vanna_agent()
