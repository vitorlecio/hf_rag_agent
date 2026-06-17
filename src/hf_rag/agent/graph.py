import chromadb
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from hf_rag.agent.prompts import SYSTEM_PROMPT
from hf_rag.agent.query_rewriter import rewrite_query
from hf_rag.agent.tools import create_tools
from hf_rag.config import (
    CHROMA_DIR,
    COLLECTION_OPENAI,
    GENERATOR_MODEL,
    make_openai_embedding_fn,
)
from hf_rag.retrieval.base import Retriever
from hf_rag.retrieval.dense import DenseRetriever
from hf_rag.retrieval.reranking import RerankingRetriever


def build_agent(use_reranking: bool = False) -> CompiledStateGraph:
    """use_reranking defaults to False — hf-eval showed dense-only beats dense+rerank
    on this corpus (MRR 0.776 vs 0.715), see README."""
    llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0.1)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_OPENAI,
        metadata={"hnsw:space": "cosine"},
    )
    dense = DenseRetriever(collection, make_openai_embedding_fn())
    retriever: Retriever = RerankingRetriever(dense) if use_reranking else dense
    tools = create_tools(retriever)

    tool_call_tracker = ToolCallLimitMiddleware(
        thread_limit=None, run_limit=2, exit_behavior="continue"
    )
    summarization_middleware = SummarizationMiddleware(
        model=GENERATOR_MODEL,
        trigger=[("tokens", 2000), ("messages", 20)],
        keep=("messages", 20),
    )

    checkpointer = MemorySaver()
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[tool_call_tracker, summarization_middleware],
    )


def ask(agent: CompiledStateGraph, question: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    history = agent.get_state(config).values.get("messages", [])
    standalone_question = rewrite_query(history, question)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": standalone_question}]},
        config=config,
    )
    return result["messages"][-1].content
