from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from hf_rag.config import GENERATOR_MODEL

_REWRITE_PROMPT = """Given the conversation so far and the latest user question, rewrite the latest \
question into a single, standalone question that can be understood without the conversation \
history. Preserve the user's intent exactly; do not answer the question, just reformulate it. \
If the question is already standalone, return it unchanged, with no extra commentary.

Conversation so far:
{history}

Latest question: {question}

Standalone question:"""


def rewrite_query(
    history: list[BaseMessage], question: str, model: str = GENERATOR_MODEL
) -> str:
    """Condense `question` plus prior turns into a standalone query for retrieval. No-op on the first turn."""
    if not history:
        return question

    transcript = "\n".join(f"{m.type}: {m.content}" for m in history)
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(_REWRITE_PROMPT.format(history=transcript, question=question))
    return response.content.strip()
