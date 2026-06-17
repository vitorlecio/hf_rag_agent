from langchain_core.tools import BaseTool, tool

from hf_rag.retrieval.base import Retriever, RetrievalResult


def create_tools(retriever: Retriever) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    def search_docs(query: str) -> tuple[str, list[RetrievalResult]]:
        """Search the Hugging Face Transformers documentation for information relevant to the query."""
        results = retriever.retrieve(query)

        if not results:
            return "No relevant documentation found.", []

        sections = []
        for r in results:
            header = f"[{r.page_title}" + (f" / {r.heading}]" if r.heading else "]")
            sections.append(f"{header}\n{r.content}")

        serialized = "\n\n---\n\n".join(sections)
        return serialized, results

    return [search_docs]
