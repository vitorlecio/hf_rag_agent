from hf_rag.agent.query_rewriter import rewrite_query


class TestRewriteQuery:
    def test_returns_question_unchanged_when_no_history(self) -> None:
        assert (
            rewrite_query([], "How do I fine-tune a model?")
            == "How do I fine-tune a model?"
        )
