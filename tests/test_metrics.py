import pytest

from hf_rag.eval.metrics import BootstrapCI, bootstrap_ci, hit_rate, mrr, precision_at_k


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"b"}) == pytest.approx(0.5)

    def test_third_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_relevant_result(self) -> None:
        assert mrr(["a", "b", "c"], {"z"}) == pytest.approx(0.0)

    def test_empty_retrieved(self) -> None:
        assert mrr([], {"a"}) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first_rank(self) -> None:
        # "b" is rank 2, "c" is rank 3 — MRR uses the highest-ranked hit
        assert mrr(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit(self) -> None:
        assert hit_rate(["a", "b"], {"b"}) == 1.0

    def test_miss(self) -> None:
        assert hit_rate(["a", "b"], {"z"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert hit_rate([], {"a"}) == 0.0

    def test_all_relevant(self) -> None:
        assert hit_rate(["a", "b"], {"a", "b"}) == 1.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}) == pytest.approx(0.5)

    def test_none_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"z"}) == pytest.approx(0.0)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_dataclass(self) -> None:
        ci = bootstrap_ci([1.0, 0.5, 0.0, 1.0, 0.5])
        assert isinstance(ci, BootstrapCI)

    def test_mean_matches_input(self) -> None:
        scores = [0.2, 0.4, 0.6, 0.8]
        ci = bootstrap_ci(scores)
        assert ci.mean == pytest.approx(0.5)

    def test_lower_leq_mean_leq_upper(self) -> None:
        scores = [float(i) / 10 for i in range(11)]
        ci = bootstrap_ci(scores)
        assert ci.lower <= ci.mean <= ci.upper

    def test_perfect_scores_zero_width(self) -> None:
        ci = bootstrap_ci([1.0] * 20)
        assert ci.lower == pytest.approx(1.0)
        assert ci.upper == pytest.approx(1.0)

    def test_deterministic_with_seed(self) -> None:
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        ci_a = bootstrap_ci(scores)
        ci_b = bootstrap_ci(scores)
        assert ci_a.lower == ci_b.lower
        assert ci_a.upper == ci_b.upper
