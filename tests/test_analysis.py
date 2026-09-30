import numpy as np
import pandas as pd

from gateway_bench import analysis as A
from gateway_bench.pricing import PriceBook
from gateway_bench.scoring import extract_choice, extract_final_number, parse_judge, score_mc, score_numeric


def test_bootstrap_ci_contains_median():
    x = np.arange(1, 101)
    est, lo, hi = A.bootstrap_ci(x, n=2000)
    assert lo <= est <= hi
    assert abs(est - 50.5) < 1e-9


def test_holm_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    adj = A.holm(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert adj[0] == 0.04  # 4 * 0.01


def test_mcnemar_identical_is_one():
    s = pd.Series([True, False, True])
    assert A.mcnemar_exact(s, s) == 1.0


def test_pareto_mask():
    cost = [1, 2, 3, 2.5]
    qual = [0.5, 0.8, 0.9, 0.7]
    assert list(A.pareto_mask(cost, qual)) == [True, True, True, False]


def test_aiq_flat_line():
    # one point at the cheapest cost with quality 0.8 -> area 0.8 over the range
    assert abs(A.aiq([(1.0, 0.8)], 1.0, 3.0) - 0.8) < 1e-6


def test_pricebook_prefers_reported_and_longest_key():
    pb = PriceBook({"gpt-4.1": {"input": 2, "output": 8}, "gpt-4.1-mini": {"input": 0.4, "output": 1.6}},
                   {"gw": 0.1})
    assert pb.cost({"cost_reported": 0.5})[0] == 0.5
    c, src = pb.cost({"system": "gw", "model_served": "openai/gpt-4.1-mini", "input_tokens": 1_000_000, "output_tokens": 0})
    assert src == "list_price" and abs(c - 0.44) < 1e-9


def test_scorers():
    assert extract_final_number("so the total is 1,234. #### 1,234") == 1234
    assert score_numeric("The answer is 42", "42")
    assert extract_choice("Answer: C") == "C"
    assert score_mc("I think (B) is right.\nAnswer: B", "B")
    assert parse_judge("INCORRECT") is False and parse_judge("correct") is True
