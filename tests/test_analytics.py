from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from analytics import (matrix_diagnostics, numeric_profile, optimize_allocation,
                       principal_components, pure_nash_equilibria, zero_sum_equilibrium)


def test_pca_matches_independent_sklearn_and_reconstructs():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(30, 4)))
    frame[3] = frame[0] * 2 + frame[1] * .3
    result = principal_components(frame)
    reference = PCA().fit(result["standardized"])
    np.testing.assert_allclose(result["variance_ratio"], reference.explained_variance_ratio_, atol=1e-12)
    reconstructed = result["scores"].to_numpy() @ result["components"].to_numpy().T
    np.testing.assert_allclose(reconstructed, result["standardized"], atol=1e-12)
    assert result["rank"] == 3
    assert np.max(np.abs(result["loadings"])) <= 1 + 1e-12


def test_pca_reports_missing_and_removes_constant_columns():
    frame = pd.DataFrame({"a": [1, 2, 4, np.inf], "b": [4, 2, 3, 1], "constant": [1] * 4})
    result = principal_components(frame)
    assert result["dropped_rows"] == 1
    assert result["constant_columns"] == ["constant"]
    assert list(result["scores"].index) == [0, 1, 2]


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"a": [1, 1, 1], "b": [0, 0, 0]}),
                                   pd.DataFrame({"a": [1, 2], "b": [3, 4]})])
def test_pca_rejects_insufficient_information(frame):
    with pytest.raises(ValueError):
        principal_components(frame)


def test_profile_does_not_treat_invalid_values_as_zero():
    numeric, summary = numeric_profile(pd.DataFrame({"x": [2, "bad", np.inf, None]}))
    assert numeric.x.isna().sum() == 3
    assert summary.loc["x", "mean"] == 2


def test_svd_known_diagonal_and_zero_matrix():
    result = matrix_diagnostics(np.diag([4, 3, 0]), 1)
    assert result["rank"] == 2 and result["nullity"] == 1
    assert np.isinf(result["condition"])
    assert result["relative_error"] == pytest.approx(.6)
    np.testing.assert_allclose(result["energy"], [.64, 1, 1])
    assert matrix_diagnostics(np.zeros((3, 2)))["relative_error"] == 0


@pytest.mark.parametrize("matrix", [[], [[np.nan]], [[np.inf]], [1, 2, 3]])
def test_nonfinite_or_invalid_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        matrix_diagnostics(matrix)
    with pytest.raises(ValueError):
        zero_sum_equilibrium(matrix)


def candidates():
    return pd.DataFrame([
        ("a", "p", "ALLOW", 9, 4), ("a", "q", "ALLOW", 8, 3),
        ("b", "p", "ALLOW", 7, 4), ("b", "q", "RESTRICTED", 1000, 1),
        ("c", "p", "FORBID", 1000, 1), ("c", "q", "ALLOW", 6, 2),
        ("d", "q", "UNKNOWN", 1000, 1),
    ], columns=["customer_id", "product_id", "suitability_level", "utility", "cost"])


@pytest.mark.parametrize("budget,limit", [(0, 3), (5, 3), (7, 2), (50, 1), (50, 0)])
def test_allocation_matches_exhaustive_search_and_enforces_constraints(budget, limit):
    frame = candidates()
    capacities = {"p": 1, "q": 1}
    result = optimize_allocation(frame, budget, limit, capacities)
    best = 0
    for bits in product([0, 1], repeat=len(frame)):
        chosen = frame.loc[np.array(bits, dtype=bool)]
        if (len(chosen) <= limit and chosen.cost.sum() <= budget
                and chosen.suitability_level.eq("ALLOW").all()
                and not chosen.customer_id.duplicated().any()
                and all(chosen.product_id.value_counts().get(p, 0) <= cap for p, cap in capacities.items())):
            best = max(best, chosen.utility.sum())
    assert result.objective == best
    assert result.selected.suitability_level.eq("ALLOW").all()
    assert result.cost <= budget


def test_allocation_rejects_bad_capacity_cost_and_duplicate_pairs():
    frame = candidates()
    for invalid in [pd.concat([frame, frame.iloc[:1]]), frame.assign(cost=-1), frame.assign(utility=np.nan)]:
        with pytest.raises(ValueError):
            optimize_allocation(invalid, 10, 2, {"p": 1, "q": 1})
    for capacity in [-1, .5, np.nan]:
        with pytest.raises(ValueError):
            optimize_allocation(frame, 10, 2, {"p": capacity, "q": 1})
    with pytest.raises(ValueError):
        optimize_allocation(frame, 10, 2, {"p": 1})


@pytest.mark.parametrize("a,probability,value", [([[1, -1], [-1, 1]], .5, 0),
        ([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], 1/3, 0)])
def test_known_mixed_equilibria(a, probability, value):
    result = zero_sum_equilibrium(a)
    np.testing.assert_allclose(result["row_strategy"], probability, atol=1e-9)
    np.testing.assert_allclose(result["column_strategy"], probability, atol=1e-9)
    assert result["value"] == pytest.approx(value, abs=1e-9)
    assert result["duality_gap"] < 1e-8


def test_negative_rectangular_game_and_saddle_point():
    result = zero_sum_equilibrium([[-3, -1, -4], [-4, -2, -5]])
    assert result["value"] == pytest.approx(-4)
    assert result["row_strategy"][0] == pytest.approx(1)
    assert result["column_strategy"][2] == pytest.approx(1)


def test_pure_nash_includes_ties_and_distinguishes_no_pure_equilibrium():
    assert pure_nash_equilibria([[3, 0], [5, 1]], [[3, 5], [0, 1]]) == [(1, 1)]
    assert len(pure_nash_equilibria(np.zeros((2, 2)), np.zeros((2, 2)))) == 4
    a = np.array([[1, -1], [-1, 1]])
    assert pure_nash_equilibria(a, -a) == []
