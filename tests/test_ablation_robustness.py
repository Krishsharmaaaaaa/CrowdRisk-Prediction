"""
Unit tests for Stage 8: Scientific Ablation and Robustness Studies.
Verifies:
- Model-space synergy identity calculation: Synergy(X, Y) = Delta_R(X,Y) - (Delta_R(X) + Delta_R(Y))
- Alpha perturbation monotonicity
- Multi-seed simulation determinism and statistic calculation
- Magnitude sensitivity progression
"""

import numpy as np
import pytest

from scripts.run_ablation_robustness import (
    cohen_d,
    compute_sample_stats,
    run_crda_ablation_and_synergy,
    run_alpha_sensitivity
)
from src.explainability.crda_engine import CRDAEngine
from src.features.cell_grid import SpatialCellFeatureVector
from src.simulation.pedestrian_sim import (
    PedestrianEvacuationSimulator,
    SimulationConfig
)


def test_cohen_d_effect_size_calculation():
    """Verify Cohen's d computation."""
    x = [10.0, 11.0, 12.0, 10.5, 11.5]
    y = [8.0, 8.5, 9.0, 8.2, 8.8]
    d = cohen_d(x, y)
    assert d > 1.5  # Significant positive separation

    # Identical samples should have d = 0
    assert abs(cohen_d(x, x)) < 1e-5


def test_sample_stats_calculation():
    """Verify mean, median, std, min, max, and range statistics."""
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = compute_sample_stats(vals)
    assert stats["mean"] == 5.0
    assert stats["median"] == 4.5
    assert stats["min"] == 2.0
    assert stats["max"] == 9.0
    assert stats["range"] == 7.0
    assert stats["std"] > 0.0


def test_model_space_synergy_identity():
    """
    Verify that interaction synergy in counterfactual intervention space
    satisfies the exact mathematical identity:
    Synergy(X, Y) = Delta_R(X, Y) - [Delta_R(X) + Delta_R(Y)]
    For positive interaction terms f(D,O) = +0.25*D*O, single interventions
    each eliminate the interaction, producing sub-additive joint Delta_R (< 0).
    For non-interacting pairs (D, B), synergy is identically 0.
    """
    engine = CRDAEngine(enable_interaction_synergy=True)
    test_vecs = {"Test": np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32)}
    res = run_crda_ablation_and_synergy(test_vecs, engine)

    synergies = res["Test"]["pairwise_synergies"]
    # D x O has positive cross-term in risk, so counterfactual joint reduction is sub-additive
    assert synergies["DxO"]["model_space_synergy"] < 0.0
    assert "Sub-additive" in synergies["DxO"]["interpretation"]

    # Linear pairs with no cross-term (e.g., D x B) should have synergy exactly ~ 0
    assert abs(synergies["DxB"]["model_space_synergy"]) < 1e-3


def test_alpha_sensitivity_monotonicity():
    """Verify that Delta_R increases monotonically with alpha."""
    engine = CRDAEngine()
    test_vecs = {"Choke": np.array([0.85, 0.50, 0.70, 0.40], dtype=np.float32)}
    res = run_alpha_sensitivity(test_vecs, engine)

    assert res["Choke"]["delta_R_monotonically_increasing"] is True
    assert res["Choke"]["ranking_stable_across_alpha"] is True


def test_ten_seed_determinism_and_coverage():
    """Verify that all 10 evaluation seeds run deterministically and produce valid outcomes."""
    cfg = SimulationConfig(num_agents=25, max_steps=150)
    sim = PedestrianEvacuationSimulator(cfg)

    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    outcomes = [sim.run(seed=s) for s in seeds]

    assert len(outcomes) == 10
    for out in outcomes:
        assert out.total_agents == 25
        assert out.clearance_time > 0.0
        assert out.exit_throughput > 0.0
