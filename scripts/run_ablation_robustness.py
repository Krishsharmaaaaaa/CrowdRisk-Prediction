"""
Stage 8: Scientific Ablation & Robustness Study Runner.
Strictly evaluates the stability and boundary conditions of CRDA and simulation outcomes.

Investigates:
A. CRDA driver ablation (D, O, B, K, pairs, triplets, full set).
B. Model-space interaction/synergy analysis.
C. Counterfactual alpha sensitivity (alpha in [0.25, 0.50, 0.75, 1.00]).
D. Simulation robustness across 10 independent seeds (original 5 + 5 new).
E. Intervention magnitude sensitivity (mild, medium, strong).
F. Critical re-evaluation of Stage 7 minimal subset claim.
G. Top-1 agreement stability across 10 seeds.
H. Throughput metric audit (raw vs normalized throughput vs clearance time).
"""

from dataclasses import asdict
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability.crda_engine import CRDAEngine, DRIVER_KEYS, DRIVER_NAMES
from src.features.cell_grid import SpatialCellFeatureVector
from src.simulation.pedestrian_sim import (
    PedestrianEvacuationSimulator,
    SimulationConfig,
    SimulationOutcome
)


# 10 Independent Seeds (5 original + 5 new)
SEEDS_ORIGINAL = [42, 101, 202, 303, 404]
SEEDS_ADDITIONAL = [505, 606, 707, 808, 909]
SEEDS_ALL_10 = SEEDS_ORIGINAL + SEEDS_ADDITIONAL

CALM_REFERENCE = np.array([0.10, 0.05, 0.05, 0.05], dtype=np.float32)


def cohen_d(x: List[float], y: List[float]) -> float:
    """Calculates Cohen's d effect size between two independent samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd < 1e-6:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def compute_sample_stats(vals: List[float]) -> Dict[str, float]:
    """Computes comprehensive descriptive statistics."""
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), 3),
        "median": round(float(np.median(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "range": round(float(np.max(arr) - np.min(arr)), 3)
    }


def run_replicates(config: SimulationConfig, seeds: List[int], name: str) -> List[SimulationOutcome]:
    """Runs simulation across multiple random seeds."""
    outcomes = []
    for s in seeds:
        sim = PedestrianEvacuationSimulator(config)
        out = sim.run(seed=s)
        out.scenario_name = name
        outcomes.append(out)
    return outcomes


# =====================================================================
# PART A: CRDA ABLATION & PART B: INTERACTION SYNERGY
# =====================================================================
def run_crda_ablation_and_synergy(
    test_vectors: Dict[str, np.ndarray],
    crda_engine: CRDAEngine
) -> Dict[str, Any]:
    """
    Evaluates driver ablations:
    Single: D, O, B, K
    Pairs: D+O, D+B, O+B, D+K, B+K, O+K
    Triplets: D+O+B
    Quad: D+O+B+K
    Calculates exact model-space synergy: Synergy(X, Y) = Delta_R(X, Y) - [Delta_R(X) + Delta_R(Y)]
    """
    ablation_results = {}

    subsets_to_test = [
        ("D", ["D"]),
        ("O", ["O"]),
        ("B", ["B"]),
        ("K", ["K"]),
        ("D+O", ["D", "O"]),
        ("D+B", ["D", "B"]),
        ("O+B", ["O", "B"]),
        ("D+K", ["D", "K"]),
        ("B+K", ["B", "K"]),
        ("O+K", ["O", "K"]),
        ("D+O+B", ["D", "O", "B"]),
        ("D+O+B+K", ["D", "O", "B", "K"]),
    ]

    key_idx = {"D": 0, "O": 1, "B": 2, "K": 3}

    for vec_name, x_vec in test_vectors.items():
        initial_risk = crda_engine.compute_cell_risk(x_vec)
        threat_level = crda_engine._classify_threat_level(initial_risk)

        # Baseline linear comparison (frozen linear model without interactions)
        base_linear_risk = 0.35 * x_vec[0] + 0.20 * x_vec[1] + 0.30 * x_vec[2] + 0.15 * x_vec[3]

        deltas = {}
        cf_risks = {}
        safe_reached = {}

        for sub_name, keys in subsets_to_test:
            x_cf = x_vec.copy()
            for k in keys:
                idx = key_idx[k]
                x_cf[idx] = CALM_REFERENCE[idx]
            cf_r = crda_engine.compute_cell_risk(x_cf)
            dR = max(0.0, initial_risk - cf_r)
            cf_risks[sub_name] = round(float(cf_r), 4)
            deltas[sub_name] = round(float(dR), 4)
            safe_reached[sub_name] = bool(cf_r < crda_engine.high_risk_threshold)

        # Model-space Synergy Analysis
        # Synergy(X, Y) = Delta_R(X, Y) - [Delta_R(X) + Delta_R(Y)]
        pairs = [("D", "O"), ("D", "B"), ("O", "B"), ("D", "K"), ("B", "K"), ("O", "K")]
        synergy_dict = {}
        for x, y in pairs:
            pair_key = f"{x}+{y}"
            dr_pair = deltas[pair_key]
            dr_sum = deltas[x] + deltas[y]
            syn = dr_pair - dr_sum
            synergy_dict[f"{x}x{y}"] = {
                "delta_R_joint": round(float(dr_pair), 4),
                "delta_R_sum_singles": round(float(dr_sum), 4),
                "model_space_synergy": round(float(syn), 4),
                "interpretation": "Sub-additive (shared interaction eliminated by either single driver)" if syn < -1e-4 else (
                    "Super-additive" if syn > 1e-4 else "Additive (zero interaction)"
                )
            }

        # CRDA Full Explanation
        mock_cell = SpatialCellFeatureVector(
            cell_id=vec_name, row=0, col=0, x1=0, y1=0, x2=10, y2=10, area_px=100,
            D=float(x_vec[0]), O=float(x_vec[1]), B=float(x_vec[2]), K=float(x_vec[3])
        )
        expl = crda_engine.explain_cell(mock_cell, CALM_REFERENCE)

        ablation_results[vec_name] = {
            "feature_vector": [round(float(v), 3) for v in x_vec],
            "initial_risk": round(float(initial_risk), 4),
            "initial_threat_level": threat_level,
            "frozen_linear_risk": round(float(base_linear_risk), 4),
            "top_driver": expl.top_driver,
            "minimal_intervention_subset": expl.minimal_intervention_subset.subset_names,
            "minimal_intervention_cardinality": expl.minimal_intervention_subset.cardinality,
            "subsets_evaluated": {
                name: {
                    "counterfactual_risk": cf_risks[name],
                    "delta_R": deltas[name],
                    "safe_reached": safe_reached[name]
                }
                for name, _ in subsets_to_test
            },
            "pairwise_synergies": synergy_dict
        }

    return ablation_results


# =====================================================================
# PART C: COUNTERFACTUAL ALPHA SENSITIVITY
# =====================================================================
def run_alpha_sensitivity(
    test_vectors: Dict[str, np.ndarray],
    crda_engine: CRDAEngine
) -> Dict[str, Any]:
    """
    Evaluates CRDA behavior across alpha in [0.25, 0.50, 0.75, 1.00].
    Audits rank stability, minimal subset stability, and delta_R monotonicity.
    """
    alphas = [0.25, 0.50, 0.75, 1.00]
    sensitivity_results = {}

    for vec_name, x_vec in test_vectors.items():
        mock_cell = SpatialCellFeatureVector(
            cell_id=vec_name, row=0, col=0, x1=0, y1=0, x2=10, y2=10, area_px=100,
            D=float(x_vec[0]), O=float(x_vec[1]), B=float(x_vec[2]), K=float(x_vec[3])
        )

        alpha_runs = {}
        rankings = []
        minimal_subsets = []
        top_driver_deltas = []

        for a in alphas:
            expl = crda_engine.explain_cell(mock_cell, CALM_REFERENCE, alpha=a)
            driver_ranks = [attr.driver_key for attr in expl.single_driver_attributions]
            min_sub = expl.minimal_intervention_subset.subset_names

            alpha_runs[f"alpha_{a:.2f}"] = {
                "top_driver": expl.top_driver,
                "top_driver_delta_R": round(float(expl.top_driver_delta_R), 4),
                "driver_ranking": driver_ranks,
                "minimal_subset": min_sub,
                "minimal_subset_risk": round(float(expl.minimal_intervention_subset.counterfactual_risk), 4),
                "achieved_safe": expl.minimal_intervention_subset.achieved_safe_threshold
            }
            rankings.append(tuple(driver_ranks))
            minimal_subsets.append(tuple(min_sub))
            top_driver_deltas.append(expl.top_driver_delta_R)

        # Audit properties
        ranking_stable = (len(set(rankings)) == 1)
        minimal_subset_stable = (len(set(minimal_subsets)) == 1)
        # Check monotonicity of top driver delta_R with respect to alpha
        monotonic_delta_R = all(
            top_driver_deltas[i] <= top_driver_deltas[i + 1] + 1e-5
            for i in range(len(top_driver_deltas) - 1)
        )

        sensitivity_results[vec_name] = {
            "alpha_runs": alpha_runs,
            "ranking_stable_across_alpha": ranking_stable,
            "minimal_subset_stable_across_alpha": minimal_subset_stable,
            "delta_R_monotonically_increasing": monotonic_delta_R
        }

    return sensitivity_results


# =====================================================================
# PART D, F, G, H: SIMULATION ROBUSTNESS & AUDIT ACROSS 10 SEEDS
# =====================================================================
def run_simulation_robustness_10_seeds(crda_engine: CRDAEngine) -> Dict[str, Any]:
    """
    Executes full 10-seed simulation robustness evaluation across all 5 scenarios.
    Audits:
    - Statistical stability (mean +/- SD, median, range, effect sizes)
    - Re-evaluates Stage 7 minimal subset claim
    - Evaluates Top-1 agreement under 10 seeds
    - Audits Throughput vs Normalized Throughput vs Clearance Time
    """
    from scripts.run_simulation_validation import build_scenario_definitions
    scenarios = build_scenario_definitions()
    key_map = {"Density (D)": "D", "Bottleneck (B)": "B", "Kinematics (K)": "K", "Disorder (O)": "O"}

    results_10_seeds = {}

    for sc_id, sc_data in scenarios.items():
        base_cfg = sc_data["baseline"]
        total_baseline_agents = base_cfg.num_agents

        # 1. Baseline across 10 seeds
        base_outcomes = run_replicates(base_cfg, SEEDS_ALL_10, f"{sc_id}_baseline")
        base_clr = [o.clearance_time for o in base_outcomes]
        base_thr = [o.exit_throughput for o in base_outcomes]
        base_cong = [o.congestion_duration for o in base_outcomes]
        base_coll = [o.collision_count for o in base_outcomes]
        base_rem = [o.remaining_count for o in base_outcomes]

        # Normalized throughput: agents evacuated per second divided by initial baseline population fraction
        # base_norm_thr = base_thr (reference)

        # Choke vector sampling
        choke_vecs = []
        for s in SEEDS_ALL_10:
            sim = PedestrianEvacuationSimulator(base_cfg)
            sim.run(seed=s)
            choke_vecs.append(sim.get_congested_choke_driver_vector())
        mean_choke = np.mean(choke_vecs, axis=0)

        # CRDA explanation
        mock_cell = SpatialCellFeatureVector(
            cell_id="CHOKE", row=0, col=0, x1=0, y1=0, x2=10, y2=10, area_px=100,
            D=float(mean_choke[0]), O=float(mean_choke[1]), B=float(mean_choke[2]), K=float(mean_choke[3])
        )
        crda_expl = crda_engine.explain_cell(mock_cell, CALM_REFERENCE)
        crda_deltas = {attr.driver_key: attr.delta_R for attr in crda_expl.single_driver_attributions}
        crda_top_driver = crda_expl.top_driver
        crda_min_subset = crda_expl.minimal_intervention_subset.subset_names

        # 2. Interventions across 10 seeds
        int_evaluations = {}
        for int_name, int_cfg in sc_data["interventions"].items():
            int_outcomes = run_replicates(int_cfg, SEEDS_ALL_10, f"{sc_id}_{int_name}")
            int_clr = [o.clearance_time for o in int_outcomes]
            int_thr = [o.exit_throughput for o in int_outcomes]
            int_cong = [o.congestion_duration for o in int_outcomes]
            int_coll = [o.collision_count for o in int_outcomes]
            int_rem = [o.remaining_count for o in int_outcomes]

            # Throughput Metric Audit:
            # Raw throughput: evacuated / clearance_time
            # Normalized throughput: normalized to baseline population so population-reduction is not confounded:
            # normalized_flow = (evacuated / baseline_population) / clearance_time * baseline_population = evacuated / clearance_time
            # BUT per-capita egress efficiency: 1 / clearance_time
            # If population was reduced from 70 to 45, raw throughput drops simply because N=45.
            # Fixed-population efficiency: clearance_time reduction ratio:
            speedup_ratio = np.mean(base_clr) / np.mean(int_clr) if np.mean(int_clr) > 0 else 1.0

            # Paired improvements (positive = improvement)
            delta_clr = [b - i for b, i in zip(base_clr, int_clr)]
            delta_thr = [i - b for b, i in zip(base_thr, int_thr)]
            delta_cong = [b - i for b, i in zip(base_cong, int_cong)]

            d_clr = cohen_d(base_clr, int_clr)
            d_thr = cohen_d(int_thr, base_thr)

            int_evaluations[int_name] = {
                "clearance_time_stats": compute_sample_stats(int_clr),
                "delta_clearance_stats": compute_sample_stats(delta_clr),
                "cohen_d_clearance": round(float(d_clr), 2),
                "exit_throughput_stats": compute_sample_stats(int_thr),
                "delta_throughput_stats": compute_sample_stats(delta_thr),
                "cohen_d_throughput": round(float(d_thr), 2),
                "speedup_ratio": round(float(speedup_ratio), 3),
                "congestion_duration_stats": compute_sample_stats(int_cong),
                "delta_congestion_stats": compute_sample_stats(delta_cong),
                "remaining_agents_stats": compute_sample_stats(int_rem),
                "collision_count_stats": compute_sample_stats(int_coll),
                "population_change_confound": int_cfg.num_agents != total_baseline_agents,
                "agent_count": int_cfg.num_agents
            }

        # 3. Rankings under 10 seeds
        single_names = [k for k in sc_data["interventions"].keys() if k in key_map]
        crda_ranked = sorted(single_names, key=lambda k: crda_deltas.get(key_map[k], 0.0), reverse=True)
        sim_clr_ranked = sorted(single_names, key=lambda k: int_evaluations[k]["delta_clearance_stats"]["mean"], reverse=True)
        sim_thr_ranked = sorted(single_names, key=lambda k: int_evaluations[k]["delta_throughput_stats"]["mean"], reverse=True)

        top1_clr_match = (crda_ranked[0] == sim_clr_ranked[0])
        top1_thr_match = (crda_ranked[0] == sim_thr_ranked[0])

        # 4. Critical Re-evaluation of Stage 7 Minimal Subset Claim:
        # Check whether multi-intervention actually achieves highest clearance improvement among ALL interventions
        all_int_names = list(sc_data["interventions"].keys())
        all_clr_ranked = sorted(all_int_names, key=lambda k: int_evaluations[k]["delta_clearance_stats"]["mean"], reverse=True)
        top_overall_int = all_clr_ranked[0]
        multi_names = [k for k in all_int_names if "Multi" in k]
        best_multi = sorted(multi_names, key=lambda k: int_evaluations[k]["delta_clearance_stats"]["mean"], reverse=True)[0] if multi_names else None

        minimal_subset_claim_verified = (best_multi == top_overall_int)
        minimal_subset_tied = (
            abs(int_evaluations[best_multi]["delta_clearance_stats"]["mean"] - int_evaluations[top_overall_int]["delta_clearance_stats"]["mean"]) < 0.10
        ) if best_multi else False

        results_10_seeds[sc_id] = {
            "scenario_name": sc_data["name"],
            "choke_features_10_seeds": [round(float(v), 3) for v in mean_choke],
            "crda_top_driver": crda_top_driver,
            "crda_minimal_subset": crda_min_subset,
            "crda_ranked_drivers": crda_ranked,
            "sim_clearance_ranked": sim_clr_ranked,
            "sim_throughput_ranked": sim_thr_ranked,
            "top1_clearance_match_10_seeds": top1_clr_match,
            "top1_throughput_match_10_seeds": top1_thr_match,
            "baseline_stats": {
                "clearance_time": compute_sample_stats(base_clr),
                "exit_throughput": compute_sample_stats(base_thr),
                "congestion_duration": compute_sample_stats(base_cong)
            },
            "interventions": int_evaluations,
            "stage_7_minimal_subset_re_evaluation": {
                "top_overall_intervention": top_overall_int,
                "best_multi_intervention": best_multi,
                "best_multi_delta_clr": int_evaluations[best_multi]["delta_clearance_stats"]["mean"] if best_multi else None,
                "best_single_delta_clr": int_evaluations[sim_clr_ranked[0]]["delta_clearance_stats"]["mean"],
                "minimal_subset_is_strictly_best": minimal_subset_claim_verified,
                "minimal_subset_tied_with_best": minimal_subset_tied,
                "claim_status": "CONFIRMED" if minimal_subset_claim_verified else (
                    "TIED / COMPARABLE" if minimal_subset_tied else "DISCONFIRMED (Single outperformed Multi)"
                )
            }
        }

    return results_10_seeds


# =====================================================================
# PART E: INTERVENTION MAGNITUDE SENSITIVITY
# =====================================================================
def run_magnitude_sensitivity() -> Dict[str, Any]:
    """
    Evaluates 3 intervention magnitudes (mild, medium, strong) for each physical driver.
    Tests whether the direction and relative behavior remain monotonic.
    """
    seeds = SEEDS_ALL_10  # 10 seeds for statistical stability

    magnitude_defs = {
        "Density": {
            "baseline": SimulationConfig(door_width=0.8, num_agents=80, max_steps=1200),
            "variants": {
                "Mild (N=65)": SimulationConfig(door_width=0.8, num_agents=65, max_steps=1200),
                "Medium (N=50)": SimulationConfig(door_width=0.8, num_agents=50, max_steps=1200),
                "Strong (N=35)": SimulationConfig(door_width=0.8, num_agents=35, max_steps=1200),
            }
        },
        "Bottleneck": {
            "baseline": SimulationConfig(door_width=0.6, num_agents=70, max_steps=1200),
            "variants": {
                "Mild (w=0.9m)": SimulationConfig(door_width=0.9, num_agents=70, max_steps=1200),
                "Medium (w=1.4m)": SimulationConfig(door_width=1.4, num_agents=70, max_steps=1200),
                "Strong (w=2.0m)": SimulationConfig(door_width=2.0, num_agents=70, max_steps=1200),
            }
        },
        "Disorder": {
            "baseline": SimulationConfig(door_width=1.0, num_agents=65, noise_strength=0.45, max_steps=1200),
            "variants": {
                "Mild (sigma=0.30)": SimulationConfig(door_width=1.0, num_agents=65, noise_strength=0.30, max_steps=1200),
                "Medium (sigma=0.15)": SimulationConfig(door_width=1.0, num_agents=65, noise_strength=0.15, max_steps=1200),
                "Strong (sigma=0.02)": SimulationConfig(door_width=1.0, num_agents=65, noise_strength=0.02, max_steps=1200),
            }
        },
        "Kinematics": {
            "baseline": SimulationConfig(door_width=1.0, num_agents=65, desired_speed_mean=2.8, desired_speed_std=0.9, max_steps=1200),
            "variants": {
                "Mild (mu=2.0, s=0.5)": SimulationConfig(door_width=1.0, num_agents=65, desired_speed_mean=2.0, desired_speed_std=0.5, max_steps=1200),
                "Medium (mu=1.5, s=0.3)": SimulationConfig(door_width=1.0, num_agents=65, desired_speed_mean=1.5, desired_speed_std=0.3, max_steps=1200),
                "Strong (mu=1.2, s=0.1)": SimulationConfig(door_width=1.0, num_agents=65, desired_speed_mean=1.2, desired_speed_std=0.1, max_steps=1200),
            }
        }
    }

    magnitude_results = {}

    for driver_name, def_data in magnitude_defs.items():
        base_runs = run_replicates(def_data["baseline"], seeds, f"{driver_name}_base")
        base_clrs = [o.clearance_time for o in base_runs]
        base_mean_clr = float(np.mean(base_clrs))

        var_stats = {}
        clr_means = []

        for var_name, var_cfg in def_data["variants"].items():
            var_runs = run_replicates(var_cfg, seeds, f"{driver_name}_{var_name}")
            var_clrs = [o.clearance_time for o in var_runs]
            delta_clrs = [b - v for b, v in zip(base_clrs, var_clrs)]
            m_clr = float(np.mean(var_clrs))
            clr_means.append(m_clr)

            var_stats[var_name] = {
                "clearance_time": compute_sample_stats(var_clrs),
                "delta_clearance": compute_sample_stats(delta_clrs),
                "cohen_d_clearance": round(float(cohen_d(base_clrs, var_clrs)), 2)
            }

        # Monotonicity check (does clearance time decrease monotonically as intervention strengthens?)
        is_monotonic = all(clr_means[i] >= clr_means[i + 1] - 0.20 for i in range(len(clr_means) - 1))

        magnitude_results[driver_name] = {
            "baseline_mean_clearance": round(base_mean_clr, 2),
            "variants": var_stats,
            "monotonic_clearance_reduction": is_monotonic,
            "trend": "Monotonically improves egress" if is_monotonic else "Non-monotonic / saturated"
        }

    return magnitude_results


# =====================================================================
# PLOTTING
# =====================================================================
def generate_ablation_robustness_plots(
    sim_10_seeds: Dict[str, Any],
    magnitude_res: Dict[str, Any],
    alpha_res: Dict[str, Any]
):
    """Generates a 4-panel publication-grade robustness figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Stage 8: Comprehensive Scientific Ablation & Robustness Studies\n"
        "(Multi-Seed Stability, Alpha Sensitivity, Magnitude Scaling, and Stage 7 Audit)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Panel 1: Stage 7 Re-evaluation: Best Single vs Best Multi across 10 Seeds
    ax1 = axes[0, 0]
    sc_names = [v["scenario_name"].replace(" Dominant", "").replace(" Multi-Driver", "") for v in sim_10_seeds.values()]
    best_singles = [v["stage_7_minimal_subset_re_evaluation"]["best_single_delta_clr"] for v in sim_10_seeds.values()]
    best_multis = [v["stage_7_minimal_subset_re_evaluation"]["best_multi_delta_clr"] for v in sim_10_seeds.values()]

    x = np.arange(len(sc_names))
    width = 0.35
    ax1.bar(x - width/2, best_singles, width, label="Best Single Driver Intervention", color="#3b82f6", alpha=0.85, edgecolor="#1e293b")
    ax1.bar(x + width/2, best_multis, width, label="CRDA Minimal Multi-Intervention", color="#10b981", alpha=0.85, edgecolor="#1e293b")
    ax1.set_ylabel("Clearance Reduction ΔT_clr (s)", fontweight="bold")
    ax1.set_title("A. Audit of Minimal Subset Claim (10 Seeds):\nBest Single vs Multi-Driver Intervention", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=9, rotation=15)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Intervention Magnitude Scaling (Clearance Improvement)
    ax2 = axes[0, 1]
    drivers = list(magnitude_res.keys())
    for d_name in drivers:
        d_data = magnitude_res[d_name]
        var_labels = list(d_data["variants"].keys())
        d_clrs = [d_data["variants"][v]["delta_clearance"]["mean"] for v in var_labels]
        # x-axis: 1, 2, 3 (Mild, Medium, Strong)
        ax2.plot([1, 2, 3], d_clrs, marker="o", linewidth=2.2, label=f"{d_name} Intervention")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Mild", "Medium", "Strong"], fontweight="bold")
    ax2.set_xlabel("Intervention Magnitude", fontweight="bold")
    ax2.set_ylabel("Clearance Reduction ΔT_clr (s)", fontweight="bold")
    ax2.set_title("B. Physical Intervention Magnitude Sensitivity:\nEgress Scaling Across Magnitudes (10 Seeds)", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Counterfactual Alpha Monotonicity
    ax3 = axes[1, 0]
    alphas = [0.25, 0.50, 0.75, 1.00]
    for vec_name, a_data in alpha_res.items():
        runs = a_data["alpha_runs"]
        deltas = [runs[f"alpha_{a:.2f}"]["top_driver_delta_R"] for a in alphas]
        ax3.plot(alphas, deltas, marker="s", linewidth=2.0, label=f"Choke {vec_name}")
    ax3.set_xlabel("Counterfactual Perturbation Strength (α)", fontweight="bold")
    ax3.set_ylabel("Top-1 Driver Risk Reduction ΔR", fontweight="bold")
    ax3.set_title("C. Counterfactual Alpha Sensitivity:\nStrict Monotonicity & Smooth Scaling", fontweight="bold", fontsize=11)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Throughput Metric Audit (Raw Flow vs Speedup Ratio)
    ax4 = axes[1, 1]
    # For Scenario 1 (Bottleneck) and Scenario 2 (Density)
    int_labels = ["Density (D)", "Bottleneck (B)", "Kinematics (K)", "Disorder (O)", "Multi (B+D)"]
    s1_data = sim_10_seeds["Scenario_1_Bottleneck"]["interventions"]
    d_thr = [s1_data[k]["delta_throughput_stats"]["mean"] for k in s1_data.keys() if k in int_labels or "Multi" in k][:5]
    speedups = [s1_data[k]["speedup_ratio"] for k in s1_data.keys() if k in int_labels or "Multi" in k][:5]
    x_sub = np.arange(len(d_thr))
    ax4.bar(x_sub - 0.18, d_thr, 0.35, label="Raw Throughput Change ΔΦ (agents/s)", color="#ef4444", alpha=0.85)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_sub + 0.18, speedups, color="#6366f1", marker="D", linewidth=2.0, label="Speedup Ratio (T_base / T_int)")
    ax4.set_ylabel("Raw Throughput Change ΔΦ (/s)", color="#dc2626", fontweight="bold")
    ax4_twin.set_ylabel("Clearance Speedup Ratio", color="#4f46e5", fontweight="bold")
    ax4.set_xticks(x_sub)
    ax4.set_xticklabels(["Density\n(confounded)", "Bottleneck", "Kinematics", "Disorder", "Multi\n(B+D)"], fontsize=8)
    ax4.set_title("D. Throughput Metric Audit (Scenario 1):\nPopulation Confound on Raw Throughput vs Egress Speedup", fontweight="bold", fontsize=11)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = "results/ablation_robustness.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Robustness visualization saved to {plot_path}")


# =====================================================================
# MAIN RUNNER
# =====================================================================
def run_stage_8_ablation_robustness():
    print("=" * 90)
    print("STAGE 8: SCIENTIFIC ABLATION & ROBUSTNESS STUDIES")
    print("Systematic Stability, Alpha Sensitivity, Multi-Seed & Magnitude Scaling Audit")
    print("=" * 90)

    crda_engine = CRDAEngine(moderate_risk_threshold=0.30, high_risk_threshold=0.60)

    # 1. Representative Choke Vectors for Ablation and Alpha Studies
    test_vectors = {
        "Bottleneck_Choke": np.array([0.86, 0.24, 0.20, 0.28], dtype=np.float32),
        "Density_Choke": np.array([0.90, 0.24, 0.16, 0.35], dtype=np.float32),
        "Disorder_Choke": np.array([0.86, 0.23, 0.18, 0.29], dtype=np.float32),
        "Kinematics_Choke": np.array([0.82, 0.18, 0.15, 0.36], dtype=np.float32),
        "Compound_Choke": np.array([0.87, 0.28, 0.21, 0.31], dtype=np.float32)
    }

    # PART A & B: Ablation and Interaction Synergy
    print("\n[1/5] Running CRDA Driver Ablation & Pairwise Synergy Analysis...")
    ablation_res = run_crda_ablation_and_synergy(test_vectors, crda_engine)

    # PART C: Alpha Sensitivity
    print("\n[2/5] Running Counterfactual Alpha Sensitivity (alpha in [0.25, 0.50, 0.75, 1.00])...")
    alpha_res = run_alpha_sensitivity(test_vectors, crda_engine)

    # PART D, F, G, H: 10-Seed Simulation Robustness & Stage 7 Claim Audit
    print(f"\n[3/5] Running 10-Seed Simulation Robustness (Seeds: {SEEDS_ALL_10})...")
    sim_10_res = run_simulation_robustness_10_seeds(crda_engine)

    # PART E: Magnitude Sensitivity
    print("\n[4/5] Running Physical Intervention Magnitude Sensitivity (Mild, Medium, Strong)...")
    magnitude_res = run_magnitude_sensitivity()

    # Consolidate all machine-readable results
    full_stage_8_data = {
        "study_metadata": {
            "stage": "Stage 8",
            "title": "Ablation and Robustness Studies",
            "random_seeds_original": SEEDS_ORIGINAL,
            "random_seeds_additional": SEEDS_ADDITIONAL,
            "random_seeds_total": SEEDS_ALL_10,
            "alphas_evaluated": [0.25, 0.50, 0.75, 1.00]
        },
        "part_a_crda_ablation": ablation_res,
        "part_b_pairwise_synergies": {k: v["pairwise_synergies"] for k, v in ablation_res.items()},
        "part_c_alpha_sensitivity": alpha_res,
        "part_d_simulation_robustness_10_seeds": sim_10_res,
        "part_e_magnitude_sensitivity": magnitude_res
    }

    json_path = "results/ablation_robustness.json"
    with open(json_path, "w") as f:
        json.dump(full_stage_8_data, f, indent=4)
    print(f"[OK] Machine-readable results saved to {json_path}")

    # Plot generation
    print("\n[5/5] Generating Robustness Verification Figures...")
    generate_ablation_robustness_plots(sim_10_res, magnitude_res, alpha_res)

    # Statistical Reporting Table
    print("\n" + "=" * 115)
    print("STAGE 8 ROBUSTNESS SUMMARY: 10-SEED VALIDATION & STAGE 7 CLAIM AUDIT")
    print("=" * 115)
    print(f"{'Scenario':<22} | {'CRDA Top':<10} | {'Sim Top Clr':<14} | {'Clr Match?':<10} | {'Best Single dClr':<16} | {'Best Multi dClr':<16} | {'Minimal Set Claim':<16}")
    print("-" * 115)
    for sc_id, sc_data in sim_10_res.items():
        sc_name = sc_data["scenario_name"].replace(" Dominant", "").replace(" Multi-Driver", "")
        crda_top = sc_data["crda_top_driver"]
        sim_top_clr = sc_data["sim_clearance_ranked"][0]
        match_str = "YES (MATCH)" if sc_data["top1_clearance_match_10_seeds"] else "NO (DISAGREE)"
        audit = sc_data["stage_7_minimal_subset_re_evaluation"]
        b_single = f"{audit['best_single_delta_clr']:+.2f}s"
        b_multi = f"{audit['best_multi_delta_clr']:+.2f}s" if audit['best_multi_delta_clr'] is not None else "N/A"
        claim_stat = audit["claim_status"]
        print(f"{sc_name:<22} | {crda_top:<10} | {sim_top_clr:<14} | {match_str:<10} | {b_single:<16} | {b_multi:<16} | {claim_stat:<16}")
    print("=" * 115)

    print("\n" + "=" * 90)
    print("INTERVENTION MAGNITUDE SENSITIVITY SUMMARY (10 Seeds)")
    print("=" * 90)
    for d_name, d_res in magnitude_res.items():
        print(f"\n{d_name} Intervention Scaling (Baseline Clearance = {d_res['baseline_mean_clearance']}s):")
        for v_name, v_data in d_res["variants"].items():
            clr = v_data["clearance_time"]["mean"]
            d_clr = v_data["delta_clearance"]["mean"]
            d = v_data["cohen_d_clearance"]
            print(f"  - {v_name:<22}: Clearance = {clr:.2f}s (ΔT = {d_clr:+.2f}s, Cohen's d = {d:+.2f})")
        print(f"  -> Monotonic Egress Trend: {d_res['trend']}")
    print("=" * 90)


if __name__ == "__main__":
    run_stage_8_ablation_robustness()
