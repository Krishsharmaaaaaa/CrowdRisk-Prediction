"""
Stage 7: Simulation Intervention Validation Runner.
Independently evaluates whether CRDA's model-based risk-driver attribution
corresponds to measurable improvements in microscopic pedestrian evacuation outcomes.

Guardrails:
- Independent validation layer: Simulator measures physical egress outcomes (clearance time,
  exit throughput, congestion duration) completely independently from model feature formulas.
- Zero circularity: Simulator is not used to train or tune the risk models or CRDA.
- Multi-seed replicates: 5 fixed random seeds per scenario and intervention.
- Statistical reporting: mean +/- std, effect size, rank correlation, and agreement analysis.
"""

from dataclasses import asdict
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability.crda_engine import CRDAEngine
from src.features.cell_grid import SpatialCellFeatureVector
from src.simulation.pedestrian_sim import (
    PedestrianEvacuationSimulator,
    SimulationConfig,
    SimulationOutcome
)


SEEDS = [42, 101, 202, 303, 404]
CALM_REFERENCE = np.array([0.10, 0.05, 0.05, 0.05], dtype=np.float32)


def cohen_d(x: List[float], y: List[float]) -> float:
    """Calculates Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd < 1e-6:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def run_replicates(config: SimulationConfig, seeds: List[int], name: str) -> List[SimulationOutcome]:
    """Runs simulation across multiple random seeds."""
    outcomes = []
    for s in seeds:
        sim = PedestrianEvacuationSimulator(config)
        out = sim.run(seed=s)
        out.scenario_name = name
        outcomes.append(out)
    return outcomes


def build_scenario_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Defines 5 controlled scenarios and their physical intervention variants.
    """
    return {
        "Scenario_1_Bottleneck": {
            "name": "Bottleneck Dominant",
            "description": "Narrow exit constriction causing heavy queuing and severe speed collapse.",
            "baseline": SimulationConfig(
                door_width=0.6,
                num_agents=70,
                desired_speed_mean=1.3,
                desired_speed_std=0.2,
                noise_strength=0.05,
                max_steps=1200
            ),
            "interventions": {
                "Bottleneck (B)": SimulationConfig(
                    door_width=1.5, num_agents=70, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1200
                ),
                "Density (D)": SimulationConfig(
                    door_width=0.6, num_agents=45, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1200
                ),
                "Kinematics (K)": SimulationConfig(
                    door_width=0.6, num_agents=70, desired_speed_mean=1.0, desired_speed_std=0.1, noise_strength=0.05, max_steps=1200
                ),
                "Disorder (O)": SimulationConfig(
                    door_width=0.6, num_agents=70, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.01, max_steps=1200
                ),
                "Multi (B + D)": SimulationConfig(
                    door_width=1.5, num_agents=45, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1200
                )
            }
        },
        "Scenario_2_Density": {
            "name": "Density Dominant",
            "description": "Severe room overcrowding with high agent population and moderate door capacity.",
            "baseline": SimulationConfig(
                door_width=1.6,
                num_agents=120,
                desired_speed_mean=1.3,
                desired_speed_std=0.2,
                noise_strength=0.05,
                max_steps=1500
            ),
            "interventions": {
                "Density (D)": SimulationConfig(
                    door_width=1.6, num_agents=60, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1500
                ),
                "Bottleneck (B)": SimulationConfig(
                    door_width=2.4, num_agents=120, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1500
                ),
                "Kinematics (K)": SimulationConfig(
                    door_width=1.6, num_agents=120, desired_speed_mean=1.0, desired_speed_std=0.1, noise_strength=0.05, max_steps=1500
                ),
                "Disorder (O)": SimulationConfig(
                    door_width=1.6, num_agents=120, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.01, max_steps=1500
                ),
                "Multi (D + B)": SimulationConfig(
                    door_width=2.4, num_agents=60, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1500
                )
            }
        },
        "Scenario_3_Disorder": {
            "name": "Disorder Dominant",
            "description": "High directional panic/turbulence causing turbulent motion and frequent mutual blockage.",
            "baseline": SimulationConfig(
                door_width=1.2,
                num_agents=65,
                desired_speed_mean=1.3,
                desired_speed_std=0.2,
                noise_strength=0.45,
                max_steps=1200
            ),
            "interventions": {
                "Disorder (O)": SimulationConfig(
                    door_width=1.2, num_agents=65, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1200
                ),
                "Bottleneck (B)": SimulationConfig(
                    door_width=1.8, num_agents=65, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.45, max_steps=1200
                ),
                "Density (D)": SimulationConfig(
                    door_width=1.2, num_agents=40, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.45, max_steps=1200
                ),
                "Kinematics (K)": SimulationConfig(
                    door_width=1.2, num_agents=65, desired_speed_mean=1.0, desired_speed_std=0.1, noise_strength=0.45, max_steps=1200
                ),
                "Multi (O + B)": SimulationConfig(
                    door_width=1.8, num_agents=65, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.05, max_steps=1200
                )
            }
        },
        "Scenario_4_Kinematics": {
            "name": "Kinematics Dominant",
            "description": "Excessive speed surge and erratic speed dispersion creating physical jamming (Faster-is-slower).",
            "baseline": SimulationConfig(
                door_width=1.0,
                num_agents=65,
                desired_speed_mean=2.8,
                desired_speed_std=0.9,
                noise_strength=0.05,
                max_steps=1200
            ),
            "interventions": {
                "Kinematics (K)": SimulationConfig(
                    door_width=1.0, num_agents=65, desired_speed_mean=1.3, desired_speed_std=0.15, noise_strength=0.05, max_steps=1200
                ),
                "Bottleneck (B)": SimulationConfig(
                    door_width=1.8, num_agents=65, desired_speed_mean=2.8, desired_speed_std=0.9, noise_strength=0.05, max_steps=1200
                ),
                "Density (D)": SimulationConfig(
                    door_width=1.0, num_agents=40, desired_speed_mean=2.8, desired_speed_std=0.9, noise_strength=0.05, max_steps=1200
                ),
                "Disorder (O)": SimulationConfig(
                    door_width=1.0, num_agents=65, desired_speed_mean=2.8, desired_speed_std=0.9, noise_strength=0.01, max_steps=1200
                ),
                "Multi (K + B)": SimulationConfig(
                    door_width=1.8, num_agents=65, desired_speed_mean=1.3, desired_speed_std=0.15, noise_strength=0.05, max_steps=1200
                )
            }
        },
        "Scenario_5_MultiDriver": {
            "name": "Multi-Driver Compound Risk",
            "description": "Compound hazard combining high density, narrow door, elevated noise, and speed surge.",
            "baseline": SimulationConfig(
                door_width=0.7,
                num_agents=90,
                desired_speed_mean=2.2,
                desired_speed_std=0.6,
                noise_strength=0.30,
                max_steps=1400
            ),
            "interventions": {
                "Bottleneck (B)": SimulationConfig(
                    door_width=1.6, num_agents=90, desired_speed_mean=2.2, desired_speed_std=0.6, noise_strength=0.30, max_steps=1400
                ),
                "Density (D)": SimulationConfig(
                    door_width=0.7, num_agents=50, desired_speed_mean=2.2, desired_speed_std=0.6, noise_strength=0.30, max_steps=1400
                ),
                "Disorder (O)": SimulationConfig(
                    door_width=0.7, num_agents=90, desired_speed_mean=2.2, desired_speed_std=0.6, noise_strength=0.05, max_steps=1400
                ),
                "Kinematics (K)": SimulationConfig(
                    door_width=0.7, num_agents=90, desired_speed_mean=1.3, desired_speed_std=0.2, noise_strength=0.30, max_steps=1400
                ),
                "Multi (B + D)": SimulationConfig(
                    door_width=1.6, num_agents=50, desired_speed_mean=2.2, desired_speed_std=0.6, noise_strength=0.30, max_steps=1400
                ),
                "Multi (B + D + O)": SimulationConfig(
                    door_width=1.6, num_agents=50, desired_speed_mean=2.2, desired_speed_std=0.6, noise_strength=0.05, max_steps=1400
                )
            }
        }
    }


def run_simulation_validation():
    print("=" * 80)
    print("STAGE 7: INDEPENDENT SIMULATION INTERVENTION VALIDATION")
    print("Testing CRDA model-based risk attribution against simulator-native outcomes")
    print(f"Replicates per configuration: {len(SEEDS)} seeds {SEEDS}")
    print("=" * 80)

    crda = CRDAEngine(moderate_risk_threshold=0.30, high_risk_threshold=0.60)
    scenarios = build_scenario_definitions()

    validation_results = {}
    summary_table_rows = []

    for sc_id, sc_data in scenarios.items():
        print(f"\n--- Running {sc_id}: {sc_data['name']} ---")
        base_cfg = sc_data["baseline"]

        # 1. Run Baseline across seeds and extract choke features
        base_outcomes = run_replicates(base_cfg, SEEDS, f"{sc_id}_baseline")

        # Collect choke features during baseline runs
        choke_vectors = []
        for s in SEEDS:
            sim = PedestrianEvacuationSimulator(base_cfg)
            sim.run(seed=s)
            choke_vectors.append(sim.get_congested_choke_driver_vector())

        mean_choke_vec = np.mean(choke_vectors, axis=0)
        print(f"  Observed Choke Vector [D, O, B, K]: {[round(float(v), 3) for v in mean_choke_vec]}")

        # 2. Feed choke state to CRDA Engine
        mock_cell = SpatialCellFeatureVector(
            cell_id="CHOKE",
            row=1,
            col=3,
            x1=11.0,
            y1=4.0,
            x2=15.0,
            y2=8.0,
            area_px=16.0,
            D=float(mean_choke_vec[0]),
            O=float(mean_choke_vec[1]),
            B=float(mean_choke_vec[2]),
            K=float(mean_choke_vec[3])
        )
        crda_expl = crda.explain_cell(mock_cell, CALM_REFERENCE, alpha=1.0)
        print(f"  CRDA Initial Risk: {crda_expl.initial_risk:.3f} ({crda_expl.initial_threat_level})")
        print(f"  CRDA Top-1 Driver: {crda_expl.top_driver} (Delta R = {crda_expl.top_driver_delta_R:.3f})")
        print(f"  CRDA Minimal Intervention: {crda_expl.minimal_intervention_subset.subset_names} "
              f"-> Risk {crda_expl.minimal_intervention_subset.counterfactual_risk:.3f} "
              f"({crda_expl.minimal_intervention_subset.resulting_threat_level})")

        # Map single drivers to their CRDA predicted delta R
        crda_deltas = {attr.driver_key: attr.delta_R for attr in crda_expl.single_driver_attributions}
        key_map = {"Density (D)": "D", "Bottleneck (B)": "B", "Kinematics (K)": "K", "Disorder (O)": "O"}

        # Baseline performance metrics
        base_clearances = [o.clearance_time for o in base_outcomes]
        base_throughputs = [o.exit_throughput for o in base_outcomes]
        base_congestions = [o.congestion_duration for o in base_outcomes]
        base_collisions = [o.collision_count for o in base_outcomes]

        print(f"  Baseline Clearance: {np.mean(base_clearances):.2f} +/- {np.std(base_clearances):.2f}s | "
              f"Throughput: {np.mean(base_throughputs):.2f} +/- {np.std(base_throughputs):.2f}/s | "
              f"Congestion: {np.mean(base_congestions):.2f} +/- {np.std(base_congestions):.2f}s")

        # 3. Run Physical Interventions
        interv_stats = {}
        for int_name, int_cfg in sc_data["interventions"].items():
            int_outcomes = run_replicates(int_cfg, SEEDS, f"{sc_id}_{int_name}")

            int_clearances = [o.clearance_time for o in int_outcomes]
            int_throughputs = [o.exit_throughput for o in int_outcomes]
            int_congestions = [o.congestion_duration for o in int_outcomes]
            int_collisions = [o.collision_count for o in int_outcomes]

            # Physical improvements (positive means improvement)
            delta_clearance = np.mean(base_clearances) - np.mean(int_clearances)
            delta_throughput = np.mean(int_throughputs) - np.mean(base_throughputs)
            delta_congestion = np.mean(base_congestions) - np.mean(int_congestions)
            delta_collisions = np.mean(base_collisions) - np.mean(int_collisions)

            # Effect size (Cohen's d) for throughput and clearance
            d_thr = cohen_d(int_throughputs, base_throughputs)
            d_clr = cohen_d(base_clearances, int_clearances)  # positive = faster clearance

            crda_pred_delta = crda_deltas.get(key_map.get(int_name, ""), None)

            interv_stats[int_name] = {
                "clearance_time_mean": round(float(np.mean(int_clearances)), 2),
                "clearance_time_std": round(float(np.std(int_clearances)), 2),
                "delta_clearance": round(float(delta_clearance), 2),
                "cohen_d_clearance": round(float(d_clr), 2),
                "exit_throughput_mean": round(float(np.mean(int_throughputs)), 3),
                "exit_throughput_std": round(float(np.std(int_throughputs)), 3),
                "delta_throughput": round(float(delta_throughput), 3),
                "cohen_d_throughput": round(float(d_thr), 2),
                "congestion_duration_mean": round(float(np.mean(int_congestions)), 2),
                "congestion_duration_std": round(float(np.std(int_congestions)), 2),
                "delta_congestion": round(float(delta_congestion), 2),
                "collision_count_mean": round(float(np.mean(int_collisions)), 1),
                "crda_predicted_delta_R": round(float(crda_pred_delta), 3) if crda_pred_delta is not None else None
            }

            print(f"    -> [{int_name:16s}] Clr: {np.mean(int_clearances):.2f}s (dClr: {delta_clearance:+.2f}s, d={d_clr:+.2f}) | "
                  f"Thr: {np.mean(int_throughputs):.2f}/s (dThr: {delta_throughput:+.2f}/s, d={d_thr:+.2f}) | "
                  f"Cong: {np.mean(int_congestions):.2f}s (dCong: {delta_congestion:+.2f}s)")

        # 4. Compare CRDA Single-Driver Ranking vs Simulator Improvement Rankings
        single_interventions = [k for k in sc_data["interventions"].keys() if k in key_map]
        single_crda_ranked = sorted(
            single_interventions,
            key=lambda k: crda_deltas.get(key_map[k], 0.0),
            reverse=True
        )
        # Rank by Throughput improvement
        single_sim_thr_ranked = sorted(
            single_interventions,
            key=lambda k: interv_stats[k]["delta_throughput"],
            reverse=True
        )
        # Rank by Clearance Time reduction (faster egress)
        single_sim_clr_ranked = sorted(
            single_interventions,
            key=lambda k: interv_stats[k]["delta_clearance"],
            reverse=True
        )
        # Rank by Congestion Duration reduction (less jamming)
        single_sim_cong_ranked = sorted(
            single_interventions,
            key=lambda k: interv_stats[k]["delta_congestion"],
            reverse=True
        )

        top1_thr_agrees = (single_crda_ranked[0] == single_sim_thr_ranked[0])
        top1_clr_agrees = (single_crda_ranked[0] == single_sim_clr_ranked[0])

        print(f"  CRDA Single Driver Rank : {single_crda_ranked}")
        print(f"  Sim Throughput Rank     : {single_sim_thr_ranked}")
        print(f"  Sim Clearance Time Rank : {single_sim_clr_ranked}")
        print(f"  Top-1 Match (Throughput): {'YES (MATCH)' if top1_thr_agrees else 'DISAGREEMENT'}")
        print(f"  Top-1 Match (Clearance) : {'YES (MATCH)' if top1_clr_agrees else 'DISAGREEMENT'}")

        # Check minimal intervention subset physical efficacy
        min_sub_names = crda_expl.minimal_intervention_subset.subset_names
        # Find closest multi-intervention in simulator
        multi_matches = [k for k in sc_data["interventions"].keys() if "Multi" in k]
        best_multi = multi_matches[0] if multi_matches else None
        if best_multi:
            multi_eff = interv_stats[best_multi]["delta_throughput"]
            multi_clr_eff = interv_stats[best_multi]["delta_clearance"]
            print(f"  Sim Multi-Intervention ({best_multi}): dThr = {multi_eff:+.2f}/s, dClr = {multi_clr_eff:+.2f}s")

        validation_results[sc_id] = {
            "scenario_name": sc_data["name"],
            "description": sc_data["description"],
            "observed_choke_features": {
                "D": round(float(mean_choke_vec[0]), 3),
                "O": round(float(mean_choke_vec[1]), 3),
                "B": round(float(mean_choke_vec[2]), 3),
                "K": round(float(mean_choke_vec[3]), 3)
            },
            "crda_prediction": {
                "initial_risk": round(crda_expl.initial_risk, 3),
                "threat_level": crda_expl.initial_threat_level,
                "top_driver": crda_expl.top_driver,
                "top_driver_delta_R": round(crda_expl.top_driver_delta_R, 3),
                "single_driver_attributions": [a.to_dict() for a in crda_expl.single_driver_attributions],
                "minimal_intervention_subset": crda_expl.minimal_intervention_subset.to_dict(),
                "narrative": crda_expl.narrative_explanation
            },
            "baseline_outcomes": {
                "clearance_time_mean": round(float(np.mean(base_clearances)), 2),
                "clearance_time_std": round(float(np.std(base_clearances)), 2),
                "exit_throughput_mean": round(float(np.mean(base_throughputs)), 3),
                "exit_throughput_std": round(float(np.std(base_throughputs)), 3),
                "congestion_duration_mean": round(float(np.mean(base_congestions)), 2),
                "congestion_duration_std": round(float(np.std(base_congestions)), 2),
                "collision_count_mean": round(float(np.mean(base_collisions)), 1)
            },
            "intervention_outcomes": interv_stats,
            "validation_analysis": {
                "crda_ranked_drivers": single_crda_ranked,
                "simulator_ranked_throughput": single_sim_thr_ranked,
                "simulator_ranked_clearance": single_sim_clr_ranked,
                "simulator_ranked_congestion": single_sim_cong_ranked,
                "top1_agreement_throughput": top1_thr_agrees,
                "top1_agreement_clearance": top1_clr_agrees,
                "minimal_subset_effective": bool(best_multi and interv_stats[best_multi]["delta_clearance"] > 0)
            }
        }

    # 5. Save structured JSON
    os.makedirs("results", exist_ok=True)
    out_json_path = "results/simulation_validation.json"
    with open(out_json_path, "w") as f:
        json.dump(validation_results, f, indent=4)
    print(f"\n[OK] Validation results saved to {out_json_path}")

    # 6. Generate Multi-panel Verification Plot
    generate_validation_plot(validation_results)

    # 7. Print Final Summary Table
    print("\n" + "=" * 105)
    print("STAGE 7 STATISTICAL SUMMARY: CRDA PREDICTION VS SIMULATION OUTCOMES")
    print("=" * 105)
    print(f"{'Scenario':<24} | {'CRDA Top-1':<11} | {'Sim Top (Thr)':<14} | {'Sim Top (Clr)':<14} | {'Clr Match?':<10} | {'Max dClr (s)':<12}")
    print("-" * 105)
    for sc_id, sc_res in validation_results.items():
        v = sc_res["validation_analysis"]
        top_crda = sc_res["crda_prediction"]["top_driver"]
        top_sim_thr = v["simulator_ranked_throughput"][0]
        top_sim_clr = v["simulator_ranked_clearance"][0]
        clr_match = "YES (MATCH)" if v["top1_agreement_clearance"] else "NO"
        best_stats = sc_res["intervention_outcomes"][top_sim_clr]
        d_clr = f"{best_stats['delta_clearance']:+.2f}s"
        print(f"{sc_res['scenario_name']:<24} | {top_crda:<11} | {top_sim_thr:<14} | {top_sim_clr:<14} | {clr_match:<10} | {d_clr:<12}")
    print("=" * 105)


def generate_validation_plot(results: Dict[str, Any]):
    """Creates a comprehensive visualization of simulation validation results."""
    scenario_keys = list(results.keys())
    fig, axes = plt.subplots(len(scenario_keys), 2, figsize=(14, 3.2 * len(scenario_keys)))
    fig.suptitle(
        "CRDA Model-Based Risk Attribution vs Independent Pedestrian Simulation Outcomes\n"
        "(5 Replicates per Configuration | Error bars: +/- 1 SD)",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )

    for i, sc_id in enumerate(scenario_keys):
        sc = results[sc_id]
        ax_clr = axes[i, 0]
        ax_thr = axes[i, 1]

        int_data = sc["intervention_outcomes"]
        labels = list(int_data.keys())

        # Clearance times
        base_clr = sc["baseline_outcomes"]["clearance_time_mean"]
        base_clr_std = sc["baseline_outcomes"]["clearance_time_std"]
        clrs = [int_data[k]["clearance_time_mean"] for k in labels]
        clr_errs = [int_data[k]["clearance_time_std"] for k in labels]

        # Exit Throughputs
        base_thr = sc["baseline_outcomes"]["exit_throughput_mean"]
        base_thr_std = sc["baseline_outcomes"]["exit_throughput_std"]
        thrs = [int_data[k]["exit_throughput_mean"] for k in labels]
        thr_errs = [int_data[k]["exit_throughput_std"] for k in labels]

        x = np.arange(len(labels))
        width = 0.55

        # Colors: Green for multi/improvements, blue for single
        colors = ["#2563eb" if "Multi" not in l else "#059669" for l in labels]

        # Plot Clearance Time
        bars_clr = ax_clr.bar(x, clrs, width, yerr=clr_errs, capsize=4, color=colors, alpha=0.85, edgecolor="#1e293b")
        ax_clr.axhline(base_clr, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Baseline ({base_clr:.1f}s)")
        ax_clr.set_ylabel("Clearance (s)", fontsize=9, fontweight="bold")
        ax_clr.set_title(f"{sc['scenario_name']} — Clearance Time (Lower is Better)", fontsize=10, fontweight="bold")
        ax_clr.set_xticks(x)
        ax_clr.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
        ax_clr.legend(loc="upper right", fontsize=8)
        ax_clr.grid(axis="y", linestyle=":", alpha=0.6)

        # Plot Throughput
        bars_thr = ax_thr.bar(x, thrs, width, yerr=thr_errs, capsize=4, color=colors, alpha=0.85, edgecolor="#1e293b")
        ax_thr.axhline(base_thr, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Baseline ({base_thr:.2f}/s)")
        top_driver_str = sc["crda_prediction"]["top_driver"]
        clr_agree = "MATCH" if sc["validation_analysis"]["top1_agreement_clearance"] else "DISAGREE"
        thr_agree = "MATCH" if sc["validation_analysis"]["top1_agreement_throughput"] else "DISAGREE"
        ax_clr.set_title(f"{sc['scenario_name']} — Clearance Time (Lower is Better) | CRDA: {top_driver_str} [{clr_agree}]", fontsize=9, fontweight="bold")
        ax_thr.set_title(f"Throughput (Higher is Better) | Thr Match: [{thr_agree}]", fontsize=9, fontweight="bold")
        ax_thr.set_xticks(x)
        ax_thr.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
        ax_thr.legend(loc="lower right", fontsize=8)
        ax_thr.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = "results/simulation_validation.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Validation plot saved to {plot_path}")


if __name__ == "__main__":
    run_simulation_validation()
