"""
Controlled Evaluation of Interaction-Capable Interpretable Risk Models vs. Baseline Linear Fusion.
Evaluates:
- Baseline Linear Risk Fusion (0.45*Panic + 0.40*Bottleneck + 0.15*Density)
- Interpretable Additive + Pairwise Interaction Model (GAM/Logistic Interaction with L2 regularization)
- Shallow Interaction-Capable Tree Ensemble (Gradient Boosting, max_depth=2)
- Driver Group Ablations: [D], [O], [B], [K], [D+O], [D+B], [O+B], [D+O+B+K], [D+O+B+K + Interactions]
Under strict Leave-One-Scene-Out (LOSO) and Leave-One-Clip-Out (LOGO) cross-validation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_driver_representations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes normalized D, O, B, K drivers and baseline heuristic risk components
    from tabular video dataset features.
    """
    out = df.copy()

    # 1. Density Driver D (persons/Mpx and global density)
    ref_density = 200.0
    out["D"] = np.clip(out["density"], 0.0, 1.0)

    # 2. Directional Disorder Driver O (circular variance + directional entropy)
    circ_var = np.clip(out["direction_variance"], 0.0, 1.0)
    entropy_norm = np.clip(out["direction_entropy"] / np.log2(8.0), 0.0, 1.0)
    out["O"] = np.clip(0.5 * circ_var + 0.5 * entropy_norm, 0.0, 1.0)

    # 3. Bottleneck Driver B (density * jamming / speed drop)
    free_speed = 4.0
    speed_drop = np.clip(np.maximum(0.0, 1.0 - (out["mean_speed"] / free_speed)), 0.0, 1.0)
    stopped_ratio = np.clip(1.0 - out["moving_ratio"], 0.0, 1.0)
    jamming = 0.6 * speed_drop + 0.4 * stopped_ratio
    out["B"] = np.clip(out["D"] * jamming, 0.0, 1.0)

    # 4. Kinematic Instability Driver K (speed variance, acceleration, irregularity, flow)
    norm_spd_var = np.clip(np.sqrt(out["speed_variance"]) / 4.0, 0.0, 1.0)
    norm_accel = np.clip(out["mean_acceleration"] / 4.0, 0.0, 1.0)
    norm_irreg = np.clip(out["trajectory_irregularity"] / np.pi, 0.0, 1.0)
    norm_flow = np.clip((out["flow_mean"] + np.sqrt(out["flow_variance"])) / 3.0, 0.0, 1.0)
    out["K"] = np.clip(
        0.35 * norm_spd_var + 0.25 * norm_accel + 0.15 * norm_irreg + 0.25 * norm_flow,
        0.0,
        1.0
    )

    # Baseline Heuristic Panic and Baseline Risk Score
    panic_heur = np.clip(
        0.25 * np.clip(out["mean_speed"] / 5.0, 0.0, 1.0) +
        0.20 * norm_spd_var +
        0.20 * entropy_norm +
        0.20 * np.clip(out["flow_mean"] / 3.0, 0.0, 1.0) +
        0.15 * np.clip(out["flow_variance"] / 4.0, 0.0, 1.0),
        0.0,
        1.0
    )
    out["panic_heur"] = panic_heur
    out["risk_baseline"] = np.clip(
        0.45 * panic_heur + 0.40 * out["B"] + 0.15 * out["D"],
        0.0,
        1.0
    )

    return out


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Computes comprehensive classification, calibration, and error metrics."""
    y_pred = (y_score >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        roc_auc = 0.50
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def run_cross_validation_experiment(
    df: pd.DataFrame,
    group_col: str,
    feature_sets: Dict[str, List[str]],
    use_interactions: Dict[str, bool],
    model_types: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Executes Leave-One-Group-Out CV across all models on the specified grouping column.
    """
    groups = df[group_col].astype(str).values
    logo = LeaveOneGroupOut()
    y = df["label"].values

    results: Dict[str, Dict] = {}

    for model_name, feat_cols in feature_sets.items():
        is_interaction = use_interactions.get(model_name, False)
        m_type = model_types.get(model_name, "logistic")

        fold_metrics = []
        all_y_test = []
        all_y_scores = []

        for train_idx, test_idx in logo.split(df, y, groups=groups):
            X_tr = df.iloc[train_idx][feat_cols].values
            X_te = df.iloc[test_idx][feat_cols].values
            y_tr = y[train_idx]
            y_te = y[test_idx]

            if m_type == "baseline_rule":
                # Direct evaluation of existing linear formula
                y_score = df.iloc[test_idx]["risk_baseline"].values
            elif m_type == "logistic":
                if is_interaction and len(feat_cols) > 1:
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    X_tr_poly = poly.fit_transform(X_tr)
                    X_te_poly = poly.transform(X_te)
                else:
                    X_tr_poly = X_tr
                    X_te_poly = X_te

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_poly)
                X_te_s = scaler.transform(X_te_poly)

                clf = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=42)
                clf.fit(X_tr_s, y_tr)
                y_score = clf.predict_proba(X_te_s)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_te))

            elif m_type == "gradient_boosting":
                # Shallow GBM with max_depth=2 (strictly pairwise tree splits)
                clf = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=42
                )
                clf.fit(X_tr, y_tr)
                y_score = clf.predict_proba(X_te)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_te))

            metrics = evaluate_predictions(y_te, y_score)
            fold_metrics.append(metrics)
            all_y_test.extend(y_te)
            all_y_scores.extend(y_score)

        # Compute aggregate averages
        summary = {
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
            "mean_precision": float(np.mean([m["precision"] for m in fold_metrics])),
            "std_precision": float(np.std([m["precision"] for m in fold_metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in fold_metrics])),
            "std_recall": float(np.std([m["recall"] for m in fold_metrics])),
            "mean_f1": float(np.mean([m["f1_score"] for m in fold_metrics])),
            "std_f1": float(np.std([m["f1_score"] for m in fold_metrics])),
            "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "std_roc_auc": float(np.std([m["roc_auc"] for m in fold_metrics])),
            "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
            "mean_brier_score": float(np.mean([m["brier_score"] for m in fold_metrics])),
            "mean_fpr": float(np.mean([m["false_positive_rate"] for m in fold_metrics])),
            "mean_fnr": float(np.mean([m["false_negative_rate"] for m in fold_metrics])),
            "overall_pooled_metrics": evaluate_predictions(np.array(all_y_test), np.array(all_y_scores)),
            "fold_details": fold_metrics
        }
        results[model_name] = summary

    return results


def analyze_driver_interactions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fits logistic regression with all pairwise cross terms on full dataset
    to inspect empirical interaction coefficients and associative weights.
    """
    driver_cols = ["D", "O", "B", "K"]
    X = df[driver_cols].values
    y = df["label"].values

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(driver_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=42)
    clf.fit(X_scaled, y)

    coefs = clf.coef_[0]

    interaction_summary = {}
    for name, coef in zip(feature_names, coefs):
        is_cross = " " in name
        interaction_summary[name] = {
            "coefficient": float(round(coef, 4)),
            "odds_ratio": float(round(np.exp(coef), 4)),
            "is_pairwise_interaction": is_cross,
            "interpretation": (
                "Synergistic positive risk association" if coef > 0.1 else
                ("Antagonistic negative risk association" if coef < -0.1 else "Negligible independent association")
            )
        }

    return interaction_summary


def plot_model_comparisons(
    loso_results: Dict[str, Dict],
    logo_results: Dict[str, Dict],
    output_png: Path
):
    """Generates comparison bar chart between Baseline and proposed models."""
    models = list(loso_results.keys())
    loso_f1 = [loso_results[m]["mean_f1"] for m in models]
    loso_auc = [loso_results[m]["mean_roc_auc"] for m in models]
    logo_f1 = [logo_results[m]["mean_f1"] for m in models]
    logo_auc = [logo_results[m]["mean_roc_auc"] for m in models]

    x = np.arange(len(models))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, loso_f1, width, label="LOSO Mean F1", color="#2b5c8f")
    ax.bar(x - 0.5 * width, loso_auc, width, label="LOSO ROC-AUC", color="#4e88c7")
    ax.bar(x + 0.5 * width, logo_f1, width, label="LOGO Mean F1", color="#d95f02")
    ax.bar(x + 1.5 * width, logo_auc, width, label="LOGO ROC-AUC", color="#fdae6b")

    ax.set_ylabel("Metric Score [0.0 - 1.0]")
    ax.set_title("Risk Fusion Models & Driver Group Ablation Comparison\n(Grouped Validation to Prevent Temporal Leakage)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Interaction-Capable Risk Models vs Frozen Baseline.")
    parser.add_argument("--dataset", type=str, default="data/processed/features_dataset.csv", help="Feature CSV path.")
    parser.add_argument("--output-json", type=str, default="results/risk_model_comparison.json", help="Output comparison JSON.")
    parser.add_argument("--output-png", type=str, default="results/risk_model_comparison.png", help="Output comparison plot.")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 5: RISK MODEL & INTERACTION CAPABILITY EVALUATION")
    print("=" * 70)

    raw_df = pd.read_csv(args.dataset)
    print(f"Loaded {len(raw_df)} frames across scenes: {raw_df['scene'].unique().tolist()}")

    df = compute_driver_representations(raw_df)

    # Define Candidate Models & Driver Group Ablations
    feature_sets = {
        "A. Baseline Linear (0.45P+0.4B+0.15D)": ["panic_heur", "B", "D"],
        "B. Driver D (Density Only)": ["D"],
        "C. Driver O (Disorder Only)": ["O"],
        "D. Driver B (Bottleneck Only)": ["B"],
        "E. Driver K (Kinematics Only)": ["K"],
        "F. Pair [D + O]": ["D", "O"],
        "G. Pair [D + B]": ["D", "B"],
        "H. Pair [O + B]": ["O", "B"],
        "I. Additive [D + O + B + K]": ["D", "O", "B", "K"],
        "J. GAM Pairwise Interactions [D,O,B,K + Cross]": ["D", "O", "B", "K"],
        "K. Shallow GBM (Depth=2 Trees)": ["D", "O", "B", "K"]
    }

    use_interactions = {
        "J. GAM Pairwise Interactions [D,O,B,K + Cross]": True
    }

    model_types = {
        "A. Baseline Linear (0.45P+0.4B+0.15D)": "baseline_rule",
        "K. Shallow GBM (Depth=2 Trees)": "gradient_boosting"
    }
    for k in feature_sets:
        if k not in model_types:
            model_types[k] = "logistic"

    print("\n1. Running Leave-One-Scene-Out (LOSO) Cross-Validation (Lawn vs Indoor vs Plaza)...")
    loso_results = run_cross_validation_experiment(
        df=df,
        group_col="scene",
        feature_sets=feature_sets,
        use_interactions=use_interactions,
        model_types=model_types
    )

    print("\n2. Running Leave-One-Clip-Out (LOGO) Cross-Validation (11 Video Sequences)...")
    logo_results = run_cross_validation_experiment(
        df=df,
        group_col="clip_id",
        feature_sets=feature_sets,
        use_interactions=use_interactions,
        model_types=model_types
    )

    print("\n3. Analyzing Pairwise Interaction Effects...")
    interaction_effects = analyze_driver_interactions(df)

    # Print Formatted Results Table
    print("\n" + "=" * 90)
    print(f"{'Model / Driver Representation':<46} | {'LOSO F1':<9} | {'LOSO AUC':<9} | {'LOGO F1':<9} | {'LOGO AUC':<9}")
    print("=" * 90)
    for m in feature_sets:
        l_f1 = loso_results[m]["mean_f1"]
        l_auc = loso_results[m]["mean_roc_auc"]
        g_f1 = logo_results[m]["mean_f1"]
        g_auc = logo_results[m]["mean_roc_auc"]
        print(f"{m:<46} | {l_f1:.4f}    | {l_auc:.4f}    | {g_f1:.4f}    | {g_auc:.4f}")
    print("=" * 90)

    print("\nPairwise Interaction Coefficients (Full Dataset Logistic Model):")
    for feat, info in interaction_effects.items():
        print(f" - {feat:<12}: Coef = {info['coefficient']:+.4f} (Odds Ratio = {info['odds_ratio']:.4f}) -> {info['interpretation']}")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    full_output = {
        "evaluation_protocol": "Leave-One-Scene-Out (LOSO, 3 folds) and Leave-One-Clip-Out (LOGO, 11 folds)",
        "total_samples": len(df),
        "class_distribution": df["label"].value_counts().to_dict(),
        "leave_one_scene_out_results": loso_results,
        "leave_one_clip_out_results": logo_results,
        "pairwise_interaction_analysis": interaction_effects
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=4)
    print(f"\nSaved structured comparison JSON -> {out_json}")

    out_png = Path(args.output_png)
    plot_model_comparisons(loso_results, logo_results, out_png)
    print(f"Saved comparison visualization plot -> {out_png}")


if __name__ == "__main__":
    main()
