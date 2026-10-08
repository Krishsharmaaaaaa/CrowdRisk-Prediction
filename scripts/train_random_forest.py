"""
Random Forest Model Training and Validation Script for Crowd Panic Classification.
Evaluates panic detection using rigorous grouped validation to prevent temporal data leakage:
1. Leave-One-Scene-Out (LOSO): Evaluates across 3 independent scenes (Lawn, Indoor, Plaza).
2. GroupKFold (Clip-Wise): Evaluates across 11 independent video sequences.
"""

import argparse
import json
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


FEATURE_COLUMNS = [
    "mean_speed",
    "speed_variance",
    "max_speed",
    "mean_acceleration",
    "direction_entropy",
    "direction_variance",
    "moving_ratio",
    "density",
    "density_change",
    "trajectory_irregularity",
    "flow_mean",
    "flow_variance",
    "high_flow_ratio"
]


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_name: str,
    results_dir: Path
):
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    print("\n" + "=" * 65)
    print(f"RUNNING VALIDATION: Grouping by '{group_name}' ({n_groups} unique groups)")
    print(f"Groups: {list(unique_groups)}")
    print("=" * 65)

    logo = LeaveOneGroupOut() if n_groups <= 11 else GroupKFold(n_splits=5)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_roc_aucs = []
    fold_details = []

    all_y_test = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_group_names = list(np.unique(groups[test_idx]))

        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_test))
        else:
            y_proba = y_pred.astype(float)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        if len(np.unique(y_test)) > 1:
            roc_auc = float(roc_auc_score(y_test, y_proba))
            fold_roc_aucs.append(roc_auc)
        else:
            roc_auc = None

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        fold_info = {
            "fold": fold_idx + 1,
            "test_groups": test_group_names,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_class_distribution": {int(k): int(v) for k, v in train_dist.items()},
            "test_class_distribution": {int(k): int(v) for k, v in test_dist.items()},
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        fold_details.append(fold_info)

        print(f"Fold {fold_idx + 1:2d} [Test: {test_group_names}]: "
              f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}" +
              (f", ROC-AUC={roc_auc:.4f}" if roc_auc is not None else ", ROC-AUC=N/A"))

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_prec = float(np.mean(fold_precisions))
    std_prec = float(np.std(fold_precisions))
    mean_rec = float(np.mean(fold_recalls))
    std_rec = float(np.std(fold_recalls))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    mean_roc_auc = float(np.mean(fold_roc_aucs)) if fold_roc_aucs else None

    print("-" * 65)
    print(f"SUMMARY FOR '{group_name}' VALIDATION:")
    print(f"Mean Accuracy:  {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Mean Precision: {mean_prec:.4f} (+/- {std_prec:.4f})")
    print(f"Mean Recall:    {mean_rec:.4f} (+/- {std_rec:.4f})")
    print(f"Mean F1 Score:  {mean_f1:.4f} (+/- {std_f1:.4f})")
    if mean_roc_auc is not None:
        print(f"Mean ROC-AUC:   {mean_roc_auc:.4f}")
    print("-" * 65)

    # Confusion matrix
    cm = confusion_matrix(all_y_test, all_y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix ({group_name} split)")
    fig.colorbar(im)
    classes = ["Normal", "Panic / Escape"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = results_dir / f"confusion_matrix_{group_name}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC curve
    if len(np.unique(all_y_test)) > 1 and len(all_y_proba) > 0:
        fpr, tpr, _ = roc_curve(all_y_test, all_y_proba)
        cv_auc = roc_auc_score(all_y_test, all_y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Group CV ROC (AUC = {cv_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Panic Detection ROC ({group_name})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = results_dir / f"roc_curve_{group_name}.png"
        plt.savefig(roc_path, dpi=200)
        plt.close()

    return {
        "group_name": group_name,
        "num_groups": n_groups,
        "groups": list(unique_groups),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_precision": mean_prec,
        "std_precision": std_prec,
        "mean_recall": mean_rec,
        "std_recall": std_rec,
        "mean_f1_score": mean_f1,
        "std_f1_score": std_f1,
        "mean_roc_auc": mean_roc_auc,
        "fold_details": fold_details
    }


def train_panic_rf(
    dataset_csv: str = "data/processed/features_dataset.csv",
    model_output_path: str = "models/panic_rf.joblib",
    results_dir: str = "results"
):
    csv_path = Path(dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {dataset_csv}. Run scripts/extract_features.py first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples from {csv_path}")

    class_counts = df["label"].value_counts().to_dict()
    print(f"Overall Class Distribution: Normal (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%), "
          f"Abnormal/Panic (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    X = df[FEATURE_COLUMNS].fillna(0.0).values
    y = df["label"].values

    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = {
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "total_samples": len(df),
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
        "evaluations": {}
    }

    # 1. Leave-One-Scene-Out Evaluation (Lawn vs Indoor vs Plaza)
    if "scene" in df.columns and len(df["scene"].unique()) >= 2:
        scene_groups = df["scene"].astype(str).values
        scene_res = run_grouped_cv(X, y, scene_groups, "leave_one_scene_out", res_dir)
        metrics_output["evaluations"]["leave_one_scene_out"] = scene_res

    # 2. Leave-One-Clip-Out / GroupKFold Evaluation (11 Video Sequences)
    clip_col = "clip_id" if "clip_id" in df.columns else ("video_id" if "video_id" in df.columns else "video_name")
    if clip_col in df.columns and len(df[clip_col].unique()) >= 2:
        clip_groups = df[clip_col].astype(str).values
        clip_res = run_grouped_cv(X, y, clip_groups, "leave_one_clip_out", res_dir)
        metrics_output["evaluations"]["leave_one_clip_out"] = clip_res

    # 3. Fit Final Production Model on Full Dataset
    print("\nFitting final Random Forest baseline on full dataset...")
    final_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )
    final_clf.fit(X, y)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, model_output_path)
    print(f"Final model saved -> {model_output_path}")

    metrics_output["feature_importances"] = {
        feat: float(imp) for feat, imp in zip(FEATURE_COLUMNS, final_clf.feature_importances_)
    }

    metrics_path = res_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=4)
    print(f"Full metrics saved -> {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest Classifier using Grouped Video Validation.")
    parser.add_argument("--dataset", type=str, default="data/processed/features_dataset.csv", help="Input dataset CSV.")
    parser.add_argument("--model-out", type=str, default="models/panic_rf.joblib", help="Output model path.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for metric artifacts.")
    args = parser.parse_args()

    train_panic_rf(args.dataset, args.model_out, args.results_dir)


if __name__ == "__main__":
    main()
