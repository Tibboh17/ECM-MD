import csv
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BIN_CLASSES = ("terrain", "mine")
NAME2ID = {"terrain": 0, "mine": 1}

logger = logging.getLogger("reporting_csv")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def generate_single_run_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    report_type: str = "test",
    class_names: list = BIN_CLASSES,
):
    """
    단일 학습/평가 실행 결과를 CSV와 혼동 행렬 이미지로 저장합니다.
    기존의 리포트 집계 함수와 호환되는 형식으로 파일을 생성합니다.

    Args:
        y_true: 실제 레이블 배열.
        y_pred: 모델 예측 레이블 배열.
        output_dir: 결과물이 저장될 디렉터리 (e.g., .../reports/original/hog/).
        model_name: 사용된 모델 이름 (e.g., 'SVC').
        report_type: 리포트 종류 ('train' 또는 'test'). 파일명에 사용됩니다.
        class_names: 클래스 이름 리스트. 'terrain'이 0, 'mine'이 1이어야 합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 혼동 행렬 계산 (TN, FP, FN, TP)
    # scikit-learn의 confusion_matrix는 [[TN, FP], [FN, TP]] 순서로 결과를 반환합니다.
    cm = pd.DataFrame(
        np.bincount(2 * y_true + y_pred, minlength=4).reshape(2, 2),
        index=class_names,
        columns=class_names,
    )
    tn, fp, fn, tp = cm.values.ravel()

    # 2. 정확도 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # 3. _test_metrics.csv 파일 작성
    metrics_path = output_dir / f"{model_name}_{report_type}_metrics.csv"
    header = ["timestamp", "acc", "tn", "fp", "fn", "tp"]
    row = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "acc": f"{accuracy:.4f}",
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"{report_type.capitalize()} metrics saved to {metrics_path}")

    # 4. 혼동 행렬 이미지 저장
    cm_plot_path = output_dir / f"{model_name}_{report_type}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix ({model_name} - {report_type.capitalize()})')
    fig.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {cm_plot_path}")

    # 5. 상세 텍스트 리포트 생성
    report_path = output_dir / f"{model_name}_{report_type}_report.txt"
    misclassification_rate = 1.0 - accuracy
    per_class_metrics = _per_class_from_confmat_binary(tn, fp, fn, tp)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f" Single Run Performance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model           : {model_name}\n")
        f.write(f"Report Type     : {report_type.capitalize()}\n")
        f.write(f"Timestamp       : {row['timestamp']}\n")
        f.write("-" * 50 + "\n\n")

        f.write("--- Overall Metrics ---\n")
        f.write(f"Accuracy                : {accuracy:.4f}\n")
        f.write(f"Misclassification Rate  : {misclassification_rate:.4f}\n\n")

        f.write("--- Per-Class Metrics ---\n")
        header = f"{'':<12}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for cls_name in class_names:
            metrics = per_class_metrics[cls_name]
            f.write(f"{cls_name:<12}{metrics['Precision']:>10.4f}{metrics['Recall']:>10.4f}{metrics['F1-Score']:>10.4f}{int(metrics['Support']):>10}\n")
        f.write("\n")

        f.write("--- Confusion Matrix ---\n")
        f.write(cm.to_string())
        f.write("\n\n")
    logger.info(f"Detailed text report saved to {report_path}")

def _read_csv_rows(p):
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _per_class_from_confmat_binary(tn, fp, fn, tp):
    # mine (positive)
    p_m = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_m = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_m = (2 * p_m * r_m / (p_m + r_m)) if (p_m + r_m) > 0 else 0.0
    s_m = tp + fn

    # terrain (negative) = positive 반전
    tp_t, fp_t, fn_t, tn_t = tn, fn, fp, tp
    p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f_t = (2 * p_t * r_t / (p_t + r_t)) if (p_t + r_t) > 0 else 0.0
    s_t = tp_t + fn_t

    return {
        "mine":    {"Precision": p_m, "Recall": r_m, "F1-Score": f_m, "Support": s_m},
        "terrain": {"Precision": p_t, "Recall": r_t, "F1-Score": f_t, "Support": s_t},
    }

def _find_methods(reports_root):
    return sorted([p.name for p in reports_root.glob("*") if p.is_dir()])

def _find_models(method_dir, report_type="test"):
    models = set()
    suffix = f"_{report_type}_metrics.csv"
    for p in method_dir.glob(f"*{suffix}"): models.add(p.name.replace(suffix, ""))
    return sorted(list(models))

def _collect_results_from_csv(reports_root, scenario_name, report_type="test"):
    rows = []
    methods = _find_methods(reports_root)

    for method in methods:
        mdir = reports_root / method
        if not mdir.is_dir():
            continue

        for model in _find_models(mdir, report_type=report_type):
            metrics_csv = mdir / f"{model}_{report_type}_metrics.csv"
            metrics_rows = _read_csv_rows(metrics_csv)
            if not metrics_rows:
                logger.warning("[SKIP] no test_metrics for %s/%s", method, model)
                continue

            t = metrics_rows[0]
            acc = _to_float(t.get("acc"))
            tn = int(_to_float(t.get("tn", 0)) or 0)
            fp = int(_to_float(t.get("fp", 0)) or 0)
            fn = int(_to_float(t.get("fn", 0)) or 0)
            tp = int(_to_float(t.get("tp", 0)) or 0)

            per_cls = _per_class_from_confmat_binary(tn, fp, fn, tp)

            for cls in BIN_CLASSES:
                rows.append({
                    "Scenario": scenario_name,
                    "Feature": method,
                    "Model": model,
                    "Class": cls,
                    "Accuracy": acc if acc is not None else np.nan,
                    "Precision": float(per_cls[cls]["Precision"]),
                    "Recall": float(per_cls[cls]["Recall"]),
                    "F1-Score": float(per_cls[cls]["F1-Score"]),
                    "Support": int(per_cls[cls]["Support"]),
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "Scenario","Feature","Model","Class","Accuracy",
            "Precision","Recall","F1-Score","Support","TN","FP","FN","TP"
        ])
    
    return pd.DataFrame(rows)

def _write_summary_markdown(df, output_path, title, report_type="test"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("This report is compiled directly from CSV metrics (no legacy text parsing).\n\n")

        summary_df = df[df["Class"] == "mine"][["Scenario","Feature","Model","Accuracy","F1-Score"]].copy()
        summary_df = summary_df.rename(columns={"F1-Score": "Mine_F1_Score"})
        summary_df = summary_df.sort_values(by=["Mine_F1_Score","Accuracy"], ascending=False)
        f.write("## Overall Performance Summary (sorted by Mine F1-Score)\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        df_classes = df[df["Class"].isin(BIN_CLASSES)].copy()
        for group_name, group in df_classes.groupby("Scenario"):
            pivot_group = group.pivot_table(
                index=["Feature","Model"],
                columns="Class",
                values=["Accuracy","Precision","Recall","F1-Score","TN","FP","FN","TP"],
            ).reset_index()

            pivot_group.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in pivot_group.columns.values
            ]
            rename_map = {"Accuracy_mine": "Accuracy", "Feature_": "Feature", "Model_": "Model"}
            drop_cols = [c for c in [
                "Accuracy_terrain","TN_terrain","FP_terrain","FN_terrain","TP_terrain",
                "TN_mine","FP_mine","FN_mine","TP_mine"
            ] if c in pivot_group.columns]
            pivot_group = pivot_group.rename(columns=rename_map).drop(columns=drop_cols, errors="ignore")

            if "Accuracy" not in pivot_group.columns:
                for c in pivot_group.columns:
                    if c.startswith("Accuracy_"):
                        pivot_group = pivot_group.rename(columns={c: "Accuracy"})
                        break

            pivot_group = pivot_group.sort_values(by="Accuracy", ascending=False)
            f.write(f"## Details for: {group_name}\n\n")
            f.write(pivot_group.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

    logger.info("[REPORT] Saved training report -> %s", output_path.as_posix())

def generate_summary_report(reports_root, set_name, output_dir, report_type="test", title=None):
    reports_root = Path(reports_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = set_name
    df = _collect_results_from_csv(reports_root, scenario_name, report_type=report_type)
    if df.empty:
        logger.error(f"[CSV-REPORT] No '{report_type}_metrics.csv' files found under {reports_root.as_posix()}")
        return

    out_md = out_dir / f"{report_type}_summary_report.md"
    default_title = f"{report_type.capitalize()} Performance Report (SET={set_name})"
    _write_summary_markdown(df, out_md, title or default_title, report_type=report_type)