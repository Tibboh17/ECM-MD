import json
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1) SafeKBest
# ============================================================
class SafeKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=64, score_func=f_classif):
        self.k = int(k)
        self.score_func = score_func
        self._sel = None
        self.effective_k_ = None

    def fit(self, X, y=None):
        from sklearn.feature_selection import SelectKBest
        k_eff = "all" if self.k >= X.shape[1] else self.k
        self.effective_k_ = (X.shape[1] if k_eff == "all" else k_eff)
        self._sel = SelectKBest(self.score_func, k=k_eff).fit(X, y)
        return self

    def transform(self, X):
        return self._sel.transform(X)

# ============================================================
# 2) 튜닝 결과 로드 및 병합 유틸
# ============================================================
def load_tuned_params_json(json_path):
    if not json_path:
        return None
    p = Path(json_path)
    if not p.exists():
        logging.warning(f"Tuned params JSON not found: {p}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logging.warning("Tuned params JSON is not a dict. Ignored.")
            return None
        return data
    except Exception as e:
        logging.error(f"Failed to load tuned params JSON: {e}", exc_info=True)
        return None

def _merge(a, b):
    out = dict(a) if a else {}
    if b:
        out.update(b)
    return out

# ============================================================
# 3) 모델 생성 함수 (특징별 튜닝 결과 반영)
# ============================================================
def get_model_instance(model_name, random_state, feature_name=None, tuned_params=None):
    base_defaults = {
        "svc": {
            "probability": True,
            "class_weight": "balanced",
            "kernel": "rbf",
            "gamma": 0.01,
            "C": 0.1,
        },
        "random_forest": {
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 10,
        },
    }

    tuned_for_feature = None
    if tuned_params and feature_name:
        tuned_for_feature = tuned_params.get(str(feature_name), {}).get(model_name, None)

    if model_name == "svc":
        final_kwargs = _merge(base_defaults["svc"], tuned_for_feature)
        return SVC(**final_kwargs)

    elif model_name == "random_forest":
        final_kwargs = _merge(base_defaults["random_forest"], tuned_for_feature)
        return RandomForestClassifier(**final_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_name}")

# ============================================================
# 4) 파이프라인 생성
# ============================================================
def build_pipeline(model_instance, model_name, C_train):
    steps = [('scaler', StandardScaler())]

    if C_train.get("use_pca", False):
        keep = float(C_train.get("pca_variance", 0.95))
        steps.append(('pca', PCA(n_components=keep, random_state=C_train.get("random_state"))))
        logging.info(f"Pipeline will use PCA to retain {int(keep * 100)}% variance.")

    if C_train.get("use_kbest", False):
        k_cfg = int(C_train.get("k_best_features", 64))
        steps.append(('selector', SafeKBest(k=k_cfg, score_func=f_classif)))
        logging.info(f"Pipeline will use SelectKBest with k={k_cfg}.")

    steps.append((model_name, model_instance))
    return Pipeline(steps)

def build_pipeline_for_feature(model_name, feature_name, C_train, random_state, tuned_params=None):
    model = get_model_instance(
        model_name=model_name,
        random_state=random_state,
        feature_name=feature_name,
        tuned_params=tuned_params,
    )
    return build_pipeline(model, model_name, C_train)

# ============================================================
# 5) 평가 및 저장
# ============================================================
def _ensure_id_to_name_map(name_to_id_map):
    id_to_name_map = {}

    if isinstance(name_to_id_map, dict) and len(name_to_id_map) > 0:
        k, v = next(iter(name_to_id_map.items()))
        if isinstance(k, str) and isinstance(v, (int, np.integer)):
            id_to_name_map = {int(v): str(k) for k, v in name_to_id_map.items()}
        elif isinstance(k, (int, np.integer)) and isinstance(v, str):
            id_to_name_map = {int(k): str(v) for k, v in name_to_id_map.items()}

    if not id_to_name_map:
        id_to_name_map = {0: "terrain", 1: "mine"}

    return id_to_name_map

def _plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def evaluate_and_save_model(pipeline, X_train, y_train, X_test, y_test,
                            class_names, output_dir, method, model_name,
                            feature_path, name_to_id_map):
    logging.info(f"Training the {model_name} model...")
    logging.info(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

    pipeline.fit(X_train, y_train)
    logging.info("Model training complete.")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy:.4f}")

    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    id_to_name_map = _ensure_id_to_name_map(name_to_id_map)
    target_names = [id_to_name_map.get(int(i), str(i)) for i in unique_labels]

    report_str = classification_report(
        y_test, y_pred, labels=unique_labels,
        target_names=target_names, zero_division=0
    )

    logging.info("Classification Report:\n" + report_str)
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")

    if isinstance(name_to_id_map, dict):
        k, v = next(iter(name_to_id_map.items()))
        if isinstance(k, (int, np.integer)) and isinstance(v, str):
            name_to_id_map = {v: int(k) for k, v in name_to_id_map.items()}

    with open(output_dir / "class_map.json", 'w', encoding='utf-8') as f:
        json.dump(name_to_id_map, f, indent=2, ensure_ascii=False)

    with open(output_dir / "report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Report for: {method.upper()} with {model_name.upper()} model\n")
        f.write(f"Source: {feature_path.parent.name}/{feature_path.name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix (rows=True, cols=Pred):\n")
        f.write(np.array2string(cm, separator=', '))

    _plot_confusion_matrix(cm, target_names, output_dir / "confusion_matrix.png")
    logging.info(f"Saved results and confusion matrix to {output_dir}")

def load_model_and_map(model_dir):
    model_path = model_dir / "model.joblib"
    class_map_path = model_dir / "class_map.json"

    if not all([model_path.exists(), class_map_path.exists()]):
        logging.error(f"Missing model or class map in {model_dir}.")
        return None, None

    try:
        model = joblib.load(model_path)
        with open(class_map_path, 'r', encoding='utf-8') as f:
            name_to_id_map = json.load(f)

        id_to_name_map = _ensure_id_to_name_map(name_to_id_map)
        logging.info(f"Loaded model and class map from {model_dir}.")
        return model, id_to_name_map
    except Exception as e:
        logging.error(f"Failed to load model/class map: {e}", exc_info=True)
        return None, None