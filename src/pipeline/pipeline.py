import json
import logging
import numpy as np
import pandas as pd
import re
import os

from pathlib import Path, PurePath
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from src.pipeline.utils import setup_logging
from src.pipeline.data_loader import load_and_split_data, load_inference_features, load_npz_from_split_dir
from src.pipeline.reporting import (
    generate_training_report,
    generate_inference_report,
    analyze_predictions,
)
from src.pipeline.models import (
    build_pipeline,
    get_model_instance,
    evaluate_and_save_model,
    load_model_and_map,
    run_prediction,
    get_features_on_the_fly,
)

# ----------------------------
# Helpers for feature path
# ----------------------------
def resolve_feature_npz_strict(features_root, method, allowed=("train",)):
    root = Path(features_root)
    for sub in allowed:
        if sub not in ("train", "test"):
            continue
        p = root / sub / f"{method}.npz"
        if p.exists():
            return p
    return None

def _take_train_part(maybe_pair, expected_len=None):
    if isinstance(maybe_pair, (tuple, list)) and len(maybe_pair) == 2:
        arr = maybe_pair[0]
    else:
        arr = maybe_pair
    if expected_len is not None:
        try:
            if len(arr) != expected_len:
                return None
        except Exception:
            return None
    return arr

def get_expected_input_dim(pipe):
    try:
        for _, step in getattr(pipe, "named_steps", {}).items():
            nfi = getattr(step, "n_features_in_", None)
            if isinstance(nfi, (int, np.integer)):
                return int(nfi)
        steps = getattr(pipe, "steps", [])
        if steps:
            last_est = steps[-1][1]
            nfi = getattr(last_est, "n_features_in_", None)
            if isinstance(nfi, (int, np.integer)):
                return int(nfi)
    except Exception:
        pass
    return None

# ----------------------------
# NEW: Build group map from CSVs
# ----------------------------
def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/")

def _guess_scene_from_image_path(image_path: str) -> str:
    """.../<scene>_with_terrain/<class>/<file>.png -> <scene>_with_terrain"""
    try:
        p = PurePath(image_path)
        return p.parent.parent.name
    except Exception:
        return ""

def _file_stem(image_or_filename: str) -> str:
    b = os.path.basename(str(image_or_filename))
    return os.path.splitext(b)[0]

def build_group_map_from_csv(labels_dir: Path) -> dict:
    """
    labels_dir 아래의 모든 CSV를 읽어 image_path→group 매핑을 만든다.
    group := "<scene_dir>/<file_stem>"
    - CSV에 'image_path'가 있으면 그대로 사용
    - 없고 'filename'만 있으면 CSV 파일명(stem)과 결합할 수 없으므로, scene 추정이 어려우니
      우선 filename만으로 group을 만들되, 학습 시 stem 기반 보정으로 매칭한다.
    """
    labels_dir = Path(labels_dir)
    group_map = {}
    csv_paths = sorted(labels_dir.glob("*.csv"))
    if not csv_paths:
        logging.warning(f"[GROUP MAP] No CSVs under {labels_dir}")
        return group_map

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.warning(f"[GROUP MAP] Read fail: {csv_path}: {e}")
            continue

        has_image_path = "image_path" in df.columns
        has_filename = "filename" in df.columns

        if not has_image_path and not has_filename:
            logging.warning(f"[GROUP MAP] {csv_path.name} lacks image_path/filename columns; skip.")
            continue

        if has_image_path:
            # scene 디렉토리 + 파일 stem으로 그룹 생성
            image_paths = df["image_path"].astype(str).tolist()
            for ip in image_paths:
                ipn = _norm_path(ip)
                scene = _guess_scene_from_image_path(ipn)
                stem = _file_stem(ipn)
                gid = f"{scene}/{stem}" if scene else stem
                group_map[ipn] = gid
        else:
            # filename만 있는 경우: stem 자체로 그룹을 일단 만든다(후속 stem 매칭으로 사용)
            filenames = df["filename"].astype(str).tolist()
            for fn in filenames:
                stem = _file_stem(fn)
                group_map[stem] = stem  # 키를 stem로만 보관 (fallback용)
    logging.info(f"[GROUP MAP] Built with {len(group_map)} entries from {len(csv_paths)} CSV(s).")
    return group_map

def map_groups_for_filenames(filenames, group_map: dict) -> np.ndarray:
    """
    npz filenames 배열에 대해 group_map을 조인한다.
    우선 image_path(정규화)로 찾고, 실패하면 stem으로 fallback.
    """
    groups, miss = [], 0
    for p in filenames:
        key = _norm_path(p)
        g = group_map.get(key)
        if g is None:
            stem = _file_stem(key)
            # exact stem key
            g = group_map.get(stem)
            if g is None:
                # 마지막 fallback: group_map의 키들 중 stem 같은 항목 찾기(비용은 작음)
                for k, v in group_map.items():
                    if _file_stem(k) == stem:
                        g = v
                        break
        if g is None:
            miss += 1
            g = f"__missing__/{_file_stem(key)}"
        groups.append(g)
    if miss:
        logging.warning(f"[GROUP MAP] {miss} filenames unmatched; used stem fallback/placeholder groups.")
    return np.array(groups)

def choose_groupkfold_splits(y_train: np.ndarray, groups: np.ndarray, max_splits: int = 5):
    """
    클래스별 그룹 수 제약까지 반영해 GroupKFold의 n_splits를 결정한다.
    반환: (cv, cv_kwargs, cv_name)
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    classes = np.unique(y_train)

    # 각 클래스 그룹 수 계산
    per_class_group_counts = []
    for c in classes:
        idx = (y_train == c)
        grp = np.unique(groups[idx])
        per_class_group_counts.append(len(grp))

    n_splits = min(max_splits, n_groups, *per_class_group_counts) if per_class_group_counts else 0

    if n_splits < 2:
        logging.warning("[CV] Not enough class-wise groups; fallback to StratifiedKFold(5).")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42), {}, "StratifiedKFold(5)"

    if n_splits < max_splits:
        logging.warning(f"[CV] Reducing n_splits to {n_splits} (groups={n_groups}, per-class={per_class_group_counts}).")

    return GroupKFold(n_splits=n_splits), {"groups": groups}, f"GroupKFold({n_splits})"

# ----------------------------
# K-Fold quick report (unchanged)
# ----------------------------
def run_kfold_training(config, scenario):
    project_root = Path(config.get("project_root", "."))
    results_dir = config.get("results_dir", "results")
    train_data_type = scenario["train_data"]

    features_root = project_root / results_dir / "features" / train_data_type
    out_root = project_root / results_dir / "models" / f"{train_data_type}_kfold"
    out_root.mkdir(parents=True, exist_ok=True)

    C_train = config.get("training", {})
    seed = int(C_train.get("random_state", 42))
    methods = list(C_train.get("methods", []))
    model_types = list(C_train.get("model_types", []))

    folds = sorted({p.name.split("_")[0].replace("fold","")
                    for p in features_root.glob("fold*_train")})
    if not folds:
        logging.error("No folds found under %s", features_root)
        return

    for method in methods:
        logging.info("========== [K-FOLD | %s] ==========", method.upper())
        for model_name in model_types:
            logging.info("[MODEL] %s", model_name)
            per_fold = []
            for k in folds:
                tr_dir = features_root / f"fold{k}_train"
                va_dir = features_root / f"fold{k}_val"
                X_tr, y_tr, _ = load_npz_from_split_dir(tr_dir, method)
                X_va, y_va, _ = load_npz_from_split_dir(va_dir, method)
                model = get_model_instance(model_name, seed)
                pipe = build_pipeline(model, model_name, C_train)
                pipe.fit(X_tr, y_tr)
                pr = pipe.predict(X_va)
                acc = accuracy_score(y_va, pr)
                f1m = f1_score(y_va, pr, average="macro")
                logging.info("[fold %s] ACC=%.4f | F1M=%.4f", k, acc, f1m)
                per_fold.append({"acc": float(acc), "f1_macro": float(f1m)})
            accs = [m["acc"] for m in per_fold]
            f1s = [m["f1_macro"] for m in per_fold]
            logging.info("[CV][%s|%s] ACC  %.4f ± %.4f", method, model_name, float(np.mean(accs)), float(np.std(accs)))
            logging.info("[CV][%s|%s] F1M  %.4f ± %.4f", method, model_name, float(np.mean(f1s)), float(np.std(f1s)))

# ----------------------------
# Training (uses CSV-driven groups)
# ----------------------------
def run_training(config, scenario):
    train_data_type = scenario["train_data"]
    scenario_name = f"{scenario['train_data']}-to-{scenario['infer_data']}"
    C_train_base = deepcopy(config.get("training", {}))
    C_feat_base = deepcopy(config.get("feature_extraction", {}))
    project_root = Path(config.get("project_root", "."))
    results_dir = config.get("results_dir", "results")

    logging.info(f"[TRAIN] Scenario: {scenario_name}")

    feature_base_dir = project_root / results_dir / "features" / train_data_type
    output_base_dir = project_root / results_dir / "reports" / scenario_name / "train"
    split_csv_dir = project_root / results_dir / "splits" / train_data_type  # CSV들이 여기에 있다고 가정

    # 1) CSV에서 group map 구성(한 번만)
    group_map = build_group_map_from_csv(split_csv_dir)

    for method in C_train_base.get("methods", []):
        feature_path = resolve_feature_npz_strict(feature_base_dir, method, allowed=("train",))
        if feature_path is None:
            logging.error(f"[TRAIN] Feature file not found under train/ for '{method}' at {feature_base_dir}. Abort.")
            continue

        split_data_package = load_and_split_data(
            feature_path,
            C_train_base["test_size"],
            C_train_base["random_state"]
        )
        if not split_data_package:
            logging.error(f"[TRAIN] Failed to load/split data: {feature_path}")
            continue

        (X_train, X_test, y_train, y_test), label_map, filenames_train, groups_train = split_data_package
        filenames_train = _take_train_part(filenames_train, expected_len=len(y_train))
        groups_train = _take_train_part(groups_train, expected_len=len(y_train))

        # 2) 라벨 맵 정규화
        name_to_id_map = {}
        if isinstance(label_map, dict) and len(label_map) > 0:
            k, v = next(iter(label_map.items()))
            if isinstance(k, str) and isinstance(v, (int, np.integer)):
                name_to_id_map = {str(k): int(v) for k, v in label_map.items()}
            elif isinstance(k, (int, np.integer)) and isinstance(v, str):
                name_to_id_map = {str(v): int(k) for k, v in label_map.items()}
        if not name_to_id_map:
            name_to_id_map = {"terrain": 0, "mine": 1}
        id_to_name_map = {v: k for k, v in name_to_id_map.items()}
        class_names = [id_to_name_map[i] for i in sorted(id_to_name_map.keys())]

        # 3) 그룹 보정: CSV 기반 group_map으로 반드시 대체
        if filenames_train is not None and len(filenames_train) == len(y_train):
            groups_from_csv = map_groups_for_filenames(filenames_train, group_map)
            groups_train = groups_from_csv
            logging.info("[CV] Groups loaded from CSV mapping (strict).")
        else:
            logging.warning("[CV] filenames missing/length mismatch; fallback to StratifiedKFold later.")

        # 4) CV 스킴 결정
        if groups_train is not None and len(groups_train) == len(y_train):
            cv, cv_kwargs, cv_name = choose_groupkfold_splits(y_train, groups_train, max_splits=5)
        else:
            logging.warning("[CV] groups missing/invalid; fallback to StratifiedKFold.")
            cv, cv_kwargs, cv_name = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), {}, "StratifiedKFold(5)"

        logging.info(f"[CV] Using {cv_name} to select best fixed config per model (no grid search).")

        # 5) 후보 파이프라인
        def make_pipe(model_name, use_pca=False, pca_var=0.95, use_kbest=False, k=64):
            C = deepcopy(C_train_base)
            C["use_pca"] = use_pca
            C["pca_variance"] = pca_var
            C["use_kbest"] = use_kbest
            C["k_best_features"] = k
            model = get_model_instance(model_name, C["random_state"])
            return build_pipeline(model, model_name, C), C

        for model_name in C_train_base.get("model_types", []):
            candidates = [
                ("pca95",   *make_pipe(model_name, use_pca=True,  pca_var=0.95, use_kbest=False)),
                ("pca90",   *make_pipe(model_name, use_pca=True,  pca_var=0.90, use_kbest=False)),
                ("kbest32", *make_pipe(model_name, use_pca=False, use_kbest=True, k=32)),
                ("kbest64", *make_pipe(model_name, use_pca=False, use_kbest=True, k=64)),
            ]

            best = None
            for tag, pipe, C_used in candidates:
                try:
                    f1m = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs)
                    acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
                    logging.info("[CV][%s|%s|%s] F1M %.4f ± %.4f | ACC %.4f ± %.4f",
                                 method, model_name, tag, float(f1m.mean()), float(f1m.std()),
                                 float(acc.mean()), float(acc.std()))
                    score = float(f1m.mean())
                    if (best is None) or (score > best["score"]):
                        best = {"tag": tag, "pipe": pipe, "C_used": C_used, "score": score}
                except Exception as e:
                    logging.warning(f"[CV] Failed for {model_name}|{tag}: {e}")

            if best is None:
                logging.error(f"[TRAIN] No valid candidate pipeline for {model_name} on method={method}. Skipping.")
                continue

            # 6) 전체 train으로 재학습 후 저장/홀드아웃 평가
            final_pipe = best["pipe"]
            final_pipe.fit(X_train, y_train)
            logging.info(f"[SELECT] ({model_name}) Best config = {best['tag']} (F1M={best['score']:.4f}). Refit on full train.")

            output_dir = output_base_dir / method / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("INFO", output_dir / "train.log")

            try:
                (output_dir / "feature_config.json").write_text(json.dumps(C_feat_base, indent=2), encoding="utf-8")
            except Exception as e:
                logging.warning(f"[TRAIN] Failed to write feature_config.json: {e}")

            evaluate_and_save_model(
                final_pipe,
                X_train, y_train, X_test, y_test,
                class_names,
                output_dir,
                method,
                model_name,
                feature_path,
                name_to_id_map,
            )

            try:
                (output_dir / "selected_config.json").write_text(
                    json.dumps({"model": model_name, "tag": best["tag"], "training_config": best["C_used"]}, indent=2),
                    encoding="utf-8"
                )
            except Exception as e:
                logging.warning(f"[TRAIN] Failed to write selected_config.json: {e}")

# ----------------------------
# Inference / Analysis / Reports (unchanged)
# ----------------------------
def run_inference(config, scenario):
    train_data, infer_data = scenario["train_data"], scenario["infer_data"]
    scenario_name = f"{train_data}-to-{infer_data}"
    C_infer = deepcopy(config.get("inference", {}))
    C_feat = deepcopy(config.get("feature_extraction", {}))
    project_root = Path(config.get("project_root", "."))
    results_dir = config.get("results_dir", "results")

    model_base_dir = project_root / results_dir / "reports" / scenario_name / "train"
    feature_base_dir = project_root / results_dir / "features" / infer_data
    output_base_dir = project_root / results_dir / "reports" / scenario_name / "inference"

    for method in C_infer.get("methods", []):
        for model_type in C_infer.get("model_types", []):
            output_dir = output_base_dir / method / model_type
            output_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("INFO", output_dir / f"inference_{method}_{model_type}.log")

            model, id_to_name_map = load_model_and_map(model_base_dir / method / model_type)
            if model is None:
                logging.error(f"[INFER] Missing model/map at {model_base_dir / method / model_type}.")
                continue

            snap_path = model_base_dir / method / model_type / "feature_config.json"
            if snap_path.exists():
                try:
                    C_feat = json.loads(snap_path.read_text(encoding="utf-8"))
                    logging.info("[INFER] Loaded feature_config.json snapshot for on-the-fly extraction parity.")
                except Exception as e:
                    logging.warning(f"[INFER] Failed to read feature_config.json snapshot: {e}")

            npz_path = resolve_feature_npz_strict(feature_base_dir, method, allowed=("test",))
            features, filenames = load_inference_features(npz_path) if npz_path else (None, None)

            if features is not None:
                expected_features = get_expected_input_dim(model)
                if isinstance(expected_features, int) and features.shape[1] != expected_features:
                    logging.warning(
                        f"[INFER] Feature mismatch: expected {expected_features}, got {features.shape[1]}. Falling back to on-the-fly extraction."
                    )
                    features = None

            if features is None:
                image_base_dir = project_root / "datasets" / "augmented" / infer_data
                features, filenames = get_features_on_the_fly(C_feat, image_base_dir, method)
                if features is None:
                    logging.error("[INFER] On-the-fly feature extraction failed.")
                    continue

                expected_features = get_expected_input_dim(model)
                if isinstance(expected_features, int) and features.shape[1] != expected_features:
                    logging.error(
                        f"[INFER] On-the-fly features still mismatched: expected {expected_features}, got {features.shape[1]}."
                    )
                    continue

            output_csv = output_dir / "inference_results.csv"
            run_prediction(model, features, filenames, id_to_name_map, output_csv)

def run_analysis(config, scenario):
    train_data, infer_data = scenario["train_data"], scenario["infer_data"]
    scenario_name = f"{train_data}-to-{infer_data}"
    C_analysis = config.get("analysis", {})
    project_root = Path(config.get("project_root", "."))
    results_dir = config.get("results_dir", "results")

    logging.info(f"[ANALYSIS] Scenario: {scenario_name}")
    output_base_dir = project_root / results_dir / "reports" / scenario_name / "inference"

    for model_type in C_analysis.get("model_types", []):
        for method in C_analysis.get("methods", []):
            output_dir = output_base_dir / method / model_type
            csv_path = output_dir / "inference_results.csv"
            if not csv_path.exists():
                continue
            setup_logging("INFO", output_dir / f"analysis_{method}_{model_type}.log")
            analyze_predictions(csv_path, output_dir, method, model_type)

def run_report_generation(config, scenario):
    scenario_name = f"{scenario['train_data']}-to-{scenario['infer_data']}"
    logging.info(f"[REPORT] Generating reports for scenario: {scenario_name}")
    generate_training_report(config, scenario)
    generate_inference_report(config, scenario)
    logging.info(f"[REPORT] Finished for scenario: {scenario_name}")
