# -*- coding: utf-8 -*-
"""
End-to-End Mine Detection Pipeline
Steps:
  0) Sanity & dirs
  1) Generate terrain XMLs
  2) Crop patches from XMLs
  3) Build label CSVs
  4) Split mines(=ratio) & terrain(=fixed sizes)
  5) Augment TRAIN (mine only) and merge final train/test CSV
  6) Extract features from train/test and save to NPZ (hog/lbp/gabor/sfs)
  7) Train & evaluate models per feature, save reports & summaries

All log messages are in English by request.
"""

import sys
from pathlib import Path
import csv
import json
import time
import logging

# ───────── path setup ─────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ───────── process modules ─────────
from process.terrain import generate_terrain_for_dir
from process.cropping import crop_from_xml
from process.labeling import build_label_sets
from process.splitting import split_dataset, split_dataset_by_size
from process.augmentation import augment_from_split_csv, AugConfig

# ───────── features & models ─────────
from features.combined_feature_extractor import CombinedConfig, ComprehensiveFeatureExtractor
from pipeline.models import build_pipeline_for_feature, load_tuned_params_json
from pipeline.reporting import generate_single_run_report, generate_summary_report
from pipeline.data_loader import load_npz_from_split_dir

# ───────── configuration ─────────
SET = "original"

PROJECT_ROOT = Path(".").resolve()
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_DIR = PROJECT_ROOT / "datasets"

RAW_XML_DIR = DATASET_DIR / "raw" / SET
RAW_IMG_DIR = DATASET_DIR / "raw" / SET
XML_TERRAIN = DATASET_DIR / "xml_with_terrain" / SET
CROPS_ROOT = DATASET_DIR / "crops" / SET
LABELS_DIR = RESULTS_DIR / "labels" / SET
SPLIT_DIR = RESULTS_DIR / "splits" / SET
AUG_IMG_OUT = DATASET_DIR / "augmented" / SET / "train"
AUG_LBL_DIR = RESULTS_DIR / "aug_labels" / SET
FEATURES_DIR = RESULTS_DIR / "features" / SET
REPORTS_DIR = RESULTS_DIR / "reports" / SET
MODELS_DIR = RESULTS_DIR / "models" / SET
SUMMARY_DIR = REPORTS_DIR / "summary_csv"
TUNED_PARAMS_JSON = PROJECT_ROOT / "config" / "tuned_params.json"

SEED = 2025

# ---- Terrain generation params ----
PATCH_SIZE = 40
STRIDE = 20
NUM_TERRAIN = 1500
N_CLUSTERS = 1
CENTER_EXCL = 40
IOU_EXCL = 0.1

# ---- Cropping params ----
CLASS_FILTER = ("mine", "terrain")
MARGIN = 0.0
SQUARE = True
RESIZE_TO = 64
MIN_SIZE = 8

# ---- Split params ----
TEST_RATIO = 0.2
TRAIN_SIZE_TERRAIN = 1000
TEST_SIZE_TERRAIN = 50

# ---- Augmentation params ----
AUG_PER_IMAGE = 9
AUG_INCLUDE_ORIG = False
AUG_CLASSES = ("mine",)

# ---- Feature extraction params ----
GRAYSCALE = True
RESIZE_HW = (256, 256)
WORKERS = 8
CHUNK = 512

USE_HOG = True
USE_LBP = True
USE_GABOR = True
USE_SFS = True

# ---- Training params ----
METHODS = ["hog", "lbp", "gabor"]
MODELS = ["svc", "random_forest"]
TRAIN_CFG = {
    "random_state": SEED,
    "use_pca": False,
    "pca_variance": 0.95,
    "use_kbest": False,
    "k_best_features": 64,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("end2end")

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CANON_NAME2ID = {"terrain": 0, "mine": 1}
REQUIRED = ["image", "filename", "class", "source", "group_id"]

def _timer():
    t0 = time.time()
    return lambda: time.time() - t0

def _ensure_dirs():
    dirs = [
        XML_TERRAIN, CROPS_ROOT, LABELS_DIR, SPLIT_DIR, AUG_IMG_OUT, AUG_LBL_DIR,
        FEATURES_DIR / "train", FEATURES_DIR / "test",
        REPORTS_DIR, MODELS_DIR, SUMMARY_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _check_inputs():
    missing = [p for p in [RAW_XML_DIR, RAW_IMG_DIR] if not p.exists()]
    if missing:
        for p in missing:
            log.error("Missing required path: %s", p.as_posix())
        raise FileNotFoundError("Required input path(s) are missing.")

def _read_csv_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not set(REQUIRED).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain {REQUIRED}: {path}")
        for row in reader:
            rows.append({
                "image": row["image"].replace("\\", "/").strip("/"),
                "filename": row["filename"],
                "class": row["class"].strip(),
                "source": row["source"].replace("\\", "/").strip("/"),
                "group_id": row["group_id"].replace("\\", "/").strip("/"),
            })
    return rows

def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(rows)

def _combine_label_csvs(out_csv, *inputs, dedup_on="image"):
    all_rows = []
    for p in inputs:
        if not Path(p).exists():
            log.warning("[WARN] Missing CSV to combine: %s", p.as_posix())
            continue
        all_rows.extend(_read_csv_rows(p))
    if dedup_on:
        seen, uniq = set(), []
        for r in all_rows:
            key = r.get(dedup_on, "")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        all_rows = uniq
    _write_csv(out_csv, all_rows)
    return len(all_rows)

def _cv2_read(path):
    import cv2
    flag = cv2.IMREAD_GRAYSCALE if GRAYSCALE else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(path.as_posix())
    if RESIZE_HW:
        img = cv2.resize(img, RESIZE_HW)
    return img

def _resolve_from_roots(rel_path, roots):
    clean_path = str(rel_path).replace("\\", "/").strip("/")
    for r in roots:
        p = r / clean_path
        if p.exists():
            return p
    return Path(clean_path)

def _build_extractor():
    cfg = CombinedConfig(
        use_hog=USE_HOG,
        use_lbp=USE_LBP,
        use_gabor=USE_GABOR,
        use_sfs=USE_SFS,
        max_workers=WORKERS,
        hog_mode="multiscale",
        lbp_radius_list=[2, 4, 8],
        gabor_mode="optimized",
        gabor_n_frequencies=4,
        gabor_n_orientations=4,
        gabor_patch_size=32,
        gabor_compute_phase=False,
        sfs_config=None,
    )
    return ComprehensiveFeatureExtractor(cfg)

def _save_npz_block(out_dir, feats_blocks, y, paths):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    fnames = np.array([str(p) for p in paths], dtype=object)
    for method, feats in feats_blocks.items():
        out_path = out_dir / f"{method}.npz"
        np.savez_compressed(out_path, **{
            method: feats,
            "labels": y,
            "filenames": fnames,
            "label_map": np.array(CANON_NAME2ID, dtype=object),
        })
        log.info("[SAVE] %s (%dx%d)", out_path.name, feats.shape[0], feats.shape[1])

def _extract_from_csv(split_name, csv_file, extractor, roots):
    import numpy as np
    rows = _read_csv_rows(csv_file)
    paths, labels = [], []
    for r in rows:
        p = _resolve_from_roots(r["image"], roots)
        if not p.exists():
            log.warning("[SKIP] not found: %s", r["image"])
            continue
        cls = r["class"]
        if cls not in CANON_NAME2ID:
            continue
        paths.append(p)
        labels.append(CANON_NAME2ID[cls])

    if not paths:
        log.warning("[SKIP] %s: no resolvable images", split_name)
        return

    y_all = np.array(labels, dtype=np.int32)
    log.info("[SPLIT] %s: files=%d  csv=%s", split_name, len(paths), csv_file.name)

    feature_accumulator = None
    acc_paths = []
    t0 = _timer()
    for i in range(0, len(paths), CHUNK):
        batch_paths = paths[i:i+CHUNK]
        images, y_ok, p_ok = [], [], []
        for pth, yy in zip(batch_paths, y_all[i:i+CHUNK]):
            try:
                images.append(_cv2_read(pth))
                y_ok.append(yy)
                p_ok.append(pth)
            except Exception as e:
                log.warning("[SKIP] read fail: %s (%s)", pth.as_posix(), e)
        if not images:
            continue

        extracted = extractor.extract_batch(images)
        if feature_accumulator is None:
            feature_accumulator = {k: [v] for k, v in extracted.items()}
            feature_accumulator["labels"] = [np.array(y_ok, dtype=np.int32)]
        else:
            for k, v in extracted.items():
                feature_accumulator[k].append(v)
            feature_accumulator["labels"].append(np.array(y_ok, dtype=np.int32))

        acc_paths.extend(p_ok)
        log.info("[%s] chunk %d: %d images", split_name, (i // CHUNK) + 1, len(p_ok))

    if feature_accumulator is None or not acc_paths:
        return

    feats_blocks = {k: np.concatenate(vs, axis=0)
                    for k, vs in feature_accumulator.items() if k != "labels"}
    y_all = np.concatenate(feature_accumulator["labels"], axis=0)
    out_dir = FEATURES_DIR / split_name
    _save_npz_block(out_dir, feats_blocks, y_all, acc_paths)
    log.info("[OK] %s feature extraction done in %.2fs", split_name, t0())

def main():
    from sklearn.metrics import precision_recall_fscore_support
    _check_inputs()
    _ensure_dirs()

    log.info("=" * 80)
    log.info("Start End-to-End Pipeline")
    log.info("=" * 80)

    # 1) Terrain generation
    log.info("[STEP 1] Generate terrain XMLs")
    t = _timer()
    generate_terrain_for_dir(
        xml_dir=RAW_XML_DIR,
        image_dir=RAW_IMG_DIR,
        out_dir=XML_TERRAIN,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        num_terrain=NUM_TERRAIN,
        n_clusters=N_CLUSTERS,
        central_exclusion_percent=CENTER_EXCL,
        iou_exclusion_thresh=IOU_EXCL,
        seed=SEED,
    )
    log.info("[OK] Terrain XMLs -> %s (%.2fs)", XML_TERRAIN.as_posix(), t())

    # 2) Cropping
    log.info("[STEP 2] Crop patches from XMLs")
    xmls = sorted(XML_TERRAIN.glob("*.xml"))
    if not xmls:
        log.warning("No XML files in %s. Skipping cropping.", XML_TERRAIN.as_posix())
    else:
        for x in xmls:
            crop_from_xml(
                xml_path=str(x),
                image_dir=str(RAW_IMG_DIR),
                out_root=str(CROPS_ROOT),
                class_filter=CLASS_FILTER,
                margin=MARGIN,
                square=SQUARE,
                resize_to=RESIZE_TO,
                min_size=MIN_SIZE,
                seed=SEED,
            )

    # 3) Label CSVs
    build_label_sets(
        crops_root=CROPS_ROOT,
        out_dir=LABELS_DIR,
        make_all=True,
        make_per_class=True,
        make_per_source=False,
        print_summary=True,
    )

    # 4) Split
    mine_csv = LABELS_DIR / "mine.csv"
    terrain_csv = LABELS_DIR / "terrain.csv"
    split_dataset(
        input_csv=str(mine_csv),
        output_dir=str(SPLIT_DIR),
        seed=SEED,
        test_ratio=TEST_RATIO,
        train_name="mine_train.csv",
        test_name="mine_test.csv",
    )
    split_dataset_by_size(
        input_csv=str(terrain_csv),
        output_dir=str(SPLIT_DIR),
        train_size=TRAIN_SIZE_TERRAIN,
        test_size=TEST_SIZE_TERRAIN,
        seed=SEED,
        train_name="terrain_train.csv",
        test_name="terrain_test.csv",
    )

    # 5) Augment & combine
    mine_train_csv = SPLIT_DIR / "mine_train.csv"
    aug_stats = augment_from_split_csv(
        split_csv=str(mine_train_csv),
        crops_root=str(CROPS_ROOT),
        out_root=str(AUG_IMG_OUT),
        labels_out_dir=str(AUG_LBL_DIR),
        num_augmentations=AUG_PER_IMAGE,
        cfg=AugConfig(),
        seed=SEED,
        include_original=AUG_INCLUDE_ORIG,
        classes_to_augment=AUG_CLASSES,
    )
    aug_csv = aug_stats.get("labels_csv") if aug_stats else None
    final_test_csv = SPLIT_DIR / "test.csv"
    _combine_label_csvs(final_test_csv, SPLIT_DIR / "mine_test.csv", SPLIT_DIR / "terrain_test.csv")
    final_train_csv = SPLIT_DIR / "train.csv"
    train_inputs = [SPLIT_DIR / "mine_train.csv", SPLIT_DIR / "terrain_train.csv"]
    if aug_csv:
        train_inputs.append(Path(aug_csv))
    _combine_label_csvs(final_train_csv, *train_inputs, dedup_on="image")

    # 6) Feature Extraction
    extractor = _build_extractor()
    roots = [AUG_IMG_OUT, CROPS_ROOT]
    _extract_from_csv("train", final_train_csv, extractor, roots)
    _extract_from_csv("test", final_test_csv, extractor, roots)

    # 7) Train & Evaluate
    tuned_params = load_tuned_params_json(TUNED_PARAMS_JSON)
    train_dir = FEATURES_DIR / "train"
    test_dir = FEATURES_DIR / "test"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        method_report_dir = REPORTS_DIR / method
        method_model_dir = MODELS_DIR / method
        method_report_dir.mkdir(parents=True, exist_ok=True)
        method_model_dir.mkdir(parents=True, exist_ok=True)
        try:
            X_tr, y_tr, _ = load_npz_from_split_dir(train_dir, method)
            X_te, y_te, _ = load_npz_from_split_dir(test_dir, method)
        except FileNotFoundError:
            log.warning("[SKIP] method=%s missing NPZ", method)
            continue
        for model_name in MODELS:
            pipe = build_pipeline_for_feature(
                model_name=model_name,
                feature_name=method,
                C_train=TRAIN_CFG.copy(),
                random_state=SEED,
                tuned_params=tuned_params,
            )
            pipe.fit(X_tr, y_tr)
            y_pred_tr = pipe.predict(X_tr)
            generate_single_run_report(
                y_true=y_tr, y_pred=y_pred_tr,
                output_dir=method_report_dir, model_name=model_name, report_type="train"
            )
            y_pred = pipe.predict(X_te)
            generate_single_run_report(
                y_true=y_te, y_pred=y_pred,
                output_dir=method_report_dir, model_name=model_name, report_type="test"
            )

    generate_summary_report(
        reports_root=REPORTS_DIR.as_posix(),
        set_name=SET,
        output_dir=SUMMARY_DIR.as_posix(),
        report_type="test",
        title=f"Test Performance Report (SET={SET})"
    )
    generate_summary_report(
        reports_root=REPORTS_DIR.as_posix(),
        set_name=SET,
        output_dir=SUMMARY_DIR.as_posix(),
        report_type="train",
        title=f"Train Performance Report (SET={SET})"
    )
    log.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()