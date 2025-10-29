# data_loader.py
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_feature_key(npz_obj) -> str:
    """
    Return the key for the feature matrix inside an npz.
    Prefer project helper `utils.find_feature_key` if available; otherwise fallback.
    """
    # Try project helper first (optional)
    try:
        from utils import find_feature_key  # type: ignore
        k = find_feature_key(npz_obj)
        if k:
            return k
    except Exception:
        pass

    # Fallback: common keys
    for k in ("X", "features", "feat", "data"):
        if k in npz_obj.files:
            return k

    # Last resort: first non-meta array
    meta = {"labels", "y", "filenames", "paths", "label_map"}
    for k in npz_obj.files:
        if k not in meta:
            return k

    raise KeyError("Could not determine feature key in NPZ.")


def _extract_filenames(npz_obj, n_rows: int):
    """Get filenames/paths if present; otherwise synthesize stable row IDs."""
    if "filenames" in npz_obj.files:
        return npz_obj["filenames"]
    if "paths" in npz_obj.files:
        return npz_obj["paths"]
    return np.array([f"row_{i:06d}" for i in range(n_rows)], dtype=object)


def _extract_label_map(npz_obj):
    """Normalize label_map to dict or None."""
    if "label_map" not in npz_obj.files:
        return None
    lm = npz_obj["label_map"]
    # common: object array holding a dict
    try:
        return lm.item() if hasattr(lm, "item") else dict(lm)
    except Exception:
        try:
            return dict(lm.tolist())
        except Exception:
            return None


def _build_groups_from_filenames(filenames) -> np.ndarray:
    """
    Build instance-level groups from filename stems.
    Matches e.g. 'mine_0001' or 'terrain_0042'; falls back to full stem.
    """
    pat = re.compile(r"(mine|terrain)_\d+")
    groups = []
    for f in filenames:
        stem = Path(str(f)).stem
        m = pat.search(stem)
        groups.append(m.group(0) if m else stem)
    return np.asarray(groups, dtype=object)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_and_split_data(
    feature_path,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_mode: bool = False,
):
    """
    Load a single NPZ feature file and split into train/test with GroupShuffleSplit.

    WARNING:
      - DO NOT use in CV mode. Set `cv_mode=True` to hard-block accidental use.
      - For cross-validation, use pre-split loaders (foldK_train/foldK_val).

    Returns:
      (X_train, X_test, y_train, y_test), label_map, groups_train, groups_test
      or None on failure.
    """
    # Guard for CV mode
    if cv_mode:
        raise RuntimeError(
            "load_and_split_data() is disabled in CV mode. "
            "Use pre-split loaders (foldK_train/foldK_val)."
        )

    # Path checks
    if not isinstance(feature_path, Path):
        feature_path = Path(feature_path)
    if not feature_path.exists():
        logging.error(f"[ERROR] Feature file not found: {feature_path}")
        return None
    if not (0.0 < float(test_size) < 1.0):
        logging.warning(f"[WARN] Invalid test_size={test_size}. Fallback to 0.2")
        test_size = 0.2

    # Load NPZ
    try:
        npz = np.load(feature_path, allow_pickle=True)
    except Exception as e:
        logging.error(f"[ERROR] Failed to load NPZ: {feature_path} ({e})")
        return None

    # Resolve keys & extract arrays
    try:
        feat_key = _resolve_feature_key(npz)
        X = npz[feat_key]
        y = npz["labels"] if "labels" in npz.files else npz["y"]  # supports 'labels' or 'y'
        filenames = _extract_filenames(npz, len(y))
        label_map = _extract_label_map(npz)

        logging.info(
            f"[INFO] Loaded features: key='{feat_key}', N={len(y)}, "
            f"D={X.shape[1] if X.ndim == 2 else 'NA'} from '{feature_path.name}'"
        )
    except Exception as e:
        logging.error(f"[ERROR] Failed to parse arrays from NPZ: {e}")
        return None

    # Basic shape checks
    if len(X) != len(y):
        logging.error(f"[ERROR] Length mismatch: X={len(X)} vs y={len(y)}")
        return None

    # Groups
    groups = _build_groups_from_filenames(filenames)

    # Group-aware split
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
    except Exception as e:
        logging.error(f"[ERROR] Group split failed: {e}")
        return None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    logging.info(
        f"[INFO] Split done | train={len(train_idx)} test={len(test_idx)} "
        f"(groups: train={len(set(g_train))}, test={len(set(g_test))})"
    )

    return (X_train, X_test, y_train, y_test), label_map, g_train, g_test


def load_inference_features(feature_path: Path):
    """
    Load pre-computed features for inference.
    Returns:
      features (np.ndarray or None), filenames (list[str] or None)
    """
    if not isinstance(feature_path, Path):
        feature_path = Path(feature_path)

    if not feature_path.exists():
        logging.warning(
            f"[WARN] Pre-computed feature file not found: {feature_path}. "
            f"Will attempt on-the-fly extraction."
        )
        return None, None

    try:
        npz = np.load(feature_path, allow_pickle=True)
        feat_key = _resolve_feature_key(npz)
        features = npz[feat_key]
        names_arr = _extract_filenames(npz, len(features))
        filenames = [Path(str(f)).name for f in names_arr]
        logging.info(f"[INFO] Loaded {len(filenames)} pre-computed features from {feature_path}")
        return features, filenames
    except Exception as e:
        logging.warning(
            f"[WARN] Could not load pre-computed features from {feature_path}: {e}. "
            f"Will attempt on-the-fly extraction."
        )
        return None, None


def load_npz_from_split_dir(split_dir, method):
    """
    Load features saved per split/fold.
    Expected file: <split_dir>/<method>.npz
      - keys: X (float32 ndarray), y (int or str), label_map (optional)
    Returns:
      X, y, label_map(dict or None)
    """
    split_dir = Path(split_dir)
    npz_path = split_dir / f"{method}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    # Be lenient about key names, but prefer explicit "X" / "y"
    X = data["X"] if "X" in data.files else data[_resolve_feature_key(data)]
    y = data["y"] if "y" in data.files else data["labels"]
    label_map = _extract_label_map(data)
    return X, y, label_map
