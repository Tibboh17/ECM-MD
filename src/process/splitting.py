import csv
import numpy as np
from pathlib import Path

# -------------------- IO utils --------------------
def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "filename", "class", "source", "group_id"])
        w.writeheader()
        w.writerows(rows)

def _read_rows_from_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"image", "filename", "class", "source", "group_id"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in r:
            rows.append({
                "image": row["image"].replace("\\", "/").strip("/"),
                "filename": row["filename"],
                "class": row["class"],
                "source": row["source"],
                "group_id": row["group_id"],
            })
    if not rows:
        raise ValueError(f"No rows in CSV: {csv_path}")
    return rows

# -------------------- core helpers --------------------
def _permute_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx

# -------------------- Mode 1: ratio split --------------------
def split_dataset(input_csv, output_dir, seed=42, test_ratio=0.2,
                  train_name="train.csv", test_name="test.csv"):
    """
    Randomly split the GIVEN INPUT CSV into train/test by ratio (default test_ratio=0.2).
    Writes CSVs with custom filenames if provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows_from_csv(input_csv)
    n = len(rows)
    idx = _permute_indices(n, seed)

    n_test = int(round(n * float(test_ratio)))
    n_test = max(1, min(n - 1, n_test))  # ensure both non-empty

    test_rows = [rows[i] for i in idx[:n_test]]
    train_rows = [rows[i] for i in idx[n_test:]]

    train_csv = output_dir / train_name
    test_csv = output_dir / test_name
    _write_csv(train_csv, train_rows)
    _write_csv(test_csv, test_rows)

    print("[OK] CSV splits written to:", output_dir)
    print("  mode       : ratio")
    print("  test_ratio :", test_ratio)
    print(f"  {train_name:<12}: {len(train_rows)}")
    print(f"  {test_name:<12}: {len(test_rows)}")

# -------------------- Mode 2: fixed sizes --------------------
def split_dataset_by_size(input_csv, output_dir, train_size, test_size, seed=42,
                          train_name="train.csv", test_name="test.csv"):
    """
    Randomly split the GIVEN INPUT CSV into train/test by absolute sizes.
    - train_size, test_size: positive integers; train_size + test_size <= total rows
    - train_name, test_name: output CSV filenames (default train.csv/test.csv)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_size is None or test_size is None:
        raise ValueError("Both train_size and test_size must be provided.")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive integers.")

    rows = _read_rows_from_csv(input_csv)
    n = len(rows)

    if train_size + test_size > n:
        raise ValueError(
            f"Requested sizes exceed dataset size: "
            f"train({train_size}) + test({test_size}) > total({n})"
        )

    idx = _permute_indices(n, seed)

    train_rows = [rows[i] for i in idx[:train_size]]
    test_rows = [rows[i] for i in idx[train_size:train_size + test_size]]

    train_csv = output_dir / train_name
    test_csv = output_dir / test_name
    _write_csv(train_csv, train_rows)
    _write_csv(test_csv, test_rows)

    print("[OK] CSV splits written to:", output_dir)
    print("  mode       : fixed_sizes")
    print(f"  {train_name:<12}: {len(train_rows)}")
    print(f"  {test_name:<12}: {len(test_rows)}")

def merge_split_csvs(split_dir, out_train="train.csv", out_test="test.csv"):
    """
    Merge all *_train.csv and *_test.csv under split_dir into two combined CSVs.
    """
    split_dir = Path(split_dir)
    train_files = sorted(split_dir.glob("*_train.csv"))
    test_files  = sorted(split_dir.glob("*_test.csv"))

    if not train_files and not test_files:
        print(f"[WARN] No train/test CSVs found under {split_dir}")
        return None

    def collect(files):
        all_rows = []
        for f in files:
            try:
                rows = _read_rows_from_csv(f)
                all_rows.extend(rows)
            except Exception as e:
                print(f"[WARN] Skipped {f.name}: {e}")
        return all_rows

    train_rows = collect(train_files)
    test_rows  = collect(test_files)

    out_train_path = split_dir / out_train
    out_test_path  = split_dir / out_test

    if train_rows:
        _write_csv(out_train_path, train_rows)
        print(f"[OK] Merged {len(train_files)} train CSVs → {out_train_path} (rows={len(train_rows)})")
    if test_rows:
        _write_csv(out_test_path, test_rows)
        print(f"[OK] Merged {len(test_files)} test CSVs  → {out_test_path} (rows={len(test_rows)})")

    return out_train_path, out_test_path