import csv
from pathlib import Path

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _iter_images(root, recursive=True, pattern=None, valid_ext=None):
    root = Path(root)
    exts = set((valid_ext or _VALID_EXTS))
    exts = {e.lower() for e in exts}

    if pattern:
        files = [p for p in root.glob(pattern) if p.is_file()]
    else:
        if recursive:
            files = [p for p in root.rglob("*") if p.is_file()]
        else:
            files = [p for p in root.glob("*") if p.is_file()]

    for p in files:
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p

def _scan_crops(crops_root, recursive=True, pattern=None, valid_ext=None, print_summary=True):
    crops_root = Path(crops_root)
    rows = []
    per_class = {}
    per_source = {}

    source_dirs = [d for d in crops_root.glob("*") if d.is_dir() and not d.name.startswith(".")]
    for sdir in sorted(source_dirs, key=lambda p: p.name):
        source = sdir.name
        class_dirs = [c for c in sdir.glob("*") if c.is_dir() and not c.name.startswith(".")]
        for cdir in sorted(class_dirs, key=lambda p: p.name):
            cls = cdir.name
            for p in _iter_images(cdir, recursive=recursive, pattern=pattern, valid_ext=valid_ext):
                fname = p.name
                image_rel = f"{source}/{cls}/{fname}"
                rows.append((image_rel, fname, cls, source, source))
                per_class[cls] = per_class.get(cls, 0) + 1
                per_source[source] = per_source.get(source, 0) + 1

    rows.sort(key=lambda r: (r[3], r[2], r[1]))  # source, class, filename

    seen = set()
    de_duplicated = []
    for r in rows:
        key = r[0]  # image
        if key in seen:
            continue
        seen.add(key)
        de_duplicated.append(r)
    rows = de_duplicated

    if print_summary:
        total = len(rows)
        print(f"[OK] Scanned crops under {crops_root} (rows={total})")
        for src in sorted(per_source):
            print(f"  - source={src}: {per_source[src]}")
        for cls in sorted(per_class):
            print(f"  - class={cls}: {per_class[cls]}")

    classes = sorted({r[2] for r in rows})
    sources = sorted({r[3] for r in rows})
    return rows, classes, sources

def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "filename", "class", "source", "group_id"])
        w.writerows(rows)

def build_label_sets(
    crops_root,
    out_dir,
    make_all=True,
    make_per_class=True,
    make_per_source=False,
    recursive=True,
    pattern=None,
    valid_ext=None,
    print_summary=True,
):
    crops_root = Path(crops_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, classes, sources = _scan_crops(
        crops_root=crops_root,
        recursive=recursive,
        pattern=pattern,
        valid_ext=valid_ext,
        print_summary=print_summary,
    )

    if not rows:
        print(f"[WARN] No valid images found under {crops_root}. Nothing to write.")
        return {}

    written = {}

    if make_all:
        all_csv = out_dir / "all_labels.csv"
        _write_csv(all_csv, rows)
        written["all"] = str(all_csv)
        print(f"[OK] Wrote all labels → {all_csv} (rows={len(rows)})")

    if make_per_class and classes:
        by_cls = {}
        for r in rows:
            by_cls.setdefault(r[2], []).append(r)
        for cls in classes:
            cls_rows = by_cls.get(cls, [])
            cls_csv = out_dir / f"{cls}.csv"
            _write_csv(cls_csv, cls_rows)
            written[f"class:{cls}"] = str(cls_csv)
            print(f"[OK] Wrote class labels → {cls_csv} (rows={len(cls_rows)})")

    if make_per_source and sources:
        by_src = {}
        for r in rows:
            by_src.setdefault(r[3], []).append(r)
        for src in sources:
            src_rows = by_src.get(src, [])
            # 파일명 충돌 방지: 공백/슬래시 제거
            safe_src = src.replace("/", "_").replace(" ", "_")
            src_csv = out_dir / f"source_{safe_src}.csv"
            _write_csv(src_csv, src_rows)
            written[f"source:{src}"] = str(src_csv)
            print(f"[OK] Wrote source labels → {src_csv} (rows={len(src_rows)})")

    return written
