# src/process/augmentation.py
import csv
import random
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("augmentation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

_VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------- I/O helpers ----------------
def _load_image(p):
    return Image.open(p).convert("RGB")

def _save_image(img, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def _resolve_path(crops_root, image_path):
    p = Path(image_path)
    if p.is_absolute():
        return p
    return Path(crops_root) / p

# ---------------- Aug config & ops ----------------
class AugConfig:
    def __init__(self,
                 hflip_p=0.5,
                 vflip_p=0.2,
                 rot_choices=(0, 90, 180, 270),
                 jitter_brightness=(0.8, 1.2),
                 jitter_contrast=(0.8, 1.2),
                 blur_p=0.2,
                 noise_p=0.2):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot_choices = rot_choices
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.blur_p = blur_p
        self.noise_p = noise_p

def _rand_uniform(a, b):
    return a + (b - a) * random.random()

def _apply_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def _apply_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def _apply_noise(img):
    arr = np.array(img).astype(np.float32)
    sigma = _rand_uniform(2.0, 8.0)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_augmentations(img, cfg):
    angle = random.choice(cfg.rot_choices)
    aug = img.rotate(angle)

    if random.random() < cfg.hflip_p:
        aug = ImageOps.mirror(aug)
    if random.random() < cfg.vflip_p:
        aug = ImageOps.flip(aug)

    aug = _apply_brightness(aug, _rand_uniform(cfg.jitter_brightness[0], cfg.jitter_brightness[1]))
    aug = _apply_contrast(aug, _rand_uniform(cfg.jitter_contrast[0], cfg.jitter_contrast[1]))

    if random.random() < cfg.blur_p:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=_rand_uniform(0.5, 1.5)))

    if random.random() < cfg.noise_p:
        aug = _apply_noise(aug)

    return aug

def augment_from_split_csv(
    split_csv,
    crops_root,
    out_root,
    labels_out_dir=None,
    num_augmentations=None,
    cfg=None,
    seed=None,
    limit=None,
    name_prefix=None,
    include_original=True,
    classes_to_augment=("mine",)
):
    np.random.seed(seed if seed is not None else 0)
    cfg = cfg or AugConfig()

    split_csv = Path(split_csv)
    crops_root = Path(crops_root)
    out_root = Path(out_root)

    if not split_csv.exists():
        raise FileNotFoundError(split_csv.as_posix())

    if name_prefix is None:
        name_prefix = split_csv.stem

    df = pd.read_csv(split_csv)

    if {"image", "class", "source"}.issubset(df.columns):
        rows = df.copy()
    elif {"image_path", "label", "group"}.issubset(df.columns):
        rows = df.rename(columns={"image_path": "image", "label": "class", "group": "source"})
    else:
        raise ValueError(
            f"{split_csv} must contain columns: image,class,source "
            f"(optionally group_id,filename) or legacy (image_path,label,group)"
        )

    rows["image"] = rows["image"].astype(str).str.replace("\\\\", "/", regex=True).str.strip().str.strip("/")
    rows["class"] = rows["class"].astype(str).str.strip()
    rows["source"] = rows["source"].astype(str).str.replace("\\\\", "/", regex=True).str.strip().str.strip("/")

    if "group_id" not in rows.columns:
        rows["group_id"] = rows["source"]
    else:
        rows["group_id"] = rows["group_id"].astype(str).str.replace("\\\\", "/", regex=True).str.strip().str.strip("/")

    if "filename" not in rows.columns:
        rows["filename"] = rows["image"].apply(lambda p: Path(str(p)).name)
    else:
        rows["filename"] = rows["filename"].astype(str).str.replace("\\\\", "/", regex=True).apply(lambda p: Path(p).name)

    if limit:
        rows = rows.head(int(limit))

    label_writer = None
    label_csv_path = None
    f = None
    if labels_out_dir:
        labels_out_dir = Path(labels_out_dir)
        labels_out_dir.mkdir(parents=True, exist_ok=True)
        label_csv_path = labels_out_dir / f"{split_csv.stem}_aug.csv"
        f = open(label_csv_path, "w", newline="", encoding="utf-8")
        label_writer = csv.writer(f)
        label_writer.writerow(["image", "filename", "class", "source", "group_id"])

    logger.info("[AUG] split=%s rows=%d out_root=%s classes_to_augment=%s",
                split_csv.name, len(rows), out_root.as_posix(), list(classes_to_augment))

    n_ok, n_skip, n_aug_mine, n_orig_only = 0, 0, 0, 0

    for _, r in rows.iterrows():
        rel_in  = str(r["image"]).strip().strip("/")
        cls     = str(r["class"]).strip()
        src_lbl = str(r["source"]).strip().strip("/")
        gid_lbl = str(r["group_id"]).strip().strip("/")
        fname   = str(r["filename"]).strip()
        src_dir = Path(src_lbl).name

        in_path = crops_root / rel_in

        if not in_path.exists():
            alt_path = crops_root / src_lbl / cls / fname
            if alt_path.exists():
                in_path = alt_path

        if not in_path.exists():
            logger.warning("[SKIP] not found: %s", in_path.as_posix())
            n_skip += 1
            continue

        try:
            img = _load_image(in_path)
            stem = Path(fname).stem

            out_dir = out_root / src_dir / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            if include_original:
                fn = f"{name_prefix}__{stem}__orig.png"
                out_path = out_dir / fn
                _save_image(img, out_path)
                if label_writer:
                    rel_out = f"{src_dir}/{cls}/{fn}"
                    label_writer.writerow([rel_out, fn, cls, src_lbl, gid_lbl])
                n_orig_only += (cls not in classes_to_augment)

            if cls in classes_to_augment and (num_augmentations or 0) > 0:
                for k in range(num_augmentations):
                    aug = apply_augmentations(img, cfg)
                    fn = f"{name_prefix}__{stem}__aug{k+1}.png"
                    out_path = out_dir / fn
                    _save_image(aug, out_path)
                    if label_writer:
                        rel_out = f"{src_dir}/{cls}/{fn}"
                        label_writer.writerow([rel_out, fn, cls, src_lbl, gid_lbl])
                n_aug_mine += 1

            n_ok += 1

        except Exception as e:
            logger.warning("[SKIP] failed %s: %s", in_path.as_posix(), e)
            n_skip += 1

    if f:
        f.close()
        logger.info("[AUG] labels written -> %s", label_csv_path.as_posix())

    logger.info("[DONE] Augmented rows=%d (mine_aug=%d), orig_only_rows=%d, skipped=%d",
                n_ok, n_aug_mine, n_orig_only, n_skip)
    return {"ok": n_ok, "skip": n_skip, "labels_csv": str(label_csv_path) if label_csv_path else None}