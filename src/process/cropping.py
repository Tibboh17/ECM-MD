import cv2
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
Image.MAX_IMAGE_PIXELS = None  # Prevent DecompressionBombError

def _parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    w = int(float(root.find('size/width').text))
    h = int(float(root.find('size/height').text))
    fname = (root.findtext('filename') or '').strip()
    if not fname:
        fname = Path(xml_path).stem

    objects = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        bb = obj.find('bndbox')
        xmin = int(float(bb.findtext('xmin')))
        ymin = int(float(bb.findtext('ymin')))
        xmax = int(float(bb.findtext('xmax')))
        ymax = int(float(bb.findtext('ymax')))
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return root, fname, (w, h), objects

def _load_image(image_dir, xml_path, filename):
    base = Path(xml_path).parent if image_dir is None else Path(image_dir)
    stem = Path(filename).stem
    for ext in ["", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        cand = base / f"{stem}{ext}"
        if cand.exists():
            return np.array(Image.open(cand).convert("RGB")), cand
    raise FileNotFoundError(f"Image not found for XML: {xml_path} / filename: {filename}")

def _expand_bbox(b, W, H, margin):
    x1, y1, x2, y2 = b
    if margin <= 0:
        return max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    w = x2 - x1; h = y2 - y1
    dx = int(round(w * margin)); dy = int(round(h * margin))
    x1 -= dx; y1 -= dy; x2 += dx; y2 += dy
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)

def _square_bbox(b, W, H):
    x1, y1, x2, y2 = b
    w = x2 - x1; h = y2 - y1
    if w == h:
        return x1, y1, x2, y2
    if w > h:
        pad = (w - h) / 2
        y1 = int(round(y1 - pad))
        y2 = int(round(y2 + pad))
    else:
        pad = (h - w) / 2
        x1 = int(round(x1 - pad))
        x2 = int(round(x2 + pad))
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    return x1, y1, x2, y2

def _crop(img, b):
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, img.shape[1]-1))
    y1 = max(0, min(y1, img.shape[0]-1))
    x2 = max(x1+1, min(x2, img.shape[1]))
    y2 = max(y1+1, min(y2, img.shape[0]))
    return img[y1:y2, x1:x2].copy()

def _maybe_resize(patch, resize_to):
    if not resize_to:
        return patch
    if isinstance(resize_to, int):
        size = (resize_to, resize_to)
    else:
        size = tuple(resize_to)
        if len(size) != 2:
            raise ValueError("resize_to must be int or (w,h)")
    return cv2.resize(patch, size, interpolation=cv2.INTER_AREA)

def iter_objects(xml_path):
    _, _, _, objects = _parse_voc(xml_path)
    for name, bb in objects:
        yield name, bb

def crop_from_xml(
    xml_path,
    image_dir=None,
    out_root="datasets/crops",
    class_filter=None,
    margin=0.0,
    square=False,
    resize_to=None,
    min_size=4,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
    xml_path = Path(xml_path)
    root, filename, (W, H), objects = _parse_voc(xml_path)
    img, img_path = _load_image(image_dir, xml_path, filename)
    base_out = Path(out_root) / xml_path.stem
    base_out.mkdir(parents=True, exist_ok=True)

    counters = {}
    saved = 0
    logger.info("Cropping from %s (image=%s, objects=%d)", xml_path.name, img_path.name, len(objects))

    for name, bb in objects:
        if class_filter and name not in class_filter:
            continue
        b = _expand_bbox(bb, W, H, margin)
        if square:
            b = _square_bbox(b, W, H)
        patch = _crop(img, b)
        if patch.shape[0] < min_size or patch.shape[1] < min_size:
            logger.debug("Skip tiny crop: %s %s", name, str(b))
            continue
        patch = _maybe_resize(patch, resize_to)

        out_dir = base_out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        counters[name] = counters.get(name, 0) + 1
        idx = counters[name]
        out_path = out_dir / f"{name}_{idx:04d}.png"
        Image.fromarray(patch).save(out_path)
        saved += 1

    logger.info("Saved %d crops under %s", saved, base_out)
    for k, v in counters.items():
        logger.info("  - %s: %d", k, v)
    return str(base_out)
