import cv2
import random
import logging
import shutil
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

# -----------------------
# Logger
# -----------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -----------------------
# Utilities
# -----------------------
def _parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    w = int(float(root.find('size/width').text))
    h = int(float(root.find('size/height').text))
    fname = (root.findtext('filename') or '').strip()
    if not fname:
        fname = Path(xml_path).stem

    boxes = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip().lower()
        bb = obj.find('bndbox')
        xmin = int(float(bb.findtext('xmin')))
        ymin = int(float(bb.findtext('ymin')))
        xmax = int(float(bb.findtext('xmax')))
        ymax = int(float(bb.findtext('ymax')))
        boxes.append((name, (xmin, ymin, xmax, ymax)))
    return root, fname, (w, h), boxes

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = a_area + b_area - inter
    return inter / u if u > 0 else 0.0

def _central_exclusion_box(w, h, percent):
    """Exclude a vertical band across the center with full height. percent: 0~100 (width %)"""
    margin = (100 - percent) / 2.0 / 100.0
    x1 = int(w * margin)
    x2 = int(w * (1 - margin))
    return (x1, 0, x2, h)

def _in_box_center(bx, band):
    """Return True if the center of bx lies inside band."""
    x1, y1, x2, y2 = bx
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bx1, by1, bx2, by2 = band
    return (bx1 <= cx < bx2) and (by1 <= cy < by2)

def _extract_patch_features(patch_gray, n_points=16, radius=2):
    lbp = local_binary_pattern(patch_gray, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    g_mean = float(np.mean(gmag))
    g_std = float(np.std(gmag))

    lap = cv2.Laplacian(patch_gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    mean = float(np.mean(patch_gray))
    std = float(np.std(patch_gray))

    return np.hstack([hist, [g_mean, g_std, lap_var, mean, std]]).astype(np.float32)

def _load_image_gray(img_path):
    """Robust loader: EXIF orientation, alpha drop, 16-bit scaling."""
    im = Image.open(img_path)
    im = ImageOps.exif_transpose(im)

    # Drop alpha if present
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")

    # 16-bit → 8-bit linear scaling
    if im.mode == "I;16" or (im.mode == "I" and np.array(im).dtype in (np.int32, np.int64)):
        arr16 = np.array(im, dtype=np.uint16)
        if arr16.max() > 0:
            arr8 = (arr16.astype(np.float32) / arr16.max() * 255.0).astype(np.uint8)
        else:
            arr8 = arr16.astype(np.uint8)
        img_gray = arr8
    else:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        arr = np.array(im)
        if arr.ndim == 3:
            img_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = arr.astype(np.uint8)

    return img_gray, im.size  # (W_img, H_img)

def _scale_box_to_image(bx, xml_size, img_size):
    (W_xml, H_xml) = xml_size
    (W_img, H_img) = img_size
    sx = W_img / float(W_xml)
    sy = H_img / float(H_xml)
    x1, y1, x2, y2 = bx
    x1 = int(round(x1 * sx)); x2 = int(round(x2 * sx))
    y1 = int(round(y1 * sy)); y2 = int(round(y2 * sy))
    # clamp
    x1 = max(0, min(W_img - 1, x1)); x2 = max(0, min(W_img, x2))
    y1 = max(0, min(H_img - 1, y1)); y2 = max(0, min(H_img, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def _kmeans_labels(feats, k, seed):
    if len(feats) < 2 or k < 2:
        return np.zeros(len(feats), dtype=np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(X)

# -----------------------
# Core functions
# -----------------------
def generate_terrain_labels(
    xml_path,
    image_dir=None,
    out_suffix="_with_terrain",
    patch_size=60,
    stride=40,
    num_terrain=None,
    n_clusters=6,
    central_exclusion_percent=40,
    iou_exclusion_thresh=0.10,
    overlap_ok=0.1,   # allow tiny overlaps between selected boxes
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    root, filename, (W_xml, H_xml), boxes = _parse_voc(xml_path)

    # count mines & existing terrain
    mine_boxes = [b for name, b in boxes if name == "mine"]
    terrain_boxes = [b for name, b in boxes if name == "terrain"]
    mine_count = len(mine_boxes)
    if num_terrain is None:
        num_terrain = max(1, mine_count * 5)
    logger.info(f"{Path(xml_path).name}: mine={mine_count}, terrain_existing={len(terrain_boxes)}, terrain_target={num_terrain}")

    # image path resolution
    if image_dir is None:
        img_path = Path(xml_path).with_name(filename)
        if not img_path.exists():
            # try common extensions using same basename
            for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                p = Path(xml_path).with_suffix(ext)
                if p.exists():
                    img_path = p
                    break
    else:
        img_path = Path(image_dir) / filename
        if not img_path.exists():
            # try with different extensions under image_dir
            stem = Path(filename).stem
            for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                p = Path(image_dir) / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break

    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # robust load
    img_gray, (W_img, H_img) = _load_image_gray(img_path)

    # --- 금지영역 구성: '기뢰 박스'와 '중앙 밴드'를 분리 ---
    forbidden_mines = [_scale_box_to_image(b, (W_xml, H_xml), (W_img, H_img)) for b in mine_boxes]
    central_band = _central_exclusion_box(W_img, H_img, central_exclusion_percent)

    # candidate patches on real image grid
    cand_boxes, feats = [], []
    for y in range(0, max(1, H_img - patch_size + 1), stride):
        for x in range(0, max(1, W_img - patch_size + 1), stride):
            bx = (x, y, x + patch_size, y + patch_size)

            # ✅ 방법 A: 중앙 밴드에는 '패치 중심점'이 들어오면 무조건 제외
            if central_exclusion_percent > 0 and _in_box_center(bx, central_band):
                continue

            # 기존처럼 기뢰 박스와는 IoU로 충돌 제거
            if any(_iou(bx, fb) > iou_exclusion_thresh for fb in forbidden_mines):
                continue

            patch = img_gray[y:y + patch_size, x:x + patch_size]
            if patch.shape[:2] != (patch_size, patch_size):
                continue
            feats.append(_extract_patch_features(patch))
            cand_boxes.append(bx)

    if not cand_boxes:
        logger.warning(f"No terrain candidates found in {xml_path}")
        return None

    feats = np.asarray(feats, dtype=np.float32)
    k = min(n_clusters, len(cand_boxes))
    labels = _kmeans_labels(feats, k, seed)

    # per-cluster selection (first pass)
    selected = []
    used = list(forbidden_mines)  # 중앙 밴드는 '중심 제외'로 이미 처리했으므로 여기엔 포함하지 않음
    per = max(1, int(np.ceil(num_terrain / max(1, k))))

    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        np.random.shuffle(idxs)
        count = 0
        for j in idxs:
            bx = cand_boxes[j]
            if any(_iou(bx, u) > overlap_ok for u in used):
                continue
            selected.append(bx)
            used.append(bx)
            count += 1
            if len(selected) >= num_terrain or count >= per:
                break
        if len(selected) >= num_terrain:
            break

    # second pass: fill remaining slots from all candidates if needed
    if len(selected) < num_terrain:
        all_idxs = list(range(len(cand_boxes)))
        np.random.shuffle(all_idxs)
        for j in all_idxs:
            bx = cand_boxes[j]
            if any(_iou(bx, u) > overlap_ok for u in used):
                continue
            selected.append(bx)
            used.append(bx)
            if len(selected) >= num_terrain:
                break

    # append to XML (image-size coords must be mapped back to XML size if different)
    def _scale_box_to_xml(bx_img, xml_size, img_size):
        (W_xml, H_xml) = xml_size
        (W_img, H_img) = img_size
        sx = W_xml / float(W_img)
        sy = H_xml / float(H_img)
        x1, y1, x2, y2 = bx_img
        x1 = int(round(x1 * sx)); x2 = int(round(x2 * sx))
        y1 = int(round(y1 * sy)); y2 = int(round(y2 * sy))
        # clamp
        x1 = max(0, min(W_xml - 1, x1)); x2 = max(0, min(W_xml, x2))
        y1 = max(0, min(H_xml - 1, y1)); y2 = max(0, min(H_xml, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    for (xmin_i, ymin_i, xmax_i, ymax_i) in selected[:num_terrain]:
        xmin, ymin, xmax, ymax = _scale_box_to_xml((xmin_i, ymin_i, xmax_i, ymax_i), (W_xml, H_xml), (W_img, H_img))
        obj = ET.Element('object')
        ET.SubElement(obj, 'name').text = 'terrain'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bb = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bb, 'xmin').text = str(int(xmin))
        ET.SubElement(bb, 'ymin').text = str(int(ymin))
        ET.SubElement(bb, 'xmax').text = str(int(xmax))
        ET.SubElement(bb, 'ymax').text = str(int(ymax))
        root.append(obj)

    out_path = Path(xml_path).with_name(Path(xml_path).stem + out_suffix + ".xml")
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)
    logger.info(f"Saved terrain-added XML -> {out_path}")
    return str(out_path)

def generate_terrain_for_dir(
    xml_dir,
    image_dir,
    out_dir,
    pattern="*.xml",
    patch_size=60,
    stride=40,
    num_terrain=None,
    n_clusters=6,
    central_exclusion_percent=40,
    iou_exclusion_thresh=0.10,
    overlap_ok=0.02,
    seed=42
):
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_list = sorted(xml_dir.glob(pattern))
    if not xml_list:
        logger.error(f"No XML found in {xml_dir}")
        return []

    outputs = []
    for i, xml_path in enumerate(xml_list, 1):
        logger.info(f"[{i}/{len(xml_list)}] Generating terrain: {xml_path.name}")
        out_xml = generate_terrain_labels(
            xml_path=str(xml_path),
            image_dir=str(image_dir),
            out_suffix="_with_terrain",
            patch_size=patch_size,
            stride=stride,
            num_terrain=num_terrain,
            n_clusters=n_clusters,
            central_exclusion_percent=central_exclusion_percent,
            iou_exclusion_thresh=iou_exclusion_thresh,
            overlap_ok=overlap_ok,
            seed=seed,
        )
        if out_xml:
            dst = out_dir / Path(out_xml).name
            try:
                shutil.move(out_xml, dst)
            except Exception:
                # fallback: same volume replace
                Path(out_xml).replace(dst)
            outputs.append(str(dst))
            logger.info(f"MOVED XML -> {dst}")

    logger.info(f"DONE: {len(outputs)} terrain XML(s) created at {out_dir}")
    return outputs