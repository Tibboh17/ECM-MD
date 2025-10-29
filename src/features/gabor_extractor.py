import math
import logging
import warnings
import numpy as np
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- optional deps (safe import) ----
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from skimage.filters import gabor as sk_gabor, gabor_kernel as sk_gabor_kernel
    from skimage import exposure as sk_exposure
    SKIMAGE_GABOR_AVAILABLE = True
except Exception:
    SKIMAGE_GABOR_AVAILABLE = False

# =========================================================
# Utilities (pure)
# =========================================================
def rgb_to_grayscale_pure(image):
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])  # BT.709
    return image

def adaptive_hist_eq_pure(image, nbins=256):
    x = image.astype(np.float64)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    hist, bins = np.histogram(x.flatten(), nbins, [0, 1])
    cdf = hist.cumsum()
    cdf = cdf / (cdf[-1] + 1e-12)
    out = np.interp(x.flatten(), bins[:-1], cdf).reshape(image.shape)
    return out.astype(np.float64)

def create_gabor_kernel_pure(frequency, theta, sigma_x, sigma_y, n_stds=3.0, offset=0.0, kernel_size=None):
    if kernel_size is None:
        size_x = int(2 * n_stds * sigma_x + 1)
        size_y = int(2 * n_stds * sigma_y + 1)
        kernel_size = max(size_x, size_y)
        if kernel_size % 2 == 0:
            kernel_size += 1

    c = kernel_size // 2
    x = np.arange(-c, c + 1, dtype=np.float64)
    y = np.arange(-c, c + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    ct = math.cos(theta)
    st = math.sin(theta)
    x_theta = X * ct + Y * st
    y_theta = -X * st + Y * ct

    gauss = np.exp(-0.5 * ((x_theta / sigma_x) ** 2 + (y_theta / sigma_y) ** 2))
    sinus = np.exp(1j * (2 * math.pi * frequency * x_theta + offset))
    return gauss * sinus  # complex kernel


def apply_gabor_filter_pure(image, kernel):
    real_response = convolve(image, np.real(kernel), mode="constant")
    imag_response = convolve(image, np.imag(kernel), mode="constant")
    mag = np.sqrt(real_response ** 2 + imag_response ** 2)
    phase = np.arctan2(imag_response, real_response)
    return mag, phase

# =========================================================
# Filter bank
# =========================================================
class GaborConfig:
    def __init__(self, frequency=0.1, theta=0.0, sigma_x=2.0, sigma_y=2.0, n_stds=3.0, offset=0.0):
        self.frequency = max(0.01, min(float(frequency), 1.0))
        self.theta = float(theta) % (2 * np.pi)
        self.sigma_x = max(0.5, min(float(sigma_x), 10.0))
        self.sigma_y = max(0.5, min(float(sigma_y), 10.0))
        self.n_stds = float(n_stds)
        self.offset = float(offset)

class OptimizedGaborBank:
    """
    Optimized bank to manage multi-frequency, multi-orientation Gabor filters.
    """

    def __init__(self,
                 n_frequencies=6,
                 n_orientations=8,
                 frequency_range=(0.01, 0.3),
                 sigma_range=(1.0, 4.0),
                 compute_phase=True,
                 use_parallel=True):
        self.n_frequencies = int(n_frequencies)
        self.n_orientations = int(n_orientations)
        self.frequency_range = tuple(frequency_range)
        self.sigma_range = tuple(sigma_range)
        self.compute_phase = bool(compute_phase)
        self.use_parallel = bool(use_parallel)

        self.frequencies = np.logspace(np.log10(self.frequency_range[0]),
                                       np.log10(self.frequency_range[1]),
                                       self.n_frequencies)
        self.orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        self.sigmas = np.linspace(self.sigma_range[0], self.sigma_range[1], self.n_frequencies)

        self.filter_bank = self._create_filter_bank()
        logger.info(f"[INIT] Gabor bank: {self.n_frequencies} freqs × {self.n_orientations} orients = {len(self.filter_bank)} filters")

    def _create_filter_bank(self):
        bank = []
        for i, freq in enumerate(self.frequencies):
            for theta in self.orientations:
                cfg = GaborConfig(frequency=freq, theta=theta, sigma_x=self.sigmas[i], sigma_y=self.sigmas[i])
                if SKIMAGE_GABOR_AVAILABLE:
                    try:
                        kernel = sk_gabor_kernel(
                            frequency=cfg.frequency,
                            theta=cfg.theta,
                            sigma_x=cfg.sigma_x,
                            sigma_y=cfg.sigma_y,
                            n_stds=cfg.n_stds,
                            offset=cfg.offset
                        )
                    except Exception as e:
                        logger.debug(f"[GABOR] skimage gabor_kernel failed ({e}); using pure fallback")
                        kernel = create_gabor_kernel_pure(cfg.frequency, cfg.theta, cfg.sigma_x, cfg.sigma_y, cfg.n_stds, cfg.offset)
                else:
                    kernel = create_gabor_kernel_pure(cfg.frequency, cfg.theta, cfg.sigma_x, cfg.sigma_y, cfg.n_stds, cfg.offset)
                bank.append((kernel, cfg))
        return bank

    def apply_single_filter(self, image, kernel):
        try:
            if np.iscomplexobj(kernel):
                return apply_gabor_filter_pure(image, kernel)
            real_response = convolve(image, np.real(kernel), mode="constant")
            imag_response = convolve(image, np.imag(kernel), mode="constant") if hasattr(kernel, "imag") else np.zeros_like(real_response)
            mag = np.sqrt(real_response ** 2 + imag_response ** 2)
            phase = np.arctan2(imag_response, real_response)
            return mag, phase
        except Exception as e:
            logger.warning(f"[WARN] Gabor application failed: {e}")
            return np.zeros_like(image), np.zeros_like(image)

    def _prep_image(self, image):
        img = image
        if img.ndim == 3:
            if CV2_AVAILABLE:
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                except Exception:
                    img = rgb_to_grayscale_pure(img)
            else:
                img = rgb_to_grayscale_pure(img)
        if SKIMAGE_GABOR_AVAILABLE:
            try:
                img = sk_exposure.equalize_adapthist(img)
            except Exception:
                img = adaptive_hist_eq_pure(img)
        else:
            img = adaptive_hist_eq_pure(img)
        return img.astype(np.float64)

    def extract_filter_responses(self, image):
        """
        Returns: dict { f{freq:.3f}_t{theta:.2f}_mag: 2D, (optional) _phase: 2D }
        """
        img = self._prep_image(image)
        responses = {}

        def process(entry):
            kernel, cfg = entry
            name = f"f{cfg.frequency:.3f}_t{cfg.theta:.2f}"
            try:
                if SKIMAGE_GABOR_AVAILABLE:
                    try:
                        mag, ph = sk_gabor(img, frequency=cfg.frequency, theta=cfg.theta,
                                           sigma_x=cfg.sigma_x, sigma_y=cfg.sigma_y)
                    except Exception:
                        mag, ph = self.apply_single_filter(img, kernel)
                else:
                    mag, ph = self.apply_single_filter(img, kernel)
                return name, mag, ph
            except Exception as e:
                logger.warning(f"[WARN] Filter {name} failed: {e}")
                z = np.zeros_like(img)
                return name, z, z

        if self.use_parallel and len(self.filter_bank) > 4:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(process, ent) for ent in self.filter_bank]
                for fut in as_completed(futures):
                    name, mag, ph = fut.result()
                    responses[f"{name}_mag"] = mag
                    if self.compute_phase:
                        responses[f"{name}_phase"] = ph
        else:
            for ent in self.filter_bank:
                name, mag, ph = process(ent)
                responses[f"{name}_mag"] = mag
                if self.compute_phase:
                    responses[f"{name}_phase"] = ph

        return responses

# =========================================================
# Feature extractor (pipeline-facing)
# =========================================================
class GaborFeatureExtractor:
    """
    Pipeline-facing extractor
    - expose extract_comprehensive_features(image) -> (D,)
    """

    def __init__(self,
                 n_frequencies=6,
                 n_orientations=8,
                 patch_size=32,
                 frequency_min=0.02,
                 frequency_max=0.35,
                 compute_phase=False,
                 use_parallel=True,
                 sigma_min=1.0,
                 sigma_max=4.0):
        self.patch_size = int(patch_size)
        self.bank = OptimizedGaborBank(
            n_frequencies=n_frequencies,
            n_orientations=n_orientations,
            frequency_range=(frequency_min, frequency_max),
            sigma_range=(sigma_min, sigma_max),
            compute_phase=compute_phase,
            use_parallel=use_parallel,
        )
        logger.info(f"[INIT] GaborFeatureExtractor ready (patch={self.patch_size}, phase={compute_phase})")

    # ---- stats helpers ----
    def _skew(self, x):
        if x.size < 2:
            return 0.0
        m = np.mean(x); s = np.std(x)
        if s == 0: return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    def _kurt(self, x):
        if x.size < 2:
            return 0.0
        m = np.mean(x); s = np.std(x)
        if s == 0: return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3.0)

    def _entropy(self, x):
        if x.size == 0:
            return 0.0
        h, _ = np.histogram(x, bins=64, density=True)
        h = h + 1e-10
        return float(-np.sum(h * np.log2(h)))

    def _stat_features(self, responses):
        feats = []
        for name, arr in responses.items():
            if name.endswith("_phase") and f"{name[:-6]}mag" not in responses:
                continue
            if arr.size == 0:
                feats.append(np.zeros(8, np.float32)); continue
            mean = np.mean(arr); std = np.std(arr)
            mx = np.max(arr); mn = np.min(arr)
            sk = self._skew(arr); ku = self._kurt(arr)
            energy = np.sum(arr ** 2); ent = self._entropy(arr)
            feats.append(np.array([mean, std, mx, mn, sk, ku, energy, ent], np.float32))
        return np.concatenate(feats) if feats else np.zeros(0, np.float32)

    def _spatial_features_one(self, arr):
        ps = self.patch_size
        h, w = arr.shape
        means, stds = [], []
        for i in range(0, h, ps):
            for j in range(0, w, ps):
                patch = arr[i:i + ps, j:j + ps]
                if patch.size:
                    means.append(np.mean(patch)); stds.append(np.std(patch))
        if not means:
            return np.zeros(4, np.float32)
        means = np.array(means); stds = np.array(stds)
        return np.array([means.mean(), means.std(), stds.mean(), stds.std()], np.float32)

    def _spatial_block(self, responses):
        feats = []
        for name, arr in responses.items():
            if arr.size == 0: continue
            feats.append(self._spatial_features_one(arr))
        return np.concatenate(feats) if feats else np.zeros(0, np.float32)

    def _directional_features(self, responses):
        groups = {}
        for name, arr in responses.items():
            if "_mag" not in name: continue
            try:
                theta = float(name.split("_")[1][1:])
            except Exception:
                continue
            groups.setdefault(theta, []).append(arr)

        feats = []
        for theta, lst in groups.items():
            if not lst: continue
            avg = np.mean(np.stack(lst, axis=0), axis=0)
            energy = np.sum(avg ** 2)
            conc = np.max(avg) / (np.mean(avg) + 1e-10)
            feats.append(np.array([energy, conc, float(avg.mean()), float(avg.std())], np.float32))
        return np.concatenate(feats) if feats else np.zeros(0, np.float32)

    def extract_comprehensive_features(self, image):
        """
        Extract combined Gabor features (statistical + spatial + directional).
        Returns 1D float32 array.
        """
        responses = self.bank.extract_filter_responses(image)

        blocks = []
        try:
            s = self._stat_features(responses)
            if s.size: blocks.append(s)
        except Exception as e:
            logger.warning(f"[WARN] Statistical features failed: {e}")

        try:
            sp = self._spatial_block(responses)
            if sp.size: blocks.append(sp)
        except Exception as e:
            logger.warning(f"[WARN] Spatial features failed: {e}")

        try:
            d = self._directional_features(responses)
            if d.size: blocks.append(d)
        except Exception as e:
            logger.warning(f"[WARN] Directional features failed: {e}")

        feat = np.concatenate(blocks).astype(np.float32) if blocks else np.zeros(0, np.float32)
        logger.info(f"[INFO] Gabor feature dim: {feat.size}")
        return feat

# =========================================================
# Adaptive (optional; keeps your functionality)
# =========================================================
class AdaptiveGaborExtractor:
    """
    Adaptive pipeline that analyzes frequency content then selects params dynamically.
    """

    def __init__(self):
        self.base_extractor = GaborFeatureExtractor()

    def analyze_image_frequency_content(self, image):
        if image.ndim == 3:
            if CV2_AVAILABLE:
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except Exception:
                    image = rgb_to_grayscale_pure(image)
            else:
                image = rgb_to_grayscale_pure(image)

        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        mag = np.abs(f_shift)

        h, w = image.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        low = np.sum(mag[dist < min(h, w) * 0.1])
        mid = np.sum(mag[(dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)])
        high = np.sum(mag[dist >= min(h, w) * 0.3])
        total = low + mid + high

        if total > 0:
            dominant = self._find_dominant_frequency(mag, dist)
            spread = np.std(dist[mag > np.percentile(mag, 95)])
            return {
                "low_freq_ratio": low / total,
                "mid_freq_ratio": mid / total,
                "high_freq_ratio": high / total,
                "dominant_frequency": dominant,
                "frequency_spread": spread,
            }
        return {
            "low_freq_ratio": 0.0,
            "mid_freq_ratio": 0.0,
            "high_freq_ratio": 0.0,
            "dominant_frequency": 0.0,
            "frequency_spread": 0.0,
        }

    def _find_dominant_frequency(self, magnitude_spectrum, distance):
        max_d = int(np.max(distance))
        profile = []
        for d in range(1, max_d):
            mask = (distance >= d - 0.5) & (distance < d + 0.5)
            profile.append(np.mean(magnitude_spectrum[mask]) if np.any(mask) else 0.0)
        if profile:
            dom_d = int(np.argmax(profile) + 1)
            dom_f = dom_d / max_d
            return min(dom_f, 0.5)
        return 0.1

    def select_optimal_gabor_params(self, ch):
        if ch["high_freq_ratio"] > 0.4:
            n_frequencies = 8; frequency_range = (0.05, 0.5)
        elif ch["low_freq_ratio"] > 0.6:
            n_frequencies = 4; frequency_range = (0.01, 0.2)
        else:
            n_frequencies = 6; frequency_range = (0.02, 0.3)

        if ch["frequency_spread"] > 10:
            n_orientations = 12
        elif ch["frequency_spread"] < 5:
            n_orientations = 4
        else:
            n_orientations = 8

        return n_frequencies, n_orientations, frequency_range

    def extract_adaptive_features(self, image):
        ch = self.analyze_image_frequency_content(image)
        n_freq, n_orient, freq_range = self.select_optimal_gabor_params(ch)
        extractor = GaborFeatureExtractor(n_frequencies=n_freq, n_orientations=n_orient,
                                          frequency_min=freq_range[0], frequency_max=freq_range[1])
        feat = extractor.extract_comprehensive_features(image)
        logger.info(f"[INFO] Adaptive Gabor complete: {feat.size} dims (freq={n_freq}, orient={n_orient})")
        return feat