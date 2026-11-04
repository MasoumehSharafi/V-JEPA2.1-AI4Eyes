"""Cornea mask and fixation-ring mask for the corneal-staining photos, computed at native 1920×1080.

The cornea mask is a classical ray-cast boundary (not a trained model), `CORNEA_MASK_VERSION`. The fixation ring
(12 bright dots projected on the cornea) is found by a RANSAC circle fit and later inpainted, so the model never
sees it. Masks can be cached next to each image (<name>.cornea.png / <name>.ring.png, see build_masks()).
"""
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

CORNEA_MASK_VERSION = "placeholder-rays-v1"

# Fixation ring: 12 near-white dots (BGR ~ 255,255,205) on a circle of radius 23–41 px (measured on
# all 72 images). Fluorescein staining is green with low blue, so high B *and* high G isolates the dots.
RING_MIN_B, RING_MIN_G = 150, 150
RING_DOT_AREA = (8, 250)
RING_RADIUS = (18, 50)
RING_INLIER_TOL = 2.5
RING_MIN_DOTS = 6
RING_BAND = 8          # px either side of the fitted circle, covers dot radius (~4 px) + margin
RING_DILATE = 3


@dataclass
class RingFit:
    center: tuple      # (x, y) float
    radius: float
    n_dots: int


def _circle_from_3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    A = 2 * np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]], dtype=float)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    b = np.array([x2**2 - x1**2 + y2**2 - y1**2, x3**2 - x1**2 + y3**2 - y1**2], dtype=float)
    c = np.linalg.solve(A, b)
    return c, float(np.linalg.norm(p[0] - c))


def _ring_candidates(img_bgr):
    b, g, _ = cv2.split(img_bgr)
    m = ((b > RING_MIN_B) & (g > RING_MIN_G)).astype(np.uint8)
    _, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = (area >= RING_DOT_AREA[0]) & (area <= RING_DOT_AREA[1])
    return m, cents[1:][keep]


def fit_ring(img_bgr, seed=0, iters=200):
    """RANSAC circle fit over bright-blue blob centroids. Returns RingFit or None."""
    _, c = _ring_candidates(img_bgr)
    rng = np.random.default_rng(seed)
    best = None
    for i in range(len(c)):
        nb = np.where(np.linalg.norm(c - c[i], axis=1) < 2 * RING_RADIUS[1])[0]
        if len(nb) < 3:
            continue
        for _ in range(iters):
            f = _circle_from_3(c[rng.choice(nb, 3, replace=False)])
            if f is None or not RING_RADIUS[0] < f[1] < RING_RADIUS[1]:
                continue
            inl = np.abs(np.linalg.norm(c - f[0], axis=1) - f[1]) < RING_INLIER_TOL
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), f[0], f[1])
    if best is None or best[0] < RING_MIN_DOTS:
        return None
    return RingFit(center=(float(best[1][0]), float(best[1][1])), radius=float(best[2]), n_dots=best[0])


def ring_mask(img_bgr, ring):
    """Thin band on the fitted circle ∪ any bright-blue blob touching it (catches motion-smeared dots)."""
    h, w = img_bgr.shape[:2]
    band = np.zeros((h, w), np.uint8)
    cx, cy = (int(round(v)) for v in ring.center)
    cv2.circle(band, (cx, cy), int(round(ring.radius)), 1, thickness=2 * RING_BAND)
    bright, _ = _ring_candidates(img_bgr)
    n, lab = cv2.connectedComponents(bright, connectivity=8)
    touching = np.unique(lab[(band > 0) & (bright > 0)])
    touching = touching[touching > 0]
    m = band | np.isin(lab, touching).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_DILATE + 1,) * 2))
    return m.astype(bool)


# ---- cornea (placeholder) -------------------------------------------------------------------------
# Star-shaped boundary around the ring centre: along each of N_RAYS rays, the first *sustained* rise in
# green brightness (limbus nasally/temporally, lid margin / tear meniscus superiorly/inferiorly).
# Region-growing was tried first and leaked through gaps in dim lid outlines; a ray boundary cannot leak.

BLUR_SIGMA = 4
N_RAYS = 360
RAY_R = (60, 640)       # px sampled along each ray
INNER_R = 60            # first samples along each ray give the corneal (dark) reference level
ALPHA = 0.4             # threshold = inner + ALPHA * (ray max - inner)
SUSTAIN = 20            # px the profile must stay above threshold (ignores isolated staining dots)
MIN_R = 130             # hits closer than this are treated as missing rays
MEDIAN_DEG = 31         # angular median filter
MAX_R = 560
OUTLIER_X = 1.3         # cap rays beyond OUTLIER_X * median radius (dim-lid lobes)
ERODE = 8               # pull the mask off the limbus / lid margin


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def cornea_boundary(img_bgr, ring, rmask):
    """Radius (px) of the cornea boundary along each ray, and the fraction of rays that found an edge."""
    from scipy.ndimage import median_filter

    g = cv2.inpaint(img_bgr[..., 1], rmask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), BLUR_SIGMA)
    cx, cy = ring.center
    th = np.deg2rad(np.arange(N_RAYS) * 360.0 / N_RAYS)
    rs = np.arange(*RAY_R)
    X = (cx + np.outer(np.cos(th), rs)).astype(np.float32)
    Y = (cy + np.outer(np.sin(th), rs)).astype(np.float32)
    P = cv2.remap(g, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    inner = np.median(P[:, :INNER_R])
    rb = np.full(N_RAYS, np.nan)
    kernel = np.ones(SUSTAIN, int)
    for a in range(N_RAYS):
        above = (P[a] > inner + ALPHA * (P[a].max() - inner)).astype(int)
        hit = np.nonzero(np.convolve(above, kernel, "valid") == SUSTAIN)[0]
        if len(hit) and rs[hit[0]] >= MIN_R:
            rb[a] = rs[hit[0]]
    ok = ~np.isnan(rb)
    hit_frac = float(ok.mean())
    if ok.sum() < N_RAYS // 4:
        return None, hit_frac
    ang = np.arange(N_RAYS)
    rb = np.interp(ang, np.r_[ang[ok] - N_RAYS, ang[ok], ang[ok] + N_RAYS], np.r_[rb[ok], rb[ok], rb[ok]])
    rb = median_filter(rb, size=MEDIAN_DEG, mode="wrap")
    rb = np.minimum(rb, min(MAX_R, OUTLIER_X * np.median(rb)))
    return rb, hit_frac


def cornea_mask(img_bgr, ring, rmask=None):
    """Returns (mask bool[H,W], info dict). PLACEHOLDER — see module docstring."""
    h, w = img_bgr.shape[:2]
    if rmask is None:
        rmask = ring_mask(img_bgr, ring)
    rb, hit_frac = cornea_boundary(img_bgr, ring, rmask)
    info = {"ray_hit_frac": hit_frac}
    if rb is None:
        return np.zeros((h, w), bool), {**info, "ok": False, "reason": "too few rays found an edge"}

    cx, cy = ring.center
    th = np.deg2rad(np.arange(N_RAYS) * 360.0 / N_RAYS)
    pts = np.stack([cx + rb * np.cos(th), cy + rb * np.sin(th)], 1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    # Convex hull: dense staining is bright and stops rays early, carving dents exactly where the
    # signal is. The visible cornea (disc ∩ lid aperture) is near-convex, so the hull restores them.
    cv2.fillPoly(m, [cv2.convexHull(pts)], 1)
    m = cv2.erode(m, _disk(ERODE)).astype(bool)

    info.update(ok=True, r_min=float(rb.min()), r_med=float(np.median(rb)), r_max=float(rb.max()))
    if hit_frac < 0.6:
        info.update(ok=False, reason=f"only {hit_frac:.0%} of rays found an edge")
    return m, info


def compute_masks(img_bgr):
    """Both masks for one image.

    Without a fixation ring (e.g. a rejected capture with the eye closed) the cornea mask is the centre-square
    fallback, the ring mask is empty and ok=False.
    """
    ring = fit_ring(img_bgr)
    if ring is None:
        cmask = fallback_mask(img_bgr.shape)
        return cmask, np.zeros(cmask.shape, bool), {"ok": False, "reason": "fixation ring not found",
                                                    "version": CORNEA_MASK_VERSION}
    rmask = ring_mask(img_bgr, ring)
    cmask, info = cornea_mask(img_bgr, ring, rmask)
    info.update(ring_cx=ring.center[0], ring_cy=ring.center[1], ring_r=ring.radius, ring_dots=ring.n_dots,
                version=CORNEA_MASK_VERSION)
    return cmask, rmask, info


FALLBACK_SIDE = 900     # px; ≈ median cornea crop side at 1920×1080


def fallback_mask(shape, center=None, side=FALLBACK_SIDE):
    """Centre-square fallback cornea mask (centred on `center` if given)."""
    h, w = shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    m = np.zeros((h, w), bool)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    m[max(0, y0):max(0, y0 + side), max(0, x0):max(0, x0 + side)] = True
    return m


def cache_paths(image_path):
    p = str(image_path)
    return p.replace(".png", ".cornea.png"), p.replace(".png", ".ring.png")


def masks_for(image_path):
    """(cornea mask, ring mask) of one staining photo: the cached masks if both exist, else computed."""
    cp, rp = cache_paths(image_path)
    cached = os.path.exists(cp) and os.path.exists(rp)
    c = cv2.imread(cp, cv2.IMREAD_GRAYSCALE) if cached else None
    r = cv2.imread(rp, cv2.IMREAD_GRAYSCALE) if cached else None
    if c is None or r is None:
        c, r, _ = compute_masks(cv2.imread(str(image_path)))
        return c, r
    return c > 0, r > 0


def build_masks(image_paths):
    """Compute and cache the masks of every image that has no cache yet → list of info dicts."""
    out = []
    for p in image_paths:
        cp, rp = cache_paths(p)
        if os.path.exists(cp) and os.path.exists(rp):
            continue
        c, r, info = compute_masks(cv2.imread(str(p)))
        cv2.imwrite(cp, c.astype(np.uint8) * 255)
        cv2.imwrite(rp, r.astype(np.uint8) * 255)
        out.append({"image_path": str(p), **info})
    return out
