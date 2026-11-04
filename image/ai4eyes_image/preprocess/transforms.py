"""Image → model input. Square crop around a mask (the cornea, or the whole frame), fixation ring inpainted,
resized to 448, ImageNet-normalised; plus the training augmentation and the token-keep mask for masked pooling.

- to_rgb_448()   eval view (no augmentation)
- prepare() + render_train()   training views: the crop is prepared once per image, each epoch draws a new view
- token_keep()   which ViT tokens lie inside the mask (and off the ring) → only those are pooled
"""
import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONE_RES = 448
CROP_PAD = 0.04          # fractional padding around the cornea bbox
TOKEN_MIN_FRAC = 0.5     # keep a token when >=50% of its footprint is inside the mask


def square_crop_box(cmask, pad=CROP_PAD):
    """Square (x0, y0, side) centred on the cornea-mask bbox. May extend past the image; crop() pads."""
    ys, xs = np.nonzero(cmask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * pad)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop(arr, box):
    """Crop with zero padding where the square leaves the frame."""
    x0, y0, side = box
    h, w = arr.shape[:2]
    out = np.zeros((side, side) + arr.shape[2:], arr.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def _resize_mask(m, res):
    return cv2.resize(m.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)


def _normalise(rgb01):
    rgb = (rgb01.astype(np.float32) - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def to_rgb_448(img_bgr, cmask, rmask, res=BACKBONE_RES, train=False, rng=None, aug=None):
    """Returns (tensor [3,res,res] ImageNet-normalised, cornea_frac [res,res], ring_frac [res,res]).

    The ring is inpainted, not zeroed: a zeroed pixel normalises to ≈ −2 and would itself be a
    constant dark-ring confound. The *_frac maps are area-averaged masks in [0, 1].
    train=True applies augmentation (requires rng); val/test must call with train=False.
    """
    if train:
        return render_train(prepare(img_bgr, cmask, rmask), rng, aug, res)
    img = cv2.inpaint(img_bgr, rmask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    box = square_crop_box(cmask)
    img = cv2.resize(crop(img, box), (res, res), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return _normalise(rgb), _resize_mask(crop(cmask, box), res), _resize_mask(crop(rmask, box), res)


# ---- training augmentation ----------------------------------------------------------------
# Applied to every class equally. Deliberately EXCLUDED: blur, downscaling, vertical flip, cutout —
# the first three erase or distort 3.7–7.4 px punctate dots, vertical flip is anatomically impossible
# (swaps upper/lower lid), and cutout can delete the only stained region.
AUG_DEFAULTS = {
    "hflip_p": 0.5,          # OD <-> OS mirror, anatomically plausible
    "rotate_deg": 15.0,      # uniform in [-15, 15]
    "brightness": 0.10,      # multiplicative, uniform in [1-b, 1+b]
    "contrast": 0.10,        # about the per-image mean, uniform in [1-c, 1+c]
    "gamma": 0.15,           # log-uniform in [1/(1+g), 1+g]
    "green_gain": 0.10,      # green channel only (fluorescein intensity), uniform in [1-g, 1+g]
}


def prepare(img_bgr, cmask, rmask, max_rot_deg=15.0):
    """Inpaint the ring once at native res and cut a crop large enough to rotate by ±max_rot_deg
    without exposing corners. Cache the result per image; render_train() samples from it each epoch."""
    img = cv2.inpaint(img_bgr, rmask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    x0, y0, side = square_crop_box(cmask)
    t = np.deg2rad(max_rot_deg)
    big = int(np.ceil(side * (np.cos(t) + np.sin(t)))) + 2
    bx0, by0 = x0 - (big - side) // 2, y0 - (big - side) // 2
    bbox = (bx0, by0, big)
    return {"img": crop(img, bbox), "cm": crop(cmask.astype(np.uint8), bbox), "rm": crop(rmask.astype(np.uint8), bbox),
            "side": side, "off": x0 - bx0}


def render_train(prep, rng, aug=None, res=BACKBONE_RES):
    """One random augmented view → same outputs as to_rgb_448(). Geometric ops move the masks too."""
    a = {**AUG_DEFAULTS, **(aug or {})}
    img, cm, rm, side, off = prep["img"], prep["cm"], prep["rm"], prep["side"], prep["off"]
    big = img.shape[0]
    ang = float(rng.uniform(-a["rotate_deg"], a["rotate_deg"])) if a["rotate_deg"] else 0.0
    if ang:
        M = cv2.getRotationMatrix2D(((big - 1) / 2, (big - 1) / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (big, big), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cm = cv2.warpAffine(cm, M, (big, big), flags=cv2.INTER_NEAREST)
        rm = cv2.warpAffine(rm, M, (big, big), flags=cv2.INTER_NEAREST)
    sl = np.s_[off:off + side, off:off + side]
    img = cv2.resize(img[sl], (res, res), interpolation=cv2.INTER_AREA)
    cf, rf = _resize_mask(cm[sl], res), _resize_mask(rm[sl], res)
    if rng.random() < a["hflip_p"]:
        img, cf, rf = img[:, ::-1], cf[:, ::-1], rf[:, ::-1]

    rgb = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb *= rng.uniform(1 - a["brightness"], 1 + a["brightness"])
    c = rng.uniform(1 - a["contrast"], 1 + a["contrast"])
    rgb = (rgb - rgb.mean()) * c + rgb.mean()
    rgb = np.clip(rgb, 0, 1) ** np.exp(rng.uniform(-np.log1p(a["gamma"]), np.log1p(a["gamma"])))
    rgb[..., 1] *= rng.uniform(1 - a["green_gain"], 1 + a["green_gain"])
    rgb = np.clip(rgb, 0, 1)
    return _normalise(rgb), np.ascontiguousarray(cf), np.ascontiguousarray(rf)


def token_keep(cornea_frac, ring_frac, patch=16, min_frac=TOKEN_MIN_FRAC):
    """Bool [grid*grid], row-major like the ViT token order: >=min_frac inside the cornea, no ring pixels."""
    res = cornea_frac.shape[0]
    g = res // patch
    c = cornea_frac.reshape(g, patch, g, patch).mean(axis=(1, 3))
    r = ring_frac.reshape(g, patch, g, patch).max(axis=(1, 3))
    return ((c >= min_frac) & (r == 0)).reshape(-1)
