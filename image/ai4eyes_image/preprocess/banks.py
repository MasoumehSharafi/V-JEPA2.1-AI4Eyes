"""Image banks: every image of a sign decoded and preprocessed once, then served as eval batches (fixed view) or
train batches (a new augmented view per call).

- CorneaBank     corneal staining: square crop around the cornea mask, fixation ring inpainted, only tokens
                 inside the cornea pooled.
- FullFrameBank  every other sign: the whole frame in a square box (zero-padded), every token at least half
                 inside the frame pooled.
Both return (x [B,3,448,448], keep [B,(448/16)^2] bool).
"""
import cv2
import numpy as np
import torch

from ai4eyes_image.preprocess.masks import masks_for
from ai4eyes_image.preprocess.transforms import prepare, render_train, to_rgb_448, token_keep


class CorneaBank:
    def __init__(self, d, res=448, patch=16):
        self.res, self.patch, self.items = res, patch, {}
        for r in d.itertuples():
            img = cv2.imread(r.image_path)
            assert img is not None, r.image_path
            c, g = masks_for(r.image_path)
            x, cf, rf = to_rgb_448(img, c, g, res=res)
            self.items[r.sample_id] = {"prep": prepare(img, c, g), "x": x,
                                       "keep": torch.from_numpy(token_keep(cf, rf, patch=patch))}

    def eval_batch(self, ids):
        return (torch.stack([self.items[i]["x"] for i in ids]), torch.stack([self.items[i]["keep"] for i in ids]))

    def train_batch(self, ids, rng, aug):
        xs, ks = [], []
        for i in ids:
            x, cf, rf = render_train(self.items[i]["prep"], rng, aug, self.res)
            k = token_keep(cf, rf, patch=self.patch)
            if not k.any():   # extreme rotation can push a tiny mask off-grid; fall back to all tokens
                k[:] = True
            xs.append(x)
            ks.append(torch.from_numpy(k))
        return torch.stack(xs), torch.stack(ks)


class FullFrameBank(CorneaBank):
    """Frames are first downscaled to `prescale`×res; the mask is the whole frame, there is no ring."""

    def __init__(self, d, res=448, patch=16, prescale=2):
        self.res, self.patch, self.items = res, patch, {}
        for r in d.itertuples():
            img = cv2.imread(r.image_path)
            assert img is not None, r.image_path
            s = prescale * res / max(img.shape[:2])
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            frame, none = np.ones(img.shape[:2], bool), np.zeros(img.shape[:2], bool)
            x, cf, rf = to_rgb_448(img, frame, none, res=res)
            self.items[r.sample_id] = {"prep": prepare(img, frame, none), "x": x,
                                       "keep": torch.from_numpy(token_keep(cf, rf, patch=patch))}


def make_bank(sign, d):
    from ai4eyes_image.signs import SIGNS
    return CorneaBank(d) if SIGNS[sign].crop == "cornea" else FullFrameBank(d)
