"""Unlabelled images for SSL, one manifest row per image (schema: ai4eyes_image.pretrain.manifest).

Each item is read from S3 (`key`) or from disk (`local_path`, e.g. video frames), cropped to its own box (cornea
crop for staining, whole frame otherwise; the same crops the downstream signs see), the fixation ring inpainted,
then RandomResizedCrop inside the box + h-flip → tensor [3,res,res] ImageNet-normalised.
"""
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ai4eyes_image.data.s3 import s3_client, s3_read_image
from ai4eyes_image.preprocess.masks import RingFit, ring_mask
from ai4eyes_image.preprocess.transforms import IMAGENET_MEAN, IMAGENET_STD, crop


class SSLImages(Dataset):
    def __init__(self, manifest, bucket, res, augment, train=True, seed=0):
        self.m = manifest.reset_index(drop=True)
        self.bucket, self.train, self.seed = bucket, train, seed
        self.res, self.scale, self.ratio = res, tuple(augment["rrc_scale"]), tuple(augment["rrc_ratio"])
        self.hflip = augment["hflip_p"]
        self.epoch = 0
        self._client = None

    def __len__(self):
        return len(self.m)

    def set_epoch(self, e):
        self.epoch = e

    def _rng(self, idx):
        # train: new randomness every epoch; eval: fixed per item
        return np.random.default_rng([self.seed, idx] if not self.train else [self.seed, self.epoch, idx])

    def _rrc(self, rng, side):
        """RandomResizedCrop parameters (torchvision semantics) inside a side×side box."""
        area = side * side
        for _ in range(10):
            a = area * rng.uniform(*self.scale)
            r = math.exp(rng.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            w, h = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
            if 0 < w <= side and 0 < h <= side:
                return int(rng.integers(0, side - w + 1)), int(rng.integers(0, side - h + 1)), w, h
        return 0, 0, side, side

    def read(self, r):
        local = r.get("local_path") if isinstance(r.get("local_path"), str) else None
        if local:
            img = cv2.imread(local)
            if img is None:
                raise IOError(f"could not read {local}")
            return img
        if self._client is None:
            self._client = s3_client()      # one client per DataLoader worker
        return s3_read_image(self._client, self.bucket, r.key)

    def __getitem__(self, idx):
        r = self.m.iloc[idx]
        rng = self._rng(idx)
        img = self.read(r)
        x0, y0, side = int(r.box_x0), int(r.box_y0), int(r.box_side)
        dx, dy, w, h = self._rrc(rng, side) if self.train else (0, 0, side, side)
        # crop first (cheap), then inpaint the ring inside the crop only
        c = crop(img, (x0 + dx, y0 + dy, max(w, h)))[:h, :w]      # zero-pads if the box leaves the frame
        if r.ring_ok:
            rm = ring_mask(img, RingFit((r.ring_cx, r.ring_cy), r.ring_r, int(r.ring_dots))).astype(np.uint8)
            rmc = crop(rm, (x0 + dx, y0 + dy, max(w, h)))[:h, :w]
            if rmc.any():
                c = cv2.inpaint(np.ascontiguousarray(c), rmc, 5, cv2.INPAINT_TELEA)
        interp = cv2.INTER_AREA if max(w, h) >= self.res else cv2.INTER_CUBIC
        c = cv2.resize(c, (self.res, self.res), interpolation=interp)
        if self.train and rng.random() < self.hflip:
            c = c[:, ::-1]
        rgb = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy())
