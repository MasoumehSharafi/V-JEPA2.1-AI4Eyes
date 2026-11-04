"""Load the saved per-sign models and predict images (5-seed ensemble).

    from ai4eyes_image.inference import SignModel
    m = SignModel("corneal_staining", device="cuda")      # backbone + 5 LoRA/head weight sets, loaded once
    out = m.predict(d)        # d: DataFrame with sample_id, image_path → probabilities [N, K] or mm [N]
"""
import json

import numpy as np
import torch

from ai4eyes_image import paths
from ai4eyes_image.config import load_yaml, sign_config
from ai4eyes_image.preprocess.banks import make_bank
from ai4eyes_image.train.trainer import build_model, predict


class SignModel:
    def __init__(self, sign, device="cuda", config=paths.PKG_ROOT / "configs" / "supervised.yaml"):
        d = paths.CHECKPOINT_DIR / "signs" / sign
        self.sign, self.device = sign, device
        self.meta = json.loads((d / "meta.json").read_text())
        c = self.meta["config"]
        if "backbone_weights" not in c["model"]:      # checkpoints written by the original research code
            c = sign_config(load_yaml(config), sign)
        self.cfg = c
        self.model = build_model(c, device)
        self.seeds = [torch.load(d / f"seed{s}.pt", map_location=device) for s in self.meta["seeds"]]
        self._load(self.seeds[0])
        self.model.eval()

    def _load(self, sd):
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        frozen = {n for n, p in self.model.named_parameters() if not p.requires_grad}
        assert not unexpected and set(missing) <= frozen, (self.sign, unexpected, set(missing) - frozen)

    def outputs(self, z):
        if self.meta["kind"] == "regression":
            mu = self.meta.get("train_mean_mm", self.meta.get("dev_mean_mm"))
            sd = self.meta.get("train_sd_mm", self.meta.get("dev_sd_mm"))
            return z.squeeze(1).float().numpy() * sd + mu
        return torch.softmax(z.float(), 1).numpy()

    @torch.no_grad()
    def predict_bank(self, bank, ids, n_seeds=None, bs=16):
        P = []
        for sd in self.seeds[:n_seeds]:
            self._load(sd)
            P.append(self.outputs(predict(self.model, bank, ids, self.device, bs, self.device == "cuda")))
        return np.mean(P, 0)

    def predict(self, d, n_seeds=None):
        return self.predict_bank(make_bank(self.sign, d), list(d.sample_id), n_seeds)
