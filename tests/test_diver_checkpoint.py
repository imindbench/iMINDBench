"""DIVER shape-cache creation must preserve the checkpoint architecture."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def test_diver_checkpoint_with_fresh_and_existing_shape_cache(tmp_path):
    pytest.importorskip("mup")
    pytest.importorskip("jaxtyping")
    from imindbench.models.diver_components.diver import DIVER
    from imindbench.models.diver_model import DIVERModel

    cfg = OmegaConf.load(Path(__file__).parents[1] / "imindbench/conf/model/diver.yaml")
    cfg.depth = 1
    cfg.upstream_ckpt = str(tmp_path / "checkpoint.pt")
    cfg.model_dir = str(tmp_path / "shapes")
    backbone = DIVER(d_model=cfg.width, e_layer=1, mup=True, patch_size=50)
    torch.save({"module": backbone.state_dict()}, cfg.upstream_ckpt)

    for _ in range(2):
        model = DIVERModel(cfg, OmegaConf.create({"provider": "neuroprobev2"}))
        batch = model.prepare_batch(
            {"x": torch.randn(2, 3, 500), "y": torch.tensor([0, 1])}
        )
        model.build_model(batch["x"].shape[1:], 2, device="cpu").eval()
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(model.model.backbone.state_dict()[name], value)
        with torch.no_grad():
            logits = model.forward(batch["x"], **batch["model_kwargs"])
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()
