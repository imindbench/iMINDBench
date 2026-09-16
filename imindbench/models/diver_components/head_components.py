from types import SimpleNamespace

import torch
from mup import MuReadout
from torch import nn

from .model_builders import MakeModelIgnoreDataInfoList


def create_multi_heads_generic(configs):
    heads_dict = {}
    for head_name, head_dict in configs.items():
        head_config = SimpleNamespace(**head_dict)
        heads_dict[head_name] = MakeModelIgnoreDataInfoList(
            nn.Linear(head_config.d_model, head_config.out_dim)
            if not getattr(head_config, "use_mup", True)
            else MuReadout(head_config.d_model, head_config.out_dim, output_mult=1.0)
        )

    return MultiHeadOnSameInput(heads_dict)


class MultiHeadOnSameInput(nn.Module):
    def __init__(self, heads: dict[str, nn.Module] | list[nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x, data_info_list=None):
        outputs = {}
        for name, head in self.heads.items():
            outputs[name + "_output"] = head(x, data_info_list=data_info_list)
        return outputs


def getMultiMultiheadDict_fromDummy(
    input_dim, d_model, loss_configs, hidden_projection_cnt=None
):
    if hidden_projection_cnt is None:
        hidden_projcetion = [int(d_model * 4)]
    else:
        hidden_projcetion = [int(d_model * 4) for i in range(hidden_projection_cnt)]
    dummy_tensor = torch.rand((1, input_dim))
    tot_params = {}

    if "patchwise_time_domain" in loss_configs:
        tot_params["time_head"] = {
            "d_model": d_model,
            "out_dim": input_dim,
            "n_heads": d_model // 32,
            "d_ff": int(4 * d_model),
            "projection_hidden": hidden_projcetion,
            "activation": "gelu",
            "dropout": 0.1,
            "e_layers": 1,
        }
    if "patchwise_fft_amplitude" in loss_configs:
        from .target_representation_functions import fft_transform

        loss_config = loss_configs["patchwise_fft_amplitude"]
        dummy_transformed = fft_transform(dummy_tensor, **loss_config["fft_params"])
        out_dim = dummy_transformed.shape[-1]
        tot_params["fft_amplitude_head"] = {
            "d_model": d_model,
            "out_dim": out_dim,
            "n_heads": d_model // 32,
            "d_ff": int(4 * d_model),
            "projection_hidden": hidden_projcetion,
            "activation": "gelu",
            "dropout": 0.1,
            "e_layers": 1,
        }
    if "patchwise_multiscale_stft_amplitude" in loss_configs:
        from .target_representation_functions import multiscale_stft_transform

        print("amp start")
        loss_config = loss_configs["patchwise_multiscale_stft_amplitude"]
        dummy_transformed = multiscale_stft_transform(
            dummy_tensor,
            multiscale_stft_params=loss_config["stft_params_list"],
            loss_type=loss_config["loss_type"],
        )
        out_dim = dummy_transformed.shape[-1]
        tot_params["multiscale_stft_amplitude_head"] = {
            "d_model": d_model,
            "out_dim": out_dim,
            "n_heads": d_model // 32,
            "d_ff": int(4 * d_model),
            "projection_hidden": hidden_projcetion,
            "activation": "gelu",
            "dropout": 0.1,
            "e_layers": 1,
        }
    return tot_params
