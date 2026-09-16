# Ported from BaRISTA: https://github.com/ShanechiLab/BaRISTA
# USC research/non-profit license: LICENSES/BaRISTA-LICENSE.md

from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from imindbench.models.barista_components.atlas import DestrieuxAseg


class SpatialEncoderMeta:
    def __init__(self, subject_session_spatial_groups=None):
        """Metadata object with subject session information for spatial encoding."""
        pass

    @property
    def num_region_info(self):
        return 1

    @property
    def embedding_table_configs(self):
        configs = {}
        for i in range(self.num_region_info):
            atlas_enum = DestrieuxAseg
            n_embeddings_for_components_set = len(atlas_enum)
            padding_indices_set = atlas_enum.UNKNOWN.value

            configs[i] = {
                "num_embeddings": n_embeddings_for_components_set,
                "padding_idx": padding_indices_set,
            }

        return configs


class BaseSpatialEncoder(ABC, nn.Module):
    """Abstract class definition for spatial encoding modules.

    Implement this interface to try new spatial encoding approaches in the tokenizer.
    """

    _SUBJ_SESH_QUERY_HASH_STR = "{0}_queryvec"

    def __init__(
        self,
        dim_h: int,
        spatial_encoder_meta: SpatialEncoderMeta,
    ):
        super().__init__()
        self.dim_h = dim_h
        self.spatial_encoder_meta = spatial_encoder_meta

    @abstractmethod
    def _encode(self, x: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def _get_position_encoding(
        self,
        x: torch.tensor,
        *args,
        **kwargs,
    ) -> torch.tensor:
        pass

    def forward(
        self,
        x: torch.tensor,
        subject_session: str,
        region_enum_ids: torch.tensor,
        timepoints: int = 1,
        mask: torch.tensor = None,
    ) -> torch.tensor:
        """
        Args:
            x: torch.tensor of shape (B, T*R, D). Time-space interleaved tokens of dim D.

        Returns:
            A torch.tensor of shape (B, T*R, D) that is the encoding corresponding to
                the input token x.
        """
        region_enum_ids = region_enum_ids.unsqueeze(0).to(x.device)
        session_PE = self._get_position_encoding(x, region_enum_ids)
        assert x.shape[-1] == session_PE.shape[-1], (
            f"Region dimension mismatch: {x.shape[-1]} vs {session_PE.shape[-1]}."
        )

        position_encoding = einops.repeat(
            session_PE, "r d -> b (t r) d", b=x.shape[0], t=timepoints
        )

        if mask is not None:
            position_encoding = position_encoding[:, mask, :]

        assert x.shape == position_encoding.shape, (
            "Output position encoding does not match in shape"
        )
        return position_encoding


class EmbeddingTable(BaseSpatialEncoder):
    def __init__(
        self,
        dim_h: int,
        spatial_encoder_meta: SpatialEncoderMeta,
        embedding_max_dim: float | None = None,
        embedding_init_scale: float = 1.0,
    ):
        """A lookup table of different embeddings for different spatial fields."""
        super().__init__(dim_h, spatial_encoder_meta)

        # Create the embeddings.
        self.subcomponent_embedding_info = (
            self.spatial_encoder_meta.embedding_table_configs
        )

        self.subcomponent_embeddings = nn.ModuleDict()
        for (
            subcomponent_ind,
            subcomponent_config,
        ) in self.subcomponent_embedding_info.items():
            subcomponent_dim = self.dim_h

            self.subcomponent_embeddings[str(subcomponent_ind)] = nn.Embedding(
                subcomponent_config["num_embeddings"],
                subcomponent_dim,
                padding_idx=subcomponent_config["padding_idx"],
                max_norm=embedding_max_dim,
            )

            self.init_weights_for_embeddings(
                self.subcomponent_embeddings[str(subcomponent_ind)],
                embedding_init_scale,
            )

    def init_weights_for_embeddings(
        self, embedding_table: nn.Embedding, embedding_init_scale: float = 1.0
    ):
        nn.init.normal_(embedding_table.weight, std=embedding_init_scale)
        embedding_table._fill_padding_idx_with_zero()

    def _transform_query_vector(self, query_vector: torch.Tensor):
        return query_vector.to(torch.float).T

    def _get_position_encoding(
        self,
        _: torch.tensor,
        region_enum_ids: str,
        *args,
        **kwargs,
    ) -> torch.tensor:
        """Returns the encoding vector based on a subject session query."""
        single_session_PE = self._encode(region_enum_ids)
        return single_session_PE


class EmbeddingTablePool(EmbeddingTable):
    def _encode(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor of shape (B, T*R, D). Time-space interleaved tokens of dim D.

        Returns:
            A torch.tensor of shape (B, T*R, D) that is the encoding corresponding to
                the input token. If token has multiple spatial fields, the encoding for
                each of these fields will be summed together before being return (e.g.,
                x,y,z LPI coordinates).
        """
        PE = torch.zeros((x.shape[0], x.shape[1], self.dim_h), device=x.get_device())
        for subcomponent_ind in range(x.shape[0]):
            subcomponent_x = x[subcomponent_ind, ...]
            PE[subcomponent_ind, ...] = self.subcomponent_embeddings[
                str(subcomponent_ind)
            ](subcomponent_x)
        return torch.sum(PE, axis=0)


def create_spatial_encoder(
    dim_h: int,
    embedding_max_dim=None,
    embedding_init_scale=1.0,
) -> BaseSpatialEncoder:
    """Creates the spatial encoder and the cached spatial encoding information needed during forward passes."""
    spatial_encoder_meta = SpatialEncoderMeta()

    spatial_encoder = EmbeddingTablePool(
        dim_h, spatial_encoder_meta, embedding_max_dim, embedding_init_scale
    )

    return spatial_encoder
