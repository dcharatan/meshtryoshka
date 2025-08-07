from jaxtyping import Float
from torch import Tensor

from .model import Model
from .model_meshtryoshka import ModelMeshtryoshka, ModelMeshtryoshkaCfg

ModelCfg = ModelMeshtryoshkaCfg

MODELS: dict[str, type[Model]] = {
    "meshtryoshka": ModelMeshtryoshka,
}


def get_model(
    cfg: ModelCfg,
    extrinsics: Float[Tensor, "image 4 4"],
    intrinsics: Float[Tensor, "image 3 3"],
    image_shape: tuple[int, int],
    point_cloud: Float[Tensor, "point 3"] | None,
) -> Model:
    return MODELS[cfg.name](cfg, extrinsics, intrinsics, image_shape, point_cloud)
