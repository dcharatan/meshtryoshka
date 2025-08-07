from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor, nn

from .initializer import Initializer


@dataclass(frozen=True)
class SphereInitializerCfg:
    name: Literal["sphere"]
    radius: float


class SphereInitializer(Initializer):
    def __init__(
        self,
        cfg: SphereInitializerCfg,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg

    def get_sdf_modify(
        self,
        sdf_in: Float[Tensor, "*batch"],
        xyz: Float[Tensor, "*batch xyz=3"],
        step: int,
    ) -> Float[Tensor, "*batch"]:
        return sdf_in + (xyz - 0.5).norm(dim=-1) - self.cfg.radius
