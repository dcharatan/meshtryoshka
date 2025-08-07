from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor, nn


class Initializer(nn.Module, ABC):
    @abstractmethod
    def get_sdf_modify(
        self,
        sdf_in: Float[Tensor, "*batch"],
        xyz: Float[Tensor, "*batch xyz=3"],
        step: int,
    ) -> Float[Tensor, "*batch"]:
        pass
