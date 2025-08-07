from jaxtyping import Float
from torch import Tensor

from ..model.contraction import Contraction
from .initializer import Initializer
from .initializer_point_cloud import PointCloudInitializer, PointCloudInitializerCfg
from .initializer_sphere import SphereInitializer, SphereInitializerCfg

InitializerCfg = PointCloudInitializerCfg | SphereInitializerCfg

INITIALIZER: dict[str, type[Initializer]] = {
    "point_cloud": PointCloudInitializer,
    "sphere": SphereInitializer,
}


def get_initializer(
    cfg: InitializerCfg,
    contraction: Contraction,
    point_cloud: Float[Tensor, "point 3"] | None,
) -> Initializer:
    return INITIALIZER[cfg.name](
        cfg,
        contraction=contraction,
        point_cloud=point_cloud,
    )
