from dataclasses import dataclass

from jaxtyping import Float32, Int32
from torch import Tensor


@dataclass
class Scene:
    vertices: Float32[Tensor, "vertex xyz=3"]
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"]
    signed_distances: Float32[Tensor, " vertex"]
    faces: Int32[Tensor, "face corner=3"]
    voxel_indices: Int32[Tensor, " face"]
    level_sets: Float32[Tensor, " level_set"]
    face_boundaries: Int32[Tensor, " level_set"]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]


@dataclass
class SectionParams:
    view_axis: int
    max_value: float
    min_value: float
