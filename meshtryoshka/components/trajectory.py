from math import radians

import torch
from jaxtyping import Float
from torch import Tensor


def rotation_around_axis(
    angles: Float[Tensor, " T"],
    axis: Float[Tensor, "3"] | Float[Tensor, "T 3"],
) -> Float[Tensor, "T 3 3"]:
    """
    angles: (T,)
    axis: either (3,) or (T,3)
    returns: (T,3,3)
    """
    # make axis time-batched
    if axis.ndim == 1:
        axis = axis[None].expand(angles.shape[0], -1)  # → (T,3)
    else:
        assert axis.shape[0] == angles.shape[0], "axis and angles must share T"

    # normalize
    axis = axis / axis.norm(dim=1, keepdim=True)  # (T,3)
    ux, uy, uz = axis.unbind(-1)  # each (T,)

    # skew-sym matrices K for each t
    K = torch.stack(
        [
            torch.stack([torch.zeros_like(ux), -uz, uy], dim=-1),
            torch.stack([uz, torch.zeros_like(ux), -ux], dim=-1),
            torch.stack([-uy, ux, torch.zeros_like(ux)], dim=-1),
        ],
        dim=-2,
    )  # (T,3,3)

    # outer products u u^T
    uuT = axis[:, :, None] * axis[:, None, :]  # (T,3,3)

    i = torch.eye(3, device=angles.device, dtype=angles.dtype)[None]  # (1,3,3)
    c = torch.cos(angles)[:, None, None]  # (T,1,1)
    s = torch.sin(angles)[:, None, None]  # (T,1,1)
    oc = (1 - torch.cos(angles))[:, None, None]  # (T,1,1)

    return c * i + s * K + oc * uuT  # (T,3,3)


@torch.no_grad()
def generate_spin_wave_loop(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    num_frames: int,
    spin_axis_idx: int = 2,  # 0=X,1=Y,2=Z
    tilt_wave_periods: int = 2,
    tilt_amplitude_deg: float = 7.0,
    tilt_amplitude_bias: float = -10.0,
) -> Float[Tensor, "*#batch num_frames 4 4"]:
    """
    Spins the camera around world‐axis (spin_axis_idx) by 360°, and
    simultaneously tilts up/down in a sine wave (about camera’s right axis).
    """
    # --- pick global spin axis ---
    axis_map = {
        0: torch.tensor(
            [1.0, 0.0, 0.0], device=extrinsics.device, dtype=extrinsics.dtype
        ),
        1: torch.tensor(
            [0.0, 1.0, 0.0], device=extrinsics.device, dtype=extrinsics.dtype
        ),
        2: torch.tensor(
            [0.0, 0.0, 1.0], device=extrinsics.device, dtype=extrinsics.dtype
        ),
    }
    spin_axis = axis_map[spin_axis_idx]

    # time fraction t in [0,1)
    t = torch.linspace(
        0.0, 1.0, steps=num_frames, device=extrinsics.device, dtype=extrinsics.dtype
    )

    # --- build spin rotations ---
    spin_angles = 2 * torch.pi * t  # (T,)
    R_spin = rotation_around_axis(spin_angles, spin_axis)  # (T,3,3)
    R_spin_T = R_spin.transpose(-1, -2)  # (T,3,3)

    # batch- and time-expand extrinsics
    batch_dims = extrinsics.shape[:-2]  # (*batch,)
    T = num_frames
    E = extrinsics.unsqueeze(-3).expand(*batch_dims, T, 4, 4).clone()
    Re = E[..., :3, :3]  # (*batch, T, 3,3)
    te = E[..., :3, 3]  # (*batch, T, 3)

    # apply spin
    R_spin_T_exp = R_spin_T.view((1,) * len(batch_dims) + (T, 3, 3))
    Re_spun = torch.matmul(Re, R_spin_T_exp)

    # --- compute per-frame camera right axes in world space ---
    # cam2world = Re_spun^T
    cam2world = Re_spun.transpose(-1, -2)  # (*batch, T, 3,3)
    # right axis = first column of cam2world
    axis_dirs = cam2world[..., :, 0]  # (*batch, T, 3)

    # --- build tilt rotations ---
    amp_rad = radians(tilt_amplitude_deg)
    tilt_angles = -amp_rad * torch.sin(2 * torch.pi * tilt_wave_periods * t) + radians(
        tilt_amplitude_bias
    )  # (T,)
    # flatten batch*time so we can reuse rotation_around_axis
    flat_axes = axis_dirs.reshape(-1, 3)  # (B*T,3)
    flat_angles = tilt_angles.unsqueeze(0).expand(*batch_dims, T).reshape(-1)  # (B*T,)
    R_tilt_flat = rotation_around_axis(flat_angles, flat_axes)  # (B*T,3,3)
    R_tilt = R_tilt_flat.view(*batch_dims, T, 3, 3)
    R_tilt_T = R_tilt.transpose(-1, -2)  # (*batch,T,3,3)

    # apply tilt
    Re_final = torch.matmul(Re_spun, R_tilt_T)

    # write back
    E[..., :3, :3] = Re_final
    E[..., :3, 3] = te  # translation unchanged

    return E
