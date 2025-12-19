import torch

from typing import TYPE_CHECKING
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def done_timeout(
        env:    "DirectRLEnv",
    ) -> torch.Tensor:
    """
    termination on episode timeout

    Input:
        - env:  environment instance
    Output:
        - whether episode length has reached its maximum
    """
    return env.episode_length_buf >= env.max_episode_length - 1


def done_baseContact(
        env:    "DirectRLEnv"
    ) -> torch.Tensor:
    """
    termination on constraint: undesired base contact with ground

    Input:
        - env:    environment instance
    Output:
        - whether base contact force exceeds 1.0N threshold at any point in contact history
    """
    net_contact_forces = env._contact_sensor.data.net_forces_w_history
    cstr_termination_contacts = torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, env._base_id], dim=-1), dim=1)[0] > 1.0, 
            dim=1,
    )
    return cstr_termination_contacts


def done_baseHeight(
        robot:  Articulation,
        thres:  float
    ) -> torch.Tensor:
    """
    termination on constraint: base height lower than threshold

    Input:
        - robot:  robot instance
        - thres:  minimum base height threshold
    Output:
        - whether base height is lower than threshold
    """
    return robot.data.root_pos_w[:, 2] < thres


def done_baseUpsidedown(
        robot:  Articulation,
    ) -> torch.Tensor:
    """
    termination on constraint: base upsidedown

    Input:
        - robot:  robot instance
    Output:
        - whether projected gravity on base z-axis is positive
    """
    return robot.data.projected_gravity_b[:, 2] > 0