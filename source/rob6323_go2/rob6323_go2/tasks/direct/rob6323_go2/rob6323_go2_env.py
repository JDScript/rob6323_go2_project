# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from .mdp.rewards import *
from .mdp.terminations import *
from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # Linear(xy) & Angular(z/yaw) Velocity Commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids = []
        self._feet_ids_sensor = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            sensor_id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids.append(id_list[0])
            self._feet_ids_sensor.append(sensor_id_list[0])

        # Variables - PD Controller
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits
        self.fs = torch.zeros(self.num_envs, 12, device=self.device)
        self.uv = torch.zeros(self.num_envs, 12, device=self.device)

        # Variables - Raibert Heuristic
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Variables - Action History Buffer
        self._last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_cmd_linVel",
                "track_cmd_angVel",
                "body_orient",
                "body_pose",
                "dof_vel",
                "dof_torque",
                "action_rate",
                "bouncing",
                "raibert_heuristic",
                "foot_clearance",
                "tracking_contacts",
            ]
        }

        # Debug Visualization
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """
        Returns the feet positions in the world frame of shape (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add height scanner
        if hasattr(self.cfg, "height_scanner"):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # Option: Direct Position Control
        # self.robot.set_joint_position_target(self.desired_joint_pos)

        # Option: PD Control
        torques = torch.clip(
            (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel),
            -self.torque_limits,
            self.torque_limits,
        )

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # check height scanner and append to observation
        if hasattr(self, "_height_scanner"):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
            obs = torch.cat([obs, height_data], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Update gait state
        self._step_contact_targets()

        # Rewards dict
        rewards = {
            "track_cmd_linVel": reward_trackCMD_linVel(self, self.robot) * self.cfg.rewScale_cmd_linVel,
            "track_cmd_angVel": reward_trackCMD_angVel(self, self.robot) * self.cfg.rewScale_cmd_angVel,
            "body_orient": reward_bodyOrient(self.robot) * self.cfg.rewScale_body_orient,
            "body_pose": reward_bodyPose(self.robot) * self.cfg.rewScale_body_pose,
            "dof_vel": reward_dofVel(self.robot) * self.cfg.rewScale_dofVel,
            "dof_torque": reward_dofTorque(self.robot) * self.cfg.rewScale_dofTorque,
            "action_rate": reward_actionRate(self, self.cfg.action_scale) * self.cfg.rewScale_actionRate,
            "bouncing": reward_bouncing(self.robot) * self.cfg.rewScale_bounce,
            "raibert_heuristic": reward_raibertHeuristic(self, self.robot) * self.cfg.rewScale_raibertHeuristic,
            "foot_clearance": reward_footClearance(self) * self.cfg.rewScale_feetClearance,
            "tracking_contacts": reward_trackContacts(self) * self.cfg.rewScale_trackContacts,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Update Action History Buffer
        self._last_actions = torch.roll(self._last_actions, 1, 2)
        self._last_actions[:, :, 0] = self._actions[:]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = done_timeout(self)

        cstr_termination_contacts = done_baseContact(self)
        cstr_base_height_min = done_baseHeight(self.robot, self.cfg.base_height_min)
        cstr_upsidedown = done_baseUpsidedown(self.robot)
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample actuator friction model parameters
        self.fs[env_ids] = torch.zeros_like(self.fs[env_ids]).uniform_(0.0, 0.3)
        self.uv[env_ids] = torch.zeros_like(self.uv[env_ids]).uniform_(0.0, 2.5)

        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """
        set visibility of markers

        note: parent only deals with callbacks. not their visibility
        """

        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        elif hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """
        check if robot is initialized

        note: this is needed in-case the robot is de-initialized. we can't access the data
        """

        if not self.robot.is_initialized:
            return
        # get marker location
        # - base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # - resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts the XY base velocity command to arrow direction rotation
        """

        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _step_contact_targets(self) -> None:
        """
        For Raibert Heuristic Gait Shaping.
        Advances the gait clock and calculates where the feet should be based on the command velocity
        """

        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, kappa
        ).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
        )

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR


class Rob6323Go2Env_FM(Rob6323Go2Env):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

    def _apply_action(self) -> None:
        torques = torch.clip(
            (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel),
            -self.torque_limits,
            self.torque_limits,
        )

        # Apply Friction Model
        torques -= self.fs * torch.tanh(self.robot.data.joint_vel / 0.1)  # tau_stiction
        torques -= self.uv * self.robot.data.joint_vel  # tau_viscous

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_rewards(self) -> torch.Tensor:
        # Update gait state
        self._step_contact_targets()

        # Rewards dict
        rewards = {
            "track_cmd_linVel": reward_trackCMD_linVel(self, self.robot) * self.cfg.rewScale_cmd_linVel,
            "track_cmd_angVel": reward_trackCMD_angVel(self, self.robot) * self.cfg.rewScale_cmd_angVel,
            "body_orient": reward_bodyOrient(self.robot) * self.cfg.rewScale_body_orient,
            "body_pose": reward_bodyPose(self.robot) * self.cfg.rewScale_body_pose,
            "dof_vel": reward_dofVel(self.robot) * self.cfg.rewScale_dofVel,
            "dof_torque": reward_dofTorque(self.robot) * self.cfg.rewScale_dofTorque_FM,
            "action_rate": reward_actionRate(self, self.cfg.action_scale) * self.cfg.rewScale_actionRate_FM,
            "bouncing": reward_bouncing(self.robot) * self.cfg.rewScale_bounce,
            "raibert_heuristic": reward_raibertHeuristic(self, self.robot) * self.cfg.rewScale_raibertHeuristic,
            "foot_clearance": reward_footClearance(self) * self.cfg.rewScale_feetClearance,
            "tracking_contacts": reward_trackContacts(self) * self.cfg.rewScale_trackContacts,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Update Action History Buffer
        self._last_actions = torch.roll(self._last_actions, 1, 2)
        self._last_actions[:, :, 0] = self._actions[:]

        return reward
