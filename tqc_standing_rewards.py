"""Sequential Standing Reward System - Stage-based progression for natural standing."""

import attrs
import jax
import jax.numpy as jnp
import ksim
import math

import xax
from ksim.types import Trajectory, PhysicsModel
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_geom_data_idx_from_name, get_body_data_idx_from_name


@attrs.define(frozen=True, kw_only=True)
class MuJoCoStandupHeightReward(ksim.Reward):
    """Exact MuJoCo Standup v5 height reward: uph_cost = height / dt"""

    dt: float = attrs.field(default=0.02)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]  # z-coordinate (torso height)
        return height / self.dt


@attrs.define(frozen=True, kw_only=True)
class MuJoCoUprightReward(ksim.Reward):
    """Simple MuJoCo-style upright posture reward."""

    dt: float = attrs.field(default=0.02)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        quat = trajectory.qpos[..., 3:7]  # Base orientation quaternion
        qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # Z-axis alignment (how upright the torso is)
        uprightness = 1 - 2 * (qx * qx + qy * qy)

        return uprightness / self.dt


@attrs.define(frozen=True, kw_only=True)
class SimpleUprightReward(ksim.Reward):
    """Simple upright reward that's always positive."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        quat = trajectory.qpos[..., 3:7]  # Base orientation
        # Convert quaternion to rotation matrix and get Z-axis alignment
        local_z = jnp.array([0.0, 0.0, 1.0])
        global_z = xax.rotate_vector_by_quat(local_z, quat)

        # Return Z-component: 1.0 when perfectly upright, 0.0 when completely sideways
        return jnp.maximum(0.0, global_z[..., 2])  # Ensure non-negative


@attrs.define(frozen=True, kw_only=True)
class SimpleHeadUprightReward(ksim.Reward):
    """Simple reward for keeping head upright."""

    imu_body_idx: int = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Simple upright head reward."""

        # Get IMU body orientation
        imu_quat = trajectory.xquat[..., self.imu_body_idx, :]

        # Convert quaternion to get local Z-axis in global frame
        local_z = jnp.array([0.0 , 0.0, 1.0 ])
        global_z = xax.rotate_vector_by_quat(local_z, imu_quat)

        # Reward when head Z-axis points up (1.0 = perfectly upright, 0.0 = sideways)
        uprightness = jnp.maximum(0.0, global_z[..., 2])

        # Square it for sharper reward
        return uprightness ** 2

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            imu_body_name: str = "imu",
            scale: float = 1.0,
            scale_by_curriculum: bool = False,
    ) -> "SimpleHeadUprightReward":
        # Get IMU body index
        imu_body_idx = get_body_data_idx_from_name(physics_model, imu_body_name)

        return cls(
            imu_body_idx=imu_body_idx,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class FootContactReward(ksim.Reward):
    """Reward for keeping feet in contact with the ground."""

    foot_geom_indices: tuple[int, ...] = attrs.field()
    floor_geom_indices: tuple[int, ...] = attrs.field()
    contact_threshold: float = attrs.field(default=0.1)  # Minimum contact force

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Reward based on foot height (proxy for contact)."""
        # Use base position as proxy for foot contact
        # When robot is standing, base should be at stable height
        base_height = trajectory.qpos[..., 2]  # Z-coordinate of base

        # Reward stable height (around 0.5-1.0m for standing robot)
        target_height = 0.7  # Adjust based on your robot
        height_reward = jnp.exp(-jnp.abs(base_height - target_height) / 0.1)

        # Also reward low base velocities (stable standing)
        base_vel = trajectory.qvel[..., :3]  # Linear velocity
        vel_norm = jnp.linalg.norm(base_vel, axis=-1)
        stability_reward = jnp.exp(-vel_norm / 0.1)

        # Combine both rewards
        total_reward = height_reward * stability_reward
        return total_reward

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            foot_geom_names: tuple[str, ...],
            floor_geom_names: tuple[str, ...],
            scale: float = 1.0,
            contact_threshold: float = 0.1,
            scale_by_curriculum: bool = False,
    ) -> "FootContactReward":
        # Get geom indices
        foot_indices = tuple([get_geom_data_idx_from_name(physics_model, name) for name in foot_geom_names])
        floor_indices = tuple([get_geom_data_idx_from_name(physics_model, name) for name in floor_geom_names])

        return cls(
            foot_geom_indices=foot_indices,
            floor_geom_indices=floor_indices,
            contact_threshold=contact_threshold,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ContactPenalty(ksim.Reward):
    """Penalty for unwanted body parts touching the ground."""

    geom_indices: tuple[int, ...] = attrs.field()
    floor_geom_indices: tuple[int, ...] = attrs.field()
    contact_threshold: float = attrs.field(default=0.05)

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Penalize base getting too close to ground (proxy for unwanted contact)."""
        # Use base height as proxy for body contact
        base_height = trajectory.qpos[..., 2]  # Z-coordinate

        # Penalize if base is too low (indicates body parts touching ground)
        min_height = 0.3  # Minimum safe height
        contact_penalty = jnp.where(
            base_height < min_height,
            jnp.exp(-(base_height - min_height) / 0.05),  # Sharp penalty below threshold
            0.0  # No penalty above threshold
        )

        return contact_penalty

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            geom_names: tuple[str, ...],
            floor_geom_names: tuple[str, ...] = ("floor",),
            scale: float = -1.0,
            contact_threshold: float = 0.05,
            scale_by_curriculum: bool = False,
    ) -> "ContactPenalty":
        # Get geom indices
        geom_indices = tuple([get_geom_data_idx_from_name(physics_model, name) for name in geom_names])
        floor_indices = tuple([get_geom_data_idx_from_name(physics_model, name) for name in floor_geom_names])

        return cls(
            geom_indices=geom_indices,
            floor_geom_indices=floor_indices,
            contact_threshold=contact_threshold,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class FootStabilityReward(ksim.Reward):
    """Reward for stable foot placement and minimal foot movement."""

    foot_body_indices: tuple[int, ...] = attrs.field()
    velocity_threshold: float = attrs.field(default=0.1)
    position_stability_weight: float = attrs.field(default=0.5)

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Reward stable base position and low angular velocity."""
        # Use base linear velocity for stability
        base_linvel = trajectory.qvel[..., :3]  # Linear velocity
        linvel_norm = jnp.linalg.norm(base_linvel, axis=-1)
        velocity_reward = jnp.exp(-linvel_norm / self.velocity_threshold)

        # Use base angular velocity for rotational stability
        base_angvel = trajectory.qvel[..., 3:6]  # Angular velocity
        angvel_norm = jnp.linalg.norm(base_angvel, axis=-1)
        angular_stability = jnp.exp(-angvel_norm / 0.1)

        # Combine both stability measures
        total_reward = velocity_reward * angular_stability
        return total_reward

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            foot_body_names: tuple[str, ...],
            scale: float = 1.0,
            velocity_threshold: float = 0.1,
            position_stability_weight: float = 0.5,
            scale_by_curriculum: bool = False,
    ) -> "FootStabilityReward":
        from ksim.utils.mujoco import get_body_data_idx_from_name

        # Get body indices for feet
        foot_indices = tuple([get_body_data_idx_from_name(physics_model, name) for name in foot_body_names])

        return cls(
            foot_body_indices=foot_indices,
            velocity_threshold=velocity_threshold,
            position_stability_weight=position_stability_weight,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class MirrorSymmetryReward(ksim.Reward):
    """Reward for perfect mirror symmetry - left +ve = right -ve."""

    tolerance: float = attrs.field(default=0.2)

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Reward when left and right joints are exact opposites."""
        joint_pos = trajectory.qpos[..., 7:]  # Skip base position/orientation (first 7)

        # For perfect mirror symmetry: left_joint + right_joint should = 0
        # Joint pairs: (right_idx, left_idx)
        symmetry_rewards = []

        # Arms - all should mirror
        symmetry_rewards.append(
            jnp.exp(-jnp.abs(joint_pos[..., 0] + joint_pos[..., 5]) / self.tolerance))  # shoulder_pitch
        symmetry_rewards.append(
            jnp.exp(-jnp.abs(joint_pos[..., 1] + joint_pos[..., 6]) / self.tolerance))  # shoulder_roll
        symmetry_rewards.append(
            jnp.exp(-jnp.abs(joint_pos[..., 2] + joint_pos[..., 7]) / self.tolerance))  # shoulder_yaw
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 3] + joint_pos[..., 8]) / self.tolerance))  # elbow
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 4] + joint_pos[..., 9]) / self.tolerance))  # wrist

        # Legs - all should mirror
        symmetry_rewards.append(
            jnp.exp(-jnp.abs(joint_pos[..., 10] + joint_pos[..., 15]) / self.tolerance))  # hip_pitch
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 11] + joint_pos[..., 16]) / self.tolerance))  # hip_roll
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 12] + joint_pos[..., 17]) / self.tolerance))  # hip_yaw
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 13] + joint_pos[..., 18]) / self.tolerance))  # knee
        symmetry_rewards.append(jnp.exp(-jnp.abs(joint_pos[..., 14] + joint_pos[..., 19]) / self.tolerance))  # ankle

        # Return average symmetry across all joint pairs
        stacked_rewards = jnp.stack(symmetry_rewards, axis=-1)  # Shape: (batch_size, 10)
        return (jnp.mean(stacked_rewards, axis=-1) ** 2) * 10 # Shape: (batch_size,) - preserves batch dimension

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            scale: float = 1.0,
            tolerance: float = 0.3,
            scale_by_curriculum: bool = False,
    ) -> "MirrorSymmetryReward":
        return cls(
            tolerance=tolerance,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )