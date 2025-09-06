"""Sequential Standing Reward System - Stage-based progression for natural standing."""

import attrs
import jax
import jax.numpy as jnp
import ksim
import math

import xax
from ksim.types import Trajectory, PhysicsModel, PhysicsState
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_geom_data_idx_from_name, get_body_data_idx_from_name, get_qpos_data_idxs_by_name


@attrs.define(frozen=True, kw_only=True)
class MuJoCoStandupHeightReward(ksim.Reward):
    """Enhanced MuJoCo Standup v5 height reward with target height and stabilization."""
    #dt: float = attrs.field(default=0.02)
    target_height: float = attrs.field(default=0.95)  # Target standing height
    max_reward_height: float = attrs.field(default=0.96)  # Cap reward above this height
    stabilization_zone: float = attrs.field(default=0.1)  # Zone around target for stabilization

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]  # z-coordinate (base height)

        # Original MuJoCo reward: height / dt (encourages getting higher)
        base_reward = height * 10.0
        # Add stabilization component when near target
        distance_from_target = jnp.abs(height - self.target_height)

        # Bonus for being close to target height
        stabilization_bonus = jnp.where(
            distance_from_target < self.stabilization_zone,
            (self.stabilization_zone - distance_from_target) / self.stabilization_zone * 5.0,  # Up to 5.0 bonus
            0.0
        )

        # Reduce reward if too high (prevent excessive jumping)
        height_cap_factor = jnp.where(
            height > self.max_reward_height,
            jnp.exp(-(height - self.max_reward_height) / 0.1),  # Exponential decay above max
            1.0
        )

        # Combine: base MuJoCo reward + stabilization bonus + height cap
        total_reward = (base_reward + stabilization_bonus) * height_cap_factor

        # Debug prints
        #jax.debug.print("Height: {}, Base reward: {}, Stabilization: {}, Cap factor: {}, Total: {}",
        #                height[0], base_reward[0], stabilization_bonus[0],
        #                height_cap_factor[0], total_reward[0])

        return total_reward


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
        total = uprightness ** 2
        # Square it for sharper reward
        return total

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
    foot_body_indices: tuple[int, ...] = attrs.field()
    floor_z: float = attrs.field()
    contact_threshold: float = attrs.field(default=0.05)

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Reward for both feet maintaining ground contact."""
        foot_rewards = []

        for i, foot_idx in enumerate(self.foot_body_indices):
            foot_pos = trajectory.xpos[..., foot_idx, :]
            foot_z = foot_pos[..., 2]
            foot_height = foot_z - self.floor_z

            # Strong reward for foot being on or very close to ground
            contact_reward = jnp.exp(-jnp.maximum(foot_height, 0.0) / self.contact_threshold)
            foot_rewards.append(contact_reward)

        # For both feet on ground, multiply rewards instead of averaging
        # This ensures maximum reward only when BOTH feet are in contact
        foot_rewards_stacked = jnp.stack(foot_rewards, axis=-1)

        # Option 1: Multiply rewards (both feet must be down)
        both_feet_contact = jnp.prod(foot_rewards_stacked, axis=-1)

        # Option 2: Minimum of the two (weakest link approach)
        # both_feet_contact = jnp.min(foot_rewards_stacked, axis=-1)

        return both_feet_contact

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            foot_body_names: tuple[str, ...],  # Use body names (this works!)
            floor_geom_names: tuple[str, ...],
            scale: float = 1.0,
            contact_threshold: float = 0.05,
            scale_by_curriculum: bool = False,
    ) -> "FootContactReward":
        # Get foot body indices (not geom indices)
        foot_body_indices = tuple([
            get_body_data_idx_from_name(physics_model, name)
            for name in foot_body_names
        ])

        # Get floor Z position
        floor_indices = tuple([
            get_geom_data_idx_from_name(physics_model, name)
            for name in floor_geom_names
        ])
        floor_pos = physics_model.geom_pos[floor_indices[0]]
        floor_z = float(floor_pos[2])

        # Debug
        #jax.debug.print("Foot body indices: {}", foot_body_indices)
        #jax.debug.print("Floor Z coordinate: {}", floor_z)

        return cls(
            foot_body_indices=foot_body_indices,
            floor_z=floor_z,
            contact_threshold=contact_threshold,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ContactPenalty(ksim.Reward):
    """Penalty for unwanted body parts touching the ground."""
    body_indices: tuple[int, ...] = attrs.field()  # Changed from geom_indices
    floor_z: float = attrs.field()
    contact_threshold: float = attrs.field(default=0.05)

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Penalize specific body parts getting too close to ground."""

        # Check each unwanted body part for contact
        contact_penalties = []

        for i, body_idx in enumerate(self.body_indices):
            # Get body part position from trajectory.xpos
            body_pos = trajectory.xpos[..., body_idx, :]
            body_z = body_pos[..., 2]

            #jax.debug.print("Body part {} Z: {}", i, body_z[0])

            # Distance from body part to floor
            height_above_floor = body_z - self.floor_z
            #jax.debug.print("Body part {} height above floor: {}", i, height_above_floor[0])

            # Penalty for being too close to floor
            min_safe_height = 0.15  # Minimum safe height for body parts
            penalty = jnp.where(
                height_above_floor < min_safe_height,
                jnp.exp(-(height_above_floor - min_safe_height) / 0.02),  # Sharp penalty when close
                0.0  # No penalty when safe distance
            )

            contact_penalties.append(penalty)

        # Sum penalties (worse if multiple body parts contact)
        total_penalty = jnp.sum(jnp.stack(contact_penalties, axis=-1), axis=-1)

        #jax.debug.print("Total contact penalty: {}", total_penalty[0])

        return total_penalty

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            body_names: tuple[str, ...],  # Changed from geom_names
            floor_geom_names: tuple[str, ...] = ("floor",),
            scale: float = -1.0,
            contact_threshold: float = 0.05,
            scale_by_curriculum: bool = False,
    ) -> "ContactPenalty":
        # Get body indices (not geom indices)
        body_indices = tuple([
            get_body_data_idx_from_name(physics_model, name)
            for name in body_names
        ])

        # Get floor Z position
        floor_indices = tuple([
            get_geom_data_idx_from_name(physics_model, name)
            for name in floor_geom_names
        ])
        floor_pos = physics_model.geom_pos[floor_indices[0]]
        floor_z = float(floor_pos[2])

        # Debug
        #jax.debug.print("Body indices: {}", body_indices)
        #jax.debug.print("Floor Z coordinate: {}", floor_z)

        return cls(
            body_indices=body_indices,
            floor_z=floor_z,
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
        total =  (jnp.mean(stacked_rewards, axis=-1) ** 2) * 10 # Shape: (batch_size,) - preserves batch dimension
        return total

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




@attrs.define(frozen=True, kw_only=True)
class ConditionalJointPositionReward(ksim.Reward):
    """Joint position reward that only activates when robot is standing at correct height."""
    min_height: float = attrs.field(default=0.85)
    max_height: float = attrs.field(default=1.05)
    tolerance: float = attrs.field(default=0.3)  # Similar to symmetry reward

    def get_reward(self, trajectory: Trajectory) -> Array:
        """Joint position reward that ramps up when robot reaches standing height."""

        # Get current base height
        base_height = trajectory.qpos[..., 2]

        # Height-based activation (simplified)
        height_factor = jnp.where(
            (base_height >= self.min_height) & (base_height <= self.max_height),
            1.0,
            0.0
        )

        # Get joint positions (same as symmetry reward)
        joint_pos = trajectory.qpos[..., 7:]  # Skip base position/orientation (first 7)

        # Target joint positions (hardcoded indices like symmetry reward)
        # Based on your target_joint_positions dict, in order:
        target_positions = jnp.array([
            0.0,                    # right_shoulder_pitch
            math.radians(-10.0),    # right_shoulder_roll
            0.0,                    # right_shoulder_yaw
            math.radians(90.0),     # right_elbow
            0.0,                    # right_wrist
            0.0,                    # left_shoulder_pitch
            math.radians(10.0),     # left_shoulder_roll
            0.0,                    # left_shoulder_yaw
            math.radians(-90.0),    # left_elbow
            0.0,                    # left_wrist
            math.radians(-20.0),    # right_hip_pitch
            math.radians(0.0),      # right_hip_roll
            0.0,                    # right_hip_yaw
            math.radians(-50.0),    # right_knee
            math.radians(30.0),     # right_ankle
            math.radians(20.0),     # left_hip_pitch
            math.radians(0.0),      # left_hip_roll
            0.0,                    # left_hip_yaw
            math.radians(50.0),     # left_knee
            math.radians(-30.0),    # left_ankle
        ])

        # Calculate joint position rewards (same pattern as symmetry)
        joint_rewards = []
        for i in range(len(target_positions)):
            joint_error = jnp.abs(joint_pos[..., i] - target_positions[i])
            joint_reward = jnp.exp(-joint_error / self.tolerance)
            joint_rewards.append(joint_reward)

        # Stack and average (same as symmetry reward)
        stacked_rewards = jnp.stack(joint_rewards, axis=-1)
        joint_reward_total = jnp.mean(stacked_rewards, axis=-1)

        # Apply height-based activation
        final_reward = joint_reward_total * height_factor * 10.0

        # Debug output
        #jax.debug.print("Base height: {:.3f}", base_height[0])
        #jax.debug.print("Height factor: {:.3f}", height_factor[0])
        #jax.debug.print("Joint reward total: {:.6f}", joint_reward_total[0])
        #jax.debug.print("Final conditional reward: {:.3f}", final_reward[0])

        return final_reward

    @classmethod
    def create(
            cls,
            physics_model: PhysicsModel,
            min_height: float = 0.65,
            max_height: float = 1.00,
            tolerance: float = 0.3,
            scale: float = 1.0,
            scale_by_curriculum: bool = False,
    ) -> "ConditionalJointPositionReward":
        return cls(
            min_height=min_height,
            max_height=max_height,
            tolerance=tolerance,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )