"""TQC Humanoid Standing Task - HoST Style with staged progression."""

import asyncio
import math

import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim import ConstantCurriculum

from tqc_config import TQCHumanoidConfig
from tqc_data_structures import TQCInputs, TQCVariables
from tqc_networks import TqcModel
from tqc_task import TQCHumanoidTask
from tqc_utils import ZEROS

# Import the new HoST-style rewards
from tqc_standing_rewards import (

    MuJoCoStandupHeightReward, MuJoCoUprightReward, SimpleUprightReward, FootContactReward, ContactPenalty,
    FootStabilityReward, MirrorSymmetryReward, SimpleHeadUprightReward, ConditionalJointPositionReward)

# Keep some useful existing rewards
from pathlib import Path


class TQCHumanoidStandingTask(TQCHumanoidTask[TQCHumanoidConfig]):
    """TQC humanoid standing task with HoST-style staged progression."""

    def get_tqc_variables(
        self,
        model: TqcModel,
        tqc_inputs: TQCInputs,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[TQCVariables, PyTree]:
        """Get TQC variables for the given inputs."""
        dummy_carry = jnp.zeros(1)

        # Get quantiles from all critics
        all_quantiles = []
        for critic in model.critics:
            quantiles, _ = critic.forward(tqc_inputs.critic_observations, tqc_inputs.actions, dummy_carry)
            all_quantiles.append(quantiles)

        # Stack quantiles: (num_critics, batch_size, num_quantiles)
        quantile_values = jnp.stack(all_quantiles, axis=0)

        _, log_probs = model.actor.get_action_and_log_prob(tqc_inputs.actor_observations, rng)

        tqc_variables = TQCVariables(
            quantile_values=quantile_values,
            log_probs=log_probs,
            entropy=-log_probs,
        )

        return tqc_variables, model_carry

    def get_mujoco_model(self) -> mujoco.MjModel:
        """Get MuJoCo model."""
        #mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        mjcf_path = "./kbot/robot/robot.mjcf"
        print(f"Directory: {Path(mjcf_path).parent}")  # Add this line
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        """Get model metadata."""
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        return metadata

    def get_actuators(self, physics_model: ksim.PhysicsModel, metadata: ksim.Metadata | None = None) -> ksim.Actuators:
        """Get actuators."""
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(physics_model=physics_model, metadata=metadata)

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        """Get physics randomizers."""
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.98, scale_upper=1.02),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-1), scale_upper=math.radians(1)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        """Get disturbance events."""
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.3,
                force_range=(0.3, 0.8),  # Gentler pushes for standing
                x_angular_force=0.0,
                y_angular_force=0.0,
                z_angular_force=0.0,
                interval_range=(2.0, 8.0),  # Less frequent for standing
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        """Get reset conditions - start from various lying/sitting positions."""
        return [
            # Start from lying positions to encourage standing progression
            ksim.RandomJointPositionReset.create(
                physics_model,
                {k: v for k, v in ZEROS},
                scale=0.01,  # More variation to encourage different starting postures
            ),
            ksim.RandomJointVelocityReset(scale=0.3),  # Lower velocities for standing
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        """Enhanced observations with contact sensing for better standing control."""
        return [
            # 🕒 EXISTING OBSERVATIONS (keep all these)
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(1)),
            ksim.JointVelocityObservation(noise=math.radians(5)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.05),
                noise=math.radians(0.5),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=0.5,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(5),
            ),

            # 🦶 FOOT CONTACT OBSERVATIONS (Using actual foot geom names from your MJCF)
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=[
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1"
                ],
                foot_right_geom_names=[
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1"
                ],
                floor_geom_names=["floor" ],  # Add these to your environment
                noise=0.1,
            ),

            # 🦶 FOOT POSITION AWARENESS (Using actual body names)
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
                noise=0.01,
            ),

            # 🤲 HAND/ARM CONTACT PREVENTION (Using actual wrist/hand geom names)
            ksim.ContactObservation.create(
                physics_model=physics_model,
                geom_names=[
                    "left_wrist_collision_capsule",
                    "right_wrist_collision_capsule",
                    "left_forearm_collision_capsule",
                    "right_forearm_collision_capsule"
                ],
                contact_group="hands",
                noise=0.1,
            ),

            # 🔍 ADDITIONAL CONTACT OBSERVATIONS FOR BODY PARTS
            ksim.ContactObservation.create(
                physics_model=physics_model,
                geom_names=[
                    "torso_collision_box",
                    "head_collision_box"
                ],
                contact_group="torso_head",
                noise=0.1,
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        """Get commands."""
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        """Rebalanced rewards with height stabilization and drastic move penalties."""


        return [
            # 🎯 PRIMARY OBJECTIVES (Height-aware scaling)
            MuJoCoStandupHeightReward(
                target_height=0.95,  # Primary goal: reach standing height
                scale=50.0,  # High priority for standing up
            ),

            SimpleHeadUprightReward.create(
                physics_model=physics_model,
                imu_body_name="Torso_Side_Right",
                scale=3.0,
            ),

            # 🤖 SUPPORTING OBJECTIVES
            MirrorSymmetryReward.create(
                physics_model=physics_model,
                scale=3,  # Reduced since conditional pose reward handles this
                tolerance=0.4,
            ),

            FootContactReward.create(
                physics_model=physics_model,
                foot_body_names=(
                    "KB_D_501L_L_LEG_FOOT",
                    "KB_D_501R_R_LEG_FOOT",
                ),
                floor_geom_names=("floor",),
                scale=50.0,  # Important for standing stability
            ),

            FootStabilityReward.create(
                physics_model=physics_model,
                foot_body_names=(
                    "KB_D_501L_L_LEG_FOOT",
                    "KB_D_501R_R_LEG_FOOT"
                ),
                scale=10.0,
            ),

            # 🎯 HEIGHT STABILIZATION (NEW)
            ksim.BaseHeightRangeReward(
                z_lower=0.4,
                z_upper=1.00,
                dropoff=35.0,  # Sharp penalty outside standing range
                scale=25.0,  # Strong reward for being in standing zone
            ),

            ConditionalJointPositionReward.create(
                physics_model=physics_model,
                min_height=0.75,  # Start applying pose reward at this height
                max_height=1.00,  # Stop applying pose reward above this height
                tolerance=0.4,
                scale=40.0,  # Very strong when active
            ),

            # 🚫 DRASTIC MOVEMENT PENALTIES (INCREASED)
            ksim.AngularVelocityPenalty(index=("x", "y", "z"), scale=-0.005),
            ksim.LinearVelocityPenalty(index=("z"), scale=-0.01),

            # 🚫 LARGE ACTION PENALTIES (INCREASED)
            ksim.ActionAccelerationPenalty(scale=-0.0005),
            #ksim.ActionJerkPenalty(scale=-0.03),
            ksim.JointVelocityPenalty(scale=-0.0005),
            #ksim.JointAccelerationPenalty(scale=-0.04),
            #ksim.JointJerkPenalty(scale=-0.04),

            # 🚫 CONTACT PENALTIES
            ContactPenalty.create(
                physics_model=physics_model,
                body_names=(
                    "Torso_Side_Right",
                    #"KC_C_401R_R_UpForearmDrive",
                    #"KC_C_401L_L_UpForearmDrive",
                ),
                floor_geom_names=("floor",),
                scale=-0.1,
            ),

            # 🔋 ENERGY PENALTIES
            ksim.CtrlPenalty(scale=-0.0001),
            ksim.LinkAccelerationPenalty(scale=-0.01),
            ksim.AvoidLimitsPenalty.create(
                model=physics_model,
                factor=0.1,
                scale=-0.02
            ),
        ]



    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        """Get termination conditions - more forgiving for standing."""
        return [
            # More forgiving height limits for standing attempts
            ksim.BadZTermination(unhealthy_z_lower=-0.1, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=2.0),  # Allow some movement during standing
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        """Simple curriculum for standing task."""
        constant = ConstantCurriculum(level=1.0)
        return constant
