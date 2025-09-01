"""Main TQC task implementation with CIRCULAR GRADIENT SUPPORT."""

import asyncio
import functools
import math
import signal
import textwrap
import traceback
from abc import ABC, abstractmethod
from dataclasses import replace
from threading import Thread
from types import FrameType
from typing import Generic, TypeVar

import sys
import datetime
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim import log_joint_config_table
from ksim.debugging import JitLevel
from ksim.task.rl import RLConfig, RLLoopCarry, RLLoopConstants, RLTask, get_viewer, logger
from ksim.types import LoggedTrajectory, RewardState, Trajectory, PhysicsModel
from xax import load_ckpt
from xax.core import state

from tqc_config import TQCHumanoidConfig
from tqc_data_structures import TQCInputs, TQCVariables
from tqc_losses import (
    compute_tqc_actor_loss,
    compute_tqc_critic_loss,
    compute_tqc_temperature_loss,
    update_target_networks,
    create_quantile_levels,
)
from tqc_networks import TqcModel
from tqc_replay_buffer import ReplayBufferState, TQCReplayBuffer
from tqc_utils import (
    create_tqc_opt_state,
    extract_from_tqc_opt_state,
    ZEROS,
)
from pathlib import Path
Config = TypeVar("Config", bound=TQCHumanoidConfig)


class TQCHumanoidTask(RLTask[Config], Generic[Config], ABC):
    """Complete TQC task with circular gradient support."""

    def __init__(self, config: Config):
        # Store the original TQC batch size before any modifications
        self.tqc_batch_size = config.batch_size

        # Find a framework-compatible batch size (largest divisor of num_envs)
        framework_batch_size = self._find_framework_batch_size(config.num_envs, config.batch_size)

        # Create a modified config for the parent class to satisfy framework constraint
        framework_config = replace(config, batch_size=framework_batch_size)

        # Call parent's __init__ with framework-compatible config
        super().__init__(framework_config)

        # Store original config for TQC operations
        self.original_config = config

        print(f"Framework batch_size: {framework_batch_size} (for {config.num_envs} envs)")
        print(f"TQC training batch_size: {self.tqc_batch_size}")

        # Use functional buffer for JAX compatibility
        actor_obs_dim = 51 if config.use_acc_gyro else 45
        critic_obs_dim = 462  # Rich state representation
        action_dim = len(ZEROS)

        # Compute auto target entropy
        self.action_dim = action_dim
        self.target_entropy_value = config.get_target_entropy(action_dim)

        print(f"Target entropy: {self.target_entropy_value:.2f} " +
              f"({'auto (-action_dim)' if config.target_entropy == 'auto' else 'manual'})")
        print(f"Entropy coef: {'auto (learnable)' if config.use_auto_entropy() else 'manual'}")

        # Circular gradient setup
        if config.use_circular_gradients:
            print(f"CIRCULAR GRADIENTS: ENABLED (EXPERIMENTAL)")
            print(f"   This allows temperature gradients to flow through actor/critic losses")
            print(f"   Reduced learning rates for stability:")
            print(f"      - Actor: {config.learning_rate_actor} × {config.circular_actor_lr_scale}")
            print(f"      - Critic: {config.learning_rate_critic} × {config.circular_critic_lr_scale}")
            print(f"      - Temperature: {config.learning_rate_temp_circular}")
            print(f"   Gradient clipping: {config.gradient_clip_circular}")
        else:
            print(f"DETACHED GRADIENTS: Standard mode (recommended)")

        # Always use functional buffer for JAX compatibility
        self.replay_buffer = TQCReplayBuffer(
            config.buffer_size, actor_obs_dim, critic_obs_dim, action_dim
        )
        self.buffer_type = "functional"

        # Pre-compute quantile levels for TQC
        self.tau = create_quantile_levels(config.num_quantiles)

        print(f"Using functional TQCReplayBuffer: {config.buffer_size:,} transitions")
        print(f"Actor obs: {actor_obs_dim}D, Critic obs: {critic_obs_dim}D, Actions: {action_dim}D")
        print(f"Buffer type: {self.buffer_type} (JAX JIT compatible)")
        print(
            f"TQC: {config.num_critics} critics × {config.num_quantiles} quantiles = {config.num_critics * config.num_quantiles} total quantiles")

        if config.buffer_size > 500_000:
            print(f"Large buffer size ({config.buffer_size:,}) - functional buffers use more memory")
            print("   Consider reducing buffer_size for better memory efficiency")

    def _find_framework_batch_size(self, num_envs: int, desired_batch_size: int) -> int:
        """Find largest valid batch size that divides num_envs."""
        # Get all divisors of num_envs
        divisors = [i for i in range(1, num_envs + 1) if num_envs % i == 0]

        # Return the largest divisor that's <= desired_batch_size
        valid_divisors = [d for d in divisors if d <= desired_batch_size]

        if valid_divisors:
            return max(valid_divisors)
        else:
            # If desired batch is smaller than smallest divisor, use smallest
            return min(divisors)

    @property
    def batch_size(self) -> int:
        """Return TQC batch size for training operations."""
        return self.tqc_batch_size

    @functools.cached_property
    def rollout_num_samples(self) -> int:
        """Return TQC training samples per rollout step."""

        # Environment sample collection
        env_samples = self.rollout_length_steps * self.config.num_envs

        # Training configuration values
        gradient_steps = getattr(self.config, 'gradient_steps', 1)
        critic_updates = getattr(self.config, 'critic_updates_per_step', 1)
        tqc_batch = getattr(self, 'tqc_batch_size', self.config.batch_size)
        rollout_length = self.rollout_length_steps

        print(f"DEBUG rollout_num_samples:")
        print(f"  Environment: num_envs={self.config.num_envs}, rollout_length={rollout_length}")
        print(f"  env_samples = {self.config.num_envs} × {rollout_length} = {env_samples}")
        print(
            f"  Training: gradient_steps={gradient_steps}, critic_updates={critic_updates}, tqc_batch_size={tqc_batch}")

        # Training sample calculations
        total_critic_samples = critic_updates * tqc_batch
        total_actor_samples = tqc_batch
        samples_per_gradient_step = total_critic_samples + total_actor_samples
        total_training_samples = gradient_steps * samples_per_gradient_step

        print(f"  Critic samples per gradient step: {critic_updates} × {tqc_batch} = {total_critic_samples}")
        print(f"  Actor samples per gradient step: {tqc_batch}")
        print(f"  Total per gradient step: {total_critic_samples} + {tqc_batch} = {samples_per_gradient_step}")
        print(f"  Total training samples: {gradient_steps} × {samples_per_gradient_step} = {total_training_samples}")

        # Combined total
        combined_total = env_samples + total_training_samples
        print(f"  FINAL: env_samples({env_samples}) + training_samples({total_training_samples}) = {combined_total}")

        return combined_total

    @abstractmethod
    def get_tqc_variables(
        self,
        model: PyTree,
        tqc_inputs: TQCInputs,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[TQCVariables, PyTree]:
        """Get TQC variables for loss computation."""
        pass

    def get_optimizer(self) -> optax.GradientTransformation:
        """Get dummy optimizer for framework compatibility."""
        return optax.adam(self.config.learning_rate_actor)

    def get_tqc_optimizers(self) -> tuple[optax.GradientTransformation, ...]:
        """Get separate optimizers for TQC components with circular gradient support."""

        if self.config.use_circular_gradients:
            # 🔄 CIRCULAR MODE: Lower learning rates + aggressive gradient clipping
            print("🔄 Setting up CIRCULAR GRADIENT optimizers")

            clip_value = self.config.gradient_clip_circular

            # Scaled learning rates for stability
            actor_lr = self.config.learning_rate_actor * self.config.circular_actor_lr_scale
            critic_lr = self.config.learning_rate_critic * self.config.circular_critic_lr_scale
            temp_lr = self.config.learning_rate_temp_circular

            print(f"   🎭 Actor LR: {actor_lr:.2e} (scaled from {self.config.learning_rate_actor:.2e})")
            print(f"   🎯 Critic LR: {critic_lr:.2e} (scaled from {self.config.learning_rate_critic:.2e})")
            print(f"   🌡️  Temperature LR: {temp_lr:.2e}")

            optimizers = [
                optax.chain(
                    optax.clip_by_global_norm(clip_value),
                    optax.adam(actor_lr)
                ),
                optax.chain(
                    optax.clip_by_global_norm(clip_value),
                    optax.adam(critic_lr)
                ),
            ]

            if self.config.use_auto_entropy():
                optimizers.append(
                    optax.chain(
                        optax.clip_by_global_norm(clip_value * 0.1),  # Even more aggressive for temp
                        optax.adam(temp_lr)
                    )
                )
        else:
            clip_value = self.config.gradient_clip_norm
            actor_lr = self.config.learning_rate_actor
            critic_lr = self.config.learning_rate_critic
            temp_lr = self.config.learning_rate_temp
            # 🛡️ DETACHED MODE: Standard learning rates
            optimizers = [
                optax.chain(
                    optax.clip_by_global_norm(clip_value),
                    optax.adam(actor_lr)
                ),
                # Critics: moderate clipping
                optax.chain(
                    optax.clip_by_global_norm(clip_value),
                    optax.adam(critic_lr)
                ),
            ]

            if self.config.use_auto_entropy():
                optimizers.append(
                    optax.chain(
                        #optax.clip_by_global_norm(1),  # Very aggressive!
                        optax.adam(temp_lr)  # 10x lower just for temperature
                    )
                )

        return tuple(optimizers)

    def initialize_rl_training(
        self,
        mj_model: ksim.PhysicsModel,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopConstants, RLLoopCarry, xax.State]:
        """Override initialization to setup TQC structure with circular gradient support."""
        # Get the model and basic state setup
        rng, model_rng = jax.random.split(rng)
        models, optimizers, opt_states, state = self.load_initial_state(model_rng, load_optimizer=True)

        # Initialize TQC-specific optimizers and buffer state
        model = models[0]  # Assuming single model

        # Initialize TQC optimizers (handles circular vs detached automatically)
        tqc_optimizers = self.get_tqc_optimizers()
        actor_opt_state = tqc_optimizers[0].init(model.actor)
        critics_opt_state = tqc_optimizers[1].init(model.critics)

        # Temperature optimizer only if using auto entropy
        if self.config.use_auto_entropy():
            temp_opt_state = tqc_optimizers[2].init(model.temperature)
            opt_tuple = (actor_opt_state, critics_opt_state, temp_opt_state)
            mode_str = "CIRCULAR" if self.config.use_circular_gradients else "DETACHED"
            print(f"🌡️  Temperature optimizer initialized ({mode_str} gradients)")
        else:
            temp_opt_state = None
            opt_tuple = (actor_opt_state, critics_opt_state)
            print(f"🌡️  Fixed entropy coefficient (no temperature optimizer)")

        # Initialize buffer state
        buffer_state = self.replay_buffer.init_state()

        # Create TQC opt_state structure from the beginning
        tqc_opt_state = create_tqc_opt_state(
            actor_opt_state, critics_opt_state, temp_opt_state, buffer_state
        )

        # Replace the single opt_state with TQC structure
        tqc_opt_states = (tqc_opt_state,)  # Keep as tuple for framework compatibility

        # Log model and optimizer information
        model_size = xax.get_pytree_param_count(model)
        print(f"🎯 TQC Model size: {model_size:,} parameters")
        print(f"🔧 TQC Optimizers initialized: Actor, Critics ({self.config.num_critics})" +
              (", Temperature" if self.config.use_auto_entropy() else ""))
        print(f"📊 Buffer initialized: {self.config.buffer_size:,} capacity ({self.buffer_type})")

        # Partitions the models into mutable and static parts
        model_arrs, model_statics = (
            tuple(models)
            for models in zip(
                *(eqx.partition(model, self.model_partition_fn) for model in models),
                strict=True,
            )
        )

        # Loads the MJX model, and initializes the loop variables
        mjx_model = self.get_mjx_model(mj_model)
        randomizers = self.get_physics_randomizers(mjx_model)

        constants = RLLoopConstants(
            optimizer=tqc_optimizers,  # Store TQC optimizers in constants
            constants=self._get_constants(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_statics=model_statics,
                argmax_action=False,
            ),
        )

        carry = RLLoopCarry(
            opt_state=tqc_opt_states,  # Use TQC structure from the start
            env_states=self._get_env_state(
                rng=rng,
                rollout_constants=constants.constants,
                mj_model=mj_model,
                physics_model=mjx_model,
                randomizers=randomizers,
            ),
            shared_state=self._get_shared_state(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_arrs=model_arrs,
            ),
        )

        return constants, carry, state

    def _process_actor_observations(self, observations: xax.FrozenDict[str, Array]) -> Array:
        """Process observations for ACTOR (minimal state)."""
        time_1 = observations["timestep_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]

        # Handle both single and batched observations
        if time_1.ndim == 0:
            obs = [
                jnp.sin(time_1)[None],
                jnp.cos(time_1)[None],
                joint_pos_n,
                joint_vel_n,
                proj_grav_3,
            ]
        else:
            obs = [
                jnp.sin(time_1),
                jnp.cos(time_1),
                joint_pos_n,
                joint_vel_n,
                proj_grav_3,
            ]

        if self.config.use_acc_gyro:
            imu_acc_3 = observations["sensor_observation_imu_acc"]
            imu_gyro_3 = observations["sensor_observation_imu_gyro"]
            obs += [imu_acc_3, imu_gyro_3]

        result = jnp.concatenate(obs, axis=-1)

        if time_1.ndim == 0:
            result = result.squeeze(0)

        return result

    def _process_critic_observations(self, observations: xax.FrozenDict[str, Array]) -> Array:
        """Process observations for CRITIC (rich state)."""
        time_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        # NEW: Add contact observations to critic
        feet_contact = observations["feet_contact_observation"]  # Foot contact with ground
        feet_position = observations["feet_position_observation"]  # Foot positions
        hands_contact = observations["contact_observation_hands"]  # Hand contact (unwanted)
        torso_head_contact = observations["contact_observation_torso_head"]  # Body contact (unwanted)

        # Build observation list with scaling
        # Build observation list with scaling
        obs_parts = [
            jnp.sin(time_1) if time_1.ndim == 0 else jnp.sin(time_1),
            jnp.cos(time_1) if time_1.ndim == 0 else jnp.cos(time_1),
            dh_joint_pos_j,
            dh_joint_vel_j / 10.0,  # Scale joint velocities
            com_inertia_n,
            com_vel_n,
            imu_acc_3,
            imu_gyro_3,
            proj_grav_3,
            act_frc_obs_n / 100.0,  # Scale actuator forces
            base_pos_3,
            base_quat_4,

            # NEW CONTACT OBSERVATIONS (critic only)
            feet_contact,  # Binary foot contact info
            feet_position / 1.0,  # Foot positions (already in reasonable scale)
            hands_contact,  # Binary hand contact (should be 0)
            torso_head_contact,  # Binary torso/head contact (should be 0)
        ]

        # Flatten all observations to 1D for concatenation
        flattened_parts = []
        for part in obs_parts:
            if part.ndim == 0:  # Scalar
                flattened_parts.append(jnp.array([part]))
            else:  # Array
                flattened_parts.append(part.flatten())

        result = jnp.concatenate(flattened_parts)
        #print(f"Final result shape: {result.shape}")
        return result

    @xax.jit(static_argnames=["self"], jit_level=JitLevel.RL_CORE)
    def _add_to_buffer(
            self,
            buffer_state: ReplayBufferState,
            trajectories: Trajectory,
            rewards: RewardState,
            rng: PRNGKeyArray,
    ) -> ReplayBufferState:
        """Add trajectory data to functional replay buffer (JIT-compatible) - OPTIMIZED VERSION."""

        # Get shapes: (num_envs, time_steps, ...)
        num_envs, T = trajectories.done.shape
        total_transitions = num_envs * T

        # Create next observations efficiently
        next_obs = jax.tree.map(
            lambda x: jnp.concatenate([x[:, 1:], x[:, -1:]], axis=1),
            trajectories.obs
        )

        # Flatten ALL trajectories at once to prepare for batch processing
        def flatten_trajectory(x):
            return x.reshape(total_transitions, *x.shape[2:])

        flat_obs = jax.tree.map(flatten_trajectory, trajectories.obs)
        flat_actions = trajectories.action.reshape(total_transitions, *trajectories.action.shape[2:])
        flat_rewards = rewards.total.reshape(total_transitions)
        flat_next_obs = jax.tree.map(flatten_trajectory, next_obs)
        flat_dones = trajectories.done.reshape(total_transitions)

        # Use vmap to vectorize observation processing across the entire batch
        batch_process_actor = jax.vmap(self._process_actor_observations)
        batch_process_critic = jax.vmap(self._process_critic_observations)

        # Process ALL observations at once using vectorized functions
        actor_obs_batch = batch_process_actor(flat_obs)
        critic_obs_batch = batch_process_critic(flat_obs)
        next_actor_obs_batch = batch_process_actor(flat_next_obs)
        next_critic_obs_batch = batch_process_critic(flat_next_obs)

        # Add ALL transitions at once using the batch method
        final_buffer_state = self.replay_buffer.add_batch(
            buffer_state,
            actor_obs_batch,
            critic_obs_batch,
            flat_actions,
            flat_rewards,
            next_actor_obs_batch,
            next_critic_obs_batch,
            flat_dones
        )

        return final_buffer_state

    @xax.jit(static_argnames=["self"], jit_level=JitLevel.RL_CORE)
    def _tqc_training_step(
            self,
            model: TqcModel,
            tqc_opt_states: tuple,
            buffer_state: ReplayBufferState,
            rng: PRNGKeyArray,
    ) -> tuple[TqcModel, tuple, dict[str, Array]]:
        """Complete TQC training step with circular gradient support."""

        # 🔄 Get circular gradient flag
        use_auto_entropy_bool = self.config.use_auto_entropy()
        use_circular = self.config.use_circular_gradients
        use_auto_entropy = jnp.array(use_auto_entropy_bool, dtype=bool)

        # Get TQC optimizers
        tqc_optimizers = self.get_tqc_optimizers()

        # Use the JAX scalar for conditionals
        if use_auto_entropy_bool:  # Use Python bool for static unpacking
            actor_opt, critics_opt, temp_opt = tqc_optimizers
            actor_opt_state, critics_opt_state, temp_opt_state = tqc_opt_states
            entropy_coef = None  # Will be handled inside loss functions
        else:
            actor_opt, critics_opt = tqc_optimizers[:2]
            actor_opt_state, critics_opt_state = tqc_opt_states[:2]
            temp_opt = None
            temp_opt_state = None
            entropy_coef = jax.numpy.float32(self.config.ent_coef)

        key1, key2, key3, key4 = jax.random.split(rng, 4)

        # ===== 1. MULTIPLE CRITIC UPDATES =====
        current_critics = model.critics
        current_critics_opt_state = critics_opt_state
        all_critics_info = []

        # Train critics multiple times with different batches
        for critic_step in range(self.config.critic_updates_per_step):
            # Sample NEW batch for each critic update
            critic_key = jax.random.fold_in(key2, critic_step)
            tqc_inputs = self.replay_buffer.sample(buffer_state, self.config.batch_size, key=critic_key)

            def critics_loss_fn(critics):
                return compute_tqc_critic_loss(
                    critics=critics,
                    target_critics=model.target_critics,
                    actor=model.actor,
                    temperature=model.temperature,
                    tqc_inputs=tqc_inputs,
                    discount=self.config.discount_factor,
                    tau=self.tau,
                    top_quantiles_to_drop=self.config.top_quantiles_to_drop,
                    key=jax.random.fold_in(key3, critic_step),
                    entropy_coef=entropy_coef,
                    use_circular=use_circular,
                )

            (critics_loss_val, critics_info), critics_grads = jax.value_and_grad(
                critics_loss_fn, has_aux=True
            )(current_critics)

            critics_updates, current_critics_opt_state = critics_opt.update(
                critics_grads, current_critics_opt_state, current_critics
            )
            current_critics = optax.apply_updates(current_critics, critics_updates)
            all_critics_info.append(critics_info)

        # Average critic metrics across all updates
        averaged_critics_info = {}
        for key in all_critics_info[0].keys():
            averaged_critics_info[key] = jnp.mean(jnp.array([info[key] for info in all_critics_info]))


        # Update actor with circular gradient flag
        def actor_loss_fn(actor):
            return compute_tqc_actor_loss(
                actor=actor,
                critics=current_critics,
                temperature=model.temperature,
                actor_observations=tqc_inputs.actor_observations,
                critic_observations=tqc_inputs.critic_observations,
                top_quantiles_to_drop=self.config.top_quantiles_to_drop,
                key=key3,
                entropy_coef=entropy_coef,
                use_circular=use_circular,  # 🔄 Pass circular flag
                num_quantiles=self.config.num_quantiles,  # Add this line!
            )

        (actor_loss_val, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(model.actor)

        actor_updates, new_actor_opt_state = actor_opt.update(
            actor_grads, actor_opt_state, model.actor
        )
        new_actor = optax.apply_updates(model.actor, actor_updates)

        # Temperature update (same as before - always uses detached entropy in temp loss)
        if use_auto_entropy_bool:  # Use Python bool for static control flow
            # Get fresh log_probs from updated actor
            _, log_probs = new_actor.get_action_and_log_prob(tqc_inputs.actor_observations, key4)

            def temp_loss_fn(temperature):
                return compute_tqc_temperature_loss(
                    temperature, log_probs, self.target_entropy_value
                )

            (temp_loss_val, temp_info), temp_grads = jax.value_and_grad(
                temp_loss_fn, has_aux=True
            )(model.temperature)

            temp_updates, new_temp_opt_state = temp_opt.update(
                temp_grads, temp_opt_state, model.temperature
            )
            new_temperature = optax.apply_updates(model.temperature, temp_updates)

            # Log temperature behavior (with circular gradient info)
            current_temp = temp_info.get('temperature', 0.1)
            current_entropy = temp_info.get('entropy', 0.0)
            entropy_error = temp_info.get('entropy_error', 0.0)

            if use_circular:
                jax.debug.print("🔄 CIRCULAR | Temp: {:.4f} | Entropy: {:.3f} | Target: {:.3f} | Error: {:.3f}",
                                current_temp, current_entropy, self.target_entropy_value, entropy_error)
            else:
                jax.debug.print("🛡️ DETACHED | Temp: {:.8f} | Entropy: {:.3f} | Target: {:.3f} | Error: {:.3f}",
                                current_temp, current_entropy, self.target_entropy_value, entropy_error)


        else:
            # Fixed temperature - no update
            new_temperature = model.temperature
            new_temp_opt_state = temp_opt_state
            temp_info = {
                'temp_loss': 0.0,
                'temperature': model.temperature.temperature,
                'entropy': actor_info.get('entropy', 0.0),
                'target_entropy': self.target_entropy_value,
                'entropy_error': 0.0,
            }

        # Update target networks
        new_target_critics = update_target_networks(
            current_critics, model.target_critics, self.config.soft_update_rate
        )

        # Create updated model
        updated_model = eqx.tree_at(
            lambda m: (m.actor, m.critics, m.target_critics, m.temperature),
            model,
            (new_actor, current_critics, new_target_critics, new_temperature)
        )

        # Enhanced info with circular gradient monitoring
        all_info = {
            **averaged_critics_info,
            **actor_info,
            **temp_info,
            # 🔄 CIRCULAR GRADIENT MONITORING
            'use_circular_gradients': jnp.where(use_circular, 1.0, 0.0),
            'temperature_is_learning': jnp.where(use_auto_entropy, 1.0, 0.0),
            'entropy_adaptation_active': jnp.where(
                jnp.abs(temp_info.get('entropy_error', 0)) > 0.05, 1.0, 0.0
            ),
            'gradient_coupling_active': jnp.where(use_circular, 1.0, 0.0),
            'entropy_error_magnitude': jnp.abs(temp_info.get('entropy_error', 0)),
            # Monitor for instability in circular mode
            'temperature_variance': jnp.var(jnp.array([temp_info.get('temperature', 0.1)])),
            'actor_loss_magnitude': jnp.abs(actor_info.get('actor_loss', 0.0)),
            'critic_loss_magnitude': jnp.abs(averaged_critics_info.get('critic_loss', 0.0)),
        }

        # Return consistent tuple structure
        if use_auto_entropy_bool:
            new_tqc_opt_states = (new_actor_opt_state, current_critics_opt_state, new_temp_opt_state)
        else:
            new_tqc_opt_states = (new_actor_opt_state, current_critics_opt_state, None)

        return updated_model, new_tqc_opt_states, all_info

    # Add this as a method to your TQCHumanoidTask class:

    def _create_empty_metrics_template(self, model, buffer_state, use_auto_entropy, use_circular):
        """Create metrics template with all required keys including circular gradient monitoring and NEW debugging fields."""

        current_temp_value = jnp.where(
            use_auto_entropy,
            model.temperature.temperature,
            0.1
        )

        # Template with ALL possible keys that might appear in either branch
        metrics_template = {
            # Buffer metrics - ensure float32 dtype
            'buffer_size': jnp.float32(buffer_state.size),
            'buffer_ptr': jnp.float32(buffer_state.ptr),
            'buffer_utilization': jnp.float32(buffer_state.size / buffer_state.max_size),
            'training_active': jnp.float32(0.0),  # Will be overridden
            'waiting_for_data': jnp.float32(0.0),  # Will be overridden
            'target_entropy': jnp.float32(self.target_entropy_value),
            'use_auto_entropy': jnp.where(use_auto_entropy, 1.0, 0.0),

            # Core loss metrics
            'critic_loss': jnp.float32(0.0),
            'mean_critic_loss': jnp.float32(0.0),
            'target_q': jnp.float32(0.0),
            'current_q_mean': jnp.float32(0.0),
            'truncated_q_mean': jnp.float32(0.0),
            'actor_loss': jnp.float32(0.0),
            'entropy': jnp.float32(0.0),
            'temp_loss': jnp.float32(0.0),
            'temperature': current_temp_value,
            'q_truncated_mean': jnp.float32(0.0),
            'q_all_mean': jnp.float32(0.0),
            'entropy_coefficient': current_temp_value,
            'entropy_error': jnp.float32(0.0),
            'temperature_is_learning': jnp.where(use_auto_entropy, 1.0, 0.0),
            'entropy_adaptation_active': jnp.float32(0.0),
            'entropy_error_magnitude': jnp.float32(0.0),
            'log_temp_raw': model.temperature.log_temp,

            # 🆕 NEW DEBUGGING FIELDS (from updated temperature loss)
            'mean_log_prob': jnp.float32(0.0),
            'std_log_prob': jnp.float32(0.0),
            'entropy_term_mean': jnp.float32(0.0),
            'temp_loss_components': jnp.float32(0.0),

            # 🔄 CIRCULAR GRADIENT MONITORING
            'use_circular_gradients': jnp.where(use_circular, 1.0, 0.0),
            'gradient_coupling_active': jnp.where(use_circular, 1.0, 0.0),
            'gradient_mode': jnp.where(use_circular, 1.0, 0.0),
            'temperature_variance': jnp.float32(0.0),
            'actor_loss_magnitude': jnp.float32(0.0),
            'critic_loss_magnitude': jnp.float32(0.0),

            # Actor-specific metrics (MUST BE INCLUDED!) - ensure float32
            'q_min_truncated': jnp.float32(0.0),
            'q_mean_truncated_critics': jnp.float32(0.0),
            'q_std_across_critics': jnp.float32(0.0),
            'q_range_across_critics': jnp.float32(0.0),
            'num_quantiles_kept': jnp.float32(self.config.num_quantiles - self.config.top_quantiles_to_drop),
            'truncation_ratio': jnp.float32(
                (self.config.num_quantiles - self.config.top_quantiles_to_drop) / self.config.num_quantiles),
        }

        # Add individual critic losses
        for i in range(self.config.num_critics):
            metrics_template[f'critic_{i}_loss'] = jnp.float32(0.0)
            # Also add the q_truncated metrics for each critic
            metrics_template[f'critic_{i}_q_truncated'] = jnp.float32(0.0)

        return metrics_template

    # Then in your update_model method, update both do_training and skip_training to use it:

    def update_model(
            self,
            *,
            constants: RLLoopConstants,
            carry: RLLoopCarry,
            trajectories: Trajectory,
            rewards: RewardState,
            rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """Complete TQC update method with circular gradient support."""
        key1, key2 = jax.random.split(rng)

        # Extract TQC components
        tqc_opt_state = carry.opt_state[0]
        tqc_optimizers, buffer_state = extract_from_tqc_opt_state(tqc_opt_state)

        if tqc_optimizers is None or buffer_state is None:
            raise RuntimeError("TQC opt_state not properly initialized!")

        # Add trajectory data to replay buffer
        new_buffer_state = self._add_to_buffer(
            buffer_state, trajectories, rewards, key1
        )

        # Get current model
        model_arr = carry.shared_state.model_arrs[0]
        model_static = constants.constants.model_statics[0]
        model = eqx.combine(model_arr, model_static)

        # 🔧 COMPUTE ENTROPY COEFFICIENT (needed for both branches)
        if self.config.use_auto_entropy():
            entropy_coef_value = model.temperature.temperature
        else:
            entropy_coef_value = float(self.config.ent_coef)

        # Define training and no-training functions
        def do_training(inputs):
            carry, key2, new_buffer_state, tqc_optimizers = inputs

            # Convert config to JAX scalar
            use_auto_entropy_bool = self.config.use_auto_entropy()
            use_auto_entropy = jnp.array(use_auto_entropy_bool, dtype=bool)
            use_circular = self.config.use_circular_gradients

            metrics_list = []
            current_model = model
            current_tqc_optimizers = tqc_optimizers

            for _ in range(self.config.gradient_steps):
                key2, subkey = jax.random.split(key2)
                current_model, current_tqc_optimizers, step_metrics = self._tqc_training_step(
                    current_model, current_tqc_optimizers, new_buffer_state, subkey
                )
                metrics_list.append(step_metrics)

            # Average metrics
            averaged_metrics = {}
            for key in metrics_list[0].keys():
                averaged_metrics[key] = jnp.mean(jnp.array([m[key] for m in metrics_list]))

            # Update carry
            new_model_arr, _ = eqx.partition(current_model, eqx.is_array)
            new_shared_state = replace(
                carry.shared_state,
                model_arrs=xax.tuple_insert(carry.shared_state.model_arrs, 0, new_model_arr),
            )

            # Build updated opt state
            if use_auto_entropy_bool:
                new_tqc_opt_state = create_tqc_opt_state(
                    current_tqc_optimizers[0], current_tqc_optimizers[1],
                    current_tqc_optimizers[2], new_buffer_state
                )
            else:
                new_tqc_opt_state = create_tqc_opt_state(
                    current_tqc_optimizers[0], current_tqc_optimizers[1],
                    None, new_buffer_state
                )

            updated_carry = replace(
                carry,
                shared_state=new_shared_state,
                opt_state=(new_tqc_opt_state,),
            )

            # Create metrics using template to ensure consistent structure
            metrics = self._create_empty_metrics_template(current_model, new_buffer_state, use_auto_entropy,
                                                          use_circular)

            # Update with actual training values
            metrics.update(averaged_metrics)
            metrics.update({
                'training_active': 1.0,
                'waiting_for_data': 0.0,
            })

            return updated_carry, metrics

        def skip_training(inputs):
            carry, _, new_buffer_state, tqc_optimizers = inputs

            # Convert config method to JAX scalar
            use_auto_entropy_bool = self.config.use_auto_entropy()
            use_auto_entropy = jnp.array(use_auto_entropy_bool, dtype=bool)
            use_circular = self.config.use_circular_gradients

            # Build opt state for skip mode
            if use_auto_entropy_bool:
                new_tqc_opt_state = create_tqc_opt_state(
                    tqc_optimizers[0], tqc_optimizers[1],
                    tqc_optimizers[2], new_buffer_state
                )
            else:
                new_tqc_opt_state = create_tqc_opt_state(
                    tqc_optimizers[0], tqc_optimizers[1],
                    None, new_buffer_state
                )

            updated_carry = replace(
                carry,
                opt_state=(new_tqc_opt_state,)
            )

            # Get current temperature for logging
            model_arr = carry.shared_state.model_arrs[0]
            model_static = constants.constants.model_statics[0]
            model = eqx.combine(model_arr, model_static)

            current_temp_value = jnp.where(
                use_auto_entropy,
                model.temperature.temperature,
                0.1
            )

            # Use the same template method to ensure consistency
            metrics = self._create_empty_metrics_template(model, new_buffer_state, use_auto_entropy, use_circular)

            # Override for skip mode
            metrics.update({
                'training_active': 0.0,
                'waiting_for_data': 1.0,
                'temperature': current_temp_value,
                'entropy_coefficient': current_temp_value,
            })

            return updated_carry, metrics

        # Use conditional training
        train_condition = new_buffer_state.size >= self.config.min_buffer_size
        final_carry, metrics = jax.lax.cond(
            train_condition,
            do_training,
            skip_training,
            (carry, key2, new_buffer_state, tqc_optimizers)
        )

        # Create logged trajectory
        logged_trajectory = LoggedTrajectory(
            trajectory=jax.tree.map(lambda x: x[0], trajectories),
            rewards=jax.tree.map(lambda x: x[0], rewards),
            metrics=xax.FrozenDict({}),
        )

        return final_carry, xax.FrozenDict(metrics), logged_trajectory

    def get_model(self, key: PRNGKeyArray) -> TqcModel:
        """Create TQC model."""
        actor_obs_dim = 51 if self.config.use_acc_gyro else 45
        critic_obs_dim = 462
        action_dim = len(ZEROS)

        return TqcModel(
            key,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            num_critics=self.config.num_critics,
            num_quantiles=self.config.num_quantiles,
            # 🎭 Custom actor architecture
            actor_layer_sizes=self.config.actor_layer_sizes,
            # 🎯 Custom critic architecture
            critic_layer_sizes=self.config.critic_layer_sizes,
            # Legacy fallbacks (if custom sizes not provided)
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            initial_temp=self.config.initial_temperature,
        )

    def sample_action(
        self,
        model: TqcModel,
        model_carry: PyTree,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        """Sample action using TQC policy."""
        obs_array = self._process_actor_observations(observations)
        action, log_prob = model.actor.get_action_and_log_prob(
            obs_array, rng, deterministic=argmax
        )

        # Add joint position biases

        action = action + jnp.array([v for _, v in ZEROS])

        return ksim.Action(action=action, carry=model_carry)

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        """Get initial model carry state for TQC."""
        return None

    # ... (rest of the methods remain the same as in the original file)
    # I'll include the key checkpoint methods but truncate for space



###################################################################



    def load_initial_state(
            self,
            rng: PRNGKeyArray,
            load_optimizer: bool = True,
    ) -> tuple[tuple[PyTree, ...], tuple[optax.GradientTransformation, ...], tuple[optax.OptState, ...], xax.State] | \
         tuple[tuple[PyTree, ...], xax.State]:
        """Load TQC checkpoint using framework format."""

        # Check if we should load from checkpoint
        init_ckpt_path = self.get_init_ckpt_path()

        if init_ckpt_path is None:
            print("🔄 No checkpoint found - creating fresh TQC state")
            return self._create_fresh_tqc_state(rng, load_optimizer)

        print(f"🎯 Loading TQC checkpoint from: {init_ckpt_path}")

        try:
            return self._load_tqc_checkpoint(init_ckpt_path, rng, load_optimizer)
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"🔄 Falling back to fresh TQC state")
            return self._create_fresh_tqc_state(rng, load_optimizer)

    def _create_fresh_tqc_state(
            self,
            rng: PRNGKeyArray,
            load_optimizer: bool,
    ) -> tuple[tuple[PyTree, ...], tuple[optax.GradientTransformation, ...], tuple[optax.OptState, ...], xax.State] | \
         tuple[tuple[PyTree, ...], xax.State]:
        """Create fresh TQC models and optimizers."""

        print("🆕 Creating fresh TQC state")

        # Create the TQC model
        model = self.get_model(rng)

        if load_optimizer:
            # Create TQC optimizers
            tqc_optimizers = self.get_tqc_optimizers()

            # Initialize optimizer states
            actor_opt_state = tqc_optimizers[0].init(model.actor)
            critics_opt_state = tqc_optimizers[1].init(model.critics)

            if self.config.use_auto_entropy():
                temp_opt_state = tqc_optimizers[2].init(model.temperature)
            else:
                temp_opt_state = None

            optimizers = tqc_optimizers

            # Create TQC opt_state structure with fresh buffer
            buffer_state = self.replay_buffer.init_state()
            tqc_opt_state = create_tqc_opt_state(
                actor_opt_state, critics_opt_state, temp_opt_state, buffer_state
            )

            opt_states = (tqc_opt_state,)
        else:
            # When not loading optimizer, don't create optimizer-related objects
            optimizers = None
            opt_states = None

        # Create initial training state using framework's from_dict method
        try:
            state = xax.State.from_dict(
                num_steps=0,
                num_samples=0,
                elapsed_time_s=0.0,
                phase="train",
            )
            print("✅ State created with from_dict")
        except Exception as state_error:
            print(f"⚠️  from_dict failed: {state_error}")
            # If from_dict fails, we need to investigate the State constructor
            state = xax.State()
            print(f"🔍 Default state fields: {[attr for attr in dir(state) if not attr.startswith('_')]}")

        print("✅ Fresh TQC state created")

        # Return different number of values based on load_optimizer
        if load_optimizer:
            return (model,), optimizers, opt_states, state
        else:
            return (model,), state

    def _load_tqc_checkpoint(
            self,
            ckpt_path: Path,
            rng: PRNGKeyArray,
            load_optimizer: bool,
    ) -> tuple[tuple[PyTree, ...], tuple[optax.GradientTransformation, ...], tuple[optax.OptState, ...], xax.State] | \
         tuple[tuple[PyTree, ...], xax.State]:
        """Load TQC checkpoint using framework format."""

        # Create template objects
        template_model = self.get_model(rng)
        template_optimizers = self.get_tqc_optimizers()

        # Create template optimizer states
        template_actor_opt_state = template_optimizers[0].init(template_model.actor)
        template_critics_opt_state = template_optimizers[1].init(template_model.critics)

        if self.config.use_auto_entropy() and len(template_optimizers) > 2:
            template_temp_opt_state = template_optimizers[2].init(template_model.temperature)
            template_tqc_opt_state = create_tqc_opt_state(
                template_actor_opt_state, template_critics_opt_state,
                template_temp_opt_state, self.replay_buffer.init_state()
            )
        else:
            template_tqc_opt_state = create_tqc_opt_state(
                template_actor_opt_state, template_critics_opt_state,
                None, self.replay_buffer.init_state()
            )

        template_opt_states = (template_tqc_opt_state,)

        try:
            # Try framework loading with templates
            if load_optimizer:
                models, optimizers, opt_states, state, config = load_ckpt(
                    ckpt_path,
                    part="all",
                    model_templates=[template_model],
                    optimizer_templates=template_optimizers,
                    opt_state_templates=template_opt_states,
                )
                print("✅ Framework checkpoint loading successful")

                # Validate TQC structure
                if len(opt_states) == 1:
                    tqc_optimizers, buffer_state = extract_from_tqc_opt_state(opt_states[0])
                    if tqc_optimizers is not None and buffer_state is not None:
                        print(f"📊 Buffer restored: {buffer_state.size}/{buffer_state.max_size} transitions")
                    else:
                        print("⚠️  TQC structure validation failed, but continuing")

                return tuple(models), tuple(optimizers), tuple(opt_states), state
            else:
                # Load only model and state
                models, state, config = load_ckpt(
                    ckpt_path,
                    part="model_state_config",
                    model_templates=[template_model]
                )

                return tuple(models), state

        except Exception as framework_error:
            print(f"⚠️  Framework loading failed: {framework_error}")

            # Try manual Equinox loading (for older checkpoints)
            try:
                print("🔧 Attempting direct Equinox loading")

                # This handles the old .bin format checkpoints
                loaded_model = eqx.tree_deserialise_leaves(ckpt_path, template_model)
                print("✅ Direct model loading successful")
                print("⚠️  Note: Using old checkpoint format - training state will be reset to 0")

                if load_optimizer:
                    # Create fresh optimizers and buffer (can't restore from old format)
                    print("🔄 Creating fresh optimizers (old checkpoint format)")

                    optimizers = template_optimizers
                    actor_opt_state = template_optimizers[0].init(loaded_model.actor)
                    critics_opt_state = template_optimizers[1].init(loaded_model.critics)

                    if self.config.use_auto_entropy() and len(template_optimizers) > 2:
                        temp_opt_state = template_optimizers[2].init(loaded_model.temperature)
                        tqc_opt_state = create_tqc_opt_state(
                            actor_opt_state, critics_opt_state, temp_opt_state,
                            self.replay_buffer.init_state()
                        )
                    else:
                        tqc_opt_state = create_tqc_opt_state(
                            actor_opt_state, critics_opt_state, None,
                            self.replay_buffer.init_state()
                        )

                    opt_states = (tqc_opt_state,)
                else:
                    optimizers = None
                    opt_states = None

                # Create default state using from_dict method
                try:
                    state = xax.State.from_dict({
                        "num_steps": 0,
                        "num_samples": 0,
                        "elapsed_time_s": 0.0,
                        "phase": "train",
                    })
                except Exception:
                    state = xax.State()

                # Return different number of values based on load_optimizer
                if load_optimizer:
                    return (loaded_model,), optimizers, opt_states, state
                else:
                    return (loaded_model,), state

            except Exception as manual_error:
                print(f"❌ Manual loading also failed: {manual_error}")
                raise RuntimeError(
                    f"Could not load checkpoint {ckpt_path}. Framework error: {framework_error}, Manual error: {manual_error}")

    def save_checkpoint(
            self,
            models: tuple[PyTree, ...],
            optimizers: tuple[optax.GradientTransformation, ...],
            opt_states: tuple[optax.OptState, ...],
            state: xax.State,
    ) -> None:
        """Save TQC checkpoint with automatic cleanup of old checkpoints."""

        if not xax.is_master():
            return

        print("💾 Saving TQC checkpoint...")

        try:
            # Extract TQC-specific data for logging
            if len(opt_states) == 1:
                tqc_optimizers, buffer_state = extract_from_tqc_opt_state(opt_states[0])
                if buffer_state is not None:
                    print(f"📊 Saving buffer with {buffer_state.size}/{buffer_state.max_size} transitions")

            # Use the framework's save method directly
            ckpt_path = super().save_checkpoint(
                models=models,
                optimizers=optimizers,
                opt_states=opt_states,
                aux_data=None,
                state=state,
            )

            print(f"✅ TQC checkpoint saved to {ckpt_path}")

            # Clean up old checkpoints if we have too many
            if hasattr(self.config, 'keep_last_n_checkpoints'):
                self._cleanup_old_checkpoints()

            # Save additional TQC metadata for reference
            try:
                config_info_path = ckpt_path.parent / f"{ckpt_path.stem}_info.yaml"
                with open(config_info_path, 'w') as f:
                    f.write(f"# TQC Training Information\n")
                    f.write(f"checkpoint_path: {ckpt_path}\n")
                    f.write(f"model_type: TQC\n")
                    f.write(f"num_critics: {self.config.num_critics}\n")
                    f.write(f"num_quantiles: {self.config.num_quantiles}\n")
                    f.write(f"use_auto_entropy: {self.config.use_auto_entropy()}\n")
                    f.write(f"target_entropy: {getattr(self.config, 'target_entropy', 'N/A')}\n")
                    f.write(f"action_dim: {self.action_dim}\n")
                    f.write(f"steps: {state.num_steps}\n")
                    f.write(f"elapsed_time: {state.elapsed_time_s}\n")
                    if buffer_state is not None:
                        f.write(f"buffer_size: {buffer_state.size}\n")
                        f.write(f"buffer_max_size: {buffer_state.max_size}\n")

                print(f"📝 TQC metadata saved to {config_info_path}")
            except Exception as e:
                print(f"⚠️  Failed to save TQC metadata: {e}")

            print("🎉 TQC checkpoint save completed!")

        except Exception as e:
            print(f"❌ Failed to save TQC checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only the last N."""
        if not hasattr(self.config, 'keep_last_n_checkpoints'):
            return

        keep_last_n = self.config.keep_last_n_checkpoints
        if keep_last_n <= 0:
            return

        try:
            checkpoint_dir = self.exp_dir / "checkpoints"
            if not checkpoint_dir.exists():
                return

            # Find all checkpoint files matching the pattern ckpt.{steps}.bin
            checkpoint_files = []
            for ckpt_file in checkpoint_dir.glob("ckpt.*.bin"):
                try:
                    # Extract step number from filename
                    parts = ckpt_file.stem.split('.')
                    if len(parts) == 2 and parts[0] == "ckpt" and parts[1].isdigit():
                        step_num = int(parts[1])
                        checkpoint_files.append((step_num, ckpt_file))
                except (ValueError, IndexError):
                    continue

            # Also check for the base ckpt.bin (usually a symlink to the latest)
            base_ckpt = checkpoint_dir / "ckpt.bin"

            if len(checkpoint_files) <= keep_last_n:
                print(f"📁 Found {len(checkpoint_files)} checkpoints, keeping all (≤{keep_last_n})")
                return

            # Sort by step number and keep only the latest N
            checkpoint_files.sort(key=lambda x: x[0])  # Sort by step number
            checkpoints_to_remove = checkpoint_files[:-keep_last_n]  # Remove all but last N

            print(f"📁 Found {len(checkpoint_files)} checkpoints, removing {len(checkpoints_to_remove)} old ones")

            for step_num, checkpoint_path in checkpoints_to_remove:
                try:
                    # Remove the checkpoint file
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        print(f"🗑️  Removed checkpoint: {checkpoint_path.name}")

                    # Remove associated metadata file if it exists
                    metadata_file = checkpoint_path.parent / f"{checkpoint_path.stem}_info.yaml"
                    if metadata_file.exists():
                        metadata_file.unlink()
                        print(f"🗑️  Removed metadata: {metadata_file.name}")

                except Exception as e:
                    print(f"⚠️  Failed to remove {checkpoint_path}: {e}")

            remaining_checkpoints = [x[1].name for x in checkpoint_files[-keep_last_n:]]
            print(f"✅ Kept last {keep_last_n} checkpoints: {remaining_checkpoints}")

            # Make sure the ckpt.bin symlink points to the latest checkpoint
            if checkpoint_files and base_ckpt.exists():
                latest_checkpoint = checkpoint_files[-1][1]  # Last in sorted list
                try:
                    base_ckpt.unlink()
                    base_ckpt.symlink_to(latest_checkpoint.name)
                    print(f"🔗 Updated ckpt.bin symlink to point to {latest_checkpoint.name}")
                except Exception as e:
                    print(f"⚠️  Failed to update symlink: {e}")

        except Exception as e:
            print(f"⚠️  Failed to cleanup old checkpoints: {e}")

    # 3. Optional: Override get_init_ckpt_path to provide better loading info

    def get_init_ckpt_path(self) -> Path | None:
        """Get initial checkpoint path with better logging."""
        ckpt_path = super().get_init_ckpt_path()

        if ckpt_path is not None:
            # Check if it's the symlink or a specific checkpoint
            if ckpt_path.name == "ckpt.bin":
                if ckpt_path.is_symlink():
                    target = ckpt_path.resolve()
                    print(f"🔗 Loading from symlinked checkpoint: {target.name}")
                else:
                    print(f"📁 Loading from base checkpoint: {ckpt_path.name}")
            else:
                print(f"📁 Loading from specific checkpoint: {ckpt_path.name}")

            # Show available checkpoints for reference
            try:
                checkpoint_dir = ckpt_path.parent
                available_ckpts = sorted([f.name for f in checkpoint_dir.glob("ckpt.*.bin")])
                if available_ckpts:
                    print(f"📋 Available checkpoints: {available_ckpts}")
            except Exception:
                pass

        return ckpt_path