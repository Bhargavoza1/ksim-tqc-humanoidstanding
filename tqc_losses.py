"""TQC loss functions and training utilities - WITH CIRCULAR GRADIENT SUPPORT."""

from typing import Dict, Tuple
import functools

import chex
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array
from ksim.debugging import JitLevel

from tqc_data_structures import TQCInputs
from tqc_networks import TqcActor, QuantileCritic, Temperature


def quantile_huber_loss(u: Array, tau: Array, kappa: float = 1.0) -> Array:
    """Quantile Huber loss for robust quantile regression."""
    # Huber loss component
    huber_loss = jnp.where(
        jnp.abs(u) <= kappa,
        0.5 * u**2,
        kappa * (jnp.abs(u) - 0.5 * kappa)
    )

    # Quantile component
    quantile_weight = jnp.abs(tau - (u < 0).astype(jnp.float32))

    return quantile_weight * huber_loss


def create_quantile_levels(num_quantiles: int) -> Array:
    """Create quantile levels array outside of JIT context."""
    quantile_indices = jnp.arange(1, num_quantiles + 1, dtype=jnp.float32)
    tau = quantile_indices / (num_quantiles + 1)
    return tau.reshape(1, -1)  # Shape: (1, num_quantiles)


@xax.jit(static_argnames=["top_quantiles_to_drop", "use_circular"], jit_level=JitLevel.RL_CORE)
def compute_tqc_critic_loss(
    critics: tuple[QuantileCritic, ...],
    target_critics: tuple[QuantileCritic, ...],
    actor: TqcActor,
    temperature: Temperature,
    tqc_inputs: TQCInputs,
    discount: float,
    tau: Array,  # Pre-computed quantile levels
    top_quantiles_to_drop: int,
    key: chex.PRNGKey,
    entropy_coef: float = None,
    use_circular: bool = False,  # 🔄 NEW: Circular gradient flag
) -> tuple[Array, dict[str, Array]]:
    """Compute TQC critic loss using quantile regression - with circular gradient option."""
    dummy_carry = jnp.zeros(1)

    # 🔄 CIRCULAR vs DETACHED temperature handling
    if entropy_coef is None:
        if use_circular:
            # 🔄 CIRCULAR: Non-detached temperature (gradients flow!)
            entropy_coefficient = jnp.exp(temperature.log_temp)
            entropy_coefficient = jnp.clip(entropy_coefficient, 0.001, 1.0)
        else:
            # 🛡️ DETACHED: Standard approach (recommended)
            entropy_coefficient = temperature.temperature_detached_sb3
    else:
        entropy_coefficient = entropy_coef

    def safe_forward(critic, obs, actions, carry):
        quantiles, carry_out = critic.forward(obs, actions, carry)
        # Clip quantile values to reasonable range
        quantiles = jnp.clip(quantiles, -1000.0, 1000.0)
        # Replace NaN with zeros
        quantiles = jnp.where(jnp.isnan(quantiles), 0.0, quantiles)
        return quantiles, carry_out

    # Current quantiles from all critics
    current_quantiles = []
    for critic in critics:
        quantiles, _ = safe_forward(
            critic, tqc_inputs.critic_observations, tqc_inputs.actions, dummy_carry
        )
        current_quantiles.append(quantiles)

    # Stack quantiles: (num_critics, batch_size, num_quantiles)
    current_quantiles = jnp.stack(current_quantiles, axis=0)

    # Next actions and log probs from current policy
    next_actions, next_log_probs = actor.get_action_and_log_prob(
        tqc_inputs.next_actor_observations, key
    )
    # Clip log probs
    next_log_probs = jnp.where(jnp.isnan(next_log_probs), -5.0, next_log_probs)

    # Target quantiles from all target critics
    target_quantiles = []
    for target_critic in target_critics:
        quantiles, _ = safe_forward(
            target_critic, tqc_inputs.next_critic_observations, next_actions, dummy_carry
        )
        target_quantiles.append(quantiles)

    # Stack target quantiles: (num_critics, batch_size, num_quantiles)
    target_quantiles = jnp.stack(target_quantiles, axis=0)

    # 🔧 TQC KEY FEATURE: Truncated quantile selection
    # Concatenate all quantiles and sort to select lowest ones
    # Shape: (batch_size, num_critics * num_quantiles)
    all_target_quantiles = target_quantiles.transpose(1, 0, 2).reshape(
        target_quantiles.shape[1], -1
    )

    # Sort and truncate to keep lowest quantiles
    sorted_quantiles = jnp.sort(all_target_quantiles, axis=-1)
    num_to_keep = all_target_quantiles.shape[-1] - top_quantiles_to_drop

    # Use dynamic slice instead of [:, :num_to_keep]
    truncated_quantiles = jax.lax.dynamic_slice(
        sorted_quantiles,
        start_indices=(0, 0),
        slice_sizes=(sorted_quantiles.shape[0], num_to_keep)
    )

    # Take mean of truncated quantiles for target
    q_target_next = jnp.mean(truncated_quantiles, axis=-1)

    # 🔄 Add entropy term (circular vs detached affects gradient flow!)
    q_target_next = q_target_next - entropy_coefficient * next_log_probs

    # Compute target with clipping
    q_target = tqc_inputs.rewards + discount * (1 - tqc_inputs.dones) * q_target_next
    q_target = jax.lax.stop_gradient(q_target)

    # 🔧 STABILITY: Clip targets
    q_target = jnp.where(jnp.isnan(q_target), 0.0, q_target)

    # Expand target for quantile loss computation
    # Shape: (batch_size, 1) -> (batch_size, num_quantiles)
    num_quantiles = tau.shape[-1]
    q_target_expanded = jnp.expand_dims(q_target, axis=-1)
    q_target_expanded = jnp.repeat(q_target_expanded, num_quantiles, axis=-1)

    # Compute quantile losses for each critic
    total_loss = 0.0
    critic_losses = []

    for i, critic_quantiles in enumerate(current_quantiles):
        # Compute quantile regression loss
        # Shape: (batch_size, num_quantiles)
        u = q_target_expanded - critic_quantiles

        # Apply quantile Huber loss
        quantile_loss = quantile_huber_loss(u, tau, kappa=1.0)

        # Average over quantiles and batch
        critic_loss = jnp.mean(quantile_loss)

        # Replace NaN losses
        critic_loss = jnp.where(jnp.isnan(critic_loss), 1.0, critic_loss)
        critic_losses.append(critic_loss)
        total_loss += critic_loss

    info = {
        'critic_loss': total_loss,
        'mean_critic_loss': total_loss / len(critics),
        'target_q': jnp.mean(q_target),
        'current_q_mean': jnp.mean(current_quantiles),
        'truncated_q_mean': jnp.mean(q_target_next),
        'entropy_coefficient': entropy_coefficient,
        'gradient_mode': jnp.where(use_circular, 1.0, 0.0),  # 🔄 Track gradient mode
    }

    # Add individual critic losses to info
    for i, loss in enumerate(critic_losses):
        info[f'critic_{i}_loss'] = loss

    # 🔧 STABILITY: Sanitize info
    info = {k: jnp.where(jnp.isnan(v), 0.0, v) for k, v in info.items()}

    return total_loss, info


@xax.jit(static_argnames=["top_quantiles_to_drop", "use_circular", "num_quantiles"], jit_level=JitLevel.RL_CORE)
def compute_tqc_actor_loss(
        actor: TqcActor,
        critics: tuple[QuantileCritic, ...],
        temperature: Temperature,
        actor_observations: chex.Array,
        critic_observations: chex.Array,
        top_quantiles_to_drop: int,
        key: chex.PRNGKey,
        entropy_coef: float = None,
        use_circular: bool = False,
        num_quantiles: int = None,  # NEW: Pass as static argument
) -> tuple[Array, dict[str, Array]]:
    """TQC actor loss with per-critic truncation and minimum selection."""

    # 🌡️ Temperature coefficient handling
    if entropy_coef is None:
        if use_circular:
            # 🔄 CIRCULAR: Non-detached temperature (gradients flow)
            entropy_coefficient = jnp.exp(temperature.log_temp)
            entropy_coefficient = jnp.clip(entropy_coefficient, 0.001, 1.0)
        else:
            # 🛡️ DETACHED: Standard approach (recommended)
            entropy_coefficient = temperature.temperature_detached_sb3
    else:
        entropy_coefficient = entropy_coef

    # 🎭 Sample actions from current policy with safety
    actions, log_probs = actor.get_action_and_log_prob(actor_observations, key)
    log_probs = jnp.where(jnp.isnan(log_probs), -5.0, log_probs)

    # 🎯 Get Q-values from each critic with per-critic truncation
    dummy_carry = jnp.zeros(1)

    def safe_critic_forward(critic, obs, acts, carry):
        """Safe forward pass through critic with NaN handling."""
        quantiles, carry_out = critic.forward(obs, acts, carry)
        quantiles = jnp.clip(quantiles, -1000.0, 1000.0)
        quantiles = jnp.where(jnp.isnan(quantiles), 0.0, quantiles)
        return quantiles, carry_out

    def process_single_critic(critic):
        """Process a single critic: forward → truncate → mean."""
        quantiles, _ = safe_critic_forward(critic, critic_observations, actions, dummy_carry)

        # Sort quantiles for this critic (lowest to highest)
        sorted_quantiles = jnp.sort(quantiles, axis=-1)

        # Use the static num_quantiles parameter
        if num_quantiles is None:
            # Fallback: get from shape (but this might cause issues)
            current_num_quantiles = sorted_quantiles.shape[-1]
        else:
            current_num_quantiles = num_quantiles

        # Calculate how many quantiles to keep (now static)
        num_to_keep = current_num_quantiles - top_quantiles_to_drop

        # Ensure we keep at least 1 quantile
        num_to_keep = max(num_to_keep, 1)

        # 🔧 FIXED: Use dynamic_slice with static slice_sizes
        truncated_quantiles = jax.lax.dynamic_slice(
            sorted_quantiles,
            start_indices=(0, 0),
            slice_sizes=(sorted_quantiles.shape[0], num_to_keep)
        )

        # Take mean of truncated quantiles for this critic
        critic_mean = jnp.mean(truncated_quantiles, axis=-1)

        return critic_mean

    # Process all critics to get their truncated means
    critic_truncated_means = []
    for critic in critics:
        critic_mean = process_single_critic(critic)
        critic_truncated_means.append(critic_mean)

    # Stack to shape: (num_critics, batch_size)
    critic_truncated_means = jnp.stack(critic_truncated_means, axis=0)

    # 🎯 TQC CORE PRINCIPLE: Take minimum across critics (conservative/pessimistic)
    q_min_truncated = jnp.min(critic_truncated_means, axis=0)  # Shape: (batch_size,)

    # 🎭 Actor loss: maximize Q while maintaining entropy
    actor_loss = jnp.mean(entropy_coefficient * log_probs - q_min_truncated)
    actor_loss = jnp.where(jnp.isnan(actor_loss), 1.0, actor_loss)

    # 📊 Comprehensive logging info
    info = {
        # Core metrics
        'actor_loss': actor_loss,
        'entropy': -jnp.mean(log_probs),
        'temperature': entropy_coefficient,
        'entropy_coefficient': entropy_coefficient,

        # Q-value metrics
        'q_min_truncated': jnp.mean(q_min_truncated),
        'q_mean_truncated_critics': jnp.mean(critic_truncated_means),
        'q_std_across_critics': jnp.std(jnp.mean(critic_truncated_means, axis=-1)),
        'q_range_across_critics': jnp.max(jnp.mean(critic_truncated_means, axis=-1)) -
                                  jnp.min(jnp.mean(critic_truncated_means, axis=-1)),

        # Truncation info (using static values)
        'num_quantiles_kept': (num_quantiles or 0) - top_quantiles_to_drop,
        'truncation_ratio': ((num_quantiles or 0) - top_quantiles_to_drop) / (num_quantiles or 1),

        # Gradient mode tracking
        'gradient_mode': jnp.where(use_circular, 1.0, 0.0),
        'use_circular_gradients': jnp.where(use_circular, 1.0, 0.0),
    }

    # Add individual critic metrics for debugging
    for i in range(len(critics)):
        info[f'critic_{i}_q_truncated'] = jnp.mean(critic_truncated_means[i])

    # 🔧 SAFETY: Sanitize all metrics
    def sanitize_value(key, value):
        if 'temp' in key.lower():
            return jnp.where(jnp.isnan(value), 0.1, value)
        else:
            return jnp.where(jnp.isnan(value), 0.0, value)

    info = {k: sanitize_value(k, v) for k, v in info.items()}

    return actor_loss, info

@xax.jit(jit_level=JitLevel.RL_CORE)
def compute_tqc_temperature_loss(
        temperature: Temperature,
        log_probs: chex.Array,
        target_entropy: float,
) -> tuple[Array, dict[str, Array]]:
    """FIXED Temperature loss for TQC/SAC."""

    # Compute current entropy (should be negative for reasonable policies)
    current_entropy = -jnp.mean(log_probs)

    # Compute entropy error
    entropy_error = current_entropy - target_entropy

    # CORRECTED: Standard SAC temperature loss
    # The stop_gradient should only be on log_probs, not on the sum
    entropy_term = jax.lax.stop_gradient(log_probs) + target_entropy
    temp_loss = -jnp.mean(temperature.log_temp * entropy_term)

    # Handle potential NaNs
    temp_loss = jnp.where(jnp.isnan(temp_loss), 0.01, temp_loss)

    # More detailed logging
    detached_log_temp = jax.lax.stop_gradient(temperature.log_temp)
    current_temp_value = jnp.exp(detached_log_temp)
    current_temp_value = jnp.clip(current_temp_value, 0.000001, 10.0)

    # Debug info
    mean_log_prob = jnp.mean(log_probs)
    std_log_prob = jnp.std(log_probs)

    info = {
        'temp_loss': temp_loss,
        'temperature': current_temp_value,
        'entropy': current_entropy,
        'target_entropy': target_entropy,
        'entropy_error': entropy_error,
        'log_temp_raw': temperature.log_temp,
        # Additional debugging
        'mean_log_prob': mean_log_prob,
        'std_log_prob': std_log_prob,
        'entropy_term_mean': jnp.mean(entropy_term),
        'temp_loss_components': -temperature.log_temp * jnp.mean(entropy_term),
    }

    return temp_loss, info


@xax.jit(jit_level=JitLevel.RL_CORE)
def update_target_networks(
    online_critics: tuple[QuantileCritic, ...],
    target_critics: tuple[QuantileCritic, ...],
    tau: float
) -> tuple[QuantileCritic, ...]:
    """Soft update target networks - with stability checks."""
    # Clamp tau to reasonable range
    tau = jnp.clip(tau, 0.001, 0.1)

    def safe_update(target_param, online_param):
        # Check for NaN in parameters
        online_param = jnp.where(jnp.isnan(online_param), target_param, online_param)
        updated = tau * online_param + (1 - tau) * target_param
        # Final NaN check
        updated = jnp.where(jnp.isnan(updated), target_param, updated)
        return updated

    updated_targets = []
    for online_critic, target_critic in zip(online_critics, target_critics):
        updated_target = jax.tree.map(safe_update, target_critic, online_critic)
        updated_targets.append(updated_target)

    return tuple(updated_targets)


def get_truncated_quantiles(
    critics: tuple[QuantileCritic, ...],
    observations: chex.Array,
    actions: chex.Array,
    top_quantiles_to_drop: int,
) -> Array:
    """Helper function to get truncated quantiles from all critics."""
    dummy_carry = jnp.zeros(1)

    # Collect quantiles from all critics
    all_quantiles = []
    for critic in critics:
        quantiles, _ = critic.forward(observations, actions, dummy_carry)
        quantiles = jnp.where(jnp.isnan(quantiles), 0.0, quantiles)
        all_quantiles.append(quantiles)

    # Stack quantiles: (num_critics, batch_size, num_quantiles)
    all_quantiles = jnp.stack(all_quantiles, axis=0)

    # Concatenate and sort to select lowest quantiles
    concatenated_quantiles = all_quantiles.transpose(1, 0, 2).reshape(
        all_quantiles.shape[1], -1
    )

    sorted_quantiles = jnp.sort(concatenated_quantiles, axis=-1)
    num_to_keep = concatenated_quantiles.shape[-1] - top_quantiles_to_drop

    # Use dynamic slice instead of [:, :num_to_keep]
    truncated_quantiles = jax.lax.dynamic_slice(
        sorted_quantiles,
        start_indices=(0, 0),
        slice_sizes=(sorted_quantiles.shape[0], num_to_keep)
    )

    return jnp.mean(truncated_quantiles, axis=-1)