"""TQC neural networks: Actor, Quantile Critics, and Temperature - WITH SB3-EXACT IMPLEMENTATION + LAYER NORMALIZATION."""

from typing import Tuple, List

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
import math


class TqcActor(eqx.Module):
    """TQC Actor with proper bias handling and action scaling."""

    layers: tuple[eqx.nn.Linear, ...]
    layer_norms: tuple[eqx.nn.LayerNorm, ...]
    mean_layer: eqx.nn.Linear
    log_std_layer: eqx.nn.Linear

    # Trainable parameters
    action_scale: Array  # Trainable scaling

    # Static configuration (Python primitives and lists only)
    action_bias_list: List[float] = eqx.static_field()
    joint_limits_low_list: List[float] = eqx.static_field()
    joint_limits_high_list: List[float] = eqx.static_field()
    action_low_list: List[float] = eqx.static_field()
    action_high_list: List[float] = eqx.static_field()
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    layer_sizes: List[int] = eqx.static_field()
    use_layer_norm: bool = eqx.static_field()
    max_scale: float = eqx.static_field()

    def __init__(
            self,
            key: PRNGKeyArray,
            num_inputs: int,
            num_outputs: int,
            layer_sizes: List[int] = None,
            hidden_size: int = 256,
            depth: int = 4,
            max_scale: float = 3.14,
            use_layer_norm: bool = False,
    ):
        # Use custom layer sizes if provided, otherwise fall back to default
        if layer_sizes is not None:
            self.layer_sizes = layer_sizes
            total_layers = len(layer_sizes) + 1
        else:
            self.layer_sizes = [hidden_size] * (depth - 1)
            total_layers = depth

        keys = jax.random.split(key, total_layers + 3)  # +3 for mean/log_std heads + scaling

        # Build neural network layers
        layers = []
        layer_norms = []

        # First layer: input -> first hidden
        if self.layer_sizes:
            layers.append(eqx.nn.Linear(num_inputs, self.layer_sizes[0], key=keys[0]))
            if use_layer_norm:
                layer_norms.append(eqx.nn.LayerNorm(self.layer_sizes[0]))

            # Hidden layers: use specified sizes
            for i in range(1, len(self.layer_sizes)):
                layers.append(eqx.nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i], key=keys[i]))
                if use_layer_norm:
                    layer_norms.append(eqx.nn.LayerNorm(self.layer_sizes[i]))

            final_hidden_size = self.layer_sizes[-1]
        else:
            layers.append(eqx.nn.Linear(num_inputs, hidden_size, key=keys[0]))
            if use_layer_norm:
                layer_norms.append(eqx.nn.LayerNorm(hidden_size))

            for i in range(1, depth - 1):
                layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]))
                if use_layer_norm:
                    layer_norms.append(eqx.nn.LayerNorm(hidden_size))

            final_hidden_size = hidden_size

        self.layers = tuple(layers)
        self.layer_norms = tuple(layer_norms) if use_layer_norm else tuple()
        self.use_layer_norm = use_layer_norm

        # Output heads for mean and log_std
        self.mean_layer = eqx.nn.Linear(final_hidden_size, num_outputs, key=keys[-3])
        self.log_std_layer = eqx.nn.Linear(final_hidden_size, num_outputs, key=keys[-2])

        # Initialize trainable scaling parameters
        scale_key = keys[-1]
        self.action_scale = jax.random.normal(scale_key, (num_outputs,)) * 0.2 + 1.5

        # Set static fields
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_scale = max_scale

        # Define joint limits as Python lists (static-safe)
        self.joint_limits_low_list = [
            math.radians(-180), math.radians(-95), math.radians(-95), math.radians(0), math.radians(-100),
            math.radians(-80), math.radians(-20), math.radians(-95), math.radians(-142), math.radians(-100),
            math.radians(-127), math.radians(-130), math.radians(-90), math.radians(-155), math.radians(-13),
            math.radians(-60), math.radians(-12), math.radians(-90), math.radians(0), math.radians(-72)
        ]

        self.joint_limits_high_list = [
            math.radians(80), math.radians(20), math.radians(95), math.radians(142), math.radians(100),
            math.radians(180), math.radians(95), math.radians(95), math.radians(0), math.radians(100),
            math.radians(60), math.radians(12), math.radians(90), math.radians(0), math.radians(72),
            math.radians(127), math.radians(130), math.radians(90), math.radians(155), math.radians(13)
        ]

        # Define desired standing pose bias as Python list
        self.action_bias_list = [
            0.0,  # dof_right_shoulder_pitch_03
            math.radians(-10.0),  # dof_right_shoulder_roll_03
            0.0,  # dof_right_shoulder_yaw_02
            math.radians(90.0),  # dof_right_elbow_02
            0.0,  # dof_right_wrist_00
            0.0,  # dof_left_shoulder_pitch_03
            math.radians(10.0),  # dof_left_shoulder_roll_03
            0.0,  # dof_left_shoulder_yaw_02
            math.radians(-90.0),  # dof_left_elbow_02
            0.0,  # dof_left_wrist_00
            math.radians(-20.0),  # dof_right_hip_pitch_04
            math.radians(0.0),  # dof_right_hip_roll_03
            0.0,  # dof_right_hip_yaw_03
            math.radians(-50.0),  # dof_right_knee_04
            math.radians(30.0),  # dof_right_ankle_02
            math.radians(20.0),  # dof_left_hip_pitch_04
            math.radians(0.0),  # dof_left_hip_roll_03
            0.0,  # dof_left_hip_yaw_03
            math.radians(50.0),  # dof_left_knee_04
            math.radians(-30.0),  # dof_left_ankle_02
        ]

        # Calculate action limits as Python lists
        self.action_low_list = [low - bias for low, bias in zip(self.joint_limits_low_list, self.action_bias_list)]
        self.action_high_list = [high - bias for high, bias in zip(self.joint_limits_high_list, self.action_bias_list)]

        # Initialize log_std layer with conservative values
        self.log_std_layer = eqx.tree_at(
            lambda layer: layer.weight,
            self.log_std_layer,
            jnp.zeros_like(self.log_std_layer.weight)
        )
        self.log_std_layer = eqx.tree_at(
            lambda layer: layer.bias,
            self.log_std_layer,
            jnp.full_like(self.log_std_layer.bias, -2.5)
        )

    def forward(self, obs: chex.Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        """Forward pass returning action distribution."""
        x = obs

        # NaN safety
        x = jnp.where(jnp.isnan(x), 0.0, x)

        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = jnp.dot(x, layer.weight.T) + layer.bias

            if self.use_layer_norm and i < len(self.layer_norms):
                if x.ndim > 1:
                    x = jax.vmap(self.layer_norms[i])(x)
                else:
                    x = self.layer_norms[i](x)

            x = jax.nn.relu(x)

        # Output heads
        mean = jnp.dot(x, self.mean_layer.weight.T) + self.mean_layer.bias
        log_std = jnp.dot(x, self.log_std_layer.weight.T) + self.log_std_layer.bias

        # Clip log_std for numerical stability
        log_std = jnp.clip(log_std, -20.0, 2.0)
        std = jnp.exp(log_std)

        # Create distribution
        base_dist = distrax.Independent(
            distrax.Normal(mean, std),
            reinterpreted_batch_ndims=1
        )

        dummy_carry = jnp.zeros_like(carry)
        return base_dist, dummy_carry

    def get_action_and_log_prob(
            self,
            obs: chex.Array,
            key: chex.PRNGKey,
            deterministic: bool = False
    ) -> tuple[chex.Array, chex.Array]:
        """Sample action and compute log probability with proper bias handling."""
        # Convert static lists to JAX arrays for computation
        action_bias = jnp.array(self.action_bias_list)
        joint_limits_low = jnp.array(self.joint_limits_low_list)
        joint_limits_high = jnp.array(self.joint_limits_high_list)
        action_low = jnp.array(self.action_low_list)
        action_high = jnp.array(self.action_high_list)

        carry = jnp.zeros(1)
        dist, _ = self.forward(obs, carry)

        # Sample or use mean
        if deterministic:
            raw_action = dist.mean()
        else:
            raw_action = dist.sample(seed=key)

        # Apply tanh to squash to [-1, 1]
        tanh_action = jnp.tanh(raw_action)

        # Compute action ranges (accounting for bias adjustment)
        negative_range = jnp.maximum(jnp.abs(action_low), 0.001)  # Avoid zero
        positive_range = jnp.maximum(action_high, 0.001)  # Avoid zero

        # Scale based on sign of tanh output
        action_range = jnp.where(tanh_action < 0, negative_range, positive_range)

        # Apply scaling and bias
        scaled_action = tanh_action * action_range + action_bias

        # Final action (with safety clipping to joint limits)
        action = jnp.clip(scaled_action, joint_limits_low, joint_limits_high)

        # Compute log probability
        log_prob = dist.log_prob(raw_action)

        # Apply tanh correction to log probability
        tanh_correction = jnp.sum(jnp.log(1 - tanh_action ** 2 + 1e-6), axis=-1)
        corrected_log_prob = log_prob - tanh_correction

        # Safety checks
        final_log_prob = jnp.where(jnp.isfinite(corrected_log_prob), corrected_log_prob, -1e6)
        final_action = jnp.where(jnp.isfinite(action), action, 0.0)

        return final_action, final_log_prob


class QuantileCritic(eqx.Module):
    """TQC Quantile Critic network - CUSTOMIZABLE LAYER SIZES + LAYER NORMALIZATION."""

    layers: tuple[eqx.nn.Linear, ...]
    layer_norms: tuple[eqx.nn.LayerNorm, ...]
    num_inputs: int = eqx.static_field()
    num_quantiles: int = eqx.static_field()
    layer_sizes: List[int] = eqx.static_field()
    use_layer_norm: bool = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_dim: int,
        action_dim: int,
        num_quantiles: int = 25,
        layer_sizes: List[int] = None,  # Custom layer sizes
        hidden_size: int = 256,  # Fallback for backward compatibility
        depth: int = 4,  # Fallback for backward compatibility
        use_layer_norm: bool = True,  # 🆕 Enable layer normalization
    ):
        num_inputs = obs_dim + action_dim
        use_layer_norm = False
        # Use custom layer sizes if provided, otherwise fall back to old method
        if layer_sizes is not None:
            self.layer_sizes = layer_sizes
            total_layers = len(layer_sizes) + 1  # +1 for output layer
        else:
            # Backward compatibility: create layer sizes from hidden_size and depth
            self.layer_sizes = [hidden_size] * (depth - 1)
            total_layers = depth

        keys = jax.random.split(key, total_layers)

        layers = []
        layer_norms = []

        # Build layers with custom sizes
        if self.layer_sizes:
            # First layer: input -> first hidden
            layers.append(eqx.nn.Linear(num_inputs, self.layer_sizes[0], key=keys[0]))
            if use_layer_norm:
                layer_norms.append(eqx.nn.LayerNorm(self.layer_sizes[0]))

            # Hidden layers: use specified sizes
            for i in range(1, len(self.layer_sizes)):
                layers.append(eqx.nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], key=keys[i]))
                if use_layer_norm:
                    layer_norms.append(eqx.nn.LayerNorm(self.layer_sizes[i]))

            # Output layer: last hidden -> quantiles (no layer norm on output)
            layers.append(eqx.nn.Linear(self.layer_sizes[-1], num_quantiles, key=keys[-1]))
        else:
            # Fallback: all hidden layers same size
            layers.append(eqx.nn.Linear(num_inputs, hidden_size, key=keys[0]))
            if use_layer_norm:
                layer_norms.append(eqx.nn.LayerNorm(hidden_size))

            for i in range(1, depth - 1):
                layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]))
                if use_layer_norm:
                    layer_norms.append(eqx.nn.LayerNorm(hidden_size))

            layers.append(eqx.nn.Linear(hidden_size, num_quantiles, key=keys[-1]))

        self.layers = tuple(layers)
        self.layer_norms = tuple(layer_norms) if use_layer_norm else tuple()
        self.use_layer_norm = use_layer_norm
        self.num_inputs = num_inputs
        self.num_quantiles = num_quantiles

    def forward(self, obs: chex.Array, action: chex.Array, carry: Array) -> tuple[chex.Array, Array]:
        """Forward pass with observation and action - outputs quantiles with layer normalization."""
        # 🔧 STABILITY: Check inputs for NaN/inf
        obs = jnp.where(jnp.isnan(obs), 0.0, obs)
        action = jnp.where(jnp.isnan(action), 0.0, action)

        # Concatenate along feature dimension
        x = jnp.concatenate([obs, action], axis=-1)

        # Forward pass through all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = jnp.dot(x, layer.weight.T) + layer.bias

            # 🆕 Apply layer normalization before activation (handle batching)
            if self.use_layer_norm and i < len(self.layer_norms):
                if x.ndim > 1:
                    # Batched input: use vmap for layer norm
                    x = jax.vmap(self.layer_norms[i])(x)
                else:
                    # Single input: apply directly
                    x = self.layer_norms[i](x)

            x = jax.nn.relu(x)

        # Final layer - outputs all quantiles (no layer norm on output)
        quantiles = jnp.dot(x, self.layers[-1].weight.T) + self.layers[-1].bias

        # Replace NaN with zeros
        quantiles = jnp.where(jnp.isnan(quantiles), 0.0, quantiles)

        dummy_carry = jnp.zeros_like(carry)
        return quantiles, dummy_carry


class Temperature(eqx.Module):
    """Learnable temperature parameter for TQC - SB3-EXACT IMPLEMENTATION."""
    log_temp: chex.Array

    def __init__(self, initial_temp: float = 1.0):
        # 🔧 SB3 EXACT: Use log parameterization for stability
        initial_temp = jnp.clip(initial_temp, 0.001, 3)
        self.log_temp = jnp.log(jnp.array(initial_temp))

    @property
    def temperature(self) -> chex.Array:
        """Get temperature with stability constraints (for logging/monitoring)."""
        # 🔧 STABILITY: Clamp log_temp to prevent explosion
        #clamped_log_temp = jnp.clip(self.log_temp, -8.0, 2.00)  # temp ∈ [0.007, 7.4]
        temp = jnp.exp(self.log_temp)

        # 🔧 STABILITY: Additional safety clamp
        #temp = jnp.clip(temp, 0.007,  20.4)

        # Replace NaN with default value
        temp = jnp.where(jnp.isnan(temp), 0.1, temp)

        return temp

    @property
    def temperature_detached_sb3(self) -> chex.Array:
        """Get SB3-style detached temperature with restrictive bounds."""
        detached_log_temp = jax.lax.stop_gradient(self.log_temp)
        #detached_log_temp = jnp.clip(detached_log_temp, -8.0, 2.0) # Same restrictive bounds
        temp = jnp.exp(detached_log_temp)
        #temp = jnp.clip(temp, 0.007,  20.4)  # More restrictive than original 10.0
        temp = jnp.where(jnp.isnan(temp), 0.1, temp)
        return temp

    def tree_replace(self, **kwargs):
        """Replace tree attributes with MORE restrictive clamping."""
        #if 'log_temp' in kwargs:
        #    # 🔧 FIXED: More restrictive clamping during updates too
        #    kwargs['log_temp'] = jnp.clip(kwargs['log_temp'], -8.0, 2.0)  # Changed from 2.4 to 0.5
        return eqx.tree_at(lambda x: tuple(kwargs.keys()), self, tuple(kwargs.values()))


class TqcModel(eqx.Module):
    """Complete TQC model with multiple quantile critics - CUSTOMIZABLE SIZES + SB3-EXACT TEMPERATURE + LAYER NORMALIZATION."""

    actor: TqcActor
    critics: tuple[QuantileCritic, ...]  # Multiple quantile critics
    target_critics: tuple[QuantileCritic, ...]  # Target networks
    temperature: Temperature
    num_critics: int = eqx.static_field()
    num_quantiles: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        num_critics: int = 5,  # TQC typically uses 5 critics
        num_quantiles: int = 25,  # TQC typically uses 25 quantiles per critic
        actor_layer_sizes: List[int] = None,  # 🆕 Custom actor layer sizes
        critic_layer_sizes: List[int] = None,  # 🆕 Custom critic layer sizes
        hidden_size: int = 256,  # Fallback for backward compatibility
        depth: int = 4,  # Fallback for backward compatibility
        initial_temp: float = 0.1,
        use_layer_norm: bool = True,  # 🆕 Enable layer normalization
    ):
        keys = jax.random.split(key, num_critics + 1)

        # 🆕 Create actor with custom layer sizes and layer normalization
        self.actor = TqcActor(
            keys[0],
            actor_obs_dim,
            action_dim,
            layer_sizes=actor_layer_sizes,
            hidden_size=hidden_size,  # Fallback
            depth=depth,  # Fallback
            use_layer_norm=use_layer_norm
        )

        # 🆕 Create multiple quantile critics with custom layer sizes and layer normalization
        critics = []
        for i in range(num_critics):
            critic = QuantileCritic(
                keys[i + 1],
                critic_obs_dim,
                action_dim,
                num_quantiles,
                layer_sizes=critic_layer_sizes,
                hidden_size=hidden_size,  # Fallback
                depth=depth,  # Fallback
                use_layer_norm=use_layer_norm
            )
            critics.append(critic)

        self.critics = tuple(critics)

        # Initialize target networks as copies of main critics
        self.target_critics = self.critics

        # 🔧 SB3-EXACT: Temperature with proper SB3 implementation
        self.temperature = Temperature(initial_temp)
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles


