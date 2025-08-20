"""Utility functions and constants for TQC implementation - WITH AUTO ENTROPY SUPPORT."""

import math
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

# Safety constants
MAX_EXPONENTIAL_INPUT = 20.0
EPS = 1e-8
MAX_CONTROL_EFFORT = 100.0
MAX_JOINT_VELOCITY = 20.0

# Joint position targets for humanoid
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", 0),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", 0),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", 0),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", 0),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", 0),
    ("dof_right_hip_roll_03", 0),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", 0),
    ("dof_right_ankle_02", 0),
    ("dof_left_hip_pitch_04", 0),
    ("dof_left_hip_roll_03", 0),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", 0),
    ("dof_left_ankle_02", 0),
]

def safe_exp(x: Array, max_input: float = MAX_EXPONENTIAL_INPUT) -> Array:
    """Exponential with overflow protection."""
    clipped_x = jnp.clip(x, -max_input, max_input)
    return jnp.exp(clipped_x)


def safe_divide(numerator: Array, denominator: Array, eps: float = EPS) -> Array:
    """Division with numerical stability."""
    return numerator / (denominator + eps)


def safe_norm(x: Array, axis: int = -1, eps: float = EPS) -> Array:
    """L2 norm with numerical stability."""
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)


def create_tqc_opt_state(
    actor_opt_state: optax.OptState,
    critics_opt_state: optax.OptState,  # For multiple critics
    temp_opt_state: Optional[optax.OptState],  # Can be None if not using auto entropy
    buffer_state: Any
) -> Tuple[Union[Tuple[optax.OptState, optax.OptState], Tuple[optax.OptState, optax.OptState, optax.OptState]], Any]:
    """Create TQC optimizer state that includes buffer state.

    Args:
        actor_opt_state: Actor optimizer state
        critics_opt_state: Critics optimizer state
        temp_opt_state: Temperature optimizer state (None if not using auto entropy)
        buffer_state: Replay buffer state

    Returns:
        Tuple of (optimizer_states, buffer_state)
    """
    if temp_opt_state is not None:
        # Auto entropy mode: include temperature optimizer
        tqc_optimizers = (actor_opt_state, critics_opt_state, temp_opt_state)
    else:
        # Fixed entropy mode: no temperature optimizer
        tqc_optimizers = (actor_opt_state, critics_opt_state)

    return (tqc_optimizers, buffer_state)


def extract_from_tqc_opt_state(
    opt_state: Any
) -> Tuple[Optional[Union[Tuple[optax.OptState, optax.OptState], Tuple[optax.OptState, optax.OptState, optax.OptState]]], Optional[Any]]:
    """Extract TQC optimizers and buffer state from opt_state.

    Returns:
        Tuple of (optimizer_states, buffer_state) where optimizer_states can be
        either (actor, critics) or (actor, critics, temperature)
    """
    if isinstance(opt_state, tuple) and len(opt_state) == 2:
        tqc_optimizers, buffer_state = opt_state
        if isinstance(tqc_optimizers, tuple) and len(tqc_optimizers) in [2, 3]:
            # Valid: either (actor, critics) or (actor, critics, temperature)
            return tqc_optimizers, buffer_state
    return None, None


def is_tqc_opt_state(opt_state: Any) -> bool:
    """Check if opt_state is in TQC format."""
    tqc_optimizers, buffer_state = extract_from_tqc_opt_state(opt_state)
    return tqc_optimizers is not None and buffer_state is not None


def get_auto_target_entropy(action_dim: int) -> float:
    """Get automatic target entropy value following SB3 convention.

    Args:
        action_dim: Dimension of action space

    Returns:
        Target entropy value (-action_dim)
    """
    return -float(action_dim)