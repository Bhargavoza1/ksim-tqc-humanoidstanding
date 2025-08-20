"""TQC data structures and types."""

from dataclasses import dataclass
from typing import Mapping

import jax
from jaxtyping import Array


@jax.tree_util.register_dataclass
@dataclass
class TQCInputs:
    """Data needed for TQC training step."""
    actor_observations: Array  # 51-dim for actor
    critic_observations: Array  # 446-dim for critic
    actions: Array
    rewards: Array
    next_actor_observations: Array  # 51-dim for next actor
    next_critic_observations: Array  # 446-dim for next critic
    dones: Array



@jax.tree_util.register_dataclass
@dataclass
class TQCVariables:
    """Variables computed during TQC forward pass."""
    quantile_values: Array  # All quantiles from all critics
    log_probs: Array
    entropy: Array | None = None
    aux_losses: Mapping[str, Array] | None = None