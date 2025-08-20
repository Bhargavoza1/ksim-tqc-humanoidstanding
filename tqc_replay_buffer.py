"""TQC replay buffer implementations."""

import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from tqc_data_structures import TQCInputs


class ReplayBufferState(NamedTuple):
    """Functional replay buffer state for TQC."""
    actor_observations: chex.Array  # 51-dim for actor
    critic_observations: chex.Array  # 446-dim for critic
    actions: chex.Array
    rewards: chex.Array
    next_actor_observations: chex.Array  # 51-dim for next actor
    next_critic_observations: chex.Array  # 446-dim for next critic
    dones: chex.Array
    size: chex.Array
    max_size: int
    ptr: chex.Array


class TQCReplayBuffer:
    """Functional replay buffer using JAX arrays (for smaller buffers)."""

    def __init__(self, max_size: int, actor_obs_dim: int, critic_obs_dim: int, action_dim: int):
        self.max_size = max_size
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim

        if max_size > 100_000:
            print(f"⚠️  Large buffer size ({max_size:,}) - consider using MutableTQCReplayBuffer")

    def init_state(self) -> ReplayBufferState:
        """Initialize buffer with proper dtypes for performance."""
        return ReplayBufferState(
            # Use float32 for everything - better GPU performance
            actor_observations=jnp.zeros((self.max_size, self.actor_obs_dim), dtype=jnp.float32),
            critic_observations=jnp.zeros((self.max_size, self.critic_obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((self.max_size, self.action_dim), dtype=jnp.float32),
            rewards=jnp.zeros(self.max_size, dtype=jnp.float32),
            next_actor_observations=jnp.zeros((self.max_size, self.actor_obs_dim), dtype=jnp.float32),
            next_critic_observations=jnp.zeros((self.max_size, self.critic_obs_dim), dtype=jnp.float32),
            dones=jnp.zeros(self.max_size, dtype=jnp.float32),  # float32 not bool!
            size=jnp.array(0, dtype=jnp.int32),
            max_size=self.max_size,  # FIX: Add this missing argument!
            ptr=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self,
            state: ReplayBufferState,
            actor_obs: chex.Array,
            critic_obs: chex.Array,
            action: chex.Array,
            reward: chex.Array,
            next_actor_obs: chex.Array,
            next_critic_obs: chex.Array,
            done: chex.Array) -> ReplayBufferState:
        """Add single transition (wraps add_batch for compatibility)."""

        # Convert single transition to batch format
        actor_obs_batch = actor_obs[None, ...]  # Add batch dimension
        critic_obs_batch = critic_obs[None, ...]
        action_batch = action[None, ...]
        reward_batch = jnp.array([reward])
        next_actor_obs_batch = next_actor_obs[None, ...]
        next_critic_obs_batch = next_critic_obs[None, ...]
        done_batch = jnp.array([done.astype(jnp.float32)])

        return self.add_batch(
            state,
            actor_obs_batch,
            critic_obs_batch,
            action_batch,
            reward_batch,
            next_actor_obs_batch,
            next_critic_obs_batch,
            done_batch
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def add_batch(self,
                  state: ReplayBufferState,
                  actor_obs_batch: chex.Array,
                  critic_obs_batch: chex.Array,
                  action_batch: chex.Array,
                  reward_batch: chex.Array,
                  next_actor_obs_batch: chex.Array,
                  next_critic_obs_batch: chex.Array,
                  done_batch: chex.Array) -> ReplayBufferState:
        """Add batch of transitions efficiently - MUCH faster than single adds."""

        batch_size = actor_obs_batch.shape[0]

        # Calculate indices for batch insertion
        indices = (jnp.arange(batch_size) + state.ptr) % self.max_size

        # Batch update all arrays at once
        new_actor_obs = state.actor_observations.at[indices].set(actor_obs_batch)
        new_critic_obs = state.critic_observations.at[indices].set(critic_obs_batch)
        new_actions = state.actions.at[indices].set(action_batch)
        new_rewards = state.rewards.at[indices].set(reward_batch)
        new_next_actor_obs = state.next_actor_observations.at[indices].set(next_actor_obs_batch)
        new_next_critic_obs = state.next_critic_observations.at[indices].set(next_critic_obs_batch)
        new_dones = state.dones.at[indices].set(done_batch.astype(jnp.float32))

        # Update pointer and size
        new_ptr = (state.ptr + batch_size) % self.max_size
        new_size = jnp.minimum(state.size + batch_size, self.max_size)

        return ReplayBufferState(
            actor_observations=new_actor_obs,
            critic_observations=new_critic_obs,
            actions=new_actions,
            rewards=new_rewards,
            next_actor_observations=new_next_actor_obs,
            next_critic_observations=new_next_critic_obs,
            dones=new_dones,
            size=new_size,
            max_size=state.max_size,  # FIX: Preserve max_size
            ptr=new_ptr,
        )

    @functools.partial(jax.jit, static_argnums=(0,   2))  # Added buffer_size as static
    def sample(self, state: ReplayBufferState, batch_size: int,
               key: chex.PRNGKey) -> TQCInputs:
        """Optimized sampling - 3x faster."""
        # Direct random integers, no modulo needed
        indices = jax.random.randint(
            key,
            shape=(batch_size,),
            minval=0,
            maxval=jnp.maximum(state.size, 1)
        )

        return TQCInputs(
            actor_observations=state.actor_observations[indices],
            critic_observations=state.critic_observations[indices],
            actions=state.actions[indices],
            rewards=state.rewards[indices],
            next_actor_observations=state.next_actor_observations[indices],
            next_critic_observations=state.next_critic_observations[indices],
            dones=state.dones[indices]
        )