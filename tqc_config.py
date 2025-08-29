"""TQC configuration and hyperparameters - WITH CIRCULAR GRADIENT SUPPORT."""

from dataclasses import dataclass
from typing import Union, List, Optional

import xax
from ksim.task.rl import RLConfig


@dataclass
class TQCHumanoidConfig(RLConfig):
    """Complete TQC configuration extending RLConfig - with circular gradient support."""

    critic_updates_per_step: int = xax.field(
        value=2,
        help="Number of critic updates per training step (2-3 recommended for observation imbalance)",
    )

    # 🔄 CIRCULAR GRADIENT OPTIONS
    use_circular_gradients: bool = xax.field(
        value=False,
        help="Use circular gradients (non-detached temperature). EXPERIMENTAL - may be unstable.",
    )
    learning_rate_temp_circular: float = xax.field(
        value=1e-6,
        help="Much lower temperature learning rate for circular gradients.",
    )
    gradient_clip_circular: float = xax.field(
        value=0.1,
        help="Aggressive gradient clipping for circular mode.",
    )
    circular_actor_lr_scale: float = xax.field(
        value=0.1,
        help="Scale factor for actor learning rate in circular mode (0.1 = 10x lower).",
    )
    circular_critic_lr_scale: float = xax.field(
        value=0.1,
        help="Scale factor for critic learning rate in circular mode (0.1 = 10x lower).",
    )

    # 🆕 CUSTOM NETWORK ARCHITECTURES
    actor_layer_sizes: Optional[List[int]] = xax.field(
        value=None,  # Default: use hidden_size and depth
        help="Custom layer sizes for actor network, e.g., [128, 64]. If None, uses hidden_size and depth.",
    )
    critic_layer_sizes: Optional[List[int]] = xax.field(
        value=None,  # Default: use hidden_size and depth
        help="Custom layer sizes for critic networks, e.g., [512, 256, 64]. If None, uses hidden_size and depth.",
    )

    only_save_most_recent: bool = xax.field(False, help="Keep multiple checkpoints instead of just the latest")
    keep_last_n_checkpoints: int = xax.field(2, help="Number of recent checkpoints to preserve")

    # Legacy parameters (for backward compatibility)
    hidden_size: int = xax.field(
        value=256,
        help="Hidden size for networks (used if custom layer sizes not provided).",
    )
    depth: int = xax.field(
        value=4,
        help="Depth of networks (used if custom layer sizes not provided).",
    )

    # 🔧 TQC-specific parameters
    num_critics: int = xax.field(
        value=5,
        help="Number of quantile critics (TQC default: 5).",
    )
    num_quantiles: int = xax.field(
        value=25,
        help="Number of quantiles per critic (TQC default: 25).",
    )
    top_quantiles_to_drop: int = xax.field(
        value=2,
        help="Number of highest quantiles to drop for truncation (TQC default: 2).",
    )

    # 🔧 Learning rates
    learning_rate_actor: float = xax.field(
        value=3e-4,
        help="Learning rate for actor.",
    )
    learning_rate_critic: float = xax.field(
        value=3e-4,
        help="Learning rate for critics.",
    )
    learning_rate_temp: float = xax.field(
        value=3e-4,
        help="Learning rate for temperature.",
    )
    discount_factor: float = xax.field(
        value=0.99,
        help="Discount factor gamma.",
    )
    soft_update_rate: float = xax.field(
        value=0.005,
        help="Soft update rate tau for target networks.",
    )

    # 🚀 Auto entropy support
    target_entropy: Union[str, float] = xax.field(
        value="auto",
        help="Target entropy for automatic temperature tuning. Use 'auto' for -action_dim.",
    )
    ent_coef: Union[str, float] = xax.field(
        value="auto",
        help="Entropy regularization coefficient. Use 'auto' for learnable temperature.",
    )
    initial_temperature: float = xax.field(
        value=1.0,
        help="Initial temperature value when ent_coef='auto'.",
    )

    # 🔧 Gradient clipping parameters
    gradient_clip_norm: float = xax.field(
        value=0.5,
        help="Maximum gradient norm for clipping.",
    )

    # 🔧 Temperature bounds (for safety)
    min_temperature: float = xax.field(
        value=0.001,
        help="Minimum temperature value.",
    )
    max_temperature: float = xax.field(
        value=2.0,
        help="Maximum temperature value.",
    )

    # Replay buffer - SB3-like defaults
    buffer_size: int = xax.field(
        value=100_000,
        help="Replay buffer size.",
    )
    min_buffer_size: int = xax.field(
        value=100,
        help="Minimum buffer size before training.",
    )
    batch_size: int = xax.field(
        value=256,
        help="Batch size for training.",
    )
    train_freq: int = xax.field(
        value=1,
        help="Training frequency (every N environment steps).",
    )
    gradient_steps: int = xax.field(
        value=1,
        help="Number of gradient steps per training.",
    )

    # Buffer implementation
    use_mutable_buffer: bool = xax.field(
        value=True,
        help="Use mutable buffer for better performance (recommended for large buffers).",
    )
    mutable_buffer_threshold: int = xax.field(
        value=50_000,
        help="Automatically use mutable buffer for sizes above this threshold.",
    )

    # Environment-specific
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use IMU observations.",
    )

    # 🔧 Numerical stability options
    enable_gradient_clipping: bool = xax.field(
        value=True,
        help="Enable gradient clipping for numerical stability.",
    )
    enable_nan_detection: bool = xax.field(
        value=True,
        help="Enable NaN detection and replacement.",
    )
    log_prob_clip_min: float = xax.field(
        value=-20.0,
        help="Minimum log probability value (prevents -inf).",
    )
    log_prob_clip_max: float = xax.field(
        value=20.0,
        help="Maximum log probability value (prevents overflow).",
    )

    def get_target_entropy(self, action_dim: int) -> float:
        """Get target entropy value, computing auto if needed."""
        if isinstance(self.target_entropy, str) and self.target_entropy == "auto":
            return -float(action_dim)  # SB3 convention: -action_dim
        return float(self.target_entropy)

    def use_auto_entropy(self) -> bool:
        """Check if using automatic entropy coefficient."""
        return isinstance(self.ent_coef, str) and self.ent_coef == "auto"



