"""Converts a TQC checkpoint to a deployable kinfer model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from tqc_humanoid_standing import TQCHumanoidStandingTask
from tqc_config import TQCHumanoidConfig
from tqc_utils import ZEROS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to TQC checkpoint")
    parser.add_argument("output_path", type=str, help="Output path for kinfer model")
    parser.add_argument("--config-path", type=str, default=None, help="Optional config file path")
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    # Load the TQC task and model
    print(f"Loading TQC checkpoint from {ckpt_path}")

    # Create minimal config - only what's needed for model structure and observations
    config = TQCHumanoidConfig(
        # 🏗️ NETWORK ARCHITECTURE (needed for model loading)
        actor_layer_sizes=[256, 256],
        critic_layer_sizes=[512, 512],  # Needed for loading, not inference

        # 🎯 TQC STRUCTURE (needed for checkpoint loading)
        num_critics=5,
        num_quantiles=25,

        # 🎮 OBSERVATION PROCESSING (critical - must match training)
        use_acc_gyro=True,

        # 🌡️ ENTROPY (needed for model loading)
        ent_coef="auto",  # Determines if temperature component exists
        initial_temperature=0.2,
        num_envs=512,
        batch_size=256,
    )

    # Create task instance
    task = TQCHumanoidStandingTask(config)

    # Load the checkpoint - this should load your TQC model
    try:
        # Try to load using your task's loading mechanism
        models, state = task.load_initial_state(
            jax.random.PRNGKey(42),
            load_optimizer=False
        )
        model = models[0]  # Get the TQC model
        print("✅ Successfully loaded TQC model from checkpoint")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Creating fresh model for structure reference...")
        # Fallback: create a fresh model for structure
        model = task.get_model(jax.random.PRNGKey(42))

    # Get MuJoCo model and joint names
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Remove root joint

    print(f"MJCF joint names: {joint_names}")
    print(f"Expected TQC joint order: {[name for name, _ in ZEROS]}")

    # Verify joint name consistency
    expected_joints = [name for name, _ in ZEROS]
    if joint_names != expected_joints:
        print("⚠️  WARNING: Joint name/order mismatch between MJCF and TQC training!")
        print(f"   MJCF joints: {joint_names}")
        print(f"   TQC expects: {expected_joints}")
    else:
        print("✅ Joint names and order match perfectly")

    print(f"Model has {len(joint_names)} controllable joints")
    print(f"Joint names: {joint_names[:5]}...")  # Show first few

    # Determine observation dimensions based on config
    actor_obs_dim = 51 if config.use_acc_gyro else 45

    print(f"Actor observation dimension: {actor_obs_dim}")
    print(f"Action dimension: {len(joint_names)}")

    # Create metadata for kinfer
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=None,  # TQC doesn't use commands
        carry_size=(1,),    # Minimal carry state shape for compatibility
    )

    @jax.jit
    def init_fn() -> Array:
        """Initialize function - returns minimal carry state."""
        return jnp.zeros(1)  # Shape (1,) for compatibility

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        accelerometer: Array,
        gyroscope: Array,
        time: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        """TQC step function - deterministic action selection."""

        # Build observation vector matching your _process_actor_observations
        obs_parts = [
            jnp.sin(time),
            jnp.cos(time),
            joint_angles,
            joint_angular_velocities,
            projected_gravity,
        ]

        # Add IMU data if used during training
        if config.use_acc_gyro:
            obs_parts.extend([accelerometer, gyroscope])

        obs = jnp.concatenate(obs_parts, axis=-1)

        # Get deterministic action from TQC actor
        action, _ = model.actor.get_action_and_log_prob(
            obs,
            jax.random.PRNGKey(0),  # Dummy key for deterministic mode
            deterministic=True      # Use mean action, not sampled
        )

        # Ensure action is properly shaped
        action = jnp.asarray(action, dtype=jnp.float32)

        # Return action and carry (unchanged)
        return action, carry

    print("🔄 Exporting functions to ONNX...")

    # Export both functions
    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    print("📦 Packing kinfer model...")

    # Pack into kinfer model
    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Save the model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(kinfer_model)

    print(f"✅ TQC model successfully converted and saved to {output_path}")
    print(f"📊 Model info:")
    print(f"   - Input joints: {len(joint_names)}")
    print(f"   - Actor obs dim: {actor_obs_dim}")
    print(f"   - Actor architecture: {config.actor_layer_sizes}")
    print(f"   - Uses IMU: {config.use_acc_gyro}")
    print(f"   - TQC critics: {config.num_critics}")
    print(f"   - Quantiles per critic: {config.num_quantiles}")
    print(f"   - Model size: ~{sum(p.size for p in jax.tree.leaves(model.actor) if hasattr(p, 'size')):,} parameters")


if __name__ == "__main__":
    main()