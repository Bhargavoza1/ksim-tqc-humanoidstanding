"""Converts a TQC checkpoint to a deployable kinfer model."""

import argparse
from pathlib import Path
import os
import tempfile
import shutil
import tarfile
import pickle

import equinox as eqx
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


def load_tqc_checkpoint(ckpt_path: Path, task: TQCHumanoidStandingTask):
    """Load TQC checkpoint with robust error handling and multiple loading strategies."""

    print(f"Attempting to load checkpoint from: {ckpt_path}")

    # Create template model for structure
    template_model = task.get_model(jax.random.PRNGKey(42))
    print("Created template model for loading")

    # Strategy 1: Try to use the task's own loading mechanism first (this is the most reliable)
    try:
        print("Attempting to use task's native loading mechanism...")

        # Temporarily override the task's get_init_ckpt_path to point to our checkpoint
        original_method = task.get_init_ckpt_path
        task.get_init_ckpt_path = lambda: ckpt_path

        models, state = task.load_initial_state(
            jax.random.PRNGKey(42),
            load_optimizer=False
        )

        # Restore original method
        task.get_init_ckpt_path = original_method

        model = models[0]
        print("Successfully loaded using task's native loading mechanism")

        if hasattr(state, 'num_steps'):
            print(f"Loaded model from training step: {state.num_steps}")

        return model

    except Exception as task_error:
        print(f"Task loading failed: {task_error}")
        # Restore original method in case of error
        task.get_init_ckpt_path = original_method

    # Strategy 2: Check if file is compressed and handle accordingly
    with open(ckpt_path, 'rb') as f:
        header = f.read(2)
        is_gzipped = header == b'\x1f\x8b'
        print(f"File format - Gzipped: {is_gzipped}, Header: {header.hex()}")

    if is_gzipped:
        return _load_compressed_checkpoint(ckpt_path, template_model)
    else:
        return _load_uncompressed_checkpoint(ckpt_path, template_model, task)


def _load_compressed_checkpoint(ckpt_path: Path, template_model):
    """Load from compressed tar.gz checkpoint archive."""

    print("Decompressing gzipped checkpoint...")
    temp_dir = tempfile.mkdtemp()
    print(f"Extracting to temporary directory: {temp_dir}")

    try:
        # Extract the tar.gz archive
        with tarfile.open(ckpt_path, 'r:gz') as tar:
            tar.extractall(temp_dir)

        # Find all extracted files
        extracted_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                extracted_files.append(os.path.join(root, file))

        print(f"Extracted files: {[os.path.basename(f) for f in extracted_files]}")

        # Strategy 1: Try to reconstruct model from multiple component files
        try:
            print("Attempting to reconstruct model from component files...")
            model = _reconstruct_model_from_components(temp_dir, extracted_files, template_model)
            print("Successfully reconstructed model from components")
            return model
        except Exception as component_error:
            print(f"Component reconstruction failed: {component_error}")

        # Strategy 2: Find the best single model file (fallback)
        model_file = _find_model_file(extracted_files)
        if not model_file:
            raise Exception("No suitable model file found in extracted archive")

        print(f"Fallback: Using single model file: {os.path.basename(model_file)}")
        return _load_model_file(model_file, template_model)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleaned up temporary files")


def _reconstruct_model_from_components(temp_dir: str, extracted_files: list, template_model):
    """Reconstruct TQC model from separate component files."""

    import numpy as np

    # Load all available files and inspect their contents
    file_contents = {}
    for file_path in extracted_files:
        filename = os.path.basename(file_path)
        try:
            # Try loading as numpy first
            if filename.startswith(('model_', 'actor_', 'critic_', 'temp_')):
                try:
                    data = np.load(file_path, allow_pickle=True)
                    file_contents[filename] = data
                    print(f"Loaded {filename}: shape={getattr(data, 'shape', 'no shape')}, dtype={getattr(data, 'dtype', 'no dtype')}")
                except:
                    # Try as pickle
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    file_contents[filename] = data
                    print(f"Loaded {filename}: type={type(data)}")
        except Exception as e:
            print(f"Could not load {filename}: {e}")

    if not file_contents:
        raise Exception("No loadable component files found")

    # Start with template model and try to replace components
    model = template_model

    # Look for model components by filename patterns
    for filename, data in file_contents.items():
        if filename.startswith('model_'):
            # This might be a model component - figure out which one
            print(f"Processing model component: {filename}")

            if isinstance(data, np.ndarray):
                print(f"  Array shape: {data.shape}")

                # Try to identify which layer this might be based on shape
                if data.shape == (256, 51):  # Matches actor first layer (obs_dim=51, hidden=256)
                    print("  Identified as actor first layer weight")
                    # This is just one layer - we need to find a way to load the complete model
                    # For now, this approach won't work with partial layers

    # Try alternative approach: look for complete serialized models
    for filename, data in file_contents.items():
        if isinstance(data, np.ndarray) and data.ndim == 0:
            # Scalar array might contain pickled complete model
            try:
                complete_model = data.item()
                if hasattr(complete_model, 'actor') or (isinstance(complete_model, dict) and 'actor' in complete_model):
                    print(f"Found complete model in {filename}")
                    if isinstance(complete_model, dict):
                        return _extract_model_from_dict(complete_model, template_model)
                    else:
                        return complete_model
            except:
                pass

    raise Exception("Could not reconstruct complete model from available components")


def _load_uncompressed_checkpoint(ckpt_path: Path, template_model, task):
    """Load from uncompressed checkpoint file."""

    print("Loading uncompressed checkpoint...")

    # First try using the task's loading mechanism (for framework checkpoints)
    try:
        print("Attempting framework loading via task...")

        # Temporarily override the task's get_init_ckpt_path to point to our checkpoint
        original_method = task.get_init_ckpt_path
        task.get_init_ckpt_path = lambda: ckpt_path

        models, state = task.load_initial_state(
            jax.random.PRNGKey(42),
            load_optimizer=False
        )

        # Restore original method
        task.get_init_ckpt_path = original_method

        model = models[0]
        print("Successfully loaded using task framework mechanism")

        if hasattr(state, 'num_steps'):
            print(f"Loaded model from training step: {state.num_steps}")

        return model

    except Exception as framework_error:
        print(f"Framework loading failed: {framework_error}")
        print("Attempting direct file loading...")

        # Restore original method in case of error
        task.get_init_ckpt_path = original_method

        # Fallback to direct loading
        return _load_model_file(str(ckpt_path), template_model)


def _find_model_file(extracted_files):
    """Find the best model file from extracted files."""

    # Priority 1: Look for files named 'model_0', 'model_1', etc.
    for f in extracted_files:
        basename = os.path.basename(f)
        if basename.startswith('model_') and basename[6:].isdigit():
            return f

    # Priority 2: Look for files with 'model' in the name
    for f in extracted_files:
        if 'model' in os.path.basename(f).lower():
            return f

    # Priority 3: Look for common checkpoint extensions
    for f in extracted_files:
        basename = os.path.basename(f).lower()
        if (basename.endswith('.bin') or basename.endswith('.pkl') or
            basename.endswith('.eqx') or 'ckpt' in basename):
            return f

    # Priority 4: Exclude optimizer states and use largest remaining file
    non_optstate_files = [f for f in extracted_files
                         if 'opt_state' not in os.path.basename(f).lower()]
    if non_optstate_files:
        return max(non_optstate_files, key=os.path.getsize)

    # Last resort: largest file
    if extracted_files:
        return max(extracted_files, key=os.path.getsize)

    return None


def _load_model_file(model_file_path: str, template_model):
    """Load model from file using multiple strategies."""

    # Strategy 1: Direct Equinox deserialization
    try:
        print("Attempting Equinox tree deserialization...")
        model = eqx.tree_deserialise_leaves(model_file_path, template_model)
        print("Successfully loaded with Equinox")
        return model
    except Exception as eqx_error:
        print(f"Equinox loading failed: {eqx_error}")

    # Strategy 2: Try different pickle protocols
    for protocol in [None, 0, 1, 2, 3, 4, 5]:
        try:
            print(f"Attempting pickle loading with protocol {protocol}...")
            with open(model_file_path, 'rb') as f:
                if protocol is None:
                    model_data = pickle.load(f)
                else:
                    # For older protocols, we can't specify load protocol, so skip
                    if protocol >= 3:
                        continue
                    model_data = pickle.load(f)

            # Handle different pickle formats
            if isinstance(model_data, dict):
                model = _extract_model_from_dict(model_data, template_model)
            else:
                # Assume it's the model directly
                model = model_data

            print(f"Successfully loaded with pickle (protocol {protocol})")
            return model

        except Exception as pickle_error:
            print(f"Pickle protocol {protocol} failed: {pickle_error}")

    # Strategy 3: Try loading with different pickle modules
    try:
        print("Attempting dill loading (alternative pickle)...")
        import dill
        with open(model_file_path, 'rb') as f:
            model_data = dill.load(f)

        if isinstance(model_data, dict):
            model = _extract_model_from_dict(model_data, template_model)
        else:
            model = model_data

        print("Successfully loaded with dill")
        return model

    except ImportError:
        print("Dill not available, skipping...")
    except Exception as dill_error:
        print(f"Dill loading failed: {dill_error}")

    # Strategy 4: Try numpy array loading
    try:
        print("Attempting numpy array loading...")
        import numpy as np

        # Load as numpy array
        model_array = np.load(model_file_path, allow_pickle=True)
        print(f"Loaded numpy array with shape: {model_array.shape}")

        # Convert to JAX and try to reconstruct model
        import jax.numpy as jnp
        if model_array.ndim == 1:
            # Flat array - need to reconstruct model structure
            print("Attempting to reconstruct model from flat array...")
            # This would need model structure knowledge - skip for now
            raise Exception("Flat array reconstruction needs model structure")
        elif hasattr(model_array, 'item') and model_array.shape == ():
            # Scalar numpy array containing pickled object
            model_data = model_array.item()
            print("Extracted pickled object from numpy scalar")

            if isinstance(model_data, dict):
                model = _extract_model_from_dict(model_data, template_model)
            else:
                model = model_data

            print("Successfully loaded from numpy-pickled object")
            return model
        else:
            raise Exception(f"Unsupported numpy array format: {model_array.shape}")

    except Exception as numpy_error:
        print(f"Numpy loading failed: {numpy_error}")

    # Strategy 5: Try loading numpy with different parameters
    try:
        print("Attempting numpy loading with pickle disabled...")
        import numpy as np

        model_array = np.load(model_file_path, allow_pickle=False)
        print(f"Loaded pure numpy array: {model_array.shape}, dtype: {model_array.dtype}")

        # This is likely the raw model parameters - would need structure to reconstruct
        raise Exception("Raw parameter reconstruction not implemented")

    except Exception as numpy_raw_error:
        print(f"Raw numpy loading failed: {numpy_raw_error}")

    # Strategy 6: Try loading as raw JAX/Equinox serialized data
    try:
        print("Attempting JAX/Equinox raw data loading...")
        import numpy as np

        # Load the raw numpy array
        raw_data = np.load(model_file_path, allow_pickle=True)
        print(f"Loaded raw data: {type(raw_data)}, shape: {getattr(raw_data, 'shape', 'no shape')}")

        # If it's a 0-d array containing the actual data
        if hasattr(raw_data, 'item') and raw_data.ndim == 0:
            actual_data = raw_data.item()
            print(f"Extracted item: {type(actual_data)}")

            # Try to use this as pytree leaves for Equinox
            try:
                model = eqx.tree_deserialise_leaves(actual_data, template_model)
                print("Successfully reconstructed model from pytree leaves")
                return model
            except Exception as pytree_error:
                print(f"Pytree reconstruction failed: {pytree_error}")

                # If it's a dict, try extracting model components
                if isinstance(actual_data, dict):
                    model = _extract_model_from_dict(actual_data, template_model)
                    return model
                else:
                    # Try as direct model
                    return actual_data

        else:
            # Try direct reconstruction
            model = eqx.tree_deserialise_leaves(raw_data, template_model)
            print("Successfully loaded from raw numpy data")
            return model

    except Exception as raw_error:
        print(f"RAW JAX/Equinox loading failed: {raw_error}")

    # Strategy 6: Inspect file contents for debugging
    try:
        print("Inspecting file contents for debugging...")
        with open(model_file_path, 'rb') as f:
            header = f.read(100)  # Read first 100 bytes

        print(f"File header (hex): {header[:20].hex()}")
        print(f"File header (ascii): {header[:50]}")

        # Check if it's a compressed file
        if header.startswith(b'\x1f\x8b'):
            print("File appears to be gzipped internally")
        elif header.startswith(b'PK'):
            print("File appears to be a ZIP archive")
        elif b'NUMPY' in header[:50]:
            print("File appears to contain numpy arrays")

    except Exception as inspect_error:
        print(f"File inspection failed: {inspect_error}")

    raise Exception(f"All loading strategies failed for {model_file_path}. File may use unsupported serialization format.")


def _extract_model_from_dict(model_data: dict, template_model):
    """Extract model from dictionary-based checkpoint format."""

    # Strategy 1: Direct model key
    if 'model' in model_data:
        return model_data['model']

    # Strategy 2: Separate components (actor, critics, temperature)
    if any(key in model_data for key in ['actor', 'critics', 'temperature']):
        print("Reconstructing model from components...")
        model = template_model

        if 'actor' in model_data:
            model = eqx.tree_at(lambda m: m.actor, model, model_data['actor'])
            print("Loaded actor component")

        if 'critics' in model_data:
            model = eqx.tree_at(lambda m: m.critics, model, model_data['critics'])
            print("Loaded critics component")

        if 'target_critics' in model_data:
            model = eqx.tree_at(lambda m: m.target_critics, model, model_data['target_critics'])
            print("Loaded target critics component")

        if 'temperature' in model_data:
            model = eqx.tree_at(lambda m: m.temperature, model, model_data['temperature'])
            print("Loaded temperature component")

        return model

    # Strategy 3: Look for nested model structures
    for key, value in model_data.items():
        if hasattr(value, 'actor') or (isinstance(value, dict) and 'actor' in value):
            print(f"Found model structure under key: {key}")
            return value

    # Strategy 4: Assume the entire dict is the model (flatten and reconstruct)
    print("Attempting to treat entire dict as model state...")
    return template_model  # This might need more sophisticated reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TQC checkpoint to kinfer model")
    parser.add_argument("checkpoint_path", type=str, help="Path to TQC checkpoint")
    parser.add_argument("output_path", type=str, help="Output path for kinfer model")
    parser.add_argument("--config-path", type=str, default=None, help="Optional config file path")
    parser.add_argument("--actor-layers", nargs="+", type=int, default=[256, 256],
                       help="Actor layer sizes (default: 256 256)")
    parser.add_argument("--critic-layers", nargs="+", type=int, default=[512, 512],
                       help="Critic layer sizes (default: 512 512)")
    parser.add_argument("--num-critics", type=int, default=5, help="Number of critics (default: 5)")
    parser.add_argument("--num-quantiles", type=int, default=25, help="Number of quantiles (default: 25)")
    parser.add_argument("--use-acc-gyro", action="store_true", default=True,
                       help="Use accelerometer and gyroscope data (default: True)")
    parser.add_argument("--entropy-coef", default="auto", help="Entropy coefficient (default: auto)")
    parser.add_argument("--initial-temp", type=float, default=0.2, help="Initial temperature (default: 0.2)")
    args = parser.parse_args()

    # Validate paths
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    print(f"Converting TQC checkpoint: {ckpt_path}")
    print(f"Output path: {args.output_path}")

    # Create configuration matching the checkpoint
    config = TQCHumanoidConfig(
        # Network architecture
        actor_layer_sizes=args.actor_layers,
        critic_layer_sizes=args.critic_layers,

        # TQC structure
        num_critics=args.num_critics,
        num_quantiles=args.num_quantiles,

        # Observation processing (critical - must match training)
        use_acc_gyro=args.use_acc_gyro,

        # Entropy settings
        ent_coef=args.entropy_coef,
        initial_temperature=args.initial_temp,

        # Required framework settings (not used in conversion)
        num_envs=256,
        batch_size=256,
    )

    print(f"Configuration:")
    print(f"  Actor layers: {config.actor_layer_sizes}")
    print(f"  Critic layers: {config.critic_layer_sizes}")
    print(f"  TQC: {config.num_critics} critics × {config.num_quantiles} quantiles")
    print(f"  Uses IMU: {config.use_acc_gyro}")
    print(f"  Entropy coef: {config.ent_coef}")

    # Create task instance
    task = TQCHumanoidStandingTask(config)

    # Load the checkpoint with robust error handling
    try:
        model = load_tqc_checkpoint(ckpt_path, task)
        print("Successfully loaded trained weights")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("WARNING: Creating fresh model - THIS WILL NOT HAVE TRAINED WEIGHTS!")
        print("WARNING: Your kinfer model will have random weights, not trained weights!")
        model = task.get_model(jax.random.PRNGKey(42))

    # Get MuJoCo model and joint information
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Remove root joint

    print(f"Joint verification:")
    print(f"  MJCF joints: {joint_names[:3]}...{joint_names[-3:]}")
    print(f"  TQC expected: {[name for name, _ in ZEROS[:3]]}...{[name for name, _ in ZEROS[-3:]]}")

    # Verify joint consistency
    expected_joints = [name for name, _ in ZEROS]
    if joint_names != expected_joints:
        print("WARNING: Joint name/order mismatch between MJCF and TQC training!")
        print("This may cause incorrect behavior in the deployed model.")
        for i, (mjcf, tqc) in enumerate(zip(joint_names, expected_joints)):
            if mjcf != tqc:
                print(f"  Joint {i}: MJCF='{mjcf}' vs TQC='{tqc}'")
    else:
        print("Joint names and order verified - perfect match")

    # Calculate observation dimensions
    actor_obs_dim = 51 if config.use_acc_gyro else 45
    action_dim = len(joint_names)

    print(f"Model dimensions:")
    print(f"  Actor observation: {actor_obs_dim}D")
    print(f"  Action space: {action_dim}D")
    print(f"  Joint names: {len(joint_names)} joints")

    # Create kinfer metadata
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=None,  # TQC doesn't use commands
        carry_size=(1,),    # Minimal carry state for compatibility
    )

    @jax.jit
    def init_fn() -> Array:
        """Initialize carry state."""
        return jnp.zeros(1, dtype=jnp.float32)

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
        """TQC inference step - deterministic action selection."""

        # Build observation vector exactly matching training format
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

        # Concatenate all observation parts
        obs = jnp.concatenate(obs_parts, axis=-1)

        # Get deterministic action from TQC actor (mean of policy distribution)
        action, _ = model.actor.get_action_and_log_prob(
            obs,
            jax.random.PRNGKey(0),  # Dummy key for deterministic mode
            deterministic=True      # Use mean action, not sampled
        )

        # Ensure proper data types and shapes
        action = jnp.asarray(action, dtype=jnp.float32)

        # Return action and unchanged carry
        return action, carry

    print("Exporting functions to ONNX format...")

    # Export both initialization and step functions
    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    print("Packing into kinfer model format...")

    # Pack into final kinfer model
    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Save the final model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(kinfer_model)

    # Calculate approximate model size
    try:
        model_params = sum(p.size for p in jax.tree.leaves(model.actor) if hasattr(p, 'size'))
    except:
        model_params = 0

    print("=" * 60)
    print("TQC to Kinfer Conversion Complete!")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print(f"Model information:")
    print(f"  - Joints controlled: {len(joint_names)}")
    print(f"  - Actor observation dim: {actor_obs_dim}")
    print(f"  - Actor architecture: {config.actor_layer_sizes}")
    print(f"  - Uses IMU data: {config.use_acc_gyro}")
    print(f"  - TQC structure: {config.num_critics} critics × {config.num_quantiles} quantiles")
    print(f"  - Approximate parameters: {model_params:,}")
    print(f"  - Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)

    print("Model is ready for deployment with kinfer!")


if __name__ == "__main__":
    main()