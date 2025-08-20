"""TQC Standing Training - HoST Style with staged progression."""

from tqc_config import TQCHumanoidConfig
from tqc_humanoid_standing import TQCHumanoidStandingTask  # Import the new task
import jax.numpy as jnp



def main_standing():
    """Launch TQC standing training with HoST-style staged progression."""


    TQCHumanoidStandingTask.launch(
        TQCHumanoidConfig(

            critic_updates_per_step=4,
            # 🏗️ NETWORK ARCHITECTURE
            actor_layer_sizes=[256, 256  ],
            critic_layer_sizes=[512, 512  ],

            # 🎯 TQC PARAMETERS
            num_critics=5,
            num_quantiles=25,
            top_quantiles_to_drop=2,


            learning_rate_actor=1e-4,
            learning_rate_critic=2e-4,
            learning_rate_temp=3e-4,
            #use_circular_gradients=False,  # Use stable detached gradients
            #learning_rate_temp_circular=3e-4,
            # 🌡️ ENTROPY SETTINGS
            #target_entropy="auto",  # -action_dim
            target_entropy="auto",  # -action_dim
            ent_coef="auto",        # Learnable temperature
            initial_temperature=0.2,  # Lower initial temperature for more focused exploration


            # 📊 TRAINING SCALE
            num_envs=512,            # Good balance for standing task
            batch_size=256,
            buffer_size=100_000,     # Sufficient for standing patterns
            min_buffer_size=2000,    # Start training earlier

            # 🔄 TRAINING FREQUENCY
            train_freq=1,
            gradient_steps=8,        # Moderate gradient steps
            discount_factor=0.99,
            soft_update_rate=0.005,

            # 🕒 EPISODE LENGTH
            rollout_length_seconds=3.0,  # Longer episodes for standing success

            # 🎮 ENVIRONMENT SETTINGS
            use_acc_gyro=True,

            # for traning
            dt=0.004,
            ctrl_dt=0.02,
            iterations=6,
            ls_iterations=6,

            #for eval
            #dt=0.002,
            #ctrl_dt=0.02,
            #iterations=8,
            #ls_iterations=8,

            # 🔧 STABILITY SETTINGS
            gradient_clip_norm=0.5,
            enable_gradient_clipping=True,
            enable_nan_detection=True,

            # 💾 CHECKPOINTING
            save_every_n_seconds=500,  # Save every 3 minutes
            keep_last_n_checkpoints=3,

            # 📈 LOGGING
            valid_every_n_seconds=200,  # Validate every 5 minutes
            render_length_seconds=3.0,  # Show full standing attempts

            # 🎥 RENDERING
            render_track_body_id=0,
            render_distance=2.4,
            render_azimuth=45.0,
            render_elevation=-15.0,
            render_lookat=[0.0, 0.0, 0.6],  # Focus on standing height
        )
    )




if __name__ == "__main__":
    main_standing()
