"""VMC for 1d quantum"""

import argparse

import matplotlib.pyplot as plt
import jax

from VMC.utils import training_kernel

# jax.config.update("jax_enable_x64", True)

# Plotting Settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 600


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="1D VMC")

    parser.add_argument(
        "--log_domain", action="store_true", help="True for working in log domain"
    )
    parser.add_argument(
        "--ferminet_loss",
        action="store_true",
        help="Set this to use FermiNet style loss.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="Energy Gradient Clipping Factor, None for not clipping",
    )
    parser.add_argument(
        "--wfclip",
        type=float,
        default=None,
        help="WaveFunction Gradient Clipping Factor (to network params)",
    )
    parser.add_argument(
        "--state_indices",
        type=int,
        nargs="+",
        default=None,
        help="Choose the states' quamtum number to be trained."
        "0 for ground state, 1 for the first excited state, etc.",
    )
    parser.add_argument(
        "--ckpt_filename",
        type=str,
        default=None,
        help="Designate for continue from chekpoint",
    )
    input_args = parser.parse_args()

    log_domain = input_args.log_domain
    ferminet_loss = input_args.ferminet_loss
    clip_factor = input_args.clip
    wf_clip_factor = input_args.wfclip
    state_indices = input_args.state_indices
    ckpt_filename = input_args.ckpt_filename

    total_num_of_states = len(state_indices)

    if total_num_of_states <= 0:
        raise ValueError("Total Number of states must larger than 0!")
    if not isinstance(total_num_of_states, int):
        raise TypeError("total number of states must be integer!")

    version = "sigmoid"
    version = "FixAcceptCount"
    version = "testHarmonic3"
    version = "x^2+10x^4"
    version = "testMLPPositiveInit"
    version = "OriginHermite"
    # version = "x^4_16-x^2_2-x"
    # version = "test/testplot"
    version = "0-1Benchmark"
    version = "NonMonotonous"
    version = "tanh"
    version = "swish"
    # version = "linear"
    version = "New0-1Benchmark"
    version = "NewTanh"
    version = "NewSigmoid"
    version = "Rotated"
    version = "Rotated30"
    version = "NoRotate"

    # Global System settings
    batch_size = 200
    mlp_width = 500
    mlp_depth = 2
    init_learning_rate = 5e-3
    # NOTE: Sufficient MCMC Sampling
    # is VITAL!
    # For example, calculating 0,1
    # The suggesting batch-mc_step-step_size combination is
    # batch_size=200 with mc_step=1500 and step_size=0.3
    mc_steps = 1500
    step_size = 0.3
    thermal_step = 100
    acc_steps = 1
    init_width = 1.0
    iterations = 10000
    inference_thermal_step = 1000
    params_init_width = {"kernel": 0.0005, "bias": 0.0}

    figure_save_path = f"./figure/{version}/StateIndices{state_indices}/"

    key = jax.random.PRNGKey(42)

    # End of configuration
    training_args = {
        "key": key,
        "batch_size": batch_size,
        "state_indices": state_indices,
        "thermal_step": thermal_step,
        "acc_steps": acc_steps,
        "mc_steps": mc_steps,
        "step_size": step_size,
        "init_width": init_width,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "init_learning_rate": init_learning_rate,
        "iterations": iterations,
        "inference_thermal_step": inference_thermal_step,
        "figure_save_path": figure_save_path,
        "log_domain": log_domain,
        "ferminet_loss": ferminet_loss,
        "clip_factor": clip_factor,
        "wf_clip_factor": wf_clip_factor,
        "params_init_width": params_init_width,
        "ckpt_filename": ckpt_filename,
    }

    training_kernel(
        args=training_args,
    )


if __name__ == "__main__":
    main()
