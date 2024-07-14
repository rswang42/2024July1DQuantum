"""VMC for 1d quantum"""

import argparse

import matplotlib.pyplot as plt
import jax

from VMC.utils import training_kernel

jax.config.update("jax_enable_x64", True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="1D VMC")

    parser.add_argument(
        "--n",
        type=int,
        default="1",
        help="total number of states",
    )
    input_args = parser.parse_args()

    total_num_of_states = input_args.n

    # Plotting Settings
    plt.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams["figure.dpi"] = 600

    key = jax.random.PRNGKey(42)

    if total_num_of_states <= 0:
        raise ValueError("Total Number of states must larger than 0!")
    elif not isinstance(total_num_of_states, int):
        raise TypeError("total number of states must be integer!")

    if total_num_of_states == 1:
        # System settings
        batch_size = 2000
        thermal_step = 20
        acc_steps = 2
        mc_steps = 20
        step_size = 1.5
        num_substeps = 1  # DONT MOVE!
        init_width = 1.5
        mlp_width = 3
        mlp_depth = 3
        init_learning_rate = 2e-2
        iterations = 10000
        figure_save_path = "./figure/GS/"
        inference_batch_size = 1000
        inference_thermal_step = 50
    else:
        # System settings
        batch_size = 2000
        thermal_step = 20
        acc_steps = 2
        mc_steps = 50
        step_size = 1.5
        num_substeps = 1  # DONT MOVE!
        init_width = 3.0
        mlp_width = 3
        mlp_depth = 5
        init_learning_rate = 2e-2
        iterations = 100000
        figure_save_path = f"./figure/Excit{total_num_of_states}/"
        inference_batch_size = 10000
        inference_thermal_step = 50

    # End of configuration
    training_args = {
        "key": key,
        "batch_size": batch_size,
        "total_num_of_states": total_num_of_states,
        "thermal_step": thermal_step,
        "acc_steps": acc_steps,
        "mc_steps": mc_steps,
        "step_size": step_size,
        "num_substeps": num_substeps,
        "init_width": init_width,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "init_learning_rate": init_learning_rate,
        "iterations": iterations,
        "inference_batch_size": inference_batch_size,
        "inference_thermal_step": inference_thermal_step,
        "figure_save_path": figure_save_path,
    }

    training_kernel(
        args=training_args,
    )


if __name__ == "__main__":
    main()
