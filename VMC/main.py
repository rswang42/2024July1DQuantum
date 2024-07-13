"""VMC for 1d quantum"""

# %%
import time
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.append("../")
jax.config.update("jax_enable_x64", True)
from VMC.utils import *


if __name__ == "__main__":
    # Plotting Settings
    plt.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams["figure.dpi"] = 600

    GS = True
    GS = False

    if GS:
        key = jax.random.PRNGKey(42)
        print("=======================GS=========================")
        # System settings
        test = False
        batch_size = 50000
        state_indices = np.arange(1)
        thermal_step = 20
        acc_steps = 1
        mc_steps = 200
        step_size = 1.5
        num_substeps = 1  # DONT MOVE!
        init_width = 1.0
        init_learning_rate = 2e-2
        iterations = 6000
        figure_save_path = "./figure/GS/"

        if not os.path.isdir(figure_save_path):
            print("Creating Directory")
            os.makedirs(figure_save_path)
        else:
            # raise FileExistsError("Desitination already exists! Please check!")
            pass

        # Initialize flow
        model_flow = MLPFlow(out_dims=1)
        key, subkey = jax.random.split(key)
        x_dummy = jax.random.normal(subkey, dtype=jnp.float64)  # Dummy input data
        key, subkey = jax.random.split(key)
        params = model_flow.init(subkey, x_dummy)
        # Initial Jacobian
        init_jacobian = jax.jacfwd(lambda x: model_flow.apply(params, x))(x_dummy)
        print(f"Init Jacobian = \n{init_jacobian}")

        # Initialize Wavefunction
        wf_ansatz_obj = WFAnsatz(flow=model_flow)
        wf_ansatz = wf_ansatz_obj.wf_ansatz
        wf_vmapped = jax.vmap(wf_ansatz, in_axes=(None, 0, 0))

        # Plotting wavefunction
        print("Wavefunction (Initialization)")
        xmin = -10
        xmax = 10
        Nmesh = 2000
        xmesh = np.linspace(xmin, xmax, Nmesh, dtype=np.float64)
        mesh_interval = xmesh[1] - xmesh[0]
        plt.figure()
        for i in state_indices:
            wf_on_mesh = jax.vmap(wf_ansatz, in_axes=(None, 0, None))(params, xmesh, i)
            plt.plot(xmesh, wf_on_mesh, label=f"n={i}")
        plt.xlim([-10, 10])
        plt.legend()
        plt.title("Wavefunction (Initialization)")
        # plt.show()
        plt.savefig(
            f"{os.path.join(figure_save_path, "WavefunctionInitialization.png")}"
        )
        plt.close()
        print("Figure Saved.")

        # Local Energy Estimator
        energy_estimator = EnergyEstimator(wf_ansatz=wf_ansatz)
        if test:
            x_test_local_energy = np.random.uniform(-2, 2, state_indices.shape)
            x_test_local_energy = jnp.array(x_test_local_energy)
            local_energy = energy_estimator.local_energy(
                params, x_test_local_energy, state_indices
            )
            print("Testing Local Energy Estimator:\n")
            print(
                f"testing x: {x_test_local_energy}\n"
                f"state_indices: {state_indices}\n"
                f"Get local energy = {local_energy}"
            )

        # Metropolis, thermalization
        key, subkey = jax.random.split(key)
        init_x_batched = init_batched_x(
            key=subkey,
            batch_size=batch_size,
            num_of_orbs=state_indices.shape[0],
            init_width=init_width,
        )
        metropolis = Metropolis(wf_ansatz=wf_ansatz)
        metropolis_sample = metropolis.oneshot_sample
        metropolis_sample_batched = jax.jit(
            jax.vmap(metropolis_sample, in_axes=(0, None, 0, None, None, 0))
        )
        probability_batched = (
            jax.vmap(wf_vmapped, in_axes=(None, 0, None))(
                params, init_x_batched, state_indices
            )
            ** 2
        )
        xs = init_x_batched

        print("Thermalization...")
        pmove = 0
        t0 = time.time()
        key, xs, probability_batched, step_size, pmove = mcmc(
            steps=thermal_step,
            num_substeps=num_substeps,
            metropolis_sampler_batched=metropolis_sample_batched,
            key=key,
            xs_batched=xs,
            state_indices=state_indices,
            params=params,
            probability_batched=probability_batched,
            mc_step_size=step_size,
            pmove=pmove,
        )
        t1 = time.time()
        time_cost = t1 - t0
        print(
            f"After Thermalization:\tpmove {pmove:.2f}\t"
            f"step_size={step_size:.4f}\ttime={time_cost:.2f}s",
            flush=True,
        )

        # Loss function
        loss_energy = make_loss(
            wf_ansatz=wf_ansatz,
            local_energy_estimator=energy_estimator.local_energy,
            state_indices=state_indices,
        )
        loss_and_grad = jax.jit(
            jax.value_and_grad(loss_energy, argnums=0, has_aux=True)
        )
        (loss, energies), gradients = loss_and_grad(params, xs)
        print(f"After MCMC, with initial network, loss={loss:.2f}")
        if test:
            print(f"Energies = {energies}")

        # One Step Optimization
        optimizer = optax.adam(init_learning_rate)
        opt_state = optimizer.init(params)

        if test:
            updates, opt_state = optimizer.update(gradients, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            (loss, energies), gradients = loss_and_grad(new_params, xs)
            print(f"One step optimization with previous x, loss={loss:.2f}")
            print(f"Energies = {energies}")

        # Update
        update_obj = Update(
            mcmc_steps=mc_steps,
            state_indices=state_indices,
            batch_size=batch_size,
            num_substeps=num_substeps,
            metropolis_sampler_batched=metropolis_sample_batched,
            loss_and_grad=loss_and_grad,
            optimizer=optimizer,
            acc_steps=acc_steps,
        )
        update = jax.jit(update_obj.update)
        loss_energy_list = []
        energy_std_list = []
        for i in range(iterations):
            t0 = time.time()
            key, subkey = jax.random.split(key)
            (
                loss_energy,
                energy_std,
                xs,
                probability_batched,
                params,
                opt_state,
                pmove,
                step_size,
            ) = update(
                subkey,
                xs,
                probability_batched,
                step_size,
                params,
                opt_state,
            )
            t1 = time.time()
            print(
                f"Iter:{i}, LossE={loss_energy:.5f}({energy_std:.5f})"
                f"\t pmove={pmove:.2f}\tstepsize={step_size:.4f}"
                f"\t time={(t1-t0):.2f}s"
            )
            loss_energy_list.append(loss_energy)
            energy_std_list.append(energy_std)

        expectedenergy = 0.194
        plot_step = 10
        plt.figure()
        plt.errorbar(
            range(iterations)[::plot_step],
            loss_energy_list[::plot_step],
            energy_std_list[::plot_step],
            label="loss",
            fmt="o",
            markersize=1,
            capsize=0.4,
            elinewidth=0.3,
            linestyle="none",
        )
        plt.plot([0, iterations], [expectedenergy, expectedenergy], label="ref")
        plt.xlabel("Epoch")
        plt.ylabel("Loss,E")
        plt.xscale("log")
        plt.legend()
        # plt.show()
        plt.savefig(f"{os.path.join(figure_save_path, "Training.png")}")
        plt.close()
        print("Figure Saved.")

        # Finally Plotting Probability Density
        print("Probability Density (Trained)")
        plt.figure()
        for i in state_indices:
            wf_on_mesh = jax.vmap(wf_ansatz, in_axes=(None, 0, None))(params, xmesh, i)
            normalize_factor = (wf_on_mesh**2).sum() * mesh_interval
            prob_density = wf_on_mesh**2 / normalize_factor
            plt.plot(xmesh, prob_density, label="VMC", lw=2)
        plt.xlim([-1.5, 1.5])
        plt.legend()
        plt.ylabel(r"$\rho$")
        plt.title("Probability Density (Trained)")
        # plt.show()
        plt.savefig(
            f"{os.path.join(figure_save_path, "ProbabilityDensityAfterTraining.png")}"
        )
        plt.close()
        print("Figure Saved.")

        # Calculate last 100 epochs' energy
        last_energies = loss_energy_list[-100::]
        last_std = energy_std_list[-100::]
        energy_final = np.mean(last_energies)
        std_final = np.mean(last_std)
        print(f"After training, get energy = {energy_final:.5f}({std_final:.5f})")
        filepath = os.path.join(figure_save_path, "Energy.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Energy={energy_final:.5f}({std_final:.5f})\n")

        print("=======================GS=========================")

    else:
        key = jax.random.PRNGKey(42)
        num_of_excitation_orbs = 2
        print("======================================")
        print(f"First {num_of_excitation_orbs} Excitation States")
        print("======================================")
        # System settings
        test = False
        batch_size = 50000
        inference_batch_size = 100000
        state_indices = np.arange(num_of_excitation_orbs)
        thermal_step = 20
        inference_thermal_step = 50
        acc_steps = 1
        mc_steps = 200
        step_size = 1.5
        num_substeps = 1  # DONT MOVE!
        init_width = 1.0
        init_learning_rate = 2e-2
        iterations = 10000
        figure_save_path = f"./figure/Excit{num_of_excitation_orbs}/"

        if not os.path.isdir(figure_save_path):
            print("Creating Directory")
            os.makedirs(figure_save_path)
        else:
            # raise FileExistsError("Desitination already exists! Please check!")
            pass

        # Initialize flow
        model_flow = MLPFlow(out_dims=1)
        key, subkey = jax.random.split(key)
        x_dummy = jnp.array(-1.0, dtype=jnp.float64)
        key, subkey = jax.random.split(key)
        params = model_flow.init(subkey, x_dummy)
        # Initial Jacobian
        init_jacobian = jax.jacfwd(lambda x: model_flow.apply(params, x))(x_dummy)
        print(f"Init Jacobian = \n{init_jacobian}")

        # Initialize Wavefunction
        wf_ansatz_obj = WFAnsatz(flow=model_flow)
        wf_ansatz = wf_ansatz_obj.wf_ansatz
        wf_vmapped = jax.vmap(wf_ansatz, in_axes=(None, 0, 0))

        # Plotting wavefunction
        print("Wavefunction (Initialization)")
        xmin = -10
        xmax = 10
        Nmesh = 2000
        xmesh = np.linspace(xmin, xmax, Nmesh, dtype=np.float64)
        mesh_interval = xmesh[1] - xmesh[0]
        plt.figure()
        for i in state_indices:
            wf_on_mesh = jax.vmap(wf_ansatz, in_axes=(None, 0, None))(params, xmesh, i)
            plt.plot(xmesh, wf_on_mesh, label=f"n={i}")
        plt.xlim([-10, 10])
        plt.legend()
        plt.title("Wavefunction (Initialization)")
        # plt.show()
        plt.savefig(
            f"{os.path.join(figure_save_path, "WavefunctionInitialization.png")}"
        )
        plt.close()
        print("Figure Saved.")

        # Local Energy Estimator
        energy_estimator = EnergyEstimator(wf_ansatz=wf_ansatz)

        # Metropolis, thermalization
        key, subkey = jax.random.split(key)
        init_x_batched = init_batched_x(
            key=subkey,
            batch_size=batch_size,
            num_of_orbs=state_indices.shape[0],
            init_width=init_width,
        )
        metropolis = Metropolis(wf_ansatz=wf_ansatz)
        metropolis_sample = metropolis.oneshot_sample
        metropolis_sample_batched = jax.jit(
            jax.vmap(metropolis_sample, in_axes=(0, None, 0, None, None, 0))
        )
        probability_batched = (
            jax.vmap(wf_vmapped, in_axes=(None, 0, None))(
                params, init_x_batched, state_indices
            )
            ** 2
        )
        xs = init_x_batched

        print("Thermalization...")
        pmove = 0
        t0 = time.time()
        key, xs, probability_batched, step_size, pmove = mcmc(
            steps=thermal_step,
            num_substeps=num_substeps,
            metropolis_sampler_batched=metropolis_sample_batched,
            key=key,
            xs_batched=xs,
            state_indices=state_indices,
            params=params,
            probability_batched=probability_batched,
            mc_step_size=step_size,
            pmove=pmove,
        )
        t1 = time.time()
        time_cost = t1 - t0
        print(
            f"After Thermalization:\tpmove {pmove:.2f}\t"
            f"step_size={step_size:.4f}\ttime={time_cost:.2f}s",
            flush=True,
        )

        # Loss function
        loss_energy = make_loss(
            wf_ansatz=wf_ansatz,
            local_energy_estimator=energy_estimator.local_energy,
            state_indices=state_indices,
        )
        loss_and_grad = jax.jit(
            jax.value_and_grad(loss_energy, argnums=0, has_aux=True)
        )
        (loss, energies), gradients = loss_and_grad(params, xs)
        print(f"After MCMC, with initial network, loss={loss:.2f}")

        # Optimizer
        optimizer = optax.adam(init_learning_rate)
        opt_state = optimizer.init(params)

        # Training
        update_obj = Update(
            mcmc_steps=mc_steps,
            state_indices=state_indices,
            batch_size=batch_size,
            num_substeps=num_substeps,
            metropolis_sampler_batched=metropolis_sample_batched,
            loss_and_grad=loss_and_grad,
            optimizer=optimizer,
            acc_steps=acc_steps,
        )
        update = jax.jit(update_obj.update)
        loss_energy_list = []
        energy_std_list = []
        for i in range(iterations):
            t0 = time.time()
            key, subkey = jax.random.split(key)
            (
                loss_energy,
                energy_std,
                xs,
                probability_batched,
                params,
                opt_state,
                pmove,
                step_size,
            ) = update(
                subkey,
                xs,
                probability_batched,
                step_size,
                params,
                opt_state,
            )
            t1 = time.time()
            print(
                f"Iter:{i}, LossE={loss_energy:.5f}({energy_std:.5f})"
                f"\t pmove={pmove:.2f}\tstepsize={step_size:.4f}"
                f"\t time={(t1-t0):.2f}s"
            )
            loss_energy_list.append(loss_energy)
            energy_std_list.append(energy_std)

        # Training Curve
        plot_step = 10
        plt.figure()
        plt.errorbar(
            range(iterations)[::plot_step],
            loss_energy_list[::plot_step],
            energy_std_list[::plot_step],
            label="loss",
            fmt="o",
            markersize=1,
            capsize=0.4,
            elinewidth=0.3,
            linestyle="none",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss,E")
        plt.xscale("log")
        plt.legend()
        # plt.show()
        plt.savefig(f"{os.path.join(figure_save_path, "Training.png")}")
        plt.close()
        print("Figure Saved.")

        # Inference
        print("...Inferencing...")
        key, subkey = jax.random.split(key)
        xs_inference = init_batched_x(
            key=subkey,
            batch_size=inference_batch_size,
            num_of_orbs=state_indices.shape[0],
            init_width=init_width,
        )
        probability_batched = (
            jax.vmap(wf_vmapped, in_axes=(None, 0, None))(
                params, xs_inference, state_indices
            )
            ** 2
        )
        print("Thermalization...")
        t0 = time.time()
        key, xs_inference, probability_batched, step_size, pmove = mcmc(
            steps=inference_thermal_step,
            num_substeps=num_substeps,
            metropolis_sampler_batched=metropolis_sample_batched,
            key=key,
            xs_batched=xs_inference,
            state_indices=state_indices,
            params=params,
            probability_batched=probability_batched,
            mc_step_size=step_size,
            pmove=pmove,
        )
        t1 = time.time()
        time_cost = t1 - t0
        print(
            f"After Thermalization:\tpmove {pmove:.2f}\t"
            f"step_size={step_size:.4f}\ttime={time_cost:.2f}s",
            flush=True,
        )

        print("Mesuring Energy...")
        (_, energies_batched), _ = loss_and_grad(params, xs_inference)
        energy_levels = jnp.mean(energies_batched, axis=0)
        energy_levels_std = jnp.sqrt(
            (jnp.mean(energies_batched**2, axis=0) - energy_levels**2)
            / (acc_steps * batch_size)
        )

        filepath = os.path.join(figure_save_path, "EnergyLevels.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            for i, energyi in enumerate(energy_levels):
                f.write(f"n={i}\tenergy={energyi:.5f}({energy_levels_std[i]:.5f})\n")

        print(f"Energy levels written to {filepath}")

        print("======================================")
        print("Excitation States")
        print("======================================")
# %%
