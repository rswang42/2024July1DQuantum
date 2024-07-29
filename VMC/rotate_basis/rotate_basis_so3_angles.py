"""Rotate Basis by Euler Angles"""

# %%
import sys

import scipy.optimize


import numpy as np
import jax
import jax.numpy as jnp
import optax
import scipy
import matplotlib.pyplot as plt

sys.path.append("../")

from VMC.utils import wf_base

# Plotting Settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 600


class WFAnsatzRot:
    """The wave function ansatz for rotation
    NOTE: Here only a 2-dimensional rotation!
    """

    def wf_ansatze_gs(
        self,
        theta: float,
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            theta: the rotation angle
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        amplitude = jnp.cos(theta) * wf_base(x, 0) - jnp.sin(theta) * wf_base(x, 1)
        return amplitude

    def wf_ansatze_1st(
        self,
        theta: float,
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            theta: the rotation angle
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        amplitude = jnp.sin(theta) * wf_base(x, 0) + jnp.cos(theta) * wf_base(x, 1)
        return amplitude


class EnergyEstimator:
    """Energy Estimator
    compatible with 2-dimensional rotation
    """

    def __init__(
        self,
        wf_ansatz_obj: WFAnsatzRot,
        xmesh: jax.Array,
    ) -> None:
        self.wf_ansatz_obj = wf_ansatz_obj
        self.xmesh = xmesh

    def local_kinetic_energy(
        self,
        theta: float,
        x: jax.Array,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_kinetic: the local kinetic energy.
            NOTE: returns actually the K * psi^2, for convenience for
                future integration
        """
        # NOTE: only work for one-dimensional system!
        # For higher dimension, use jax.jacrev and jax.jvp
        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_grad_func = jax.grad(gs, argnums=1)
        gs_laplacian_func = jax.grad(gs_grad_func, argnums=1)
        gs_laplacian = gs_laplacian_func(theta, x)

        first_excited = self.wf_ansatz_obj.wf_ansatze_1st
        first_grad_func = jax.grad(first_excited, argnums=1)
        first_laplacian_func = jax.grad(first_grad_func, argnums=1)
        first_laplacian = first_laplacian_func(theta, x)

        local_kinetics = -0.5 * jnp.array(
            [gs_laplacian * gs(theta, x), first_laplacian * first_excited(theta, x)]
        )
        return local_kinetics

    def local_potential_energy(
        self,
        theta: float,
        x: jax.Array,
    ) -> jax.Array:
        """Local Potential Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_potential: the local potential energy.
            NOTE: returns actually the V * psi^2, for convenience for
                future integration
        """
        gs = self.wf_ansatz_obj.wf_ansatze_gs
        first_excited = self.wf_ansatz_obj.wf_ansatze_1st
        local_potentials = 3 * x**4 + x**3 / 2 - 3 * x**2
        return jnp.array(
            [
                local_potentials * gs(theta, x) ** 2,
                local_potentials * first_excited(theta, x) ** 2,
            ]
        )

    def local_energy(
        self,
        theta: float,
        x: jax.Array,
    ) -> jax.Array:
        """Local Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_energy: the local energy.
            NOTE: returns actually the E * psi^2, for convenience for
                future integration
        """
        kin_energy = self.local_kinetic_energy(theta, x)
        pot_energy = self.local_potential_energy(theta, x)
        local_energy = kin_energy + pot_energy
        return local_energy

    def total_energy(
        self,
        theta: float,
    ) -> jax.Array:
        """Total Energy Estimated on xmesh"""
        xmesh = self.xmesh
        interval = xmesh[1] - xmesh[0]
        loc_eng_vmapped = jax.vmap(
            self.local_energy,
            in_axes=(None, 0),
        )
        energy = loc_eng_vmapped(theta, xmesh) * interval
        energy = jnp.sum(energy, axis=0)

        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_vmapped = jax.vmap(gs, in_axes=(None, 0))
        normalize_gs = jnp.sum(gs_vmapped(theta, xmesh) ** 2 * interval)

        first = self.wf_ansatz_obj.wf_ansatze_1st
        first_vmapped = jax.vmap(first, in_axes=(None, 0))
        normalize_first = jnp.sum(first_vmapped(theta, xmesh) ** 2 * interval)

        normalize_factor = jnp.array([normalize_gs, normalize_first])
        print(
            f"energy={energy}"
            #   f"normalize_factor = {normalize_factor}")
        )
        energy = energy / normalize_factor
        return jnp.sum(energy)


def two_dim_rotate():
    """The main function of 2 dimensional rotation"""
    xmin = -10
    xmax = 10
    Nmesh = 3000
    xmesh = jnp.linspace(xmin, xmax, Nmesh)
    wavefunction_obj = WFAnsatzRot()

    energy_estimator = EnergyEstimator(wavefunction_obj, xmesh=xmesh)
    thetas = np.linspace(-np.pi, np.pi, 10)

    for theta in thetas:
        total_energy = energy_estimator.total_energy(theta)
        print(f"Total energy at theta={theta}: {total_energy}")


class WFAnsatz3Rot:
    """The wave function ansatz for rotation
    NOTE: 3 dimensional rotation
    """

    def wf_ansatze_gs(
        self,
        thetas: list[float],
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            thetas: the rotation angles
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        alpha, beta, gamma = thetas
        amplitude = (
            jnp.cos(beta) * jnp.cos(gamma) * wf_base(x, 0)
            - jnp.cos(beta) * jnp.sin(gamma) * wf_base(x, 1)
            + jnp.sin(beta) * wf_base(x, 2)
        )
        return amplitude

    def wf_ansatze_1st(
        self,
        thetas: list[float],
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            thetas: the rotation angles
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        alpha, beta, gamma = thetas
        amplitude = (
            (
                jnp.sin(alpha) * jnp.sin(beta) * jnp.cos(gamma)
                + jnp.cos(alpha) * jnp.sin(gamma)
            )
            * wf_base(x, 0)
            + (
                -jnp.sin(alpha) * jnp.sin(beta) * jnp.sin(gamma)
                + jnp.cos(alpha) * jnp.cos(gamma)
            )
            * wf_base(x, 1)
            - jnp.sin(alpha) * jnp.cos(beta) * wf_base(x, 2)
        )
        return amplitude


class EnergyEstimator3Rot:
    """Energy Estimator
    NOTE: 3 dimensional rotation
    """

    def __init__(
        self,
        wf_ansatz_obj: WFAnsatz3Rot,
        xmesh: jax.Array,
    ) -> None:
        self.wf_ansatz_obj = wf_ansatz_obj
        self.xmesh = xmesh

    def local_kinetic_energy(
        self,
        thetas: list[float],
        x: jax.Array,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_kinetic: the local kinetic energy.
            NOTE: returns actually the K * psi^2, for convenience for
                future integration
        """
        # NOTE: only work for one-dimensional system!
        # For higher dimension, use jax.jacrev and jax.jvp
        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_grad_func = jax.grad(gs, argnums=1)
        gs_laplacian_func = jax.grad(gs_grad_func, argnums=1)
        gs_laplacian = gs_laplacian_func(thetas, x)

        first_excited = self.wf_ansatz_obj.wf_ansatze_1st
        first_grad_func = jax.grad(first_excited, argnums=1)
        first_laplacian_func = jax.grad(first_grad_func, argnums=1)
        first_laplacian = first_laplacian_func(thetas, x)

        local_kinetics = -0.5 * jnp.array(
            [gs_laplacian * gs(thetas, x), first_laplacian * first_excited(thetas, x)]
        )
        return local_kinetics

    def local_potential_energy(
        self,
        thetas: list[float],
        x: jax.Array,
    ) -> jax.Array:
        """Local Potential Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_potential: the local potential energy.
            NOTE: returns actually the V * psi^2, for convenience for
                future integration
        """
        gs = self.wf_ansatz_obj.wf_ansatze_gs
        first_excited = self.wf_ansatz_obj.wf_ansatze_1st
        local_potentials = 3 * x**4 + x**3 / 2 - 3 * x**2
        return jnp.array(
            [
                local_potentials * gs(thetas, x) ** 2,
                local_potentials * first_excited(thetas, x) ** 2,
            ]
        )

    def local_energy(
        self,
        thetas: list[float],
        x: jax.Array,
    ) -> jax.Array:
        """Local Energy Estimator
        NOTE: only for 0 and 1 state!
        Returns:
            local_energy: the local energy.
            NOTE: returns actually the E * psi^2, for convenience for
                future integration
        """
        kin_energy = self.local_kinetic_energy(thetas, x)
        pot_energy = self.local_potential_energy(thetas, x)
        local_energy = kin_energy + pot_energy
        return local_energy

    def total_energy(
        self,
        thetas: list[float],
    ) -> jax.Array:
        """Total Energy Estimated on xmesh"""
        xmesh = self.xmesh
        interval = xmesh[1] - xmesh[0]
        loc_eng_vmapped = jax.vmap(
            self.local_energy,
            in_axes=(None, 0),
        )
        energy = loc_eng_vmapped(thetas, xmesh) * interval
        energy = jnp.sum(energy, axis=0)

        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_vmapped = jax.vmap(gs, in_axes=(None, 0))
        normalize_gs = jnp.sum(gs_vmapped(thetas, xmesh) ** 2 * interval)

        first = self.wf_ansatz_obj.wf_ansatze_1st
        first_vmapped = jax.vmap(first, in_axes=(None, 0))
        normalize_first = jnp.sum(first_vmapped(thetas, xmesh) ** 2 * interval)

        normalize_factor = jnp.array([normalize_gs, normalize_first])
        # print(f"energy={energy}"
        #   f"normalize_factor = {normalize_factor}")
        # )
        energy = energy / normalize_factor
        return jnp.sum(energy)


def three_dim_rotate(
    energy_estimator: EnergyEstimator3Rot,
    thetas: list[float],
):
    """The main function of 3 dimensional rotation"""
    total_energy = energy_estimator.total_energy(thetas)
    return total_energy


if __name__ == "__main__":
    xmin = -10
    xmax = 10
    Nmesh = 2000
    xmesh = jnp.linspace(xmin, xmax, Nmesh)
    interval = xmesh[1] - xmesh[0]

    wavefunction_obj = WFAnsatz3Rot()
    wf_gs = wavefunction_obj.wf_ansatze_gs
    wf_gs_vmapped = jax.vmap(wf_gs, in_axes=(None, 0))
    energy_estimator = EnergyEstimator3Rot(wavefunction_obj, xmesh=xmesh)
    thetas = [0.2, 0.0, 0.1]

    totalen = three_dim_rotate(energy_estimator, thetas)
    print(f"init thetas={thetas}\ntotal energy={totalen}")

    # def minimize_func(thetas):
    #     return three_dim_rotate(energy_estimator,thetas)

    # res = scipy.optimize.minimize(minimize_func,thetas)
    # print(f"res={res}")

    params = jnp.array(thetas)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_body(i, val):
        opt_state, params = val
        grad = jax.grad(three_dim_rotate, argnums=1)(energy_estimator, params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params)

    init_val = (opt_state, params)
    (opt_state, params) = jax.lax.fori_loop(
        0,
        20000,
        update_body,
        init_val,
    )
    totalen = three_dim_rotate(energy_estimator, params)
    print(f"after optimization\n thetas={params},\n totalenergy = {totalen}")

    plt.figure()
    wf_gs_on_mesh = wf_gs_vmapped(params, xmesh)
    normalize_gs = jnp.sum(wf_gs_vmapped(params, xmesh) ** 2 * interval)
    probability_gs = wf_gs_on_mesh / normalize_gs
    plt.plot(xmesh, probability_gs)
    plt.xlim(-3.5, 3.5)

# %%
