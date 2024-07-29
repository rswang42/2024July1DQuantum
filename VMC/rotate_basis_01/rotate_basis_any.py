"""Rotate Basis With Any Unitary Transformation"""

# %%
import sys
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
import scipy
import matplotlib.pyplot as plt
import scipy.linalg

sys.path.append("../../")

from VMC.utils import wf_base
from VMC.utils import buildH
from VMC.utils import EnergyEstimator as VMCEnergyEstimator

# Plotting Settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 600


class WFAnsatzAnyRot:
    """The wave function ansatz for rotation
    NOTE: Parameterize an arbitrary rotation
    by U = scipy.linalg.expm(A - A.T)
    """

    @staticmethod
    def wf_base_indices_vmapped(x: jax.Array, indices: jax.Array) -> jax.Array:
        """Vmapped wf base w.r.t. indices

        Args:
            x: the coordinate
            indices: (n,) the state indices, for example
                [0,1,2]

        Returns:
            (n,) The corresponding array of different excited wavefunctions.
        """
        return jax.vmap(wf_base, in_axes=(None, 0))(x, indices)

    def wf_ansatze_gs(
        self,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            params: (n,n) the EXACTLY rotation matrix A,
                n for mixing n states to each expected state.
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        A = params
        n = params.shape[0]
        rot_matrix = jax.scipy.linalg.expm(A - A.T)
        gs_rot = rot_matrix[0]
        mix_indices = jnp.array(range(n))
        wf_for_mix = self.wf_base_indices_vmapped(x, mix_indices)
        amplitude = jnp.dot(gs_rot, wf_for_mix)
        return amplitude

    def wf_ansatze_1st(
        self,
        params: np.ndarray,
        x: jax.Array,
    ) -> jax.Array:
        """The basis rotation wavefunction
        NOTE: only for 0 and 1 state!

        Args:
            params: (n,n) the EXACTLY rotation matrix A,
                n for mixing n states to each expected state.
            x: the coordinate

        Returns:
            amplitude: the wavefunction
        """
        A = params
        n = params.shape[0]
        rot_matrix = jax.scipy.linalg.expm(A - A.T)
        first_rot = rot_matrix[1]
        mix_indices = jnp.array(range(n))
        wf_for_mix = self.wf_base_indices_vmapped(x, mix_indices)
        amplitude = jnp.dot(first_rot, wf_for_mix)
        return amplitude


class EnergyEstimatorAnyRot:
    """Energy Estimator
    NOTE: Parameterize an arbitrary rotation
    by U = scipy.linalg.expm(A - A.T)
    """

    def __init__(
        self,
        wf_ansatz_obj: WFAnsatzAnyRot,
        xmesh: jax.Array,
    ) -> None:
        self.wf_ansatz_obj = wf_ansatz_obj
        self.xmesh = xmesh

    def local_kinetic_energy(
        self,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: only for 0 and 1 state!

        Args:
            params: (n,n) the EXACTLY rotation matrix A,
                n for mixing n states to each expected state.
            x: the coordinate

        Returns:
            local_kinetic: the local kinetic energy.
            NOTE: returns actually the K * psi^2, for convenience for
                future integral
        """
        # NOTE: only work for one-dimensional system!
        # For higher dimension, use jax.jacrev and jax.jvp
        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_grad_func = jax.grad(gs, argnums=1)
        gs_laplacian_func = jax.grad(gs_grad_func, argnums=1)
        gs_laplacian = gs_laplacian_func(params, x)

        first_excited = self.wf_ansatz_obj.wf_ansatze_1st
        first_grad_func = jax.grad(first_excited, argnums=1)
        first_laplacian_func = jax.grad(first_grad_func, argnums=1)
        first_laplacian = first_laplacian_func(params, x)

        local_kinetics = -0.5 * jnp.array(
            [gs_laplacian * gs(params, x), first_laplacian * first_excited(params, x)]
        )
        return local_kinetics

    def local_potential_energy(
        self,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: only for 0 and 1 state!

        Args:
            params: (n,n) the EXACTLY rotation matrix A,
                n for mixing n states to each expected state.
            x: the coordinate

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
                local_potentials * gs(params, x) ** 2,
                local_potentials * first_excited(params, x) ** 2,
            ]
        )

    def local_energy(
        self,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Local Kinetic Energy Estimator
        NOTE: only for 0 and 1 state!

        Args:
            params: (n,n) the EXACTLY rotation matrix A,
                n for mixing n states to each expected state.
            x: the coordinate

        Returns:
            local_energy: the local energy.
            NOTE: returns actually the E * psi^2, for convenience for
                future integration
        """
        kin_energy = self.local_kinetic_energy(params, x)
        pot_energy = self.local_potential_energy(params, x)
        local_energy = kin_energy + pot_energy
        return local_energy

    def total_energy(
        self,
        params: jax.Array,
    ) -> jax.Array:
        """Total Energy Estimated on xmesh"""
        xmesh = self.xmesh
        interval = xmesh[1] - xmesh[0]
        loc_eng_vmapped = jax.vmap(
            self.local_energy,
            in_axes=(None, 0),
        )
        energy = loc_eng_vmapped(params, xmesh) * interval
        energy = jnp.sum(energy, axis=0)

        gs = self.wf_ansatz_obj.wf_ansatze_gs
        gs_vmapped = jax.vmap(gs, in_axes=(None, 0))
        normalize_gs = jnp.sum(gs_vmapped(params, xmesh) ** 2 * interval)

        first = self.wf_ansatz_obj.wf_ansatze_1st
        first_vmapped = jax.vmap(first, in_axes=(None, 0))
        normalize_first = jnp.sum(first_vmapped(params, xmesh) ** 2 * interval)

        normalize_factor = jnp.array([normalize_gs, normalize_first])
        # print(f"energy={energy}"
        #   f"normalize_factor = {normalize_factor}")
        # )
        energy = energy / normalize_factor
        return jnp.sum(energy)


def test_transmatrix_ortho(A: jax.Array) -> None:
    """Test the transformation matrix generated by
    A:
    U = jax.scipy.linalg.expm(A - A.T)
    is orthogonal
    NOTE: Here A is real and hence U is orthogonal
    if A is complex then we'd expect U unitary.

    Args:
        A: (n,n) the parameterized matrix to generate
            orthogonal(unitary) rotation matrix U.
    """
    n = A.shape[0]
    U = jax.scipy.linalg.expm(A - A.T)
    np.testing.assert_array_almost_equal(np.dot(U, U.T), np.eye(n), decimal=3)


def any_rotate(energy_estimator: EnergyEstimatorAnyRot, params):
    """The main function of any dimensional rotation"""
    total_energy = energy_estimator.total_energy(params)
    return total_energy


def main():
    """Man"""

    xmin = -10
    xmax = 10
    Nmesh = 2000
    xmesh = jnp.linspace(xmin, xmax, Nmesh)
    interval = xmesh[1] - xmesh[0]

    n = 30

    wavefunction_obj = WFAnsatzAnyRot()
    wf_gs = wavefunction_obj.wf_ansatze_gs
    wf_1st = wavefunction_obj.wf_ansatze_1st
    wf_gs_vmapped = jax.vmap(wf_gs, in_axes=(None, 0))
    wf_first_vmapped = jax.vmap(wf_1st, in_axes=(None, 0))
    energy_estimator = EnergyEstimatorAnyRot(wavefunction_obj, xmesh=xmesh)

    params = jnp.eye(n)

    totalen = any_rotate(energy_estimator, params)
    print("Parameterizing U = jax.scipy.linalg.expm(A - A.T)\n")
    print(f"init A=\n{params}\ntotal energy={totalen}")

    # def minimize_func(thetas):
    #     return three_dim_rotate(energy_estimator,thetas)

    # res = scipy.optimize.minimize(minimize_func,thetas)
    # print(f"res={res}")

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_body(i, val):
        opt_state, params = val
        grad = jax.grad(any_rotate, argnums=1)(energy_estimator, params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params)

    init_val = (opt_state, params)
    (opt_state, params) = jax.lax.fori_loop(
        0,
        1000,
        update_body,
        init_val,
    )
    totalen = any_rotate(energy_estimator, params)
    print("=" * 50)
    print(f"after optimization\n A=\n{params},\n ")

    print("=" * 50)
    print("Testing Orthogonality After Optimization...")
    test_transmatrix_ortho(A=params)
    print("Passed")

    print("=" * 50)
    print("Saving Params")
    filename = "RotParams.pkl"
    with open(filename, "wb") as f:
        pickle.dump(params, f)
    print("Done")

    # Finite Differential Method
    # which would be <exact> if our mesh intervals are small enough
    potential_func = VMCEnergyEstimator.local_potential_energy
    H = buildH(potential_func, xmesh, Nmesh, interval)
    exact_eigenvalues, exact_eigenvectors = scipy.linalg.eigh(H)
    trained_states_energy = exact_eigenvalues[range(2)]
    expectedenergy = np.sum(trained_states_energy)
    print(
        f"After optimization,\nTotal energy = {totalen}\nRef Energy = {expectedenergy}"
    )

    plt.figure()
    for i in range(2):
        if i == 0:
            wf_gs_on_mesh = wf_gs_vmapped(params, xmesh)
            normalize_gs = jnp.sum(wf_gs_vmapped(params, xmesh) ** 2 * interval)
            wf_gs_on_mesh = wf_gs_on_mesh**2 / normalize_gs
            plt.plot(xmesh, wf_gs_on_mesh, label=f"Rotated-n={i}", lw=2)
        if i == 1:
            wf_1st_on_mesh = wf_first_vmapped(params, xmesh)
            normalize_1st = jnp.sum(wf_first_vmapped(params, xmesh) ** 2 * interval)
            wf_1st_on_mesh = wf_1st_on_mesh**2 / normalize_1st
            plt.plot(xmesh, wf_1st_on_mesh, label=f"Rotated-n={i}", lw=2)

        exact_wf_on_mesh = exact_eigenvectors[:, i]
        normalize_factor = (exact_wf_on_mesh**2).sum() * interval
        exact_prob_density = exact_wf_on_mesh**2 / normalize_factor
        plt.plot(xmesh, exact_prob_density, "-.", label=f"Exact-n={i}", lw=1.75)
    plt.xlim([-2.0, 2.0])
    plt.legend()
    plt.ylabel(r"$\rho$")
    plt.title("Probability Density (Trained and Compare)")


if __name__ == "__main__":
    main()
# %%
